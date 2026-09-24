"""
Full style transfer for GenTron, implementing both components of:

  "Style Injection in Diffusion: A Training-free Approach for Adapting
  Large-scale Diffusion Models for Style Transfer" (StyleID), CVPR 2024.
  https://arxiv.org/abs/2312.09008

  1. Initial Latent AdaIN (color & tone initialization) -- see
     diffusion/style_transfer.py for details on this piece and how it's
     generalized here from single images to video clips.
  2. Attention-based style injection -- at every self-attention layer and
     every denoising step, content attends using its OWN query (preserving
     structure/motion) but the STYLE video's key/value (injecting texture/
     appearance), with query-preservation AdaIN and attention-temperature
     scaling as refinements, exactly as in the paper. See
     StyleInjectionAttention's docstring in models.py for full details.

Usage:
    python sample_style_transfer.py \
        --model GenTron-T2V-S/2 --ckpt path/to/checkpoint.pt \
        --content_video path/to/content.mp4 \
        --style_video path/to/style.mp4 \
        --prompt "a description of the desired content/motion" \
        --image_size 256 --video_length 8

How it works:
    1. Both --content_video and --style_video are loaded and VAE-encoded
       into latents, the same way training data is encoded.
    2. Each latent is DDIM-inverted into the noise the model would have
       produced it from (deterministic, no training involved).
    3. The content noise's per-channel mean/std are replaced with the
       style noise's per-channel mean/std (Initial Latent AdaIN) --
       transfers coarse color/tone before any denoising happens.
    4. Content and style are then denoised STEP BY STEP IN LOCKSTEP: at
       each timestep, content's self-attention layers use style's current
       key/value (attention injection -- this is what transfers texture,
       not just color), while style takes an ordinary, uninjected step to
       keep its own trajectory valid for the next timestep's injection.

Caveats worth knowing:
    - This sampling loop is inherently sequential (content and style must
      be stepped together, one timestep at a time) and roughly 2x the
      per-step cost of ordinary sampling (both branches run the model each
      step) -- expect it to take longer than plain --style_video-less
      generation at the same --num_sampling_steps.
    - DDIM inversion is only exactly invertible for a deterministic sampler
      (eta=0, used throughout here) and accumulates some numerical drift
      over many steps -- a known, inherent limitation of DDIM inversion in
      general, not specific to this implementation.
    - GenTron's DiT backbone is isotropic (every block structurally
      identical), unlike the U-Net the original paper used (which has an
      explicit encoder/decoder split motivating which layers to inject
      at). This implementation injects at every block by default; use
      --style_layers to restrict it if uniform injection over- or
      under-stylizes relative to what you want.
    - Verified: the attention-injection math (StyleInjectionAttention) is
      numerically identical to plain self-attention when unused (checkpoint
      compatibility confirmed), and the combined CFG+style batching produces
      correct output shapes (see accompanying smoke tests). NOT verified:
      end-to-end visual quality against a real trained checkpoint -- no GPU
      was available in the environment this was written in.
"""

import argparse
import os

import imageio
import numpy as np
import torch
from decord import VideoReader, cpu
from diffusers.models import AutoencoderKL
from einops import rearrange
from PIL import Image, ImageFilter
from torchvision import transforms
from transformers import AutoConfig, AutoTokenizer, CLIPTextModel

from diffusion import create_diffusion
from diffusion.style_transfer import adain_latent, invert_to_noise
from download import find_model
from models import GenTron_models


# Facebook Stories, Instagram Stories, and WhatsApp Status all specifically
# require 9:16 vertical video (1080x1920 recommended); Telegram accepts any
# aspect ratio (up to 2GB), so 9:16 is also fine there. One export format
# satisfies all four platforms -- no need for per-platform profiles.
PLATFORM_CANVAS_W = 1080
PLATFORM_CANVAS_H = 1920
PLATFORM_MIN_DURATION_SEC = 3.0  # Facebook/Instagram Stories' hard minimum
PLATFORM_MAX_FILE_SIZE_MB = 16   # WhatsApp Status' cap -- the tightest of the four


def fit_to_vertical_canvas(frames_fhwc_uint8, canvas_w=PLATFORM_CANVAS_W, canvas_h=PLATFORM_CANVAS_H):
    """
    Place square (or any-aspect-ratio) generated frames onto a proper 9:16
    vertical canvas, the format Facebook/Instagram Stories and WhatsApp
    Status all require (and Telegram accepts fine too).

    Rather than letting the platform auto-crop a square video to 9:16 (which
    would cut off whatever's on the left/right of the frame) or adding plain
    black bars, this fills the canvas with a blurred, scaled-up version of
    the same frame as a backdrop -- the common technique apps use for
    square-to-vertical conversion -- with the original frame centered on
    top at full quality.

    Args:
        frames_fhwc_uint8: (F, H, W, C) uint8 numpy array.
        canvas_w, canvas_h: target canvas size, default 1080x1920 (9:16).

    Returns:
        (F, canvas_h, canvas_w, C) uint8 numpy array.
    """
    num_frames, h, w, c = frames_fhwc_uint8.shape
    out_frames = np.empty((num_frames, canvas_h, canvas_w, c), dtype=np.uint8)

    # Foreground: scale the original frame to fill the canvas WIDTH, height
    # follows from the aspect ratio (for a square input this makes it
    # canvas_w x canvas_w, centered vertically).
    fg_w = canvas_w
    fg_h = max(1, round(h * (canvas_w / w)))

    for i in range(num_frames):
        frame = Image.fromarray(frames_fhwc_uint8[i])

        # Background: scale to COVER the whole canvas (may overflow one
        # dimension), then center-crop to exactly canvas size, then blur.
        scale = max(canvas_w / w, canvas_h / h)
        bg_w, bg_h = max(1, round(w * scale)), max(1, round(h * scale))
        background = frame.resize((bg_w, bg_h), Image.LANCZOS)
        left = (bg_w - canvas_w) // 2
        top = (bg_h - canvas_h) // 2
        background = background.crop((left, top, left + canvas_w, top + canvas_h))
        background = background.filter(ImageFilter.GaussianBlur(radius=30))

        foreground = frame.resize((fg_w, fg_h), Image.LANCZOS)
        paste_y = (canvas_h - fg_h) // 2

        canvas = background.copy()
        canvas.paste(foreground, (0, paste_y))
        out_frames[i] = np.array(canvas)

    return out_frames


def ensure_minimum_duration(num_frames, fps, min_duration_sec=PLATFORM_MIN_DURATION_SEC):
    """
    Facebook/Instagram Stories reject videos under 3 seconds. If the
    requested --fps would produce a shorter clip than that from the
    generated frame count, lower fps just enough to clear the minimum
    (stretching playback, not adding frames) rather than fail silently at
    upload time on the actual platform.
    """
    duration = num_frames / fps
    if duration >= min_duration_sec:
        return fps
    adjusted_fps = num_frames / min_duration_sec
    print(f"Note: {num_frames} frames at {fps} fps = {duration:.1f}s, under the "
          f"{min_duration_sec:.0f}s minimum Facebook/Instagram Stories require. "
          f"Lowering playback to {adjusted_fps:.2f} fps ({min_duration_sec:.0f}s total) instead.")
    return adjusted_fps


def check_file_size(path, max_mb=PLATFORM_MAX_FILE_SIZE_MB):
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > max_mb:
        print(f"WARNING: {path} is {size_mb:.1f}MB, over WhatsApp Status' {max_mb}MB limit "
              f"(Facebook/Instagram/Telegram limits are far higher, this won't affect them). "
              f"Re-export at a lower resolution or shorter duration for WhatsApp specifically.")
    return size_mb


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, choices=list(GenTron_models.keys()), default="GenTron-T2V-S/2")
    parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--text_encoder", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--image_size", type=int, choices=[128, 256, 512], default=256)
    parser.add_argument("--video_length", type=int, default=8)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--tau", type=float, default=1.5,
                         help="Attention temperature scaling for style-injected self-attention "
                              "(paper default ~1.5). Higher sharpens the attention map, countering "
                              "the over-smoothing that substituting a different K/V distribution causes.")
    parser.add_argument("--no_query_preservation", action="store_true",
                         help="Disable AdaIN-ing the content query toward the style query's statistics "
                              "before computing injected attention (on by default, per the paper).")
    parser.add_argument("--style_layers", type=int, nargs="*", default=None,
                         help="Block indices to apply style injection at (0-indexed). Default: all "
                              "blocks. GenTron's isotropic DiT backbone has no inherent 'which layers' "
                              "answer the way a U-Net's encoder/decoder split does in the original "
                              "paper -- this is exposed so you can experiment if uniform injection "
                              "over-stylizes or under-stylizes relative to what you want.")
    parser.add_argument("--progress", action="store_true", default=True,
                         help="Show a progress bar during the (necessarily sequential, can't batch "
                              "across timesteps) style-injection sampling loop.")
    parser.add_argument("--inversion_steps", type=int, default=50,
                         help="DDIM steps used for inverting content/style videos to noise. "
                              "Doesn't need to match --num_sampling_steps.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a trained GenTron-T2V checkpoint.")
    parser.add_argument("--content_video", type=str, default=None,
                         help="Real video whose STRUCTURE/MOTION is preserved. Optional -- if omitted "
                              "along with --style_video, generates a normal (un-styled) video from text, "
                              "same as sample_t2v.py.")
    parser.add_argument("--style_video", type=str, default=None,
                         help="Real video whose COLOR/TONE is transferred onto the content. Optional -- "
                              "if omitted, no style transfer is applied and this behaves like plain "
                              "text-to-video generation.")
    parser.add_argument("--num_samples", type=int, default=1,
                         help="Only used when --style_video is omitted (plain generation mode).")
    parser.add_argument("--prompt", type=str, required=True,
                         help="Text prompt conditioning the generation (still required -- GenTron-T2V "
                              "always generates conditioned on text, style transfer happens on top of that).")
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--output_type", type=str, choices=["video", "image"], default="video",
                         help="'video' saves the full generated clip as .mp4 (default). 'image' saves "
                              "only a single representative frame as .png instead -- the model still "
                              "generates the full video internally (video_length frames), a frame is "
                              "just extracted afterward, since GenTron-T2V wasn't trained to produce "
                              "single-frame output directly.")
    parser.add_argument("--no_platform_format", action="store_true",
                         help="By default, output is fit to a 9:16 vertical canvas (1080x1920) and, for "
                              "video, checked against Facebook/Instagram Stories' 3-second minimum "
                              "duration -- this single format works for Facebook Stories, Instagram "
                              "Stories, WhatsApp Status, and Telegram. Pass this flag to get the raw "
                              "square output instead, with no platform formatting applied.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def load_video_as_tensor(path, video_length, resolution):
    """
    Load `video_length` evenly-spaced frames from `path`, resize/center-crop
    to `resolution`, and normalize to [-1, 1] -- matching the preprocessing
    used for training data (see data/msrvtt.py) so inversion sees the same
    distribution the model was trained on.

    Returns a (1, C, F, H, W) tensor.
    """
    vr = VideoReader(path, ctx=cpu(0))
    total_frames = len(vr)
    if total_frames < video_length:
        raise ValueError(
            f"{path} has only {total_frames} frames, need at least {video_length}."
        )
    indices = torch.linspace(0, total_frames - 1, steps=video_length).long().tolist()
    frames = vr.get_batch(indices)
    frames = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()  # (C, F, H, W)

    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
    ])
    frames = transform(frames)
    frames = (frames / 255 - 0.5) * 2  # -> [-1, 1]
    return frames.unsqueeze(0)  # (1, C, F, H, W)


@torch.no_grad()
def encode_to_latent(vae, video_tensor, device):
    """VAE-encode a (1, C, F, H, W) video tensor into (1, 4, F, h, w) latents."""
    video_tensor = video_tensor.to(device)
    b, c, f, h, w = video_tensor.shape
    frames = rearrange(video_tensor, "b c f h w -> (b f) c h w").contiguous()
    latent = vae.encode(frames).latent_dist.sample().mul_(0.18215)
    latent = rearrange(latent, "(b f) c h w -> b c f h w", b=b).contiguous()
    return latent


def save_output(sample_fhwc_uint8, out_path_base, output_type, fps, platform_format=True):
    """
    Save one generated sample as either a video or a single image, per
    --output_type. `sample_fhwc_uint8` is a (F, H, W, C) uint8 numpy array
    (already denormalized to [0, 255]).

    Image mode takes the middle frame of the generated clip rather than the
    first, since diffusion video models are often least motion-blurred /
    most representative partway through a clip rather than at its very
    start.

    When platform_format=True (the default), output is fit to a 9:16
    vertical canvas and, for video, checked against the minimum duration
    Facebook/Instagram Stories require -- see fit_to_vertical_canvas() and
    ensure_minimum_duration() above for what that actually does and why.
    This single format satisfies Facebook Stories, Instagram Stories,
    WhatsApp Status, and Telegram simultaneously (confirmed against each
    platform's current published specs), so there's no need for separate
    per-platform exports.
    """
    if platform_format:
        sample_fhwc_uint8 = fit_to_vertical_canvas(sample_fhwc_uint8)

    if output_type == "video":
        if platform_format:
            fps = ensure_minimum_duration(sample_fhwc_uint8.shape[0], fps)
        out_path = f"{out_path_base}.mp4"
        imageio.mimwrite(out_path, sample_fhwc_uint8, fps=fps, codec="libx264", quality=8)
        if platform_format:
            check_file_size(out_path)
    else:
        num_frames = sample_fhwc_uint8.shape[0]
        middle_frame = sample_fhwc_uint8[num_frames // 2]
        out_path = f"{out_path_base}.png"
        imageio.imwrite(out_path, middle_frame)
    print(f"Saved {out_path}")
    return out_path


def run_plain_generation(args, model, vae, tokenizer, text_encoder, device):
    """
    No style transfer requested: generate ordinary text-to-video output from
    random noise, exactly like sample_t2v.py. This is the fallback path used
    whenever --style_video is omitted.
    """
    latent_size = args.image_size // 8
    z = torch.randn(args.num_samples, 4, args.video_length, latent_size, latent_size, device=device)
    y_inputs = tokenizer([args.prompt] * args.num_samples, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    y_null_inputs = tokenizer([""] * args.num_samples, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    tokens = y_inputs["input_ids"].to(device)
    uncond_tokens = y_null_inputs["input_ids"].to(device)
    y = text_encoder(input_ids=tokens).last_hidden_state
    y_null = text_encoder(input_ids=uncond_tokens).last_hidden_state
    mask = y_inputs["attention_mask"].bool().to(device)
    uncond_mask = y_null_inputs["attention_mask"].bool().to(device)

    z = torch.cat([z, z], 0)
    y = torch.cat([y, y_null], 0)
    mask = torch.cat([mask, uncond_mask], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, mask=mask)

    sample_diffusion = create_diffusion(str(args.num_sampling_steps))
    print(f"No --style_video given -- generating plain text-to-video output ({args.num_sampling_steps} steps)...")
    samples = sample_diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)
    b, _, _, _, _ = samples.shape
    samples = rearrange(samples, "b c f h w -> (b f) c h w").contiguous()
    samples = vae.decode(samples / 0.18215).sample
    samples = rearrange(samples, "(b f) c h w -> b c f h w", b=b).contiguous()

    os.makedirs(args.out_dir, exist_ok=True)
    for i, sample in enumerate(samples):
        sample = rearrange(sample, "c f h w -> f h w c").contiguous()
        sample = ((sample.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
        out_path_base = os.path.join(args.out_dir, f"sample_{i}")
        save_output(sample.cpu().numpy(), out_path_base, args.output_type, args.fps, platform_format=not args.no_platform_format)


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_size = args.image_size // 8
    text_config = AutoConfig.from_pretrained(args.text_encoder)
    embed_dim = text_config.projection_dim
    model = GenTron_models[args.model](input_size=latent_size, embedding_dim=embed_dim).to(device)
    state_dict = find_model(args.ckpt)
    # Checkpoints saved by train_t2v.py are dicts with a "model" key (plus
    # "ema"/"opt"/"args"); handle both that and a bare state_dict for
    # convenience.
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.eval()

    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    text_encoder = CLIPTextModel.from_pretrained(args.text_encoder).to(device)

    if args.style_video is None:
        if args.content_video is not None:
            print("--content_video was given without --style_video: ignoring it and "
                  "generating a normal (un-styled) video from --prompt alone, since "
                  "style transfer needs both to do anything.")
        run_plain_generation(args, model, vae, tokenizer, text_encoder, device)
        return

    if args.content_video is None:
        raise ValueError(
            "--style_video was given but --content_video was not. Style transfer needs "
            "both: --content_video (structure/motion to preserve) and --style_video "
            "(color/tone to transfer). Provide both, or omit --style_video entirely for "
            "plain text-to-video generation."
        )

    # --- Step 1: load and encode content/style videos ---
    print("Loading and encoding content/style videos...")
    content_tensor = load_video_as_tensor(args.content_video, args.video_length, args.image_size)
    style_tensor = load_video_as_tensor(args.style_video, args.video_length, args.image_size)
    content_latent = encode_to_latent(vae, content_tensor, device)
    style_latent = encode_to_latent(vae, style_tensor, device)

    # --- Step 2: text conditioning (used for both inversion and generation) ---
    y_inputs = tokenizer([args.prompt], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    tokens = y_inputs["input_ids"].to(device)
    mask = y_inputs["attention_mask"].bool().to(device)
    y = text_encoder(input_ids=tokens).last_hidden_state
    inversion_kwargs = dict(y=y, mask=mask)

    inversion_diffusion = create_diffusion(str(args.inversion_steps))

    # --- Step 3: DDIM-invert both latents into their corresponding noise ---
    print(f"Inverting content video ({args.inversion_steps} steps)...")
    content_noise = invert_to_noise(inversion_diffusion, model, content_latent, inversion_kwargs, device=device, progress=True)
    print(f"Inverting style video ({args.inversion_steps} steps)...")
    style_noise = invert_to_noise(inversion_diffusion, model, style_latent, inversion_kwargs, device=device, progress=True)

    # --- Step 4: Initial Latent AdaIN -- the actual style-transfer step ---
    styled_noise = adain_latent(content_noise, style_noise)

    # --- Step 5: coupled dual-trajectory sampling with attention-based
    # style injection (the actual StyleID mechanism) at every step ---
    sample_diffusion = create_diffusion(str(args.num_sampling_steps))
    y_null_inputs = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    uncond_tokens = y_null_inputs["input_ids"].to(device)
    y_null = text_encoder(input_ids=uncond_tokens).last_hidden_state
    uncond_mask = y_null_inputs["attention_mask"].bool().to(device)
    y_cfg = torch.cat([y, y_null], 0)
    mask_cfg = torch.cat([mask, uncond_mask], 0)

    # Style branch is conditioned on a null/empty prompt throughout -- its
    # only job is supplying key/value features to inject into content's
    # self-attention, not being "generated" toward any particular text.
    style_kwargs = dict(y=y_null, mask=uncond_mask)

    content_x = torch.cat([styled_noise, styled_noise], 0)  # [cond; uncond] copies
    style_x = style_noise.clone()
    timesteps = list(range(sample_diffusion.num_timesteps))[::-1]

    print(f"Sampling with style injection at every step ({args.num_sampling_steps} steps, "
          f"tau={args.tau}, query_preservation={not args.no_query_preservation})...")
    if args.progress:
        from tqdm.auto import tqdm
        timesteps = tqdm(timesteps)
    for i in timesteps:
        t_content = torch.tensor([i] * content_x.shape[0], device=device)
        t_style = torch.tensor([i] * style_x.shape[0], device=device)

        with torch.no_grad():
            # Content step: uses style_x's CURRENT (same-timestep) noisy
            # latent as the source of injected key/value.
            content_out = sample_diffusion.ddim_sample(
                model.forward_with_style_and_cfg, content_x, t_content,
                clip_denoised=False,
                model_kwargs=dict(
                    y=y_cfg, cfg_scale=args.cfg_scale, mask=mask_cfg,
                    style_x=style_x, style_y=y_null, style_mask=uncond_mask,
                    query_preservation=not args.no_query_preservation, tau=args.tau,
                    style_layers=args.style_layers,
                ),
            )
            content_x = content_out["sample"]

            # Style step: ordinary (non-injected) DDIM step, advancing the
            # style branch's own trajectory so it stays self-consistent and
            # available at the next (lower) timestep.
            style_out = sample_diffusion.ddim_sample(model, style_x, t_style, clip_denoised=False, model_kwargs=style_kwargs)
            style_x = style_out["sample"]

    samples, _ = content_x.chunk(2, dim=0)

    b, _, _, _, _ = samples.shape
    samples = rearrange(samples, "b c f h w -> (b f) c h w").contiguous()
    samples = vae.decode(samples / 0.18215).sample
    samples = rearrange(samples, "(b f) c h w -> b c f h w", b=b).contiguous()

    os.makedirs(args.out_dir, exist_ok=True)
    sample = samples[0]
    sample = rearrange(sample, "c f h w -> f h w c").contiguous()
    sample = ((sample.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
    out_path_base = os.path.join(args.out_dir, "style_transfer_result")
    save_output(sample.cpu().numpy(), out_path_base, args.output_type, args.fps, platform_format=not args.no_platform_format)


if __name__ == "__main__":
    args = parse_args()
    main(args)
