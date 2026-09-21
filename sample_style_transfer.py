"""
Style transfer for GenTron using Initial Latent AdaIN, adapted from:

  "Style Injection in Diffusion: A Training-free Approach for Adapting
  Large-scale Diffusion Models for Style Transfer" (StyleID), CVPR 2024.
  https://arxiv.org/abs/2312.09008

This implements ONLY the paper's Initial Latent AdaIN (color & tone
initialization) component -- see diffusion/style_transfer.py for a detailed
explanation of what that is and how it's generalized here from single
images to video clips. The paper's other component (attention injection /
attention temperature scaling inside self-attention layers) is a separate
mechanism and is NOT implemented here.

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
       style noise's per-channel mean/std (AdaIN) -- this is the step that
       actually transfers color/tone.
    4. Ordinary DDIM sampling runs forward from that combined noise,
       conditioned on --prompt, producing the final video.

Caveats worth knowing:
    - This only transfers the coarse, low-frequency color/tone statistics
      captured in the initial latent -- NOT fine-grained texture/brushwork
      style, which the paper's (unimplemented-here) attention injection
      handles. Expect a color-grading-like effect, not a full artistic
      style transfer.
    - DDIM inversion is only exactly invertible for a deterministic sampler
      (eta=0, which p_sample_loop's underlying DDIM path uses here) and
      accumulates some numerical drift over many steps -- this is a known,
      inherent limitation of DDIM inversion in general, not specific to
      this implementation.
    - Untested against real trained weights end-to-end (no GPU available
      in the environment this was written in) -- verify on your own
      checkpoint before relying on it.
"""

import argparse
import os

import imageio
import torch
from decord import VideoReader, cpu
from diffusers.models import AutoencoderKL
from einops import rearrange
from torchvision import transforms
from transformers import AutoConfig, AutoTokenizer, CLIPTextModel

from diffusion import create_diffusion
from diffusion.style_transfer import adain_latent, invert_to_noise
from download import find_model
from models import GenTron_models


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
    parser.add_argument("--inversion_steps", type=int, default=50,
                         help="DDIM steps used for inverting content/style videos to noise. "
                              "Doesn't need to match --num_sampling_steps.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to a trained GenTron-T2V checkpoint.")
    parser.add_argument("--content_video", type=str, required=True,
                         help="Real video whose STRUCTURE/MOTION is preserved.")
    parser.add_argument("--style_video", type=str, required=True,
                         help="Real video whose COLOR/TONE is transferred onto the content.")
    parser.add_argument("--prompt", type=str, required=True,
                         help="Text prompt conditioning the generation (still required -- GenTron-T2V "
                              "always generates conditioned on text, style transfer happens on top of that).")
    parser.add_argument("--out_dir", type=str, default=".")

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

    # --- Step 5: ordinary forward sampling from the AdaIN'd noise ---
    sample_diffusion = create_diffusion(str(args.num_sampling_steps))
    z = torch.cat([styled_noise, styled_noise], 0)
    y_null_inputs = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    uncond_tokens = y_null_inputs["input_ids"].to(device)
    y_null = text_encoder(input_ids=uncond_tokens).last_hidden_state
    uncond_mask = y_null_inputs["attention_mask"].bool().to(device)
    y_cfg = torch.cat([y, y_null], 0)
    mask_cfg = torch.cat([mask, uncond_mask], 0)
    model_kwargs = dict(y=y_cfg, cfg_scale=args.cfg_scale, mask=mask_cfg)

    print(f"Sampling ({args.num_sampling_steps} steps)...")
    samples = sample_diffusion.ddim_sample_loop(
        model.forward_with_cfg, z.shape, noise=z, clip_denoised=False,
        model_kwargs=model_kwargs, progress=True, device=device,
    )
    samples, _ = samples.chunk(2, dim=0)

    b, _, _, _, _ = samples.shape
    samples = rearrange(samples, "b c f h w -> (b f) c h w").contiguous()
    samples = vae.decode(samples / 0.18215).sample
    samples = rearrange(samples, "(b f) c h w -> b c f h w", b=b).contiguous()

    os.makedirs(args.out_dir, exist_ok=True)
    sample = samples[0]
    sample = rearrange(sample, "c f h w -> f h w c").contiguous()
    sample = ((sample.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
    out_path = os.path.join(args.out_dir, "style_transfer_result.mp4")
    imageio.mimwrite(out_path, sample.cpu().numpy(), fps=args.fps, codec="libx264", quality=8)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
