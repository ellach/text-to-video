"""
Text-to-video with an open pretrained model (Wan 2.2 TI2V-5B, Apache 2.0) and an automatic company logo.

    python generate_wan.py --prompt "a man in a blue shirt talking in an office" --out_dir videos_wan

The model runs on your own GPU (weights come from Hugging Face the first time, about 20 GB; keep your own copy).
After the video is made, the logo is added automatically: put the logo image into the `logos/` folder (the newest
image there is used). Videos are saved as <name>.mp4 (plain) and <name>_logo.mp4 (with the logo).

Options (all optional):
  --prompt TEXT      what to make (repeat the flag for several videos)
  --out_dir DIR      where to save (default: videos_wan)
  --height / --width size in pixels, multiples of 32 (default 480 x 832; the model's native size is 704 x 1280)
  --frames N         number of frames, must be 4k+1 (default 49; native is 121)
  --fps N            frames per second of the saved file (default 24)
  --steps N          sampling steps (default 30; 50 is the model's default)
  --cfg X            guidance scale (default 5.0)
  --seed N           random seed (default: random)
  --negative TEXT    negative prompt
  --logo FILE / --logos_dir DIR / --no_logo / --logo_target T / --logo_random / --logo_size S   (same as generate.py)
  --model NAME       Hugging Face model id (default: Wan-AI/Wan2.2-TI2V-5B-Diffusers)
"""
import argparse
import os
import random
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DEFAULT_NEGATIVE = ("blurry, low quality, distorted, deformed, extra fingers, static, still image, "
                    "worst quality, jpeg artifacts, subtitles, watermark")


def find_logo(logo, logos_dir):
    if logo:
        if not os.path.isfile(logo):
            raise SystemExit(f"--logo file not found: {logo}")
        return logo
    if not os.path.isdir(logos_dir):
        return None
    files = [os.path.join(logos_dir, f) for f in os.listdir(logos_dir) if f.lower().endswith(IMAGE_EXT)]
    return max(files, key=os.path.getmtime) if files else None


def slug(text, n=40):
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:n] or "video"


def load_pipeline(model):
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    vae = AutoencoderKLWan.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32)   # float32 VAE: better decoding
    pipe = WanPipeline.from_pretrained(model, vae=vae, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()      # moves parts to the GPU only while they run: fits 16-24 GB cards
    return pipe


def to_uint8_frames(out):
    """Pipeline output (float array T,H,W,3 in 0..1, or a list of PIL images) -> list of HxWx3 uint8 arrays."""
    frames = []
    for f in out:
        a = np.asarray(f)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        frames.append(a[..., :3])
    return frames


def main(argv=None, pipe=None, segmenter=None):
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", action="append", required=True)
    p.add_argument("--out_dir", default="videos_wan")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--frames", type=int, default=49)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--negative", default=DEFAULT_NEGATIVE)
    p.add_argument("--logo", default=None)
    p.add_argument("--logos_dir", default="logos")
    p.add_argument("--no_logo", action="store_true")
    p.add_argument("--logo_target", default=None)
    p.add_argument("--logo_random", action="store_true")
    p.add_argument("--logo_size", type=float, default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    a = p.parse_args(argv)

    if a.height % 32 or a.width % 32:
        raise SystemExit("--height and --width must be multiples of 32")
    if (a.frames - 1) % 4:
        raise SystemExit("--frames must be 4k+1 (for example 25, 49, 81, 121)")

    import add_logo as A
    os.makedirs(a.out_dir, exist_ok=True)
    logo = None if a.no_logo else find_logo(a.logo, a.logos_dir)
    if pipe is None:
        pipe = load_pipeline(a.model)
    if logo is not None:
        if segmenter is None:
            segmenter = A.ClipSegSegmenter()
        print(f"[logo] using {logo}")
    else:
        print("[logo] skipped" + ("" if a.no_logo else f": no image found in {a.logos_dir}/"))

    outs = []
    for i, prompt in enumerate(a.prompt):
        seed = a.seed if a.seed is not None else random.randrange(2 ** 31)
        gen = None
        try:
            import torch
            gen = torch.Generator("cpu").manual_seed(seed)
        except Exception:
            pass
        out = pipe(prompt=prompt, negative_prompt=a.negative, height=a.height, width=a.width, num_frames=a.frames,
                   num_inference_steps=a.steps, guidance_scale=a.cfg, generator=gen).frames[0]
        frames = to_uint8_frames(out)
        base = os.path.join(a.out_dir, f"{i:02d}_{slug(prompt)}_s{seed}")
        A.write_video(frames, a.fps, base + ".mp4")
        print(f"[video] {base}.mp4  ({len(frames)} frames)")
        outs.append(base + ".mp4")
        if logo is None:
            continue
        res, rep = A.add_logo(frames, logo, segmenter, a.logo_target, a.logo_random, seed, a.logo_size)
        A.write_video(res, a.fps, base + "_logo.mp4")
        where = rep["surface"] if rep["mode"] == "surface" else "corner watermark"
        print(f"[logo] {base}_logo.mp4  ({where})")
        outs.append(base + "_logo.mp4")
    return outs


if __name__ == "__main__":
    main()
