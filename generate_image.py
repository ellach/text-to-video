"""
Image generation with an automatic company logo, in one command.

    python generate_image.py --prompt "a man in a blue shirt in an office, photo" --out_dir images

The image comes from an open pretrained model (Stable Diffusion XL by default, run on your own GPU). Then the logo
is added automatically, so the user does nothing except put the logo file into the `logos/` folder (the newest
image there is used). The picture without the logo is kept next to the one with it: <name>.png, <name>_logo.png.

Options (all optional):
  --prompt TEXT      what to draw (repeat the flag for several images)
  --out_dir DIR      where to save (default: images)
  --logo FILE        use this logo instead of the newest file in logos/
  --logos_dir DIR    folder to look in (default: logos)
  --no_logo          skip the logo step
  --logo_target T    force a surface, e.g. "t-shirt" (default: chosen automatically)
  --logo_random      choose randomly among the good surfaces
  --logo_size S      logo size (fraction of the surface width)
  --steps N          sampling steps (default 30)
  --cfg X            guidance scale (default 6.5)
  --size PX          image side in pixels (default 1024)
  --seed N           random seed (default: random)
  --model NAME       Hugging Face model id (default: stabilityai/stable-diffusion-xl-base-1.0)
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
DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


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
    return s[:n] or "image"


def load_pipeline(model, device=None):
    import torch
    from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
    cls = StableDiffusionXLPipeline if "xl" in model.lower() else DiffusionPipeline
    pipe = cls.from_pretrained(model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()      # fits a 16 GB GPU (Colab T4)
    return pipe


def main(argv=None, pipe=None, segmenter=None):
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", action="append", required=True)
    p.add_argument("--out_dir", default="images")
    p.add_argument("--logo", default=None)
    p.add_argument("--logos_dir", default="logos")
    p.add_argument("--no_logo", action="store_true")
    p.add_argument("--logo_target", default=None)
    p.add_argument("--logo_random", action="store_true")
    p.add_argument("--logo_size", type=float, default=None)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=6.5)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    a = p.parse_args(argv)

    os.makedirs(a.out_dir, exist_ok=True)
    logo = None if a.no_logo else find_logo(a.logo, a.logos_dir)
    if pipe is None:
        pipe = load_pipeline(a.model)
    if logo is not None:
        import add_logo as A
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
        img = pipe(prompt, num_inference_steps=a.steps, guidance_scale=a.cfg, height=a.size, width=a.size,
                   generator=gen).images[0]
        base = os.path.join(a.out_dir, f"{i:02d}_{slug(prompt)}_s{seed}")
        img.save(base + ".png")
        print(f"[image] {base}.png")
        outs.append(base + ".png")
        if logo is None:
            continue
        frame = np.array(img.convert("RGB"))
        res, rep = A.add_logo([frame], logo, segmenter, a.logo_target, a.logo_random, seed, a.logo_size)
        from PIL import Image
        Image.fromarray(res[0]).save(base + "_logo.png")
        where = rep["surface"] if rep["mode"] == "surface" else "corner watermark"
        print(f"[logo] {base}_logo.png  ({where})")
        outs.append(base + "_logo.png")
    return outs


if __name__ == "__main__":
    main()
