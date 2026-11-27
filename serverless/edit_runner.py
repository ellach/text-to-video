"""Runs Qwen-Image-Edit-2511 in its own process, so it can use a newer diffusers (/opt/edit-libs) than the video worker.
Usage: python edit_runner.py job.json   (job.json: images[], prompt, width, height, steps, seed, out)"""
import json
import sys

sys.path.insert(0, "/opt/edit-libs")
import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

MODEL = "Qwen/Qwen-Image-Edit-2511"

job = json.load(open(sys.argv[1]))
pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()          # the 20B model does not need to sit on the GPU all at once
images = [Image.open(p).convert("RGB") for p in job["images"]]
with torch.inference_mode():
    out = pipe(
        image=images, prompt=job["prompt"], negative_prompt=" ", true_cfg_scale=4.0, guidance_scale=1.0,
        num_inference_steps=int(job.get("steps", 30)), generator=torch.manual_seed(int(job.get("seed", 0))),
        num_images_per_prompt=1, width=int(job["width"]), height=int(job["height"]),
    ).images[0]
out.save(job["out"], "JPEG", quality=92)
print("done", out.size, flush=True)
