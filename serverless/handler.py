"""
RunPod serverless worker: text description (+ optional logo) -> video (Wan 2.2 TI2V-5B) with the logo placed automatically.

Input (job["input"]):
  prompt       str   required   what the video shows (max 1000 characters)
  logo_url     str   optional   https URL of the logo image (png/jpg/webp, max 10 MB). No logo -> plain video.
  aspect       str   "9:16" (default, vertical) or "16:9"
  duration_s   float seconds, 1..5 (default 5)
  quality      str   "fast" (480p), "high" (704x1280, Wan 2.2 5B) or "pro" (720p, Wan 2.2 14B: best people/motion, needs an 80 GB GPU)
  logo_target  str   optional   force a surface, e.g. "t-shirt" (default: chosen automatically)
  seed         int   optional
  type         str   "video" (default) or "image"
                     image: prompt, optional logo_url/logo_target/seed, aspect "1:1" (default), "9:16" or "16:9"
                     (Stable Diffusion XL, about 1 megapixel). Output has "image_url" (or "image_base64") instead of "video_url".

Output: {"video_url": ..., "logo": "surface:<name>" | "corner watermark" | "none", "seed": ..., "seconds": ...}
        or {"error": "..."}.
Storage: set S3_BUCKET (+ S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_REGION) to get a presigned URL.
         Without S3_BUCKET the video is returned as base64 ("video_base64"): fine for tests, too big for production.
Model:   MODEL_DIR (default /runpod-volume/models/Wan2.2-TI2V-5B-Diffusers); if it does not exist the model is downloaded
         from Hugging Face on the first job.
"""
import base64
import ipaddress
import os
import socket
import tempfile
import time
import uuid
from urllib.parse import urlparse

import numpy as np

MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models/Wan2.2-TI2V-5B-Diffusers")
HF_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
HF_ID_PRO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"      # two 14B experts (~56 GB in bf16): needs an 80 GB GPU
MAX_PROMPT = 1000
MAX_LOGO_BYTES = 10 * 1024 * 1024
FPS = 24
SIZES = {("fast", "16:9"): (480, 832), ("fast", "9:16"): (832, 480),
         ("high", "16:9"): (704, 1280), ("high", "9:16"): (1280, 704),
         ("pro", "16:9"): (720, 1280), ("pro", "9:16"): (1280, 720)}      # (height, width)
IMG_SIZES = {"1:1": (1024, 1024), "9:16": (1344, 768), "16:9": (768, 1344)}      # (height, width), SDXL-friendly sizes
IMG_MODEL = os.environ.get("IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")      # Apache-2.0, follows prompts with several people well
IMG_NEGATIVE = "blurry, low quality, distorted, deformed, extra fingers, bad anatomy, watermark, text, jpeg artifacts"
NEGATIVE = ("blurry, low quality, distorted, deformed, extra fingers, static, still image, worst quality, "
            "jpeg artifacts, subtitles, watermark")

_pipe = None
_pro_pipe = None
_img_pipe = None
_segmenter = None


class JobError(Exception):
    pass


def get_pipe():
    global _pipe, _pro_pipe
    if _pipe is None:
        if _pro_pipe is not None:
            import gc
            import torch
            _pro_pipe = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        import generate_wan as G
        _pipe = G.load_pipeline(MODEL_DIR if os.path.isdir(MODEL_DIR) else HF_ID)
    return _pipe


def get_pro_pipe():
    """Wan 2.2 14B. Only one big video model lives in memory at a time."""
    global _pipe, _pro_pipe
    if _pro_pipe is None:
        import gc
        import torch
        _pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import generate_wan as G
        _pro_pipe = G.load_pipeline(HF_ID_PRO)
    return _pro_pipe


def get_image_pipe():
    global _img_pipe
    if _img_pipe is None:
        if "flux" in IMG_MODEL.lower():
            import torch
            from diffusers import FluxPipeline
            _img_pipe = FluxPipeline.from_pretrained(IMG_MODEL, torch_dtype=torch.bfloat16)
            if torch.cuda.is_available():
                _img_pipe.enable_model_cpu_offload()
        else:
            import generate_image as GI
            _img_pipe = GI.load_pipeline(IMG_MODEL)
    return _img_pipe


def get_segmenter():
    global _segmenter
    if _segmenter is None:
        import add_logo as A
        _segmenter = A.ClipSegSegmenter()
    return _segmenter


def check_url(url):
    """https only, and never a private/internal address (the URL comes from a user)."""
    u = urlparse(url)
    if u.scheme != "https" or not u.hostname:
        raise JobError("logo_url must be an https URL")
    try:
        for info in socket.getaddrinfo(u.hostname, 443):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                raise JobError("logo_url points to a non-public address")
    except socket.gaierror:
        raise JobError("logo_url host not found")


def download_logo(url, dest_dir):
    import requests
    check_url(url)
    r = requests.get(url, stream=True, timeout=20, allow_redirects=False)
    if r.status_code != 200:
        raise JobError(f"logo_url returned HTTP {r.status_code}")
    path = os.path.join(dest_dir, "logo")
    n = 0
    with open(path, "wb") as f:
        for chunk in r.iter_content(65536):
            n += len(chunk)
            if n > MAX_LOGO_BYTES:
                raise JobError("logo file is larger than 10 MB")
            f.write(chunk)
    from PIL import Image
    try:
        im = Image.open(path)
        im.verify()
        fmt = (im.format or "").lower()
    except Exception:
        raise JobError("logo_url is not a valid image")
    if fmt not in ("png", "jpeg", "webp"):
        raise JobError("logo must be png, jpg or webp")
    final = path + "." + ("jpg" if fmt == "jpeg" else fmt)
    os.rename(path, final)
    return final


def fps_for(quality):
    return 16 if quality == "pro" else FPS      # Wan 2.2 14B is trained at 16 fps


def frames_for(duration_s, fps=None):
    n = int(round(duration_s * (fps or FPS)))
    k = max(1, min(30, int(round((n - 1) / 4))))   # Wan needs 4k+1 frames: take the nearest valid count (5 .. 121)
    return 4 * k + 1


def parse(inp):
    if not isinstance(inp, dict):
        raise JobError("input must be an object")
    prompt = str(inp.get("prompt", "")).strip()
    if not prompt:
        raise JobError("prompt is required")
    if len(prompt) > MAX_PROMPT:
        raise JobError(f"prompt is longer than {MAX_PROMPT} characters")
    kind = inp.get("type") or "video"
    if kind not in ("video", "image"):
        raise JobError('type must be "video" or "image"')
    seed = inp.get("seed")
    seed = int(seed) if seed is not None else int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
    if kind == "image":
        aspect = inp.get("aspect") or "1:1"
        if aspect not in IMG_SIZES:
            raise JobError('aspect must be "1:1", "9:16" or "16:9" for images')
        return dict(type="image", prompt=prompt, aspect=aspect, seed=seed,
                    logo_url=inp.get("logo_url") or None, logo_target=inp.get("logo_target") or None)
    aspect = inp.get("aspect") or "9:16"
    quality = inp.get("quality") or "fast"
    if quality == "pro" and os.environ.get("ENABLE_PRO") != "1":
        raise JobError('quality "pro" is switched off; use "high"')
    if (quality, aspect) not in SIZES:
        raise JobError('aspect must be "9:16" or "16:9" and quality "fast", "high" or "pro"')
    try:
        duration = float(inp.get("duration_s", 5))
    except (TypeError, ValueError):
        raise JobError("duration_s must be a number")
    if not 1 <= duration <= 5:
        raise JobError("duration_s must be between 1 and 5")
    return dict(type="video", prompt=prompt, aspect=aspect, quality=quality, duration=duration, seed=seed,
                logo_url=inp.get("logo_url") or None, logo_target=inp.get("logo_target") or None)


def store(path, upload=None, kind="video"):
    if upload:
        return {kind + "_url": upload(path)}
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        with open(path, "rb") as f:
            return {kind + "_base64": base64.b64encode(f.read()).decode()}
    import boto3
    s3 = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"), region_name=os.environ.get("S3_REGION"))
    ext, ctype = ("png", "image/png") if kind == "image" else ("mp4", "video/mp4")
    key = f"{kind}s/{uuid.uuid4().hex}.{ext}"
    s3.upload_file(path, bucket, key, ExtraArgs={"ContentType": ctype})
    return {kind + "_url": s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=7 * 86400)}


def run(inp, upload=None):
    t0 = time.time()
    p = parse(inp)
    if p["type"] == "image":
        return run_image(p, upload)
    import add_logo as A
    with tempfile.TemporaryDirectory() as tmp:
        logo = download_logo(p["logo_url"], tmp) if p["logo_url"] else None
        import torch
        h, w = SIZES[(p["quality"], p["aspect"])]
        fps = fps_for(p["quality"])
        gen = torch.Generator("cpu").manual_seed(p["seed"])
        if p["quality"] == "pro":
            out = get_pro_pipe()(prompt=p["prompt"], negative_prompt=NEGATIVE, height=h, width=w,
                                 num_frames=frames_for(p["duration"], fps), num_inference_steps=40,
                                 guidance_scale=4.0, guidance_scale_2=3.0, generator=gen).frames[0]
        else:
            out = get_pipe()(prompt=p["prompt"], negative_prompt=NEGATIVE, height=h, width=w,
                             num_frames=frames_for(p["duration"]), num_inference_steps=30, guidance_scale=5.0,
                             generator=gen).frames[0]
        import generate_wan as G
        frames = G.to_uint8_frames(out)
        report = "none"
        if logo:
            frames, rep = A.add_logo(frames, logo, get_segmenter(), p["logo_target"], False, p["seed"], None)
            report = ("surface:" + str(rep["surface"])) if rep["mode"] == "surface" else "corner watermark"
        path = os.path.join(tmp, "out.mp4")
        A.write_video(frames, fps, path)
        res = store(path, upload)
    res.update(logo=report, seed=p["seed"], seconds=round(time.time() - t0, 1))
    return res


def run_image(p, upload=None):
    t0 = time.time()
    import add_logo as A
    from PIL import Image
    with tempfile.TemporaryDirectory() as tmp:
        logo = download_logo(p["logo_url"], tmp) if p["logo_url"] else None
        import torch
        h, w = IMG_SIZES[p["aspect"]]
        gen = torch.Generator("cpu").manual_seed(p["seed"])
        if "flux" in IMG_MODEL.lower():
            # FLUX.1-schnell: 4 steps, no guidance and no negative prompt (it is distilled)
            img = get_image_pipe()(prompt=p["prompt"], height=h, width=w, num_inference_steps=4,
                                   guidance_scale=0.0, max_sequence_length=256, generator=gen).images[0]
        else:
            img = get_image_pipe()(prompt=p["prompt"], negative_prompt=IMG_NEGATIVE, height=h, width=w,
                                   num_inference_steps=30, guidance_scale=6.5, generator=gen).images[0]
        frame = np.array(img.convert("RGB"))
        report = "none"
        if logo:
            res, rep = A.add_logo([frame], logo, get_segmenter(), p["logo_target"], False, p["seed"], None)
            frame = res[0]
            report = ("surface:" + str(rep["surface"])) if rep["mode"] == "surface" else "corner watermark"
        path = os.path.join(tmp, "out.png")
        Image.fromarray(frame).save(path)
        out = store(path, upload, "image")
    out.update(logo=report, seed=p["seed"], seconds=round(time.time() - t0, 1))
    return out


def free_volume_space():
    """Delete the unused Wan 14B download (about 100 GB) and Hugging Face's temporary download cache, so the
    volume has room for the models we use. Runs once when a worker starts. Set ENABLE_PRO=1 to keep the 14B."""
    import shutil
    if os.environ.get("ENABLE_PRO") == "1":
        return
    hf = os.environ.get("HF_HOME", "/runpod-volume/hf")
    for d in (os.path.join(hf, "hub", "models--" + HF_ID_PRO.replace("/", "--")), os.path.join(hf, "xet")):
        try:
            if os.path.isdir(d):
                print("freeing space: removing", d)
                shutil.rmtree(d, ignore_errors=True)
        except Exception as e:
            print("cleanup skipped:", repr(e))


def handler(job):
    try:
        return run(job.get("input"))
    except JobError as e:
        return {"error": str(e)}
    except Exception as e:                                   # never leak internals to the app user
        print("job failed:", repr(e))
        return {"error": "generation failed: " + repr(e)[:300] if os.environ.get("DEBUG_ERRORS") else "generation failed"}


if __name__ == "__main__":
    import runpod
    free_volume_space()
    runpod.serverless.start({"handler": handler})
