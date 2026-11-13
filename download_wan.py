"""
Get the complete Wan 2.2 TI2V-5B model (every file: transformer, VAE, text encoder, tokenizer, configs) into your
project, so generate_wan.py can run from a local folder. GitHub cannot hold files this large, so the weights live
outside git (in models/, which is git-ignored) and this script puts them there.

  python download_wan.py                              download the model (diffusers format) into models/Wan2.2-TI2V-5B-Diffusers
                                                      -> used by generate_wan.py (works on any GPU)
  python download_wan.py --official                   download the model in Wan's own format into models/Wan2.2-TI2V-5B
                                                      -> used by the Wan code in wan2_2/ (needs flash-attn: A100/L4/H100, not T4)
  python download_wan.py --source YOU/my-wan-copy     download from your own Hugging Face repo instead
  python download_wan.py --upload YOU/my-wan-copy     upload your local copy to YOUR private Hugging Face repo
                                                      (needs `huggingface-cli login` or HF_TOKEN)

Options: --dest DIR (default depends on the format), --force (skip the free-space check).
"""
import argparse
import os
import shutil
import sys

ORIGINAL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DEFAULT_DEST = os.path.join("models", "Wan2.2-TI2V-5B-Diffusers")
NEEDED = ["model_index.json", "vae", "transformer", "text_encoder", "tokenizer", "scheduler"]

OFFICIAL = "Wan-AI/Wan2.2-TI2V-5B"                       # Wan's own checkpoint format (for the code in wan2_2/)
OFFICIAL_DEST = os.path.join("models", "Wan2.2-TI2V-5B")
NEEDED_OFFICIAL = ["config.json", "Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google"]


def missing_parts(dest, needed=None):
    needed = needed or NEEDED
    bad = [n for n in needed if not os.path.exists(os.path.join(dest, n))]
    if needed is NEEDED_OFFICIAL and not any(f.startswith("diffusion_pytorch_model") for f in os.listdir(dest)):
        bad.append("diffusion_pytorch_model*.safetensors")
    return bad


def repo_size_gb(api, repo):
    info = api.model_info(repo, files_metadata=True)
    return sum((s.size or 0) for s in info.siblings) / 1e9


def download(source, dest, force=False, api=None, snapshot=None, needed=None):
    if api is None or snapshot is None:
        from huggingface_hub import HfApi, snapshot_download
        api = api or HfApi()
        snapshot = snapshot or snapshot_download
    os.makedirs(dest, exist_ok=True)
    need = repo_size_gb(api, source)
    free = shutil.disk_usage(dest).free / 1e9
    print(f"model size: {need:.1f} GB | free space here: {free:.1f} GB")
    if free < need + 2 and not force:
        raise SystemExit("Not enough free space for the model. Free some space, or use --force to try anyway.")
    snapshot(source, local_dir=dest)
    bad = missing_parts(dest, needed)
    if bad:
        raise SystemExit(f"Download finished but these parts are missing in {dest}: {bad}")
    print(f"OK: complete model in {dest}")
    return dest


def upload(dest, repo, api=None, needed=None):
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    bad = missing_parts(dest, needed)
    if bad:
        raise SystemExit(f"{dest} is not a complete model (missing {bad}); download it first.")
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(folder_path=dest, repo_id=repo, repo_type="model")
    print(f"OK: uploaded {dest} to https://huggingface.co/{repo} (private)")


def main(argv=None, **kw):
    p = argparse.ArgumentParser()
    p.add_argument("--official", action="store_true", help="Wan's own checkpoint format (for the code in wan2_2/)")
    p.add_argument("--source", default=None)
    p.add_argument("--dest", default=None)
    p.add_argument("--upload", default=None, metavar="YOU/REPO")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    source = a.source or (OFFICIAL if a.official else ORIGINAL)
    dest = a.dest or (OFFICIAL_DEST if a.official else DEFAULT_DEST)
    needed = NEEDED_OFFICIAL if a.official else NEEDED
    if a.upload:
        return upload(dest, a.upload, kw.get("api"), needed)
    return download(source, dest, a.force, kw.get("api"), kw.get("snapshot"), needed)


if __name__ == "__main__":
    main()
