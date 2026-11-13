# Wan 2.2 model code (vendored)

This folder is a copy of the official Wan 2.2 source code.

- Source:  https://github.com/Wan-Video/Wan2.2
- Commit:  1ea34ff48f87168174e12956e200b1d908b1c5ff  (2026-09-21)
- License: Apache License 2.0 (see LICENSE.txt). Copyright The Alibaba Wan Team Authors.
- Copied:  `wan/` (the model code), `generate.py`, `requirements.txt`, `LICENSE.txt`.
- One local change: `wan/__init__.py` does not import `WanS2V` / `WanAnimate` (they need extra libraries that
  text/image-to-video does not use). Their files are still here.

It does NOT contain the weights (tens of GB; GitHub cannot hold them). Get them with, from the folder above this one:

    python download_wan.py --official          # -> models/Wan2.2-TI2V-5B

Run, from this folder (needs a GPU with FlashAttention support: A100, L4, H100. NOT the free Colab T4):

    pip install -r requirements.txt            # includes flash_attn, which may take several minutes to build
    python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ../models/Wan2.2-TI2V-5B \
        --offload_model True --convert_model_dtype --t5_cpu --prompt "..."

On a T4 (or any GPU without FlashAttention), use `generate_wan.py` in the folder above, which loads the same model
through the diffusers library.
