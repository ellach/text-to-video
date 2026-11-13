"""
Make garment masks for a folder of real photos (CLIPSeg, zero-shot).

  python -m logoblend.prepare_masks --images my_photos --masks my_masks --prompt "a t-shirt"

Open a few masks and delete the bad ones: the model learns from them. (SAM 2 masks are better when available.)
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--masks", required=True)
    ap.add_argument("--prompt", default="a t-shirt")
    ap.add_argument("--thresh", type=float, default=0.45)
    ap.add_argument("--min_area", type=float, default=0.05)
    ap.add_argument("--max_area", type=float, default=0.8)
    a = ap.parse_args()
    from add_logo import ClipSegSegmenter                      # reuse the same segmenter
    seg = ClipSegSegmenter()
    os.makedirs(a.masks, exist_ok=True)
    kept = 0
    for p in sorted(glob.glob(os.path.join(a.images, "*"))):
        try:
            img = np.asarray(Image.open(p).convert("RGB"))
        except Exception:
            continue
        m = seg([img], [a.prompt, "a person's face", "a person's hand"])
        mask = (m[0, 0] > a.thresh) & ~((m[1, 0] > a.thresh) | (m[2, 0] > a.thresh))
        if not (a.min_area <= mask.mean() <= a.max_area):
            continue
        Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(a.masks, os.path.splitext(os.path.basename(p))[0] + ".png"))
        kept += 1
    print(f"saved {kept} masks to {a.masks}")


if __name__ == "__main__":
    main()
