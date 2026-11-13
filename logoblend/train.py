"""
Train LogoBlend.

  Synthetic data only (a first test, no images needed):
      python -m logoblend.train --data synthetic --steps 3000 --out logoblend_synth.pt

  Real images (recommended): photos of people in clothes (or posters, boxes...) + garment masks
      python -m logoblend.prepare_masks --images my_photos --masks my_masks --prompt "a t-shirt"
      python -m logoblend.train --data folder --images my_photos --masks my_masks --logos my_logos --steps 20000 --out logoblend.pt
  --logos is optional: a folder of many different logos (random ones are drawn if it is missing).
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F

from .data import FolderDataset, SyntheticDataset
from .model import BlendNet, compose, make_input


def tv(x):
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()


def losses(net, batch):
    x = make_input(batch["frame"], batch["logo_rgb"], batch["logo_a"], batch["mask"])
    p = net(x)
    out, _ = compose(batch["frame"], batch["logo_rgb"], batch["logo_a"], p)
    # weight the loss toward the logo area (a little bit beyond it)
    w = F.max_pool2d(batch["logo_a"], 7, 1, 3)
    wsum = w.sum().clamp(min=1.0)
    wl1 = lambda a, b: ((a - b).abs() * w).sum() / (wsum * a.shape[1])
    l_img = (out - batch["target"]).abs().mean() + 5.0 * wl1(out, batch["target"])
    l_s = wl1(p["S"], batch["S"])
    # visibility: cross-entropy (L1 gives almost no gradient near 1); occluded pixels are rare, so they count more
    v = p["V"].clamp(1e-4, 1 - 1e-4)
    bce = -(batch["V"] * torch.log(v) * 1.0 + (1 - batch["V"]) * torch.log(1 - v) * 4.0)
    l_v = 2.0 * (bce * w).sum() / wsum
    l_d = wl1(p["D"], batch["D"]) * 10.0
    reg = 0.1 * p["B"].abs().mean() + 0.05 * tv(p["D"])
    return l_img + l_s + l_v + l_d + reg, {"img": l_img.item(), "S": l_s.item(), "V": l_v.item(), "D": l_d.item()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "folder"], default="synthetic")
    ap.add_argument("--images"); ap.add_argument("--masks"); ap.add_argument("--logos")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--max_shift", type=float, default=0.03)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="logoblend.pt")
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    if a.data == "synthetic":
        ds = SyntheticDataset(a.size, length=10 ** 7, logos_dir=a.logos, seed=a.seed, max_shift=a.max_shift)
    else:
        assert a.images and a.masks, "--images and --masks are required for --data folder"
        ds = FolderDataset(a.images, a.masks, a.size, a.logos, a.seed, a.max_shift)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=(a.data == "folder"), num_workers=a.workers,
                                     drop_last=True, persistent_workers=a.workers > 0)
    net = BlendNet(a.base, a.max_shift).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0)
    step = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); step = ck["step"]
        print(f"resumed from {a.resume} at step {step}")

    def save():
        tmp = a.out + ".tmp"
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "step": step,
                    "args": {"base": a.base, "max_shift": a.max_shift, "size": a.size}}, tmp)
        os.replace(tmp, a.out)

    t0, run = time.time(), {}
    net.train()
    while step < a.steps:
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            loss, parts = losses(net, batch)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            step += 1
            for k, v in parts.items():
                run[k] = run.get(k, 0) + v
            if step % 100 == 0:
                print(f"step {step:6d} loss {loss.item():.4f} " + " ".join(f"{k} {v / 100:.4f}" for k, v in run.items())
                      + f"  ({100 / (time.time() - t0):.1f} it/s)", flush=True)
                run, t0 = {}, time.time()
            if step % a.save_every == 0:
                save()
            if step >= a.steps:
                break
    save()
    print("saved", a.out)


if __name__ == "__main__":
    main()
