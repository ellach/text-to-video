"""
Training data for LogoBlend, made automatically.

A training sample is built from ANY image of an object with a mask (a person's shirt, a poster, a box...):

  1. paste a random logo on the masked region               -> the model's input (flat, like add_logo.py produces)
  2. optionally put an occluder in front (a hand, hair)      -> the logo must disappear behind it
  3. simulate how a print would really look on that cloth    -> the target:
        shading   from the image's own lighting and folds under the logo,
        bend      a small displacement along the folds,
        visibility the logo is hidden where the occluder is.

IMPORTANT (honest limit): the target is produced by a physical *simulation*, not by real printed shirts. The network
learns the simulator plus how to read the scene (where the folds, light and occluders are). To go beyond it, mix in
real images of printed garments (see README), or tune the simulator.
"""
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from .model import compose


# ------------------------------------------------------------------ helpers
def gblur(x, sigma):
    """Gaussian blur of a [B, C, H, W] (or [C, H, W]) tensor."""
    squeeze = x.dim() == 3
    if squeeze:
        x = x[None]
    r = max(1, int(3 * sigma))
    k = torch.arange(-r, r + 1, dtype=x.dtype)
    k = torch.exp(-k ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    C = x.shape[1]
    mode = "reflect" if min(x.shape[-2:]) > r else "replicate"
    x = F.pad(x, (r, r, r, r), mode=mode)
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x = F.conv2d(x, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return x[0] if squeeze else x


def random_logo(rng, size=96):
    """A random logo-like graphic (shapes + letters) as straight RGB [3,h,w] and alpha [1,h,w] in [0,1].
    Real logos from a folder are used instead when --logos is given."""
    w, h = int(size * rng.uniform(1.0, 2.2)), size
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    col = lambda: tuple(int(c) for c in rng.integers(0, 256, 3)) + (255,)
    kind = rng.integers(0, 4)
    if kind == 0:
        d.ellipse((2, 2, h - 2, h - 2), fill=col())
    elif kind == 1:
        d.rounded_rectangle((2, 2, w - 2, h - 2), int(h * 0.2), fill=col())
    elif kind == 2:
        pts = [(int(rng.uniform(0, w)), int(rng.uniform(0, h))) for _ in range(int(rng.integers(3, 7)))]
        d.polygon(pts, fill=col())
    else:
        d.ellipse((2, 2, h - 2, h - 2), outline=col(), width=max(2, h // 10))
    if rng.random() < 0.8:
        try:
            font = ImageFont.load_default(size=int(h * rng.uniform(0.3, 0.6)))
        except TypeError:
            font = ImageFont.load_default()
        text = "".join(chr(int(rng.integers(65, 91))) for _ in range(int(rng.integers(1, 5))))
        d.text((w * 0.2, h * 0.25), text, fill=col(), font=font)
    a = np.asarray(im).astype(np.float32) / 255.0
    return torch.from_numpy(a[..., :3]).permute(2, 0, 1).contiguous(), torch.from_numpy(a[..., 3:]).permute(2, 0, 1).contiguous()


def load_logo_file(path):
    a = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
    return torch.from_numpy(a[..., :3]).permute(2, 0, 1).contiguous(), torch.from_numpy(a[..., 3:]).permute(2, 0, 1).contiguous()


def place_logo(logo_rgb, logo_a, mask, rng):
    """Put the logo at a random place inside the mask -> full-frame logo layers [3,H,W], [1,H,W]; width in px."""
    _, H, W = mask.shape
    ys, xs = torch.nonzero(mask[0] > 0.5, as_tuple=True)
    if len(xs) < 50:
        ys, xs = torch.tensor([H // 2]), torch.tensor([W // 2])
    bw = float(xs.max() - xs.min() + 1)
    lw = int(max(8, bw * rng.uniform(0.2, 0.5)))
    lh = max(4, int(lw * logo_rgb.shape[1] / logo_rgb.shape[2]))
    lr = F.interpolate(logo_rgb[None] * logo_a[None], size=(lh, lw), mode="area")[0]
    la = F.interpolate(logo_a[None], size=(lh, lw), mode="area")[0]
    lr = lr / la.clamp(min=1e-4)
    i = int(rng.integers(0, len(xs)))
    cx, cy = int(xs[i]), int(ys[i])
    x0, y0 = int(np.clip(cx - lw // 2, 0, max(W - lw, 0))), int(np.clip(cy - lh // 2, 0, max(H - lh, 0)))
    layer_rgb, layer_a = torch.zeros(3, H, W), torch.zeros(1, H, W)
    h2, w2 = min(lh, H - y0), min(lw, W - x0)
    layer_rgb[:, y0:y0 + h2, x0:x0 + w2] = lr[:, :h2, :w2]
    layer_a[:, y0:y0 + h2, x0:x0 + w2] = la[:, :h2, :w2]
    return layer_rgb, layer_a, lw


def occluder(H, W, near, rng):
    """A soft blob with its own colour and texture, placed over `near` (cx, cy, radius) -> (mask [1,H,W], rgb [3,H,W])"""
    cx, cy, r = near
    ox, oy = cx + rng.uniform(-r, r), cy + rng.uniform(-r, r)
    rx, ry = r * rng.uniform(0.3, 0.8), r * rng.uniform(0.3, 0.8)
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    m = (((xx - ox) / rx) ** 2 + ((yy - oy) / ry) ** 2 < 1).float()[None]
    m = gblur(m, 1.0)
    base = torch.tensor(rng.uniform(0.1, 0.95, 3), dtype=torch.float32).view(3, 1, 1)
    tex = gblur(torch.from_numpy(rng.normal(0, 0.12, (3, H, W)).astype(np.float32)), 2.0)
    return m, (base + tex).clamp(0, 1)


# ------------------------------------------------------------------ the simulator
def simulate(img, mask, logo_rgb, logo_a, rng, max_shift=0.03, with_occluder=None):
    """img [3,H,W] 0-1, mask [1,H,W] -> a training sample (tensors)."""
    _, H, W = img.shape
    lrgb, la, lw = place_logo(logo_rgb, logo_a, mask, rng)

    # lighting and folds of the cloth, read from the image itself (before any occluder is added)
    gray = img.mean(0, keepdim=True)
    low = gblur(gray, max(3.0, 0.25 * lw))
    mid = gblur(gray, max(2.5, 0.06 * lw))
    inside = gray[mask > 0.5]
    ref = float(inside.median()) if inside.numel() > 20 else float(gray.mean())
    base = (low / max(ref, 1e-3)).clamp(0.8, 1.2)
    detail = ((mid / low.clamp(min=1e-3)).clamp(0.55, 1.4) ** 0.6).clamp(0.8, 1.25)
    tint_src = img[:, mask[0] > 0.5].mean(1) if inside.numel() > 20 else img.mean((1, 2))
    tint = 1.0 + 0.25 * (tint_src / tint_src.mean().clamp(min=1e-3) - 1.0)
    S = (base * detail).expand(3, H, W) * tint.view(3, 1, 1)

    # bend: along the strongest folds, a small smooth displacement
    gx = mid[:, :, 2:] - mid[:, :, :-2]; gx = F.pad(gx, (1, 1, 0, 0))
    gy = mid[:, 2:, :] - mid[:, :-2, :]; gy = F.pad(gy, (0, 0, 1, 1))
    g = torch.cat([gx, gy], 0)
    mag = g.norm(dim=0, keepdim=True)
    D = rng.uniform(0.2, 1.0) * max_shift * g / (mag + 1e-3) * torch.tanh(mag / 0.02)
    D = gblur(D, 2.0).clamp(-max_shift, max_shift)

    # occluder in front of the cloth, over the logo
    O = torch.zeros(1, H, W)
    frame = img.clone()
    use_occ = (rng.random() < 0.5) if with_occluder is None else with_occluder
    if use_occ and la.sum() > 0:
        ys, xs = torch.nonzero(la[0] > 0.5, as_tuple=True)
        cx, cy = float(xs.float().mean()), float(ys.float().mean())
        O, ocol = occluder(H, W, (cx, cy, max(6.0, lw * 0.5)), rng)
        frame = frame * (1 - O) + ocol * O
    V = 1 - O

    fields = {"S": S[None], "B": torch.zeros(1, 3, H, W), "D": D[None], "V": V[None]}
    target, _ = compose(frame[None], lrgb[None], la[None], fields)
    return {"frame": frame, "logo_rgb": lrgb, "logo_a": la, "mask": mask, "target": target[0],
            "S": S, "D": D, "V": V}


# ------------------------------------------------------------------ images
def synth_scene(H, W, rng):
    """A procedural 'garment': coloured cloth with folds and lighting on a plain background (for tests and for
    bootstrapping; real images are better)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = np.stack([np.full((H, W), rng.uniform(0.3, 0.9), np.float32)] * 3, 0) * (1 + 0.1 * (xx / W - 0.5))
    cloth = rng.uniform(0.1, 0.9, 3).astype(np.float32)[:, None, None]
    folds = np.zeros((H, W), np.float32)
    for _ in range(int(rng.integers(2, 5))):
        kx, ky, ph = rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12), rng.uniform(0, 6.28)
        folds += rng.uniform(0.05, 0.16) * np.sin(kx * xx + ky * yy + ph)
    light = 1 + rng.uniform(-0.3, 0.3) * (xx / W - 0.5) + rng.uniform(-0.3, 0.3) * (yy / H - 0.5)
    tex = rng.normal(0, 0.02, (3, H, W)).astype(np.float32)
    shirt = cloth * (light + folds)[None] + tex
    x0, x1 = int(W * rng.uniform(0.08, 0.2)), int(W * rng.uniform(0.8, 0.92))
    y0, y1 = int(H * rng.uniform(0.1, 0.25)), int(H * rng.uniform(0.85, 0.97))
    m = np.zeros((H, W), np.float32); m[y0:y1, x0:x1] = 1
    img = np.where(m[None] > 0, shirt, bg)
    return torch.from_numpy(np.clip(img, 0, 1)), torch.from_numpy(m)[None]


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, size=96, length=2000, logos_dir=None, seed=0, max_shift=0.03):
        self.size, self.length, self.seed, self.max_shift = size, length, seed, max_shift
        self.logos = sorted(glob.glob(os.path.join(logos_dir, "*"))) if logos_dir else []

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        img, mask = synth_scene(self.size, self.size, rng)
        lr, la = load_logo_file(self.logos[int(rng.integers(len(self.logos)))]) if self.logos else random_logo(rng)
        return simulate(img, mask, lr, la, rng, self.max_shift)


class FolderDataset(torch.utils.data.Dataset):
    """Real images: <images>/name.jpg with a garment mask <masks>/name.png (white = garment).
    Make masks with prepare_masks.py. Logos come from --logos (a folder) or are random."""

    def __init__(self, images, masks, size=128, logos_dir=None, seed=0, max_shift=0.03, repeat=20):
        self.items = []
        for p in sorted(glob.glob(os.path.join(images, "*"))):
            m = os.path.join(masks, os.path.splitext(os.path.basename(p))[0] + ".png")
            if os.path.isfile(m):
                self.items.append((p, m))
        if not self.items:
            raise SystemExit(f"No image+mask pairs found in {images} / {masks}")
        self.size, self.seed, self.max_shift, self.repeat = size, seed, max_shift, repeat
        self.logos = sorted(glob.glob(os.path.join(logos_dir, "*"))) if logos_dir else []

    def __len__(self):
        return len(self.items) * self.repeat

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        p, m = self.items[i % len(self.items)]
        img = Image.open(p).convert("RGB"); mk = Image.open(m).convert("L")
        s = self.size                                                    # random square crop, then resize
        w, h = img.size; side = int(min(w, h) * rng.uniform(0.5, 1.0))
        x0, y0 = int(rng.integers(0, w - side + 1)), int(rng.integers(0, h - side + 1))
        box = (x0, y0, x0 + side, y0 + side)
        img = torch.from_numpy(np.asarray(img.crop(box).resize((s, s), Image.BILINEAR)).astype(np.float32) / 255).permute(2, 0, 1)
        mask = torch.from_numpy((np.asarray(mk.crop(box).resize((s, s), Image.NEAREST)) > 127).astype(np.float32))[None]
        if rng.random() < 0.5:
            img, mask = img.flip(-1), mask.flip(-1)
        lr, la = load_logo_file(self.logos[int(rng.integers(len(self.logos)))]) if self.logos else random_logo(rng)
        return simulate(img, mask, lr, la, rng, self.max_shift)
