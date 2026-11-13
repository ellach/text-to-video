"""
Put a company logo onto a generated video, automatically.

  python add_logo.py --video samples/x.mp4 --logo my_logo.png

By default the script looks for good surfaces in the video (t-shirt, poster, sign, billboard, screen, box, wall),
picks the best one, and places the logo there, following the surface through the frames. If nothing suitable is found
it adds a small corner watermark, so you always get a result.

Optional:
  --target "t-shirt"   force a surface (any text; CLIPSeg finds it)
  --random             pick randomly among the good surfaces instead of always the best one (see --seed)
  --size 0.4           logo width as a fraction of the surface width (watermark: fraction of the video width)
  --keep_bg            keep a white logo background instead of making it transparent
  --simple             old behaviour: logo only follows the surface box (no flow tracking, no fabric shading)
  --fold 0.6           how much the logo bends with folds (0 = rigid)
  --blend_ckpt FILE    use a trained LogoBlend model to light, bend and occlude the logo (natural mode only)
"""
import argparse
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image

# surface name -> (CLIPSeg prompt, where on the surface the logo goes: (x, y) as a fraction of the surface box)
CANDIDATES = {
    "t-shirt":   ("a t-shirt",   (0.5, 0.38)),
    "poster":    ("a poster",    (0.5, 0.5)),
    "sign":      ("a sign",      (0.5, 0.5)),
    "billboard": ("a billboard", (0.5, 0.5)),
    "screen":    ("a screen",    (0.5, 0.5)),
    "box":       ("a box",       (0.5, 0.5)),
    "wall":      ("a wall",      (0.5, 0.35)),
}
WORK_SIDE = 320       # masks are computed at this size (long side)
THRESH = 0.4          # mask probability threshold
MIN_AREA = 0.02       # a surface must cover at least 2% of the frame to count
MIN_PRESENCE = 0.5    # ... in at least half of the frames
AREA_CAP = 0.4        # larger surfaces do not score higher
MAX_SURFACE_AREA = 0.6   # a 'surface' covering most of the frame is the background, not a shirt or a poster
MAX_LOGO_FRAC = 0.3      # the logo is never wider than this fraction of the video width
BODY_PROMPTS = ["a person's face", "a person's hand"]     # kept off the garment mask so the logo lands on cloth
APPAREL_WORDS = ("shirt", "tee", "hoodie", "jacket", "sweater", "dress", "jersey", "polo", "blouse")


# ------------------------------------------------------------------ video io
def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise SystemExit(f"Could not read any frames from {path}")
    return frames, fps


def write_video(frames, fps, path):
    h, w = frames[0].shape[:2]
    tmp = tempfile.mkdtemp()
    raw = os.path.join(tmp, "raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    if shutil.which("ffmpeg"):   # H.264 plays everywhere (browsers, Colab, phones)
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-pix_fmt", "yuv420p", path])
        if r.returncode == 0:
            shutil.rmtree(tmp, ignore_errors=True)
            return
    print("ffmpeg not available: wrote mp4v video (may not play in a browser)")
    shutil.move(raw, path)
    shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ logo
def load_logo(path, keep_bg=False):
    img = Image.open(path).convert("RGBA")
    a = np.array(img).astype(np.float32)
    if not keep_bg and a[..., 3].min() >= 250:     # no transparency: treat a near-white background as transparent
        d = 255.0 - a[..., :3].min(axis=2)         # 0 for white, large for colored/dark pixels
        a[..., 3] = np.clip((d - 15.0) / 45.0, 0, 1) * 255.0
    ys, xs = np.where(a[..., 3] > 8)               # crop to the visible part
    if len(xs) == 0:
        raise SystemExit("The logo has no visible pixels.")
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return a[..., :3] / 255.0, a[..., 3] / 255.0   # rgb [0,1], alpha [0,1]


# ------------------------------------------------------------------ segmentation
class ClipSegSegmenter:
    """Text-prompted segmentation (CIDAS/clipseg-rd64-refined, downloaded from Hugging Face on first use)."""

    def __init__(self, device=None):
        import torch
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        name = "CIDAS/clipseg-rd64-refined"
        self.proc = CLIPSegProcessor.from_pretrained(name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(name).to(self.device).eval()

    def __call__(self, frames, prompts):
        """frames: list of HxWx3 uint8 -> probabilities [P, T, H, W] in [0, 1]"""
        torch = self.torch
        h, w = frames[0].shape[:2]
        out = np.zeros((len(prompts), len(frames), h, w), np.float32)
        with torch.no_grad():
            for t, f in enumerate(frames):
                inp = self.proc(text=prompts, images=[f] * len(prompts), padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**inp).logits                      # [P, 352, 352]
                if logits.dim() == 2:
                    logits = logits[None]
                p = torch.sigmoid(logits)[:, None]
                p = torch.nn.functional.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)[:, 0]
                out[:, t] = p.cpu().numpy()
        return out


# ------------------------------------------------------------------ scoring and tracking
def score_surface(masks):
    """masks: bool [T, H, W] -> (score, presence, mean_area, stability)"""
    T = masks.shape[0]
    area = masks.reshape(T, -1).mean(1)
    present = area >= MIN_AREA
    presence = float(present.mean())
    if presence == 0:
        return 0.0, 0.0, 0.0, 0.0
    if float(area[present].mean()) > MAX_SURFACE_AREA:      # whole-frame backgrounds never win
        return 0.0, presence, float(area[present].mean()), 0.0
    mean_area = float(np.minimum(area[present], AREA_CAP).mean())
    ious = []
    for t in range(1, T):
        if present[t] and present[t - 1]:
            a, b = masks[t - 1], masks[t]
            ious.append((a & b).sum() / max((a | b).sum(), 1))
    stability = float(np.mean(ious)) if ious else 0.5
    score = presence * np.sqrt(mean_area / AREA_CAP) * (0.5 + 0.5 * stability)
    return float(score), presence, mean_area, stability


def largest_box(mask):
    """bool [H, W] -> (x0, y0, x1, y1) of the largest connected region, or None"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[i, :4]
    return np.array([x, y, x + w, y + h], np.float32)


def track_boxes(masks, window=5):
    """Per-frame boxes for a surface. Missing frames are filled from neighbours, then smoothed over time."""
    boxes = [largest_box(m) if m.mean() >= MIN_AREA else None for m in masks]
    idx = [i for i, b in enumerate(boxes) if b is not None]
    if not idx:
        return None
    for i in range(len(boxes)):
        if boxes[i] is None:
            boxes[i] = boxes[min(idx, key=lambda j: abs(j - i))].copy()
    b = np.stack(boxes)                                    # [T, 4]
    k = min(window, len(b)) | 1
    pad = k // 2
    bp = np.pad(b, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(bp[:, c], np.ones(k) / k, mode="valid") for c in range(4)], axis=1)


def is_apparel(name):
    return any(w in name.lower() for w in APPAREL_WORDS)


def remove_body(shirt, body):
    """shirt, body: bool [T, h, w]. Takes faces and hands out of the garment mask and keeps its biggest piece, so the
    logo is placed on the cloth and not on the face. If the body mask swallows the garment it is ignored."""
    out = shirt.copy()
    k = np.ones((7, 7), np.uint8)
    for t in range(len(shirt)):
        total = int(shirt[t].sum())
        m = shirt[t] & ~cv2.dilate(body[t].astype(np.uint8), k).astype(bool)
        if total == 0 or m.sum() < 0.3 * total:
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n > 1:
            m = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        out[t] = m
    return out


def choose_surface(scores, rng, randomize, min_score=0.15):
    ok = {k: v for k, v in scores.items() if v[1] >= MIN_PRESENCE and v[0] >= min_score}
    if not ok:
        return None
    best = max(v[0] for v in ok.values())
    good = [k for k, v in ok.items() if v[0] >= 0.6 * best]
    if randomize and len(good) > 1:
        w = np.array([ok[k][0] for k in good])
        return good[int(rng.choice(len(good), p=w / w.sum()))]
    return max(ok, key=lambda k: ok[k][0])


# ------------------------------------------------------------------ drawing
def paste(frame, rgb, alpha, cx, cy, shade_from=None, opacity=0.95):
    """Alpha-blend the logo (centered at cx, cy) into the frame. shade_from: bool mask of the surface for lighting."""
    H, W = frame.shape[:2]
    lh, lw = alpha.shape
    x0, y0 = int(round(cx - lw / 2)), int(round(cy - lh / 2))
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + lw, W), min(y0 + lh, H)
    if X1 <= X0 or Y1 <= Y0:
        return frame
    lrgb = rgb[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0].copy()
    la = alpha[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] * opacity
    region = frame[Y0:Y1, X0:X1].astype(np.float32) / 255.0
    if shade_from is not None:                              # match the surface's lighting (soft shadows and highlights)
        gray = region.mean(2)
        ksz = max(3, (min(lw, lh) // 3) | 1)
        local = cv2.GaussianBlur(gray, (ksz, ksz), 0)
        ref = np.median(frame[shade_from].astype(np.float32).mean(1) / 255.0) if shade_from.any() else local.mean()
        shade = np.clip(local / max(ref, 1e-3), 0.6, 1.15)
        lrgb = np.clip(lrgb * shade[..., None], 0, 1)
    out = region * (1 - la[..., None]) + lrgb * la[..., None]
    frame = frame.copy()
    frame[Y0:Y1, X0:X1] = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return frame


def resize_logo(rgb, alpha, width):
    width = max(int(round(width)), 4)
    height = max(int(round(width * alpha.shape[0] / alpha.shape[1])), 4)
    return (cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA),
            cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA))


def apply_logo_box(frames, rgb, alpha, masks, anchor, size, max_frac=MAX_LOGO_FRAC):
    """Simple mode: logo follows the surface box (position and scale only). masks: bool [T, hs, ws]"""
    H, W = frames[0].shape[:2]
    hs, ws = masks.shape[1:]
    sx, sy = W / ws, H / hs
    boxes = track_boxes(masks)
    out = []
    for t, f in enumerate(frames):
        x0, y0, x1, y1 = boxes[t]
        bw, bh = (x1 - x0) * sx, (y1 - y0) * sy
        lw = min(size * bw, max_frac * W)
        if lw * alpha.shape[0] / alpha.shape[1] > 0.6 * bh:        # keep tall logos inside short surfaces
            lw = 0.6 * bh * alpha.shape[1] / alpha.shape[0]
        r, a = resize_logo(rgb, alpha, lw)
        cx = x0 * sx + anchor[0] * bw
        cy = y0 * sy + anchor[1] * bh
        surf = cv2.resize(masks[t].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        out.append(paste(f, r, a, cx, cy, shade_from=surf))
    return out



# ------------------------------------------------------------------ natural mode: flow tracking + fabric shading
TEXTURE_MIN = 1.5      # surface texture (std of fine detail, 0-255) needed to trust optical flow


def surface_tilt(mask):
    """Tilt of the surface's long axis from vertical, in degrees (0 if it has no clear long axis)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return 0.0
    c = np.cov(np.stack([xs, ys]).astype(np.float64))
    w, v = np.linalg.eigh(c)
    if w[1] < 1.15 * w[0]:
        return 0.0
    ax = v[:, 1]                                              # long axis
    a = abs(float(np.degrees(np.arctan2(ax[0], ax[1]))))
    return min(a, 180.0 - a)


def surface_texture(gray_small, mask):
    """How much fine detail the surface has (flat colour -> flow is unreliable)."""
    hf = gray_small.astype(np.float32) - cv2.GaussianBlur(gray_small.astype(np.float32), (0, 0), 1.5)
    inner = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return float(hf[inner].std()) if inner.sum() > 20 else 0.0


def regularize_flow(flow, mask, fold, sigma):
    """Robust motion of the surface: one similarity transform (move/rotate/scale, fitted with RANSAC on the surface)
    plus a smoothed, damped residual that lets the logo bend with folds."""
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h:4, 0:w:4]
    sel = mask[ys, xs]
    pts = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    if len(pts) >= 12:
        dst = pts + flow[pts[:, 1].astype(int), pts[:, 0].astype(int)]
        M, _ = cv2.estimateAffinePartial2D(pts, dst, method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if M is not None:
            px = M[0, 0] * gx + M[0, 1] * gy + M[0, 2] - gx
            py = M[1, 0] * gx + M[1, 1] * gy + M[1, 2] - gy
            param = np.stack([px, py], -1).astype(np.float32)
            resid = cv2.GaussianBlur(flow - param, (0, 0), sigma) * fold
            return param + resid
    return cv2.GaussianBlur(flow, (0, 0), sigma) * fold


def flow_maps(gray_small, ref, masks, fold, sigma):
    """maps[t][y, x] = where pixel (x, y) of frame t was in the reference frame (small-image coordinates).
    Built by chaining dense optical flow outwards from the reference frame in both directions."""
    T, (h, w) = len(gray_small), gray_small[0].shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    maps = [None] * T
    maps[ref] = np.stack([gx, gy], -1)
    for step in (1, -1):
        t = ref + step
        while 0 <= t < T:
            prev = t - step
            flow = cv2.calcOpticalFlowFarneback(gray_small[t], gray_small[prev], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = regularize_flow(flow, masks[t], fold, sigma).astype(np.float32)
            mx = cv2.remap(maps[prev][..., 0], gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            my = cv2.remap(maps[prev][..., 1], gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            maps[t] = np.stack([mx, my], -1)
            t += step
    return maps


def composite_natural(frame, layer, surf, t):
    """Blend the warped logo layer (H, W, 4; premultiplied-free rgb in 0-1 + alpha) into the frame so that it
    takes the fabric's lighting, the footage's softness/grain and a trace of its colour cast."""
    a = layer[..., 3]
    ys, xs = np.where(a > 0.02)
    if len(xs) == 0:
        return frame
    H, W = a.shape
    lw = max(xs.max() - xs.min(), 8)
    sig_low = max(3.0, 0.25 * lw)
    pad = int(3 * sig_low) + 4
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)
    reg = frame[y0:y1, x0:x1].astype(np.float32) / 255.0
    gray = reg.mean(2)
    lay = layer[y0:y1, x0:x1]
    rgb, al = lay[..., :3], lay[..., 3]

    # softness: premultiplied blur so edges do not get dark fringes
    prem = cv2.GaussianBlur(rgb * al[..., None], (0, 0), 0.7)
    al = cv2.GaussianBlur(al, (0, 0), 0.7)
    rgb = prem / np.maximum(al[..., None], 1e-4)

    # lighting: slow brightness of the fabric + folds/creases (mid-frequency detail relative to the slow part)
    sm = surf[y0:y1, x0:x1]
    low = cv2.GaussianBlur(gray, (0, 0), sig_low)
    mid = cv2.GaussianBlur(gray, (0, 0), max(2.5, 0.06 * lw))      # crease scale, coarse enough not to amplify footage noise
    ref = float(np.median(gray[sm])) if sm.sum() > 20 else float(gray.mean())
    base = np.clip(low / max(ref, 1e-3), 0.8, 1.2)
    detail = np.clip(mid / np.maximum(low, 1e-3), 0.55, 1.4) ** 0.6
    detail = np.clip(detail, 0.8, 1.25)
    shade = (base * detail)[..., None]

    # colour cast of the surface, kept subtle
    if sm.sum() > 20:
        mean = reg[sm].mean(0)
        tint = 1.0 + 0.25 * (mean / max(mean.mean(), 1e-3) - 1.0)
    else:
        tint = np.ones(3, np.float32)

    col = np.clip(rgb * shade * tint, 0, 1)
    # grain: the same amount of fine noise the footage has on the surface
    hf = gray - cv2.GaussianBlur(gray, (0, 0), 1.5)
    noise = float(hf[sm].std()) if sm.sum() > 20 else float(hf.std())
    noise = min(noise * 0.35, 0.02)
    col = np.clip(col + np.random.default_rng(t).normal(0, noise, col.shape[:2])[..., None], 0, 1)

    opa = (al * 0.95)[..., None]
    out = reg * (1 - opa) + col * opa
    frame = frame.copy()
    frame[y0:y1, x0:x1] = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return frame


def apply_logo_natural(frames, small, rgb, alpha, masks, anchor, size, fold=0.6, max_frac=MAX_LOGO_FRAC, blender=None):
    """Logo that follows the surface (moving, turning, bending with folds) and takes its lighting."""
    T = len(frames)
    H, W = frames[0].shape[:2]
    hs, ws = masks.shape[1:]
    sx, sy = W / ws, H / hs
    boxes = track_boxes(masks)
    area = masks.reshape(T, -1).mean(1)
    # reference frame: among the frames where the surface is large and clear, the one closest to upright, so the
    # logo sits straight on the garment and then turns with it in the other frames
    good = [t for t in range(T) if area[t] >= 0.85 * area.max()]
    ref = min(good, key=lambda t: (surface_tilt(masks[t]), -area[t]))

    # place the logo once, in the reference frame, on a full-size transparent layer
    x0, y0, x1, y1 = boxes[ref]
    bw, bh = (x1 - x0) * sx, (y1 - y0) * sy
    lw = min(size * bw, max_frac * W)
    if lw * alpha.shape[0] / alpha.shape[1] > 0.6 * bh:
        lw = 0.6 * bh * alpha.shape[1] / alpha.shape[0]
    r, a = resize_logo(rgb, alpha, lw)
    layer0 = np.zeros((H, W, 4), np.float32)
    lh, lwi = a.shape
    cx0, cy0 = x0 * sx + anchor[0] * bw, y0 * sy + anchor[1] * bh
    px, py = int(round(cx0 - lwi / 2)), int(round(cy0 - lh / 2))
    X0, Y0, X1, Y1 = max(px, 0), max(py, 0), min(px + lwi, W), min(py + lh, H)
    if X1 <= X0 or Y1 <= Y0:
        return frames
    layer0[Y0:Y1, X0:X1, :3] = r[Y0 - py:Y1 - py, X0 - px:X1 - px]
    layer0[Y0:Y1, X0:X1, 3] = a[Y0 - py:Y1 - py, X0 - px:X1 - px]

    gray_small = [cv2.cvtColor(cv2.resize(f, (ws, hs), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY) for f in frames]
    maps = flow_maps(gray_small, ref, masks, fold, sigma=max(2.0, 0.15 * lw / sx))

    out = []
    for t, f in enumerate(frames):
        mp = maps[t]
        mx = cv2.resize(mp[..., 0], (W, H), interpolation=cv2.INTER_LINEAR) * sx
        my = cv2.resize(mp[..., 1], (W, H), interpolation=cv2.INTER_LINEAR) * sy
        layer = cv2.remap(layer0, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # keep the logo on its surface: flow can drift, so pull it back toward the surface box
        bx0, by0, bx1, by1 = boxes[t]
        bwt = (bx1 - bx0) * sx
        tx, ty = bx0 * sx + anchor[0] * (bx1 - bx0) * sx, by0 * sy + anchor[1] * (by1 - by0) * sy
        al = layer[..., 3]
        if al.sum() > 1e-3:
            yy, xx = np.mgrid[0:H, 0:W]
            cx, cy = (al * xx).sum() / al.sum(), (al * yy).sum() / al.sum()
            textured = surface_texture(gray_small[t], masks[t]) >= TEXTURE_MIN
            # flat surface: nothing to track, stick to the box. Otherwise the logo may drift a little, never far
            radius = min(0.12 * bwt, 0.5 * lw) if textured else 0.0
            dx, dy = tx - cx, ty - cy
            dist = float(np.hypot(dx, dy))
            if dist > radius:
                k = (dist - radius) / dist
                M = np.float32([[1, 0, dx * k], [0, 1, dy * k]])
                layer = cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        surf = cv2.resize(masks[t].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        # blender: an optional trained LogoBlend model that lights, bends and occludes the logo; else the hand-made blend
        out.append(blender(f, layer, surf) if blender is not None else composite_natural(f, layer, surf, t))
    return out


def apply_watermark(frames, rgb, alpha, size):
    H, W = frames[0].shape[:2]
    r, a = resize_logo(rgb, alpha, size * W)
    m = int(0.03 * W)
    cx, cy = W - m - r.shape[1] / 2, H - m - r.shape[0] / 2
    return [paste(f, r, a, cx, cy, opacity=0.85) for f in frames]


# ------------------------------------------------------------------ main
def add_logo(frames, logo_path, segmenter, target=None, randomize=False, seed=0, size=None, keep_bg=False,
             natural=True, fold=0.6, max_logo_frac=MAX_LOGO_FRAC, blender=None):
    rgb, alpha = load_logo(logo_path, keep_bg)
    H, W = frames[0].shape[:2]
    s = WORK_SIDE / max(H, W)
    small = [cv2.resize(f, (max(int(W * s), 8), max(int(H * s), 8)), interpolation=cv2.INTER_AREA) for f in frames]

    if target:
        names, prompts, anchors = [target], [target], {target: (0.5, 0.45)}
        if target in CANDIDATES:
            prompts, anchors = [CANDIDATES[target][0]], {target: CANDIDATES[target][1]}
    else:
        names = list(CANDIDATES)
        prompts = [CANDIDATES[n][0] for n in names]
        anchors = {n: CANDIDATES[n][1] for n in names}

    probs = segmenter(small, prompts + BODY_PROMPTS)               # [P + 2, T, h, w]
    body = (probs[len(prompts)] > THRESH) | (probs[len(prompts) + 1] > THRESH)
    masks = {n: probs[i] > THRESH for i, n in enumerate(names)}
    scores = {n: score_surface(masks[n]) for n in names}
    rng = np.random.default_rng(seed)
    pick = choose_surface(scores, rng, randomize) if not target else (
        target if scores[target][1] >= MIN_PRESENCE else None)

    report = {"scores": scores, "surface": pick}
    if pick is None:
        report["mode"] = "watermark"
        return apply_watermark(frames, rgb, alpha, size or 0.16), report
    report["mode"] = "surface"
    surface_mask, anchor = masks[pick], anchors[pick]
    if is_apparel(pick):                                           # on clothes: not on the face or hands, on the chest
        surface_mask = remove_body(surface_mask, body)
        anchor = (0.5, 0.38)
    if natural and len(frames) > 1:
        return apply_logo_natural(frames, small, rgb, alpha, surface_mask, anchor, size or 0.4, fold, max_logo_frac, blender), report
    return apply_logo_box(frames, rgb, alpha, surface_mask, anchor, size or 0.4, max_logo_frac), report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--logo", required=True)
    p.add_argument("--out", default=None, help="default: <video>_logo.mp4")
    p.add_argument("--target", default=None, help="optional surface, e.g. 't-shirt'. Default: chosen automatically")
    p.add_argument("--random", action="store_true", help="choose randomly among the good surfaces")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=float, default=None, help="logo width as a fraction of the surface (default 0.4; watermark 0.16)")
    p.add_argument("--keep_bg", action="store_true")
    p.add_argument("--simple", action="store_true", help="only move/scale the logo with the surface box (no flow tracking or fabric shading)")
    p.add_argument("--fold", type=float, default=0.6, help="how much the logo bends with folds, 0 = rigid, 1 = full (default 0.6)")
    p.add_argument("--blend_ckpt", default=None, help="a trained LogoBlend checkpoint (see logoblend/README.md)")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    frames, fps = read_video(a.video)
    seg = ClipSegSegmenter(a.device)
    blender = None
    if a.blend_ckpt:
        from logoblend.apply import LogoBlender
        blender = LogoBlender(a.blend_ckpt, a.device)
    result, rep = add_logo(frames, a.logo, seg, a.target, a.random, a.seed, a.size, a.keep_bg, not a.simple, a.fold,
                           blender=blender)

    print("surface scores (score, presence, area, stability):")
    for n, v in sorted(rep["scores"].items(), key=lambda kv: -kv[1][0]):
        print(f"  {n:10s} {v[0]:.2f}  {v[1]:.2f}  {v[2]:.2f}  {v[3]:.2f}")
    if rep["mode"] == "surface":
        print(f"-> logo placed on: {rep['surface']}")
    else:
        print("-> no suitable surface found" + (f" for '{a.target}'" if a.target else "") + ": added a corner watermark")
    out = a.out or os.path.splitext(a.video)[0] + "_logo.mp4"
    write_video(result, fps, out)
    print("saved", out)


if __name__ == "__main__":
    main()
