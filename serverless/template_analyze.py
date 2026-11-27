"""Template analysis for the Ad maker (task "analyze").

Input : an ad picture the user uploaded as a template.
Output: the ad's LAYERS, in the same JSON the browser already builds ads from:
          background (colour + vertical gradient), text blocks (words' position, colour, size, alignment, role),
          the main photo (cut-out or framed), small photos, logos, flat shapes (like a white backing shape) and
          small decorations (confetti, stars...) cropped out as transparent pictures.

Models (all open, run here on the GPU):
  OCR            EasyOCR (English + Hebrew)       words and where they are
  Detection      Grounding DINO (HF transformers) person / product / logo boxes from a text prompt
  Segmentation   SAM (HF transformers)            exact shape of each box + automatic masks for shapes & decorations
Colours, gradients, text colour, alignment and roles are computed from pixels with numpy / OpenCV.
"""
import base64
import io

import numpy as np

DINO_ID = "IDEA-Research/grounding-dino-base"
SAM_ID = "facebook/sam-vit-base"
MAX_SIDE = 1280
_models = {}


class AnalyzeError(Exception):
    pass


# ------------------------------------------------------------------ model loading
def _dev():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_ocr():
    if "ocr" not in _models:
        import easyocr
        _models["ocr"] = easyocr.Reader(["en", "he"], gpu=_dev() == "cuda", verbose=False)
    return _models["ocr"]


def get_dino():
    if "dino" not in _models:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        proc = AutoProcessor.from_pretrained(DINO_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_ID).to(_dev()).eval()
        _models["dino"] = (proc, model)
    return _models["dino"]


def get_sam():
    if "sam" not in _models:
        from transformers import SamModel, SamProcessor
        proc = SamProcessor.from_pretrained(SAM_ID)
        model = SamModel.from_pretrained(SAM_ID).to(_dev()).eval()
        _models["sam"] = (proc, model)
    return _models["sam"]


# ------------------------------------------------------------------ small helpers
def hexc(rgb):
    r, g, b = [int(max(0, min(255, round(float(v))))) for v in rgb]
    return "#%02x%02x%02x" % (r, g, b)


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def inside_frac(a, b):
    """Fraction of box a that lies inside box b."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0, min(ay1, by1) - max(ay0, by0))
    area = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
    return iw * ih / area


def png_b64(rgba):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def crop_png(rgb, mask, box, flat=False, max_side=384):
    """Transparent PNG of `box` (x0,y0,x1,y1 in pixels): the pixels under `mask`, or a flat fill with their median colour."""
    import cv2
    x0, y0, x1, y1 = [int(v) for v in box]
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    m = mask[y0:y1, x0:x1]
    if not m.any():
        return None
    px = rgb[y0:y1, x0:x1].copy()
    if flat:
        px[:] = np.median(rgb[mask], axis=0).astype(np.uint8)
    a = (m.astype(np.uint8) * 255)
    a = cv2.GaussianBlur(a, (3, 3), 0)
    rgba = np.dstack([px, a])
    s = max(rgba.shape[:2])
    if s > max_side:
        k = max_side / s
        rgba = cv2.resize(rgba, (max(1, int(rgba.shape[1] * k)), max(1, int(rgba.shape[0] * k))), interpolation=cv2.INTER_AREA)
    return png_b64(rgba)


# ------------------------------------------------------------------ 1. text
def read_text(rgb):
    """EasyOCR lines -> blocks (a block is lines of one paragraph). Boxes in pixels."""
    H, W = rgb.shape[:2]
    lines = []
    for pts, text, conf in get_ocr().readtext(rgb):
        text = (text or "").strip()
        if not text or conf < 0.2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, y0, x1, y1 = max(0, min(xs)), max(0, min(ys)), min(W, max(xs)), min(H, max(ys))
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        lines.append({"text": text, "box": [x0, y0, x1, y1]})
    lines.sort(key=lambda l: (l["box"][1], l["box"][0]))
    blocks = []
    for ln in lines:
        x0, y0, x1, y1 = ln["box"]
        h = y1 - y0
        placed = False
        for b in blocks:
            bx0, by0, bx1, by1 = b["box"]
            lh = b["lh"]
            gap = y0 - by1
            same_size = 0.62 <= h / lh <= 1.6
            overlap = min(x1, bx1) - max(x0, bx0)
            aligned = overlap > 0.3 * min(x1 - x0, bx1 - bx0)
            if same_size and aligned and -0.2 * lh <= gap <= 0.9 * lh:
                b["lines"].append(ln)
                b["box"] = [min(bx0, x0), min(by0, y0), max(bx1, x1), max(by1, y1)]
                b["lh"] = float(np.mean([l["box"][3] - l["box"][1] for l in b["lines"]]))
                placed = True
                break
        if not placed:
            blocks.append({"lines": [ln], "box": list(ln["box"]), "lh": float(h)})
    return blocks


def text_style(rgb, block):
    """Colour, ink share (boldness) and alignment of a text block, from its pixels."""
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in block["box"]]
    pad = max(3, int(0.15 * block["lh"]))
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    region = rgb[Y0:Y1, X0:X1].reshape(-1, 3).astype(np.float32)
    ring = np.ones((Y1 - Y0, X1 - X0), bool)
    ring[(y0 - Y0):(y1 - Y0), (x0 - X0):(x1 - X0)] = False
    bg = np.median(rgb[Y0:Y1, X0:X1][ring].reshape(-1, 3), axis=0) if ring.sum() > 20 else np.median(region, axis=0)
    inner = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    dist = np.linalg.norm(inner - bg, axis=1)
    ink = dist > 70
    ratio = float(ink.mean()) if len(ink) else 0.0
    if ink.sum() > 10:
        color = np.median(inner[ink], axis=0)
    else:
        color = np.array([255, 255, 255]) if bg.mean() < 128 else np.array([17, 24, 39])
    lines = block["lines"]
    if len(lines) > 1:
        L = [l["box"][0] for l in lines]
        R = [l["box"][2] for l in lines]
        C = [(l["box"][0] + l["box"][2]) / 2 for l in lines]
        spread = lambda v: max(v) - min(v)
        s = min(("left", spread(L)), ("right", spread(R)), ("center", spread(C)), key=lambda t: t[1])
        align = s[0]
    else:
        cx = (x0 + x1) / 2 / W
        align = "left" if x0 / W < 0.08 else "right" if x1 / W > 0.92 else "center" if 0.4 < cx < 0.6 else ("left" if cx < 0.5 else "right")
    return {"color": hexc(color), "ink": ratio, "align": align}


# ------------------------------------------------------------------ 2. detection + exact shapes
def detect(img):
    import torch
    proc, model = get_dino()
    prompt = "a person. a logo. a product. a bottle. a jar. a box."
    inputs = proc(images=img, text=prompt, return_tensors="pt").to(_dev())
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=0.3, text_threshold=0.25, target_sizes=[img.size[::-1]])[0]
    labels = res.get("text_labels") or res.get("labels") or []
    found = []
    for box, score, lab in zip(res["boxes"].cpu().numpy().tolist(), res["scores"].cpu().numpy().tolist(), labels):
        lab = str(lab).lower()
        kind = "person" if "person" in lab else "logo" if "logo" in lab else "product" if lab.strip() else None
        if kind:
            found.append({"kind": kind, "box": box, "score": float(score)})
    # same class, overlapping: keep the best
    found.sort(key=lambda d: -d["score"])
    keep = []
    for d in found:
        if all(not (k["kind"] == d["kind"] and iou(k["box"], d["box"]) > 0.5) for k in keep):
            keep.append(d)
    return keep


def sam_masks_for_boxes(img_np, boxes):
    """Exact mask (bool HxW) for each box, from SAM."""
    import torch
    from PIL import Image
    proc, model = get_sam()
    H, W = img_np.shape[:2]
    out = []
    for b in boxes:
        inputs = proc(Image.fromarray(img_np), input_boxes=[[list(map(float, b))]], return_tensors="pt").to(_dev())
        with torch.no_grad():
            res = model(**inputs, multimask_output=False)
        m = proc.image_processor.post_process_masks(res.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
        out.append(m[0, 0].numpy().astype(bool))
    return out


def auto_masks(img):
    """Automatic masks over the whole picture (everything SAM can find)."""
    from transformers import pipeline
    proc, model = get_sam()
    gen = pipeline("mask-generation", model=model, image_processor=proc.image_processor, device=0 if _dev() == "cuda" else -1)
    out = gen(img, points_per_batch=64, pred_iou_thresh=0.86, stability_score_thresh=0.9, crops_nms_thresh=0.7, points_per_crop=24)
    return [np.asarray(m).astype(bool) for m in out["masks"]]


# ------------------------------------------------------------------ 3. put it together
def analyze(img):
    import cv2
    from PIL import Image
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    rgb = np.asarray(img)
    H, W = rgb.shape[:2]
    fx = lambda v: round(float(v) / W, 4)
    fy = lambda v: round(float(v) / H, 4)

    blocks = read_text(rgb)
    objs = detect(img)

    # a logo box that holds text: that text is part of the logo, not an editable line
    logos = [o for o in objs if o["kind"] == "logo"]
    blocks = [b for b in blocks if not any(inside_frac(b["box"], l["box"]) > 0.6 for l in logos)]
    # a "product" that is really a piece of text, or lies inside the person: drop
    people = [o for o in objs if o["kind"] == "person"]
    prods = [o for o in objs if o["kind"] == "product" and not any(inside_frac(o["box"], p["box"]) > 0.6 for p in people)
             and not any(inside_frac(o["box"], b["box"]) > 0.5 for b in blocks)]
    people = sorted(people, key=lambda o: -((o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1])))[:1]
    prods = [o for o in prods if (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]) < 0.35 * W * H]

    known = people + prods + logos
    kmasks = sam_masks_for_boxes(rgb, [o["box"] for o in known]) if known else []
    layers = []
    taken = np.zeros((H, W), bool)       # pixels explained by text / photos / logos
    for b in blocks:
        x0, y0, x1, y1 = [int(v) for v in b["box"]]
        taken[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2] = True
    person_mask = None
    for o, m in zip(known, kmasks):
        taken |= m
        ys, xs = np.where(m)
        if not len(xs):
            continue
        bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        o["mask"] = m
        o["tight"] = [bx0, by0, bx1, by1]
        o["fill"] = float(m.sum()) / max(1.0, (bx1 - bx0) * (by1 - by0))
        if o["kind"] == "person":
            person_mask = m

    # ---- shapes and decorations from the automatic masks
    shapes, decor = [], []
    for m in auto_masks(img):
        area = float(m.sum())
        frac = area / (W * H)
        if frac < 0.00008 or frac > 0.6:
            continue
        if (m & taken).sum() > 0.5 * area:
            continue
        ys, xs = np.where(m)
        box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        spread = float(np.mean(np.std(rgb[m].astype(np.float32), axis=0)))
        if spread < 16 and frac > 0.01:
            shapes.append({"mask": m, "box": box, "frac": frac})
        elif frac < 0.02:
            decor.append({"mask": m, "box": box, "frac": frac})
    shapes.sort(key=lambda s: -s["frac"])
    shapes = shapes[:6]
    decor.sort(key=lambda d: -d["frac"])
    decor = decor[:70]

    k = max(9, int(0.03 * max(W, H))) | 1
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for s in shapes:
        m = s["mask"]
        if person_mask is not None:           # the part hidden behind the person belongs to the shape
            m = m | (person_mask & (cv2.dilate(m.astype(np.uint8), kern) > 0))
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, kern) > 0
        ys, xs = np.where(m)
        box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        src = crop_png(rgb, m, box, flat=True, max_side=512)
        if src:
            layers.append({"type": "decor", "x": fx(box[0]), "y": fy(box[1]), "w": fx(box[2] - box[0]), "h": fy(box[3] - box[1]), "src": src, "kind": "shape"})
            taken |= m
    for d in decor:
        src = crop_png(rgb, d["mask"], d["box"])
        if src:
            b = d["box"]
            layers.append({"type": "decor", "x": fx(b[0]), "y": fy(b[1]), "w": fx(b[2] - b[0]), "h": fy(b[3] - b[1]), "src": src, "kind": "decoration"})
            taken |= d["mask"]

    # ---- background: the pixels nobody claimed, as a vertical gradient
    free = ~cv2.dilate(taken.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    stops = []
    for i in range(6):
        y0, y1 = int(H * i / 6), int(H * (i + 1) / 6)
        px = rgb[y0:y1][free[y0:y1]]
        if len(px) > 200:
            stops.append([round((i + 0.5) / 6, 3), hexc(np.median(px, axis=0))])
    if not stops:
        stops = [[0.5, hexc(np.median(rgb.reshape(-1, 3), axis=0))]]
    background = hexc(np.median(rgb[free], axis=0)) if free.sum() > 200 else stops[0][1]

    # ---- photos, logos
    for o in known:
        if "tight" not in o:
            continue
        bx0, by0, bx1, by1 = o["tight"] if o["kind"] != "product" else [int(v) for v in o["box"]]
        bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(W, bx1), min(H, by1)
        base = {"x": fx(bx0), "y": fy(by0), "w": fx(bx1 - bx0), "h": fy(by1 - by0)}
        if o["kind"] == "person":
            layers.append({"type": "person", "frame": "rounded" if o["fill"] > 0.88 else "cutout", **base})
        elif o["kind"] == "product":
            layers.append({"type": "product", **base})
        else:
            layers.append({"type": "logo", "align": "center", **base})
    prod = sorted([l for l in layers if l["type"] == "product"], key=lambda l: (round(l["y"] * 8), l["x"]))   # left to right, then top to bottom
    layers = [l for l in layers if l["type"] != "product"] + prod

    # ---- text: style each block, then give it a role
    styled = []
    for b in blocks:
        st = text_style(rgb, b)
        x0, y0, x1, y1 = b["box"]
        n = len(b["lines"])
        txt = " ".join(l["text"] for l in b["lines"])
        styled.append({"b": b, "st": st, "n": n, "lh": b["lh"] / H, "text": txt, "cy": (y0 + y1) / 2 / H,
                       "digits": any(ch.isdigit() or ch == "%" for ch in txt)})
    roles = {}
    if styled:
        big = max(s["lh"] for s in styled)
        cand = [s for s in styled if s["digits"] and s["lh"] >= 0.6 * big] or [max(styled, key=lambda s: s["lh"])]
        offer = max(cand, key=lambda s: s["lh"])
        roles[id(offer)] = "offer"
        # a button text: sits inside a small wide flat shape
        for s in styled:
            if s is offer:
                continue
            bx = s["b"]["box"]
            for sh in shapes:
                sb = sh["box"]
                sw, shh = sb[2] - sb[0], sb[3] - sb[1]
                if inside_frac(bx, sb) > 0.9 and sw / max(1, shh) > 2.0 and sw < 0.6 * W and "cta" not in roles.values():
                    roles[id(s)] = "cta"
        for s in sorted(styled, key=lambda s: -s["lh"]):
            if id(s) in roles:
                continue
            r = "headline" if s["cy"] < offer["cy"] else "sub"
            if r not in roles.values():
                roles[id(s)] = r
    for s in styled:
        role = roles.get(id(s))
        if not role:
            continue
        x0, y0, x1, y1 = s["b"]["box"]
        weight = 800 if s["lh"] > 0.07 else 700 if s["st"]["ink"] > 0.2 else 500
        layers.append({"type": "text", "role": role, "x": fx(x0), "y": fy(y0), "w": fx(x1 - x0), "h": fy(y1 - y0),
                       "color": s["st"]["color"], "align": s["st"]["align"], "weight": weight,
                       "fontSize": round(s["lh"] * 1.05, 4), "uppercase": False, "lines": s["n"], "sample": s["text"][:40]})

    ratio = W / H
    aspect = "landscape" if ratio > 1.3 else "story" if ratio < 0.62 else "portrait" if ratio < 0.9 else "square"
    return {"aspect": aspect, "ratio": round(ratio, 4), "background": background, "gradient": stops, "layers": layers,
            "found": {"text": len(blocks), "persons": len(people), "products": len(prods), "logos": len(logos),
                      "shapes": len(shapes), "decorations": len(decor)}}
