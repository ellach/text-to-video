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
        # EasyOCR has no Hebrew model, but its text DETECTOR works on any script; the words are read with Tesseract below.
        _models["ocr"] = easyocr.Reader(["en"], gpu=_dev() == "cuda", verbose=False)
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


def crop_png(rgb, mask, box, flat=False, max_side=384, color=None):
    """Transparent PNG of `box` (x0,y0,x1,y1 in pixels): the pixels under `mask`, or a flat fill with their median colour."""
    import cv2
    x0, y0, x1, y1 = [int(v) for v in box]
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)
    m = mask[y0:y1, x0:x1]
    if not m.any():
        return None
    px = rgb[y0:y1, x0:x1].copy()
    if flat:
        px[:] = (np.asarray(color) if color is not None else np.median(rgb[mask], axis=0)).astype(np.uint8)
    a = (m.astype(np.uint8) * 255)
    a = cv2.GaussianBlur(a, (3, 3), 0)
    rgba = np.dstack([px, a])
    s = max(rgba.shape[:2])
    if s > max_side:
        k = max_side / s
        rgba = cv2.resize(rgba, (max(1, int(rgba.shape[1] * k)), max(1, int(rgba.shape[0] * k))), interpolation=cv2.INTER_AREA)
    return png_b64(rgba)


# ------------------------------------------------------------------ 1. text
def read_line_text(rgb, box):
    """The words inside a box, with Tesseract (Hebrew + English). Empty if Tesseract is not installed or finds nothing."""
    try:
        import pytesseract
        from PIL import Image
        x0, y0, x1, y1 = [int(v) for v in box]
        H, W = rgb.shape[:2]
        pad = 4
        crop = Image.fromarray(rgb[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)])
        return pytesseract.image_to_string(crop, lang="heb+eng", config="--psm 7").strip()
    except Exception as e:
        print("tesseract unavailable:", repr(e)[:120], flush=True)
        return ""


def read_text(rgb):
    """Text regions (EasyOCR detector, any language) -> lines (words on one row) -> blocks (lines of one paragraph). Boxes in pixels."""
    H, W = rgb.shape[:2]
    horizontal, free = get_ocr().detect(rgb, text_threshold=0.5, low_text=0.3, link_threshold=0.4, canvas_size=1280)
    boxes = []
    for x0, x1, y0, y1 in (horizontal[0] if horizontal else []):
        boxes.append([max(0, x0), max(0, y0), min(W, x1), min(H, y1)])
    for poly in (free[0] if free else []):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        boxes.append([max(0, min(xs)), max(0, min(ys)), min(W, max(xs)), min(H, max(ys))])
    boxes = [b for b in boxes if b[2] - b[0] >= 6 and b[3] - b[1] >= 8 and (b[3] - b[1]) <= 1.3 * (b[2] - b[0])]   # tall narrow boxes are sideways text on packaging
    # words on the same row -> one line
    boxes.sort(key=lambda b: b[0])
    rows = []
    for b in boxes:
        h = b[3] - b[1]
        for r in rows:
            rh = r[3] - r[1]
            vov = min(b[3], r[3]) - max(b[1], r[1])
            gap = max(0, b[0] - r[2], r[0] - b[2])
            if vov > 0.6 * min(h, rh) and 0.6 <= h / rh <= 1.6 and gap < 1.0 * max(h, rh):
                r[0], r[1], r[2], r[3] = min(r[0], b[0]), min(r[1], b[1]), max(r[2], b[2]), max(r[3], b[3])
                break
        else:
            rows.append(list(b))
    lines = [{"text": read_line_text(rgb, r), "box": r} for r in rows]
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
    res = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=0.2, text_threshold=0.2, target_sizes=[img.size[::-1]])[0]
    labels = res.get("text_labels") or res.get("labels") or []
    found = []
    for box, score, lab in zip(res["boxes"].cpu().numpy().tolist(), res["scores"].cpu().numpy().tolist(), labels):
        lab = str(lab).lower()
        kind = "person" if "person" in lab else "logo" if "logo" in lab else "product" if lab.strip() else None
        need = {"person": 0.35, "product": 0.3, "logo": 0.2}.get(kind)
        if kind and float(score) >= need:
            found.append({"kind": kind, "box": box, "score": float(score)})
    # same class, overlapping: keep the best
    found.sort(key=lambda d: -d["score"])
    keep = []
    for d in found:
        if all(not (k["kind"] == d["kind"] and iou(k["box"], d["box"]) > 0.5) for k in keep):
            keep.append(d)
    return keep


def detect_decor(img):
    """Ask Grounding DINO directly for confetti, sparkles and glowing lights (the model decides what counts)."""
    import torch
    proc, model = get_dino()
    inputs = proc(images=img, text="confetti. sparkles. glowing lights.", return_tensors="pt").to(_dev())
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=0.25, text_threshold=0.2, target_sizes=[img.size[::-1]])[0]
    boxes = res["boxes"].cpu().numpy().tolist()
    scores = res["scores"].cpu().numpy().tolist()
    keep = []
    for b, sc in sorted(zip(boxes, scores), key=lambda t: -t[1]):
        if all(iou(b, k) < 0.5 for k in keep):
            keep.append(b)
    return keep[:60]


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
def debug_picture(rgb, layers):
    """The template with a coloured box and a name on everything that was found (for checking the analysis)."""
    import cv2
    img = rgb.copy()
    H, W = img.shape[:2]
    col = {"text": (255, 40, 40), "person": (40, 80, 255), "product": (255, 150, 0), "logo": (220, 0, 220), "shape": (0, 220, 220), "decoration": (255, 230, 0)}
    for l in layers:
        kind = l.get("kind") if l["type"] == "decor" else l["type"]
        name = l.get("role") if l["type"] == "text" else (l.get("frame") and f"person ({l['frame']})") if l["type"] == "person" else kind
        c = col.get(kind, (255, 255, 255))
        x0, y0, x1, y1 = int(l["x"] * W), int(l["y"] * H), int((l["x"] + l["w"]) * W), int((l["y"] + l["h"]) * H)
        cv2.rectangle(img, (x0, y0), (x1, y1), c, 3 if kind != "decoration" else 1)
        if kind != "decoration":
            cv2.putText(img, str(name), (x0 + 4, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def analyze(img):
    """Only what the models find, with no sorting rules: text boxes (OCR detector), the person, products and logos
    (Grounding DINO), the person's exact outline (SAM) and the background colours (pixels no box covers)."""
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    rgb = np.asarray(img)
    H, W = rgb.shape[:2]
    fx = lambda v: round(float(v) / W, 4)
    fy = lambda v: round(float(v) / H, 4)

    blocks = read_text(rgb)
    objs = detect(img)
    logos = [o for o in objs if o["kind"] == "logo"]
    blocks = [b for b in blocks if not any(inside_frac(b["box"], l["box"]) > 0.6 for l in logos)]   # words inside a logo are the logo
    people = sorted([o for o in objs if o["kind"] == "person"], key=lambda o: -((o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1])))[:1]
    prods = [o for o in objs if o["kind"] == "product"]

    layers = []
    covered = np.zeros((H, W), bool)
    for b in blocks:
        x0, y0, x1, y1 = [int(v) for v in b["box"]]
        covered[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2] = True
    for o in people + prods + logos:
        x0, y0, x1, y1 = [int(v) for v in o["box"]]
        covered[max(0, y0):y1, max(0, x0):x1] = True

    # background colours from the pixels no box covers: a vertical gradient
    import cv2
    free = ~cv2.dilate(covered.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    stops = []
    for i in range(6):
        y0, y1 = int(H * i / 6), int(H * (i + 1) / 6)
        px = rgb[y0:y1][free[y0:y1]]
        if len(px) > 200:
            stops.append([round((i + 0.5) / 6, 3), hexc(np.median(px, axis=0))])
    if not stops:
        stops = [[0.5, hexc(np.median(rgb.reshape(-1, 3), axis=0))]]
    background = hexc(np.median(rgb[free], axis=0)) if free.sum() > 200 else stops[0][1]

    def box_layer(kind, o, **extra):
        x0, y0, x1, y1 = o["box"]
        return {"type": kind, "x": fx(max(0, x0)), "y": fy(max(0, y0)), "w": fx(min(W, x1) - max(0, x0)), "h": fy(min(H, y1) - max(0, y0)), **extra}

    # confetti, sparkles, glowing lights: each piece the model finds is cut out and copied to the same place
    decor_boxes = detect_decor(img)
    for b, m in zip(decor_boxes, sam_masks_for_boxes(rgb, decor_boxes) if decor_boxes else []):
        ys, xs = np.where(m)
        if len(xs) < 20:
            continue
        box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        src = crop_png(rgb, m, box)
        if src:
            layers.append({"type": "decor", "kind": "decoration", "x": fx(box[0]), "y": fy(box[1]), "w": fx(box[2] - box[0]), "h": fy(box[3] - box[1]), "src": src})
    for o in people:
        m = sam_masks_for_boxes(rgb, [o["box"]])[0]
        ys, xs = np.where(m)
        if len(xs):
            fill = float(m.sum()) / max(1.0, (xs.max() + 1 - xs.min()) * (ys.max() + 1 - ys.min()))
            o = {**o, "box": [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]}
            layers.append(box_layer("person", o, frame="rounded" if fill > 0.88 else "cutout"))
        else:
            layers.append(box_layer("person", o, frame="rounded"))
    for o in sorted(prods, key=lambda o: (round(o["box"][1] / H * 8), o["box"][0])):
        layers.append(box_layer("product", o))
    for o in logos:
        layers.append(box_layer("logo", o, align="center"))

    # text: colour, alignment and size from the pixels; the biggest block is the offer, the ones above it the headline, below it the line under
    styled = []
    for b in blocks:
        st = text_style(rgb, b)
        txt = " ".join(l["text"] for l in b["lines"])
        styled.append({"b": b, "st": st, "n": len(b["lines"]), "lh": b["lh"] / H, "text": txt, "cy": (b["box"][1] + b["box"][3]) / 2 / H})
    roles = {}
    if styled:
        offer = max(styled, key=lambda s: s["lh"])
        roles[id(offer)] = "offer"
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
            "debug": debug_picture(rgb, layers),
            "found": {"text": len(blocks), "persons": len(people), "products": len(prods), "logos": len(logos), "decorations": len(decor_boxes)}}
