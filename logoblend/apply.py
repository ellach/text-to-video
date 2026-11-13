"""
Use a trained LogoBlend model inside add_logo.py.

    python add_logo.py --video clip.mp4 --logo logo.png --blend_ckpt logoblend.pt

add_logo.py places and tracks the exact logo; for each frame this class then lights, bends and occludes it.
Only a crop around the logo goes through the network (the model works at --size px), then its predictions are
applied to the full-resolution frame, so the result keeps the video's resolution and the logo stays sharp.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model import BlendNet, compose, make_input


class LogoBlender:
    def __init__(self, ckpt, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt, map_location=self.device, weights_only=False)
        a = ck.get("args", {})
        self.size = int(a.get("size", 128))
        self.net = BlendNet(a.get("base", 24), a.get("max_shift", 0.03)).to(self.device).eval()
        self.net.load_state_dict(ck["model"])

    @torch.no_grad()
    def __call__(self, frame, layer, surf):
        """frame uint8 [H,W,3]; layer float [H,W,4] (straight RGB 0-1 + alpha); surf bool [H,W] -> uint8 [H,W,3]"""
        a = layer[..., 3]
        ys, xs = np.where(a > 0.02)
        if len(xs) == 0:
            return frame
        H, W = a.shape
        side = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 2.5) + 8            # context around the logo
        cx, cy = (xs.max() + xs.min()) // 2, (ys.max() + ys.min()) // 2
        x0, y0 = int(cx - side // 2), int(cy - side // 2)
        pad = side
        fp = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        lp = cv2.copyMakeBorder(layer.astype(np.float32), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        sp = cv2.copyMakeBorder(surf.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        sl = (slice(y0 + pad, y0 + pad + side), slice(x0 + pad, x0 + pad + side))
        crop_f = torch.from_numpy(fp[sl]).permute(2, 0, 1).float()[None].to(self.device) / 255.0
        crop_l = torch.from_numpy(lp[sl]).permute(2, 0, 1)[None].to(self.device)
        crop_m = torch.from_numpy(sp[sl]).float()[None, None].to(self.device)
        R = self.size
        small = lambda t, m="bilinear": F.interpolate(t, size=(R, R), mode=m, align_corners=False if m == "bilinear" else None)
        x = make_input(small(crop_f), small(crop_l[:, :3]), small(crop_l[:, 3:]), small(crop_m, "nearest"))
        p = self.net(x)
        big = {k: F.interpolate(v, size=(side, side), mode="bilinear", align_corners=False) for k, v in p.items()}
        out, _ = compose(crop_f, crop_l[:, :3], crop_l[:, 3:], big)
        out = (out[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        res = fp.copy(); res[sl] = out
        return res[pad:pad + H, pad:pad + W]
