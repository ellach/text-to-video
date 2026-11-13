"""
LogoBlend: a small network that makes a pasted logo look printed on the object.

It never draws the logo. The exact logo pixels come from the compositing step (add_logo.py); this network only
predicts, for every pixel of the logo area:

  S  shading     (3 channels)  how the cloth's light changes the logo's brightness and colour
  B  offset      (3 channels)  a small additive colour term (colour cast, sheen)
  D  bend        (2 channels)  a small, smooth displacement so the logo follows folds
  V  visibility  (1 channel)   how much of the logo is visible (hands, hair, objects in front of the cloth hide it)

and the final frame is computed from those, so the logo's shape and text stay exact:

  out = frame * (1 - alpha) + clamp(warp(logo, D) * S + B) * alpha,     alpha = warp(logo_alpha, D) * V
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CH = 8      # frame (3) + logo colour * alpha (3) + logo alpha (1) + garment mask (1)


def conv_block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(4, o), nn.SiLU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(4, o), nn.SiLU())


class BlendNet(nn.Module):
    def __init__(self, base=24, max_shift=0.03):
        super().__init__()
        self.max_shift = max_shift          # largest bend, as a fraction of the (square) crop size
        b = base
        self.e1, self.e2, self.e3, self.e4 = conv_block(IN_CH, b), conv_block(b, 2 * b), conv_block(2 * b, 4 * b), conv_block(4 * b, 4 * b)
        self.d3, self.d2, self.d1 = conv_block(8 * b, 2 * b), conv_block(4 * b, b), conv_block(2 * b, b)
        self.head = nn.Conv2d(b, 9, 1)
        nn.init.zeros_(self.head.weight)    # start as the identity: plain paste (S=1, B=0, D=0, V~1)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[8] = 3.0

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        e4 = self.e4(F.avg_pool2d(e3, 2))
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([up(e4, e3), e3], 1))
        d2 = self.d2(torch.cat([up(d3, e2), e2], 1))
        d1 = self.d1(torch.cat([up(d2, e1), e1], 1))
        r = self.head(d1)
        return {"S": 1 + 0.5 * torch.tanh(r[:, 0:3]),
                "B": 0.1 * torch.tanh(r[:, 3:6]),
                "D": self.max_shift * torch.tanh(r[:, 6:8]),
                "V": torch.sigmoid(r[:, 8:9])}


def make_input(frame, logo_rgb, logo_a, mask):
    """frame [B,3,H,W], logo_rgb straight colour [B,3,H,W], logo_a [B,1,H,W], mask [B,1,H,W] -> [B,8,H,W]"""
    return torch.cat([frame, logo_rgb * logo_a, logo_a, mask], 1)


def compose(frame, logo_rgb, logo_a, p):
    """Final frame from the predicted fields p = {S, B, D, V} (all at the frame's resolution)."""
    B_, _, H, W = frame.shape
    ys = torch.linspace(-1, 1, H, device=frame.device, dtype=frame.dtype)
    xs = torch.linspace(-1, 1, W, device=frame.device, dtype=frame.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([gx, gy], -1)[None].expand(B_, -1, -1, -1)
    grid = base + 2.0 * p["D"].permute(0, 2, 3, 1)               # D is a fraction of the crop size; the grid spans 2
    warped = F.grid_sample(torch.cat([logo_rgb * logo_a, logo_a], 1), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    a_w = warped[:, 3:]
    straight = warped[:, :3] / a_w.clamp(min=1e-4)
    color = (straight * p["S"] + p["B"]).clamp(0, 1)
    alpha = (a_w * p["V"] * 0.98).clamp(0, 1)
    return frame * (1 - alpha) + color * alpha, alpha
