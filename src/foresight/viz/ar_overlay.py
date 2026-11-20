"""AR overlays for the demo videos: labels pinned to 3D points, glowing routes on the floor, a corner tag.

Everything is projected through the shot's real camera (pose + horizontal FOV), so a label stays on the glass or the
goal as the camera moves. Drawn on a 2x supersampled RGBA layer and downsampled, so lines and text are smooth.

Style (user's choice, 2026-11-19): AR labels in the scene; short words, one or two live numbers; no panels.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = Path(__file__).resolve().parents[3] / "data/raw/fonts"
HOLO = (110, 214, 255)
WARN = (255, 190, 92)
GOOD = (96, 236, 170)
INK = (236, 247, 255)
PANEL = (8, 22, 36)
TONES = {"info": HOLO, "warn": WARN, "good": GOOD}


@functools.lru_cache(maxsize=None)
def font(role: str, size: int, weight: Optional[str] = None):
    """role: 'display' (Sora) or 'text' (Inter); weight: a named instance, e.g. 'SemiBold'."""
    path = FONTS / ("Sora[wght].ttf" if role == "display" else "Inter[opsz,wght].ttf")
    f = ImageFont.truetype(str(path), size)
    try:
        f.set_variation_by_name(weight or ("SemiBold" if role == "display" else "Medium"))
    except (OSError, ValueError):
        pass
    return f


@dataclass
class Camera:
    """Pinhole camera looking from `eye` toward `target` (z up), horizontal FOV in degrees, image w x h."""
    eye: np.ndarray
    target: np.ndarray
    hfov_deg: float
    w: int
    h: int

    @classmethod
    def from_focal(cls, eye, target, focal_mm, aperture_mm, w, h):
        return cls(np.asarray(eye, float), np.asarray(target, float),
                   math.degrees(2 * math.atan(aperture_mm / 2 / focal_mm)), w, h)

    def project(self, p):
        """World points (N, 3) -> pixel (N, 2) and depth along the view (N,); depth <= 0 is behind the camera."""
        p = np.atleast_2d(np.asarray(p, float))
        f = self.target - self.eye
        f /= np.linalg.norm(f)
        r = np.cross(f, [0.0, 0.0, 1.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        d = p - self.eye
        z = d @ f
        fx = (self.w / 2) / math.tan(math.radians(self.hfov_deg) / 2)
        x = self.w / 2 + fx * (d @ r) / np.maximum(z, 1e-6)
        y = self.h / 2 - fx * (d @ u) / np.maximum(z, 1e-6)
        return np.stack([x, y], 1), z


class Overlay:
    """Collects elements for one frame, then `compose(frame)` draws them. Coordinates in output pixels."""

    def __init__(self, cam: Camera, ss: int = 2):
        self.cam, self.ss = cam, ss
        self.W, self.H = cam.w * ss, cam.h * ss
        self.glow = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        self.sharp = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        self.gd, self.sd = ImageDraw.Draw(self.glow), ImageDraw.Draw(self.sharp)

    def _s(self, xy):
        return [(float(x) * self.ss, float(y) * self.ss) for x, y in xy]

    # -------------------------------------------------------------- floor route
    def route(self, pts3d: Sequence, tone: str = "info", width: float = 7.0, alpha: float = 1.0,
              progress: float = 1.0, pulse: Optional[float] = None, dashed: bool = False, fade_tail: bool = True):
        """A glowing line on the floor through 3D points; progress in [0, 1] draws it growing; pulse in [0, 1] adds a
        bright travelling dot."""
        P = np.asarray(pts3d, float)
        if len(P) < 2 or alpha <= 0:
            return
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        L = s[-1] * float(np.clip(progress, 0, 1))
        dense_s = np.linspace(0.0, L, max(2, int(L / 0.04)))
        dense = np.stack([np.interp(dense_s, s, P[:, i]) for i in range(3)], 1)
        xy, z = self.cam.project(dense)
        ok = z > 0.15
        col = TONES[tone]
        for i in range(len(dense) - 1):
            if not (ok[i] and ok[i + 1]):
                continue
            if dashed and (int(dense_s[i] / 0.18) % 2):
                continue
            a = alpha * (0.25 + 0.75 * (i / max(1, len(dense) - 2)) if fade_tail else 1.0)
            seg_xy = self._s([xy[i], xy[i + 1]])
            self.gd.line(seg_xy, fill=col + (int(150 * a),), width=int(width * self.ss * 3.2))
            self.sd.line(seg_xy, fill=(255, 255, 255, int(235 * a)), width=max(1, int(width * self.ss * 0.45)))
            self.sd.line(seg_xy, fill=col + (int(255 * a),), width=int(width * self.ss))
        if pulse is not None and ok.any():
            k = int(np.clip(pulse, 0, 1) * (len(dense) - 1))
            if ok[k]:
                x, y = self._s([xy[k]])[0]
                r = width * self.ss * 1.6
                self.gd.ellipse((x - 3 * r, y - 3 * r, x + 3 * r, y + 3 * r), fill=col + (int(170 * alpha),))
                self.sd.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, int(255 * alpha)))

    def ring(self, p3d, radius_m: float = 0.35, tone: str = "good", alpha: float = 1.0, t: float = 0.0):
        """A pulsing ring on the floor (the goal)."""
        if alpha <= 0:
            return
        col = TONES[tone]
        for k, rr in enumerate((radius_m, radius_m * (1.0 + 0.6 * ((t * 0.8) % 1.0)))):
            a = alpha * (1.0 if k == 0 else 1.0 - ((t * 0.8) % 1.0))
            ang = np.linspace(0, 2 * math.pi, 64)
            pts = np.stack([p3d[0] + rr * np.cos(ang), p3d[1] + rr * np.sin(ang), np.full_like(ang, p3d[2])], 1)
            xy, z = self.cam.project(pts)
            if (z > 0.15).all():
                poly = self._s(xy)
                self.gd.line(poly + poly[:1], fill=col + (int(140 * a),), width=int(18 * self.ss))
                self.sd.line(poly + poly[:1], fill=col + (int(255 * a),), width=int(4 * self.ss))

    # -------------------------------------------------------------- pinned label
    def label(self, p3d, title: str, value: Optional[str] = None, sub: Optional[str] = None, tone: str = "info",
              side: str = "right", lead_px: float = 70.0, rise_px: float = 46.0, alpha: float = 1.0, scale: float = 1.0):
        """A dot on the 3D point, an angled leader, and a tag: TITLE · value (value in the tone colour), an optional
        second line. Hidden when the point is behind the camera or off screen."""
        if alpha <= 0:
            return
        xy, z = self.cam.project([p3d])
        if z[0] <= 0.15:
            return
        x, y = xy[0]
        if not (-50 <= x <= self.cam.w + 50 and -50 <= y <= self.cam.h + 50):
            return
        s, col = self.ss, TONES[tone]
        a8 = lambda v: int(v * alpha)  # noqa: E731
        # dot
        r = 7 * scale * s
        X, Y = x * s, y * s
        self.gd.ellipse((X - 4 * r, Y - 4 * r, X + 4 * r, Y + 4 * r), fill=col + (a8(120),))
        self.sd.ellipse((X - r, Y - r, X + r, Y + r), fill=(255, 255, 255, a8(255)))
        self.sd.ellipse((X - r * 1.9, Y - r * 1.9, X + r * 1.9, Y + r * 1.9), outline=col + (a8(230),),
                        width=int(2.2 * s * scale))
        # tag size first, so the leader can be flipped to keep the tag inside the frame
        ft = font("text", int(30 * scale * s), "SemiBold")
        fv = font("text", int(30 * scale * s), "Bold")
        fs = font("text", int(22 * scale * s), "Medium")
        tw = self.sd.textlength(title, font=ft)
        sep = "  ·  " if value else ""
        vw = self.sd.textlength(sep + value, font=fv) if value else 0
        sw = self.sd.textlength(sub, font=fs) if sub else 0
        pad = 16 * s * scale
        w = max(tw + vw, sw) + 2 * pad
        hgt = (30 + (30 if sub else 0) + 26) * s * scale
        margin = 24 * s
        sg = 1 if side == "right" else -1
        reach = lead_px * s * scale
        if sg > 0 and X + reach + w > self.W - margin:
            sg = -1
        elif sg < 0 and X - reach - w < margin:
            sg = 1
        rise = rise_px * s * scale
        if Y - rise - hgt / 2 < margin:
            rise = -rise  # no room above: lead downwards
        # leader: out-and-up (or down), then horizontal
        k1 = (X + sg * reach * 0.45, Y - rise)
        k2 = (k1[0] + sg * reach * 0.55, k1[1])
        self.sd.line([(X + sg * r * 1.33, Y - np.sign(rise) * r * 1.33), k1, k2], fill=col + (a8(235),),
                     width=int(2.4 * s * scale), joint="curve")
        x0 = k2[0] if sg > 0 else k2[0] - w
        x0 = float(np.clip(x0, margin, self.W - margin - w))
        y0 = float(np.clip(k2[1] - hgt / 2, margin, self.H - margin - hgt))
        self.glow.paste(Image.new("RGBA", (int(w), int(hgt)), col + (a8(60),)), (int(x0), int(y0)))
        self.sd.rounded_rectangle((x0, y0, x0 + w, y0 + hgt), radius=10 * s * scale, fill=PANEL + (a8(196),),
                                  outline=col + (a8(210),), width=int(2 * s * scale))
        # accent tick on the leader side
        tx = x0 if sg > 0 else x0 + w
        self.sd.line([(tx, y0 + 10 * s * scale), (tx, y0 + hgt - 10 * s * scale)], fill=col + (a8(255),),
                     width=int(5 * s * scale))
        ty = y0 + 12 * s * scale
        self.sd.text((x0 + pad, ty), title, font=ft, fill=INK + (a8(255),))
        if value:
            self.sd.text((x0 + pad + tw, ty), sep + value, font=fv, fill=col + (a8(255),))
        if sub:
            self.sd.text((x0 + pad, ty + 40 * s * scale), sub, font=fs, fill=(186, 210, 228, a8(255)))

    def thought(self, p3d, text: str, alpha: float = 1.0, max_w_px: float = 520.0, scale: float = 1.0):
        """A short quoted 'thought' floating above a 3D point (the VLM's note), with a thin stem."""
        if alpha <= 0:
            return
        xy, z = self.cam.project([p3d])
        if z[0] <= 0.15:
            return
        s = self.ss
        X, Y = xy[0] * s
        fq = font("text", int(31 * scale * s), "Medium")
        words, lines, cur = text.split(), [], ""
        for wd in words:
            trial = (cur + " " + wd).strip()
            if self.sd.textlength(trial, font=fq) <= max_w_px * s * scale:
                cur = trial
            else:
                lines.append(cur)
                cur = wd
        lines.append(cur)
        lh = 41 * s * scale
        pad = 18 * s * scale
        w = max(self.sd.textlength(l_, font=fq) for l_ in lines) + 2 * pad
        hgt = lh * len(lines) + 2 * pad - 8 * s * scale
        x0, y0 = X - w / 2, Y - 60 * s * scale - hgt
        self.sd.line([(X, Y), (X, y0 + hgt)], fill=HOLO + (int(200 * alpha),), width=int(2 * s * scale))
        self.sd.rounded_rectangle((x0, y0, x0 + w, y0 + hgt), radius=16 * s * scale, fill=PANEL + (int(205 * alpha),),
                                  outline=HOLO + (int(170 * alpha),), width=int(1.8 * s * scale))
        for i, l_ in enumerate(lines):
            self.sd.text((x0 + pad, y0 + pad - 4 * s * scale + i * lh), l_, font=fq, fill=INK + (int(255 * alpha),))

    # -------------------------------------------------------------- compose
    def compose(self, frame: Image.Image, corner: Optional[str] = "SIMULATED IN NVIDIA ISAAC SIM") -> Image.Image:
        base = frame.convert("RGBA").resize((self.cam.w, self.cam.h)) if frame.size != (self.cam.w, self.cam.h) else frame.convert("RGBA")
        q = 4 * self.ss  # blur the glow at quarter resolution: the same soft halo for ~1/50 of the cost
        glow = (self.glow.resize((self.W // q, self.H // q), Image.BILINEAR).filter(ImageFilter.GaussianBlur(10 * self.ss / q))
                .resize((self.cam.w, self.cam.h), Image.BILINEAR))
        sharp = self.sharp.resize((self.cam.w, self.cam.h), Image.LANCZOS)
        # additive-ish glow: screen blend of the glow layer
        g = np.asarray(glow, np.float32) / 255.0
        b = np.asarray(base, np.float32) / 255.0
        rgb = 1 - (1 - b[..., :3]) * (1 - g[..., :3] * g[..., 3:4])
        out = Image.fromarray((np.clip(np.concatenate([rgb, b[..., 3:]], -1), 0, 1) * 255).astype(np.uint8), "RGBA")
        out.alpha_composite(sharp)
        if corner:
            d = ImageDraw.Draw(out)
            fc = font("text", max(14, self.cam.w // 96), "SemiBold")
            tw = d.textlength(corner, font=fc)
            m = self.cam.w // 48
            d.text((self.cam.w - m - tw, self.cam.h - m - fc.size), corner, font=fc, fill=(232, 244, 255, 170))
        return out.convert("RGB")
