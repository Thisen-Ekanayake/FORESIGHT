"""Square 1920x1920 frames for the Isaac Sim replay (tools/isaac/render_replay.py): the RTX chase view on top; the
Go2's head camera, what the robot was doing at that moment of the recorded run, and a minimap of its route below.

Everything written on screen comes from the recorded run (`tools/isaac/export_run.py` JSON) at the frame's sim
time; the frames' robot poses come from Isaac Sim (frames.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from foresight.viz.mirror_trap_video import AMBER, BG, CYAN, DIM, GREEN, MUTED, PANEL, RED, S, TEXT, _box, _font

HEAD_H = 170
CHASE_H = 1280
BOT_Y = HEAD_H + CHASE_H + 16
BOT_H = S - BOT_Y - 16
NAMES = {"tv": "TV", "refrigerator": "fridge"}


def _state_line(tick, instructions):
    mode, lab = tick.get("mode"), NAMES.get(tick.get("target") or "", tick.get("target") or "")
    if tick.get("verifying"):
        return "Checking a suspected mirror before driving on", AMBER
    return {
        "explore": (f"Exploring: looking for the {lab}", TEXT),
        "approach": (f"Heading to the {lab} it saw", CYAN),
        "verify": (f"Close-up check: is it really the {lab}?", CYAN),
        "close_in": (f"Confirmed, closing in on the {lab}", GREEN),
        "announce": (f"Found the {lab}", GREEN),
        "done": ("Done: every instruction completed", GREEN),
        "gave_up": ("Gave up: nothing left to explore", RED),
    }.get(mode, ("", TEXT))


class IsaacReplay:
    def __init__(self, replay_dir: Path, run_json: Path, topdown: Path = None, assets: Path = None,
                 instructions=("sofa", "bed")):
        self.dir = Path(replay_dir)
        self.frames = json.loads((self.dir / "frames.json").read_text())["frames"]
        self.run = json.loads(Path(run_json).read_text())
        self.ticks = self.run["ticks"]
        self.tick_t = np.array([k["t"] for k in self.ticks])
        self.instructions = list(instructions)
        self.top, self.assets = None, None
        if topdown is not None and Path(topdown).exists():
            self.top = Image.open(topdown).convert("RGB")
            self.assets = json.loads(Path(assets).read_text())
        self.f_big, self.f_mid, self.f_small = _font("Bold", 58), _font("Medium", 40), _font("Regular", 30)

    def __len__(self):
        return len(self.frames)

    def _tick(self, t):
        return self.ticks[max(0, int(np.searchsorted(self.tick_t, t)) - 1)]

    def _minimap(self, img, xy, size, t):
        x0, y0 = xy
        if self.top is None:
            return
        c, w_m = np.array(self.assets["center"])[[0, 2]], self.assets["width_m"]
        R = self.top.size[0]
        pts = np.array([[p["pos"][0], -p["pos"][1]] for p in self.frames if p["t"] <= t] or [[0, 0]])
        # crop a square around the whole route
        allp = np.array([[p["pos"][0], -p["pos"][1]] for p in self.frames])
        lo, hi = allp.min(0) - 1.2, allp.max(0) + 1.2
        side = float(max(hi - lo))
        mid = (lo + hi) / 2
        to_px = lambda xz: ((np.asarray(xz) - c) / w_m + 0.5) * R  # noqa: E731
        a, b = to_px(mid - side / 2), to_px(mid + side / 2)
        crop = self.top.crop((int(a[0]), int(a[1]), int(b[0]), int(b[1]))).resize((size, size), Image.LANCZOS)
        img.paste(crop, (x0, y0))
        d = ImageDraw.Draw(img)
        k = size / side
        m = lambda xz: (x0 + (xz[0] - (mid[0] - side / 2)) * k, y0 + (xz[1] - (mid[1] - side / 2)) * k)  # noqa: E731
        for wa, wb in self.run["walls"]:
            d.line([m(wa), m(wb)], fill=RED, width=7)
        if len(pts) > 1:
            d.line([m(p) for p in pts], fill=CYAN, width=5)
        for f in self.run["found"]:
            if f[0] <= t:
                px, py = m(f[2])
                d.ellipse((px - 11, py - 11, px + 11, py + 11), fill=GREEN, outline=(0, 0, 0), width=2)
        px, py = m(pts[-1])
        d.ellipse((px - 13, py - 13, px + 13, py + 13), fill=(255, 255, 255), outline=(0, 0, 0), width=3)

    def frame(self, i, speed: float = 2.0):
        fr = self.frames[i]
        t = fr["t"]
        img = Image.new("RGBA", (S, S), BG + (255,))
        d = ImageDraw.Draw(img)
        d.text((40, 26), "FORESIGHT in NVIDIA Isaac Sim", font=self.f_big, fill=TEXT)
        d.text((40, 104), "The recorded run, replayed: same home (HM3D 00808), RTX ray tracing, real mirror reflections, "
                          "a Unitree Go2 walking the route", font=self.f_small, fill=MUTED)
        chase = Image.open(self.dir / "chase" / f"{i:05d}.jpg").convert("RGB")
        img.paste(chase.resize((S, CHASE_H)) if chase.size != (S, CHASE_H) else chase, (0, HEAD_H))
        # bottom: eye camera | status | minimap
        side = BOT_H
        eye = Image.open(self.dir / "eye" / f"{i:05d}.jpg").convert("RGB").resize((side, side), Image.LANCZOS)
        img.paste(eye, (16, BOT_Y))
        _box(img, (16, BOT_Y, 16 + 250, BOT_Y + 46), (0, 0, 0, 150), radius=10)
        d.text((28, BOT_Y + 8), "Go2 head camera", font=self.f_small, fill=TEXT)
        mx = S - 16 - side
        self._minimap(img, (mx, BOT_Y), side, t)
        _box(img, (mx, BOT_Y, mx + 190, BOT_Y + 46), (0, 0, 0, 150), radius=10)
        d.text((mx + 12, BOT_Y + 8), "Its route", font=self.f_small, fill=TEXT)
        cx, cw = 16 + side + 24, mx - 24 - (16 + side + 24)
        _box(img, (cx, BOT_Y, cx + cw, BOT_Y + side), PANEL + (255,), radius=18)
        tick = self._tick(t)
        k = sum(1 for f in self.run["found"] if f[0] <= t)
        y = BOT_Y + 28
        for j, lab in enumerate(self.instructions):
            done = j < k
            txt = ("Find the " if j == 0 else "Then go back to the ") + NAMES.get(lab, lab)
            d.text((cx + 28, y), txt, font=self.f_mid, fill=GREEN if done else (TEXT if j == k else DIM))
            if done:
                d.line((cx + 28, y + 26, cx + 28 + d.textlength(txt, font=self.f_mid), y + 26), fill=GREEN, width=3)
            y += 58
        msg, col = _state_line(tick, self.instructions)
        y += 16
        d.text((cx + 28, y), msg, font=self.f_mid, fill=col)
        y += 70
        d.text((cx + 28, y), f"run time {t:5.1f} s   ·   shown at {speed:g}x", font=self.f_small, fill=MUTED)
        y += 44
        d.text((cx + 28, y), f"Isaac Sim tracking error {fr['track_err_m'] * 100:.0f} cm", font=self.f_small, fill=MUTED)
        return img.convert("RGB")
