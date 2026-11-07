"""FORESIGHT showcase: a narrated square 1920x1920 overview of the project (tools/make_showcase_video.py).

Built like a platform promo: logo open -> the question -> how it works (fast and slow loops) -> captioned
fly-throughs of the HM3D homes and a mosaic of tilted 3D floor plans -> task montage (point goal, object search) ->
"same robot, different depth" (one point-goal episode driven four times with different depth: the depth-error
finding) -> the mirror encounter -> the set-A result bars -> what's next -> close.

Every robot view is re-rendered in Habitat at the pose interpolated to the frame's sim time (runs are logged at
~10 Hz). Numbers drawn or spoken come from the recorded runs, results/runs/pointnav/A/summary.json and the saved
mirror-trap / object-search runs; playback speed is badged on screen whenever it isn't 1x.
"""
from __future__ import annotations

import functools
import json
import math
import pickle
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from foresight.viz.mirror_trap_video import SR, Voice, _Track, _turbo, _Views, _wrap
from foresight.viz.showcase_assets import FLY, dollhouse, homes

S, FPS = 1920, 30
NAVY, PANEL, PANEL2 = (22, 30, 54), (32, 42, 72), (44, 56, 92)
TEXT, MUTED, DIM = (244, 246, 250), (160, 170, 196), (92, 104, 140)
BLUE, GREEN, RED, AMBER, CYAN = (64, 140, 255), (52, 211, 140), (255, 92, 97), (255, 186, 60), (120, 220, 255)
RUN_COLORS = {"gt": (235, 238, 245), "da": BLUE, "gt:scale=2.5": (178, 120, 255), "gt:noise=0.1": RED}
G = 16
FONT = "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf"
HM3D_ANN = "data/processed/scene_datasets/hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json"


@functools.lru_cache(maxsize=None)
def F(weight, size):
    f = ImageFont.truetype(FONT, size)
    f.set_variation_by_name(weight)
    return f


_PREVIEW = {"k": None}  # render only this frame index of a segment (preview mode)


def frames(n, offset=0):
    """Frame indices of a segment (or of a sub-run starting at `offset`); just the previewed one in preview mode."""
    k = _PREVIEW["k"]
    if k is None:
        return range(n)
    return [k - offset] if 0 <= k - offset < n else []


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ramp(t, t0, dur=0.5):
    return ease((t - t0) / dur)


def box(img, xy, fill, radius=18):
    x0, y0, x1, y1 = (int(round(v)) for v in xy)
    layer = Image.new("RGBA", (max(1, x1 - x0), max(1, y1 - y0)), (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle((0, 0, x1 - x0 - 1, y1 - y0 - 1), radius, fill=fill)
    img.alpha_composite(layer, (x0, y0))


def text(img, xy, s, font, fill=TEXT, alpha=1.0, anchor="la", shadow=False):
    if alpha <= 0:
        return
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if shadow:
        d.text((xy[0] + 3, xy[1] + 3), s, font=font, fill=(0, 0, 0, int(170 * alpha)), anchor=anchor)
    d.text(xy, s, font=font, fill=tuple(fill[:3]) + (int(255 * alpha),), anchor=anchor)
    if shadow:
        layer = layer.filter(ImageFilter.GaussianBlur(0.6))
    img.alpha_composite(layer)


def tracked(img, cx, y, s, font, track, fill=TEXT, alpha=1.0):
    """Letter-spaced text, centred on cx."""
    d = ImageDraw.Draw(img)
    widths = [d.textlength(c, font=font) for c in s]
    x = cx - (sum(widths) + track * (len(s) - 1)) / 2
    for c, w in zip(s, widths):
        text(img, (x, y), c, font, fill, alpha)
        x += w + track


def paste(img, im, xy, alpha=1.0):
    im = im.convert("RGBA")
    if alpha < 1.0:
        a = im.getchannel("A").point(lambda v: int(v * alpha))
        im.putalpha(a)
    img.alpha_composite(im, (int(xy[0]), int(xy[1])))


def canvas():
    return Image.new("RGBA", (S, S), NAVY + (255,))


# ---------------------------------------------------------------- logo


def logo_layer(size, progress=1.0, route_progress=None):
    """The FORESIGHT mark: an eye whose iris holds a route on a floor plan. White strokes, blue route.
    progress in [0, 1] draws the strokes in."""
    k = 2  # supersample
    W = size * k
    layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    c, r = W / 2, W * 0.46
    lw = max(2, int(W * 0.032))
    xs = np.linspace(-1, 1, 400)
    amp = 0.52 * r

    def almond(rx, ay):
        up = [(c + x * rx, c - ay * (1 - x * x) ** 1.1) for x in xs]
        return up + [(c + x * rx, c + ay * (1 - x * x) ** 1.1) for x in xs[::-1]]

    ring = Image.new("L", (W, W), 0)
    rd = ImageDraw.Draw(ring)
    rd.polygon(almond(r, amp), fill=255)
    rd.polygon(almond(r - lw * 1.6, amp - lw), fill=0)
    reveal = np.zeros((W, W), np.uint8)
    reveal[:, : int(W * min(1.0, progress * 1.7))] = 255
    ring = Image.fromarray(np.minimum(np.asarray(ring), reveal))
    layer.paste(Image.new("RGBA", (W, W), TEXT + (255,)), (0, 0), ring)
    ir = 0.36 * r
    ip = min(1.0, max(0.0, (progress - 0.35) / 0.4))
    if ip > 0:
        d.arc((c - ir, c - ir, c + ir, c + ir), -90, -90 + 360 * ip, fill=TEXT, width=lw)
    rp = min(1.0, max(0.0, (progress - 0.6) / 0.4)) if route_progress is None else route_progress
    route = np.array([(-0.62, 0.42), (-0.12, 0.42), (-0.12, -0.18), (0.42, -0.18)]) * ir + c
    if rp > 0:
        seg = np.linalg.norm(np.diff(route, axis=0), axis=1)
        L = seg.sum() * rp
        pts, acc = [tuple(route[0])], 0.0
        for a, b, s in zip(route[:-1], route[1:], seg):
            if acc + s >= L:
                pts.append(tuple(a + (b - a) * (L - acc) / s))
                break
            pts.append(tuple(b))
            acc += s
        d.line(pts, fill=BLUE, width=int(lw * 0.9), joint="curve")
        sx, sy = route[0]
        d.ellipse((sx - lw * 0.75, sy - lw * 0.75, sx + lw * 0.75, sy + lw * 0.75), fill=BLUE)
        if rp >= 1.0:
            gx, gy = route[-1]
            d.ellipse((gx - lw * 1.3, gy - lw * 1.3, gx + lw * 1.3, gy + lw * 1.3), fill=TEXT)
    return layer.resize((size, size), Image.LANCZOS)


def corner_brand(img, alpha=1.0):
    paste(img, logo_layer(64), (G * 3, 44), alpha)
    text(img, (G * 3 + 80, 58), "FORESIGHT", F("Bold", 34), TEXT, alpha)


# ---------------------------------------------------------------- shared scene pieces


class Sched:
    """Piecewise-linear map from segment time (s) to sim time (s): [(video_dur, sim_from, sim_to), ...]."""

    def __init__(self, pieces, lead=0.0):
        self.pieces, self.lead = pieces, lead
        self.length = lead + sum(p[0] for p in pieces)

    def sim(self, tv):
        tv -= self.lead
        if tv <= 0:
            return self.pieces[0][1]
        for dur, a, b in self.pieces:
            if tv <= dur:
                return a + (b - a) * tv / dur
            tv -= dur
        return self.pieces[-1][2]

    def speed(self, tv):
        tv -= self.lead
        for dur, a, b in self.pieces:
            if tv <= dur:
                return (b - a) / dur if tv > 0 else 0.0
            tv -= dur
        return 0.0

    def video_at(self, ts):
        """First segment time at which sim time reaches ts."""
        acc = self.lead
        for dur, a, b in self.pieces:
            if b > a and a <= ts <= b:
                return acc + dur * (ts - a) / (b - a)
            acc += dur
        return acc


def speed_badge(img, x1, y, speed):
    if speed <= 0 or abs(speed - 1.0) < 0.05:
        return
    s = f"{speed:.1f}× speed".replace(".0×", "×")
    w = ImageDraw.Draw(img).textlength(s, font=F("Bold", 32)) + 40
    box(img, (x1 - w, y, x1, y + 52), AMBER + (235,), radius=26)
    text(img, (x1 - w / 2, y + 26), s, F("Bold", 32), (20, 20, 24), anchor="mm")


def header(img, title, sub=None, clock=None, speed=None, alpha=1.0):
    corner_brand(img, alpha)
    text(img, (G * 3, 132), title, F("Bold", 66), TEXT, alpha)
    if sub:
        text(img, (G * 3, 214), sub, F("Regular", 34), MUTED, alpha)
    if clock is not None:
        text(img, (S - G * 3, 70), f"t = {clock:5.1f} s", F("Bold", 44), TEXT, alpha, anchor="ra")
        if speed is not None:
            speed_badge(img, S - G * 3, 132, speed)


def compass(img, cx, cy, r, bearing, label=None):
    """Goal compass like the reference: a disc, the robot's heading up, an arrow toward the goal.
    bearing: rad, + = goal to the right."""
    box(img, (cx - r, cy - r, cx + r, cy + r), (12, 16, 30, 190), radius=r)
    d = ImageDraw.Draw(img)
    d.ellipse((cx - r + 6, cy - r + 6, cx + r - 6, cy + r - 6), outline=(255, 255, 255, 120), width=3)
    f = np.array([math.sin(bearing), -math.cos(bearing)])
    n = np.array([-f[1], f[0]])
    tip = np.array([cx, cy]) + f * r * 0.66
    base = np.array([cx, cy]) - f * r * 0.30
    d.polygon([tuple(tip), tuple(base + n * r * 0.30), tuple(np.array([cx, cy]) - f * r * 0.12), tuple(base - n * r * 0.30)],
              fill=RED)
    d.polygon([(cx, cy - r + 14), (cx - 9, cy - r + 30), (cx + 9, cy - r + 30)], fill=(255, 255, 255))
    if label:
        text(img, (cx, cy + r + 34), label, F("Bold", 36), TEXT, anchor="mm", shadow=True)


def quote(img, s, y=S - 240, size=54, highlight=None):
    text(img, (S / 2, y), f"“{s}”", F("SemiBold", size), TEXT, anchor="mm", shadow=True)


def chip(img, x, y, s, color, font=None, anchor="left", fg=(255, 255, 255)):
    font = font or F("Bold", 34)
    w = ImageDraw.Draw(img).textlength(s, font=font) + 44
    x0 = x if anchor == "left" else x - w
    box(img, (x0, y, x0 + w, y + 58), color + (235,) if len(color) == 3 else color, radius=29)
    text(img, (x0 + w / 2, y + 29), s, font, fg, anchor="mm")
    return w


def depth_view(depth, w, h, dmax=8.0):
    """Turbo depth (near red, far blue), centre-cropped to w:h."""
    d = np.asarray(depth, np.float32)
    H, W = d.shape
    ch = int(round(W * h / w))
    if ch <= H:
        d = d[(H - ch) // 2:(H - ch) // 2 + ch]
    rgb = _turbo(1.0 - np.clip(np.nan_to_num(d, nan=0.0), 0, dmax) / dmax)
    return Image.fromarray(rgb).resize((w, h), Image.BILINEAR)


def fp_view(frame, w, h):
    im = Image.fromarray(frame)
    W, H = im.size
    ch = int(round(W * h / w))
    return im.crop((0, (H - ch) // 2, W, (H - ch) // 2 + ch)).resize((w, h), Image.LANCZOS)


class RobotMap:
    """Draws the navigator's own occupancy map (int8 score snapshots) over a world x/z box."""

    def __init__(self, rec, box_xz, w, h):
        self.origin, self.res, self.n = rec["map_origin"], rec["map_res"], rec["map_n"]
        (x0, z0), (x1, z1) = box_xz
        self.box, self.w, self.h = box_xz, w, h
        xs = x0 + (np.arange(w) + 0.5) / w * (x1 - x0)
        zs = z0 + (np.arange(h) + 0.5) / h * (z1 - z0)
        self.cols = np.clip(np.floor((xs - self.origin[0]) / self.res).astype(int), 0, self.n - 1)
        self.rows = np.clip(np.floor((zs - self.origin[1]) / self.res).astype(int), 0, self.n - 1)

    def to_px(self, x, z):
        (x0, z0), (x1, z1) = self.box
        return (np.asarray(x) - x0) / (x1 - x0) * self.w, (np.asarray(z) - z0) / (z1 - z0) * self.h

    def draw(self, score):
        g = score[self.rows[:, None], self.cols[None, :]]
        img = np.empty((self.h, self.w, 3), np.uint8)
        img[:] = (26, 34, 60)
        img[g < -15] = (46, 84, 132)
        img[g > 45] = (246, 247, 250)
        return Image.fromarray(img).convert("RGBA")


def robot_arrow(d, x, y, yaw, color, size=22, outline=(255, 255, 255)):
    f = np.array([-math.sin(yaw), -math.cos(yaw)])  # world forward (x, z) -> screen (x, y) on a plan
    n = np.array([-f[1], f[0]])
    c = np.array([x, y])
    d.polygon([tuple(c + f * size), tuple(c + n * size * 0.55 - f * size * 0.4), tuple(c - n * size * 0.55 - f * size * 0.4)],
              fill=color, outline=outline)


def screen_dir(view, pos, yaw):
    """Robot heading as a screen-space angle on a dollhouse view (yaw in world)."""
    f = np.array([-math.sin(yaw), -math.cos(yaw)])
    u0, v0 = view.project(pos[0], 0.05, pos[2])
    u1, v1 = view.project(pos[0] + f[0], 0.05, pos[2] + f[1])
    a = np.array([u1 - u0, v1 - v0])
    a /= max(np.linalg.norm(a), 1e-6)
    return a


def doll_arrow(d, view, pos, yaw, color, size=20):
    u, v = view.project(pos[0], 0.05, pos[2])
    f = screen_dir(view, pos, yaw)
    n = np.array([-f[1], f[0]])
    c = np.array([u, v])
    d.polygon([tuple(c + f * size), tuple(c + n * size * 0.6 - f * size * 0.45), tuple(c - n * size * 0.6 - f * size * 0.45)],
              fill=color, outline=(255, 255, 255))


def doll_path(d, view, pts, color, width=7):
    if len(pts) < 2:
        return
    u, v = view.project(pts[:, 0], np.full(len(pts), 0.05), pts[:, 2])
    d.line(list(zip(u, v)), fill=color, width=width, joint="curve")


def doll_pin(d, view, p, color, r=12, label=None, font=None):
    u, v = view.project(p[0], 0.05, p[2])
    d.ellipse((u - r, v - r, u + r, v + r), fill=color, outline=(255, 255, 255), width=3)
    if label:
        d.text((u + r + 8, v - 16), label, font=font or F("Bold", 26), fill=(255, 255, 255), stroke_width=3,
               stroke_fill=(10, 12, 20))


def fill_holes(im):
    """Fill isolated empty pixels of a splatted RGBA dollhouse from their filled neighbours."""
    from scipy import ndimage
    a = np.asarray(im).astype(np.float32)
    m = (a[..., 3] > 0).astype(np.float32)
    k = np.ones((3, 3), np.float32)
    num = np.stack([ndimage.convolve(a[..., c] * m, k, mode="constant") for c in range(3)], -1)
    den = ndimage.convolve(m, k, mode="constant")
    hole = (m == 0) & (den >= 5)
    out = a.copy()
    out[hole, :3] = num[hole] / den[hole, None]
    out[hole, 3] = 255
    return Image.fromarray(out.astype(np.uint8), "RGBA")


@functools.lru_cache(maxsize=None)
def _cut(assets: str, name: str):
    rgb = np.asarray(Image.open(Path(assets) / f"cut_{name}.png").convert("RGB"))
    meta = json.loads((Path(assets) / f"cut_{name}.json").read_text())
    return rgb, meta


def doll(assets, name, size, **kw):
    rgb, meta = _cut(str(assets), name)
    im, view = dollhouse(rgb, meta, size=size, **kw)
    return fill_holes(im), view


# ---------------------------------------------------------------- data


class PointnavRun:
    def __init__(self, path: Path):
        with open(path, "rb") as f:
            r = pickle.load(f)
        self.r, self.ep, self.label, self.source = r, r["ep"], r["label"], r["source"]
        self.track = _Track(self.ep)
        self.snap_t = r["snap_t"]
        self.end = float(self.ep.reached_goal_at if self.ep.reached_goal_at is not None else self.track.end)
        self.arrived = self.ep.reached_goal_at is not None

    def snap(self, t):
        return max(0, int(np.searchsorted(self.snap_t, t, side="right")) - 1)

    def step(self, t):
        return self.ep.steps[max(0, int(np.searchsorted(self.ep.times, t, side="right")) - 1)]

    def path_until(self, t):
        k = int(np.searchsorted(self.track.t, t, side="right"))
        pos, _ = self.track.at(t)
        return np.vstack([self.track.pos[:k], pos[None]])


def _stage(dataset, scene):
    return {"dataset": dataset, "scene": scene}


# ---------------------------------------------------------------- narration


def narration(runs, mirror, search, A):
    gt = runs["gt"]
    noise = runs["gt:noise=0.1"]
    p = noise.ep.positions
    drift = float(np.linalg.norm(p[:, [0, 2]] - p[0, [0, 2]], axis=1).max())
    pc = lambda k: round(A[k]["success"][0] * 100)
    hits = int(mirror["laya"].collided.sum())
    return {
        "question": "Can a robot find its way through a home it has never seen, using one ordinary camera? "
                    "No depth sensor. No lidar. And no map.",
        "stack1": "Every frame goes through Depth Anything V2, which estimates depth from a single image.",
        "stack2": "From that depth, FORESIGHT builds its own map as it drives, and plans a route on it.",
        "stack3": "In the background, the Qwen3-VL vision-language model describes what the camera sees.",
        "stack4": "Laya, a fast decision model, reads every description, watching for mirrors and glass, which fool depth.",
        "stack5": "And a safety layer always has the last word.",
        "homes1": "We test it in Habitat-Sim, inside real homes from the HM3D dataset, each one scanned with a Matterport camera.",
        "homes2": "Thirteen homes in all. Ten of them make up the evaluation set.",
        "task1": f"Point-goal navigation: reach a spot {_metres(runs['gt'].r['episode']['geodesic_m'])} "
                 "away by the walkable route, given only its coordinates.",
        "task2": "Object search, from words alone: find the sofa, then go back to the bed. No map, and no coordinates.",
        "depth1": "So how much does depth error matter? Here is one episode, driven four times. Only the depth changes.",
        "depth2": f"With true depth, the robot arrives in {gt.end:.0f} seconds. Depth Anything V2 arrives at the same "
                  "time. Depth read two and a half times too far arrives too, because FORESIGHT recalibrates its "
                  "scale from the floor.",
        "depth3": "But ten percent random noise, a much smaller error, fills its map with obstacles that aren't there. "
                  f"In two minutes, it never gets more than {_spoken_cm(drift)} from where it started.",
        "mirror1": "Mirrors are where camera depth really fails. The reflection looks like open floor.",
        "mirror2": "FORESIGHT flags the reflection, takes a second look, and walls it off on its own map.",
        "mirror_crash": "The depth-only robot trusts the reflection, and drives straight into it.",
        "mirror3": "Then it takes the real way round, and reaches the goal behind the mirror"
                   + (", with no collisions." if hits == 0 else "."),
        "results1": "Across a hundred and twenty episodes in ten homes, the same pattern holds.",
        "results2": f"Depth Anything reaches the goal {pc('map:da')} percent of the time, against {pc('map:gt')} with "
                    f"perfect depth. A scale error of two and a half times costs nothing. Ten percent noise drops "
                    f"success to {pc('map:gt:noise=0.1')} percent.",
        "results3": "The type of depth error predicts failure. Its size doesn't.",
        "next": "Next: a benchmark of targeted mirror and glass encounters, and a head-to-head test of what the "
                "vision-language model adds.",
        "close": "FORESIGHT. Zero-shot navigation from a single camera, tested in simulation.",
    }


def _metres(m):
    return f"{m:.0f} metres" if abs(m - round(m)) > 0.25 else f"{round(m)} metres"


def _spoken_cm(m):
    if m < 0.4:
        return "a third of a metre" if m > 0.28 else f"{round(m * 100)} centimetres"
    return f"{m:.1f} metres"


# ---------------------------------------------------------------- segments


def seg_open(n):
    for k in frames(n):
        t = k / FPS
        img = canvas()
        lg = logo_layer(520, progress=min(1.0, t / 1.8))
        paste(img, lg, ((S - 520) / 2, 470))
        tracked(img, S / 2, 1050, "FORESIGHT", F("Bold", 150), 26, TEXT, ramp(t, 1.4, 0.7))
        text(img, (S / 2, 1270), "Map-free navigation from a single camera", F("Regular", 52), MUTED, ramp(t, 2.2, 0.7),
             anchor="mm")
        yield img


def seg_question(n, cues, backdrop):
    bg = backdrop.resize((S, S), Image.LANCZOS).filter(ImageFilter.GaussianBlur(6)).convert("RGBA")
    dark = Image.new("RGBA", (S, S), NAVY + (205,))
    bg.alpha_composite(dark)
    q0 = cues["question"]
    lines = [("One ordinary camera.", q0 + 0.2), ("No depth sensor.", q0 + 5.0), ("No lidar.", q0 + 6.3), ("No map.", q0 + 7.4)]
    for k in frames(n):
        t = k / FPS
        img = bg.copy()
        corner_brand(img)
        y = 640
        for s, ts in lines:
            a = ramp(t, ts, 0.45)
            font = F("Bold", 120) if s.startswith("One") else F("Bold", 96)
            text(img, (S / 2, y + (1 - a) * 30), s, font, TEXT if s.startswith("One") else MUTED, a, anchor="mm")
            y += 190 if s.startswith("One") else 150
        yield img


def seg_stack(n, cues, thumbs):
    """How it works: fast loop (camera -> depth -> map + route -> safety -> drive) and slow loop (Qwen -> Laya -> wall)."""
    for k in frames(n):
        t = k / FPS
        img = canvas()
        header(img, "How FORESIGHT works", "One camera, two loops. Every model is used zero-shot: nothing was trained for these homes.")
        d = ImageDraw.Draw(img)
        # lanes
        text(img, (G * 3, 380), "FAST LOOP · every control tick (10 Hz)", F("Bold", 34), CYAN)
        text(img, (G * 3, 1060), "SLOW LOOP · in the background", F("Bold", 34), AMBER)
        bw, bh, y0 = 330, 470, 450
        xs = [G * 3 + i * (bw + 46) for i in range(5)]
        fast = [
            ("Camera", "one RGB frame", thumbs["rgb"], cues["stack1"]),
            ("Depth Anything V2", "metres, from one image", thumbs["depth"], cues["stack1"] + 1.2),
            ("Own map + route", "built while driving", thumbs["map"], cues["stack2"]),
            ("Safety layer", "stop or slow if too close", None, cues["stack5"]),
            ("Drive", "velocity command", None, cues["stack5"] + 0.6),
        ]
        for i, (title, sub, th, ts) in enumerate(fast):
            a = ramp(t, ts, 0.45)
            if a <= 0:
                continue
            x = xs[i]
            box(img, (x, y0 + (1 - a) * 24, x + bw, y0 + bh + (1 - a) * 24), PANEL + (int(255 * a),), 22)
            yy = y0 + (1 - a) * 24
            if th is not None:
                paste(img, th.resize((bw - 40, bw - 40)), (x + 20, yy + 20), a)
            else:
                icon = "■" if title.startswith("Safety") else "➜"
                if title.startswith("Safety"):
                    for j_, (w_, c_) in enumerate((("STOP", RED), ("SLOW", AMBER), ("GO", GREEN))):
                        text(img, (x + bw / 2, yy + 80 + j_ * 80), w_, F("Bold", 48), c_, a, anchor="mm")
                else:
                    text(img, (x + bw / 2, yy + 160), "v, ω", F("Bold", 56), TEXT, a, anchor="mm")
            text(img, (x + 22, yy + bw + 2), title, F("Bold", 34), TEXT, a)
            text(img, (x + 22, yy + bw + 52), sub, F("Regular", 27), MUTED, a)
            if i > 0:
                ax = x - 40
                d.polygon([(ax, y0 + 170 - 14), (ax + 26, y0 + 170), (ax, y0 + 170 + 14)], fill=MUTED + (int(255 * a),))
        # slow loop
        sy = 1130
        slow = [("Qwen3-VL (2B)", "describes the scene · ~2–5 s per look", thumbs["note"], cues["stack3"], 700),
                ("Laya", "reads every note in ~20 ms", thumbs["laya"], cues["stack4"], 600)]
        x = G * 3
        for title, sub, body, ts, w in slow:
            a = ramp(t, ts, 0.45)
            if a > 0:
                box(img, (x, sy + (1 - a) * 24, x + w, sy + 430 + (1 - a) * 24), PANEL + (int(255 * a),), 22)
                yy = sy + (1 - a) * 24
                text(img, (x + 26, yy + 24), title, F("Bold", 40), TEXT, a)
                text(img, (x + 26, yy + 82), sub, F("Regular", 27), MUTED, a)
                for j, line in enumerate(body):
                    text(img, (x + 26, yy + 150 + j * 44), line[0], F(line[1], 32), line[2], a)
            x += w + 46
        a = ramp(t, cues["stack5"] + 1.5, 0.6)
        text(img, (S / 2, 1700), "All models run on one 8 GB laptop GPU (RTX 4060).", F("Regular", 32), MUTED, a, anchor="mm")
        a = ramp(t, cues["stack4"] + 2.5, 0.5)
        if a > 0:  # flagged twice -> a wall on the map
            x2 = G * 3 + 700 + 46 + 600 + 46
            box(img, (x2, sy, S - G * 3, sy + 430), PANEL2 + (int(255 * a),), 22)
            text(img, (x2 + 26, sy + 24), "Flagged twice", F("Bold", 40), TEXT, a)
            text(img, (x2 + 26, sy + 82), "→ a wall on the map,", F("Regular", 32), AMBER, a)
            text(img, (x2 + 26, sy + 126), "  the route goes round", F("Regular", 32), AMBER, a)
            mx = xs[2] + bw / 2
            for yy in range(int(y0 + bh + 20), int(sy - 10), 22):
                d.line([(mx, yy), (mx, yy + 11)], fill=AMBER + (int(255 * a),), width=5)
            d.polygon([(mx - 14, y0 + bh + 34), (mx, y0 + bh + 8), (mx + 14, y0 + bh + 34)], fill=AMBER + (int(255 * a),))
        yield img


def seg_homes(n, cues, assets, mosaic_s):
    """Fly-throughs (full frame, captioned) then a mosaic of 12 dollhouses around the logo."""
    by = {h[0]: h for h in homes()}
    fly_s = (n / FPS - mosaic_s) / len(FLY)
    nf = int(round(fly_s * FPS))
    prev = None
    k_total = 0
    for name, _ in FLY:
        z = np.load(Path(assets) / f"fly_{name}.npz")
        pos, yaw = z["pos"], z["yaw"]
        views = _Views(_stage(by[name][1], by[name][2]), size=S)
        cap_scene = name.split("-")[0]
        src = "HM3D · minival" if by[name][1].endswith("pointnav.scene_dataset_config.json") else "HM3D · example home"
        for k in frames(nf, k_total):
            i = min(len(pos) - 1, int(k / max(1, nf - 1) * (len(pos) - 1)))
            frame = Image.fromarray(views.render(pos[i], float(yaw[i]))).convert("RGBA")
            if prev is not None and k < 10:
                frame = Image.blend(prev, frame, (k + 1) / 11)
            text(frame, (G * 3, S - 150), f"Home {cap_scene}", F("Bold", 56), TEXT, shadow=True)
            text(frame, (G * 3, S - 84), f"{src} · Matterport scan in Habitat-Sim", F("Regular", 34), TEXT, shadow=True)
            corner_brand(frame)
            if k == nf - 1:
                prev = frame
            yield frame
        k_total += nf
        views.close()
    # mosaic
    tiles = [h[0] for h in homes() if not h[0].startswith("00337")][:12]
    T = (S - 5 * G) // 4
    ims = {}
    for nm in tiles:
        im, _ = doll(assets, nm, (T, T), el_deg=55, az_deg=-20, pad=24)
        ims[nm] = im
    cells = [(r, c) for r in range(4) for c in range(4) if not (1 <= r <= 2 and 1 <= c <= 2)]
    m = n - k_total
    for k in frames(m, k_total):
        t = k / FPS
        img = canvas()
        for j, (nm, (r, c)) in enumerate(zip(tiles, cells)):
            a = ramp(t, 0.06 * j, 0.5)
            x, y = G + c * (T + G), G + r * (T + G)
            box(img, (x, y, x + T, y + T), PANEL + (int(255 * a),), 16)
            paste(img, ims[nm], (x, y + (1 - a) * 20), a)
            text(img, (x + 18, y + T - 46), nm.split("-")[0], F("Bold", 28), MUTED, a)
        a = ramp(t, 0.9, 0.6)
        cx0, cy0 = G + (T + G), G + (T + G)
        box(img, (cx0, cy0, cx0 + 2 * T + G, cy0 + 2 * T + G), NAVY + (int(255 * a),), 20)
        paste(img, logo_layer(360, progress=min(1.0, max(0.0, (t - 0.9) / 1.2))), (S / 2 - 180, S / 2 - 290))
        tracked(img, S / 2, S / 2 + 110, "FORESIGHT", F("Bold", 92), 16, TEXT, ramp(t, 1.6, 0.5))
        text(img, (S / 2, S / 2 + 250), "13 real homes · 10 for evaluation", F("Regular", 38), MUTED, ramp(t, 2.0, 0.5),
             anchor="mm")
        yield img


def seg_pointgoal(n, cues, views, run, t0=8.0, speed=2.5):
    goal = run.ep.goal
    for k in frames(n):
        tv = k / FPS
        ts = min(run.end, t0 + tv * speed)
        pos, yaw = run.track.at(ts)
        img = Image.fromarray(views.render(pos, yaw)).convert("RGBA")
        text(img, (G * 3, 60), "Point-goal navigation", F("Bold", 64), TEXT, shadow=True)
        text(img, (G * 3, 146), "Home 00800 · map built from the camera alone", F("Regular", 34), TEXT, shadow=True)
        speed_badge(img, S - G * 3, 66, speed)
        quote(img, f"Reach the goal: {run.r['episode']['geodesic_m']:.1f} m away")
        st = run.step(ts)
        dist = float(np.hypot(*(goal[[0, 2]] - pos[[0, 2]])))
        compass(img, S - 190, S - 300, 110, st["goal_bearing_rad"], f"{dist:.1f} m")
        yield img


def seg_search(n, cues, views, ep, clips):
    """clips: [(sim_from, sim_to, speed)], played back to back."""
    track = _Track(ep)
    events = {}
    for s in ep.steps:
        for ev in (s.get("search_events") or []):
            events.setdefault((ev[1], ev[2]), ev[0])
    total = [(b - a) / sp for a, b, sp in clips]
    for k in frames(n):
        tv = k / FPS
        acc, ts, sp = 0.0, clips[-1][1], clips[-1][2]
        for (a, b, s_), L in zip(clips, total):
            if tv < acc + L:
                ts, sp = a + (tv - acc) * s_, s_
                break
            acc += L
        pos, yaw = track.at(ts)
        img = Image.fromarray(views.render(pos, yaw)).convert("RGBA")
        text(img, (G * 3, 60), "Object search", F("Bold", 64), TEXT, shadow=True)
        text(img, (G * 3, 146), "Home 00808 · instructions in words, no map, no coordinates", F("Regular", 34), TEXT, shadow=True)
        speed_badge(img, S - G * 3, 66, sp)
        quote(img, "Find the sofa, then go back to the bed")
        y = S - 400
        for lab in ("sofa", "bed"):
            tf = events.get(("found", lab))
            if tf is not None and ts >= tf:
                chip(img, S - G * 3, y, f"✓  Found: {lab}", GREEN, anchor="right", fg=(10, 30, 20))
                y -= 76
            elif events.get(("instruction", lab), 0.0) <= ts and (lab == "sofa" or ts >= events.get(("instruction", "bed"), 1e9)):
                chip(img, S - G * 3, y, f"Searching: {lab}", (40, 50, 80), anchor="right")
                y -= 76
                break
        yield img


def seg_depth(n, cues, views, runs, sch, assets):
    order = ["gt", "da", "gt:scale=2.5", "gt:noise=0.1"]
    cw = (S - 3 * G) // 2
    top = 290
    ch = (S - top - 2 * G) // 2
    fp = (cw - 3 * G) // 2
    gt = runs["gt"]
    e = gt.r["episode"]
    st, gl = np.asarray(e["start"]), np.asarray(e["goal"])
    lo = np.minimum(st[[0, 2]], gl[[0, 2]]) - 2.0
    hi = np.maximum(st[[0, 2]], gl[[0, 2]]) + 2.0
    dh_im, dh_view = doll(assets, e["scene"], (fp, fp), el_deg=58, az_deg=0, pad=10, fit_xz=(lo, hi))
    lower_h = ch - 70 - fp - 3 * G
    cx, cz = (st[[0, 2]] + gl[[0, 2]]) / 2
    span_x = abs(st[0] - gl[0]) + 4.0
    span_z = span_x * lower_h / fp
    mbox = ((cx - span_x / 2, cz - span_z / 2), (cx + span_x / 2, cz + span_z / 2))
    maps = {k: RobotMap(runs[k].r, mbox, fp, lower_h) for k in order}
    for k in frames(n):
        tv = k / FPS
        ts = sch.sim(tv)
        img = canvas()
        header(img, "Same robot, different depth",
               f"One point-goal episode in home 00800 ({e['geodesic_m']:.1f} m route), driven four times. Only the depth changes.",
               clock=ts, speed=sch.speed(tv))
        for i, key in enumerate(order):
            run = runs[key]
            x0 = G + (i % 2) * (cw + G)
            y0 = top + (i // 2) * (ch + G)
            box(img, (x0, y0, x0 + cw, y0 + ch), PANEL + (255,), 22)
            col = RUN_COLORS[key]
            dr = ImageDraw.Draw(img)
            dr.ellipse((x0 + 24, y0 + 26, x0 + 48, y0 + 50), fill=col)
            text(img, (x0 + 62, y0 + 18), run.label, F("Bold", 38), TEXT)
            tr = min(ts, run.end)
            done = run.arrived and ts >= run.end
            if done:
                chip(img, x0 + cw - 20, y0 + 12, f"✓ Arrived · {run.end:.1f} s", GREEN, F("Bold", 30), anchor="right",
                     fg=(10, 30, 20))
            elif not run.arrived and ts >= run.end - 0.05:
                chip(img, x0 + cw - 20, y0 + 12, f"✗ Timed out · {run.end:.0f} s", RED, F("Bold", 30), anchor="right")
            elif not run.arrived and ts > 45:
                chip(img, x0 + cw - 20, y0 + 12, "No progress", RED, F("Bold", 30), anchor="right")
            pos, yaw = run.track.at(tr)
            fx, fy = x0 + G, y0 + 80
            paste(img, Image.fromarray(views.render(pos, yaw)).resize((fp, fp), Image.LANCZOS), (fx, fy))
            # dollhouse with the route so far
            dx = fx + fp + G
            pl = dh_im.copy()
            dd = ImageDraw.Draw(pl)
            doll_pin(dd, dh_view, gl, RED, 11)
            doll_path(dd, dh_view, run.path_until(tr), col, 6)
            doll_arrow(dd, dh_view, pos, yaw, col, 17)
            box(img, (dx, fy, dx + fp, fy + fp), (20, 27, 48, 255), 14)
            paste(img, pl, (dx, fy))
            # depth given, own map
            ly = fy + fp + G
            j = run.snap(tr)
            paste(img, depth_view(run.r["depth"][j], fp, lower_h), (fx, ly))
            mimg = maps[key].draw(run.r["map"][j])
            md = ImageDraw.Draw(mimg)
            path = run.r["path"][j]
            if len(path) > 1 and not done:
                u, v = maps[key].to_px(path[:, 0], path[:, 1])
                md.line(list(zip(u, v)), fill=AMBER, width=3)
            gu, gv = maps[key].to_px(gl[0], gl[2])
            md.ellipse((gu - 9, gv - 9, gu + 9, gv + 9), fill=RED, outline=(255, 255, 255), width=2)
            ru, rv = maps[key].to_px(pos[0], pos[2])
            robot_arrow(md, ru, rv, yaw, col, 15)
            paste(img, mimg, (dx, ly))
            for bx, lab in ((fx, "Depth it was given"), (dx, "Its own map")):
                box(img, (bx + 8, ly + 8, bx + 22 + ImageDraw.Draw(img).textlength(lab, font=F("Bold", 24)), ly + 46),
                    (8, 10, 20, 200), 10)
                text(img, (bx + 15, ly + 13), lab, F("Bold", 24), TEXT)
            if done:
                box(img, (x0, y0, x0 + cw, y0 + ch), (22, 30, 54, 70), 22)
        yield img


def seg_mirror(n, cues, mirror, sch, assets):
    base, laya, st = mirror["base"], mirror["laya"], mirror["st"]
    bt, lt = _Track(base), _Track(laya)
    views = _Views(st, size=768)
    pw = (S - 3 * G) // 2
    top = 290
    ph = 800
    events = [(s["t"], s["event"]) for s in laya.steps if s["event"]]
    walls_at = next(s["t"] for s in laya.steps if s["wall_added"])
    fwd = st["fwd"]
    trap = st["start"] + fwd * st["trap_ahead_m"]
    perp = np.array([-fwd[2], 0, fwd[0]]) * st["trap_width_m"] / 2
    dy = top + ph + G
    dh = S - dy - G
    pts = np.vstack([base.positions, laya.positions, st["goal"][None]])
    lo, hi = pts[:, [0, 2]].min(0) - 1.0, pts[:, [0, 2]].max(0) + 1.0
    dimg, dview = doll(assets, st["scene"], (S - 2 * G, dh), el_deg=56, az_deg=0, pad=26)
    l_end = float(laya.reached_goal_at)
    for k in frames(n):
        tv = k / FPS
        ts = sch.sim(tv)
        img = canvas()
        header(img, "Mirror encounter",
               "Home 00808. Same start, goal behind the mirror. One staged encounter: a demo, not a benchmark.",
               clock=min(ts, l_end), speed=sch.speed(tv))
        # panels
        for i, (name, ep, tr, col) in enumerate((("Depth only", base, bt, RED), ("FORESIGHT", laya, lt, BLUE))):
            x0 = G + i * (pw + G)
            end = float(ep.contact_at if ep.contact_at is not None else (ep.reached_goal_at or tr.end))
            te = min(ts, end)
            pos, yaw = tr.at(te)
            fr = Image.fromarray(views.render(pos, yaw))
            frame = fr.resize((pw, pw), Image.LANCZOS).crop((0, (pw - ph) // 2, pw, (pw - ph) // 2 + ph)).convert("RGBA")
            if ep.contact_at is not None and ts >= ep.contact_at:
                frame.alpha_composite(Image.new("RGBA", frame.size, (160, 20, 30, 110)))
            paste(img, frame, (x0, top))
            box(img, (x0 + 18, top + 18, x0 + 60 + ImageDraw.Draw(img).textlength(name, font=F("Bold", 40)), top + 82),
                (8, 10, 20, 210), 14)
            ImageDraw.Draw(img).ellipse((x0 + 32, top + 38, x0 + 52, top + 58), fill=col)
            text(img, (x0 + 62, top + 26), name, F("Bold", 40), TEXT)
            by = top + ph - 100
            if ep is base and ep.contact_at is not None and ts >= ep.contact_at:
                chip(img, x0 + pw / 2 + 200, by, "✗ Drove into the mirror", RED, F("Bold", 40), anchor="right")
            if ep is laya:
                msg = None
                last = [(t_, e_) for t_, e_ in events if t_ <= ts]
                if ts >= l_end:
                    msg = (f"✓ Goal reached · {l_end:.1f} s", GREEN, (10, 30, 20))
                elif last:
                    t_, e_ = last[-1]
                    if t_ < walls_at + 0.01 and e_ == "flag":
                        msg = ("Mirror? Taking a second look", AMBER, (30, 20, 0))
                    elif e_ == "confirmed" and ts - t_ < 4.0 and t_ <= walls_at + 0.01:
                        msg = ("Mirror confirmed: wall on its map", AMBER, (30, 20, 0))
                    elif e_ == "flag":
                        msg = ("Checking a possible mirror…", (60, 70, 100), TEXT)
                if msg:
                    w = ImageDraw.Draw(img).textlength(msg[0], font=F("Bold", 38)) + 44
                    chip(img, x0 + (pw + w) / 2, by, msg[0], msg[1], F("Bold", 38), anchor="right", fg=msg[2])
        # dollhouse with both routes
        pl = dimg.copy()
        dd = ImageDraw.Draw(pl)
        a, b = trap - perp, trap + perp
        ua, va = dview.project(a[0], 0.6, a[2])
        ub, vb = dview.project(b[0], 0.6, b[2])
        dd.line([(ua, va), (ub, vb)], fill=CYAN, width=9)
        dd.text((ub + 10, vb - 20), "mirror", font=F("Bold", 28), fill=CYAN, stroke_width=3, stroke_fill=(10, 12, 20))
        if ts >= walls_at:
            for (wa, wb) in laya.steps[-1]["map_walls"][:1]:
                u1, v1 = dview.project(wa[0], 0.05, wa[1])
                u2, v2 = dview.project(wb[0], 0.05, wb[1])
                dd.line([(u1, v1), (u2, v2)], fill=AMBER, width=8)
        doll_pin(dd, dview, st["goal"], RED, 13, "goal")
        for ep, tr, col in ((base, bt, RED), (laya, lt, BLUE)):
            end = float(ep.contact_at if ep.contact_at is not None else (ep.reached_goal_at or tr.end))
            te = min(ts, end)
            k_ = int(np.searchsorted(tr.t, te, side="right"))
            p, yw = tr.at(te)
            doll_path(dd, dview, np.vstack([tr.pos[:k_], p[None]]), col, 7)
            doll_arrow(dd, dview, p, yw, col, 18)
        box(img, (G, dy, S - G, dy + dh), PANEL + (255,), 22)
        paste(img, pl, (G, dy))
        yield img
    views.close()


def seg_results(n, cues, A):
    """Success rate (map navigator, set A) for six depth conditions, with 95% bootstrap intervals; one hue, a
    reference line at ground truth, error size (AbsRel) as a text column (not a second axis)."""
    rows = [("Ground-truth depth", "map:gt", "reference"),
            ("Depth Anything V2", "map:da", "the real estimator"),
            ("Read 2.5× too far", "map:gt:scale=2.5", "scale, floor-calibrated"),
            ("Read 0.3 m too far", "map:gt:bias_m=0.3", "bias"),
            ("Read 0.3 m too near", "map:gt:bias_m=-0.3", "bias"),
            ("10% random noise", "map:gt:noise=0.1", "zero-mean noise")]
    x_lab, x0, x1 = G * 3, 660, 1440
    x_err = 1680
    y0, rh = 470, 190
    ref = A["map:gt"]["success"][0]
    for k in frames(n):
        t = k / FPS
        img = canvas()
        header(img, "120 episodes, 10 homes", "Point-goal success with each kind of depth error. Map navigator, set A.")
        d = ImageDraw.Draw(img)
        text(img, (x0, y0 - 92), "Success rate", F("Bold", 30), MUTED)
        text(img, (x_err, y0 - 92), "Error size", F("Bold", 30), MUTED)
        text(img, (x_err, y0 - 56), "(AbsRel)", F("Regular", 26), MUTED)
        for g in (0, 25, 50, 75, 100):  # recessive grid
            gx = x0 + (x1 - x0) * g / 100
            d.line([(gx, y0 - 20), (gx, y0 + rh * len(rows) - 40)], fill=(255, 255, 255, 22), width=2)
            text(img, (gx, y0 + rh * len(rows) - 20), f"{g}%", F("Regular", 26), DIM, anchor="ma")
        for i, (lab, key, note) in enumerate(rows):
            r = A[key]
            mean, lo, hi = r["success"]
            y = y0 + i * rh
            a = ramp(t, 0.4 + 0.35 * i, 0.5)
            grow = ramp(t, 0.4 + 0.35 * i, 0.9)
            text(img, (x_lab, y + 4), lab, F("Bold", 38), TEXT, a)
            text(img, (x_lab, y + 56), note, F("Regular", 28), MUTED, a)
            bh = 46
            xe = x0 + (x1 - x0) * mean * grow
            if xe > x0 + 4:
                d.rounded_rectangle((x0, y + 18, xe, y + 18 + bh), 4, fill=BLUE + (int(255 * a),))
            if grow >= 1:
                cl, chh = x0 + (x1 - x0) * lo, x0 + (x1 - x0) * hi
                yc = y + 18 + bh / 2
                d.line([(cl, yc), (chh, yc)], fill=TEXT, width=3)
                d.line([(cl, yc - 12), (cl, yc + 12)], fill=TEXT, width=3)
                d.line([(chh, yc - 12), (chh, yc + 12)], fill=TEXT, width=3)
                text(img, (max(xe, chh) + 18, y + 20), f"{mean * 100:.0f}%", F("Bold", 40), TEXT)
            text(img, (x_err, y + 20), f"{r['absrel']:.2f}", F("Bold", 40), TEXT, a)
        # reference line at ground truth
        a = ramp(t, 0.6, 0.5)
        rx = x0 + (x1 - x0) * ref
        for yy in range(y0 - 20, y0 + rh * len(rows) - 40, 24):
            d.line([(rx, yy), (rx, yy + 12)], fill=(255, 255, 255, int(150 * a)), width=3)
        a = ramp(t, cues["results3"], 0.5)
        if a > 0:
            box(img, (G * 3, S - 290, S - G * 3, S - 140), PANEL2 + (int(255 * a),), 22)
            text(img, (S / 2, S - 215), "Error type predicts failure. Error size doesn't.", F("Bold", 54), TEXT, a, anchor="mm")
        text(img, (G * 3, S - 100),
             "95% bootstrap intervals · n = 104–112 per condition · success = within 0.36 m · simulation (Habitat-Sim, HM3D)",
             F("Regular", 26), MUTED)
        text(img, (G * 3, S - 62),
             f"In the 4 homes with mirror annotations, Depth Anything cost more: {A['S']['map:da']:.0f}% vs {A['S']['map:gt']:.0f}% (p = 0.006).",
             F("Regular", 26), MUTED)
        yield img


def seg_next(n, cues):
    items = [("Mirror & glass encounters", "a benchmark of targeted starts facing mirrors and glass panes"),
             ("What does the VLM add?", "FORESIGHT vs the depth-only robot, head to head")]
    for k in frames(n):
        t = k / FPS
        img = canvas()
        header(img, "What's next")
        for i, (a_, b_) in enumerate(items):
            a = ramp(t, 0.6 + 1.6 * i, 0.5)
            y = 700 + i * 360
            box(img, (G * 3, y, S - G * 3, y + 280), PANEL + (int(255 * a),), 24)
            text(img, (G * 3 + 50, y + 60), f"0{i + 1}", F("Bold", 90), BLUE, a)
            text(img, (G * 3 + 230, y + 70), a_, F("Bold", 58), TEXT, a)
            text(img, (G * 3 + 230, y + 160), b_, F("Regular", 38), MUTED, a)
        yield img


def seg_close(n):
    refs = ["Habitat-Sim (Savva et al. 2019; Szot et al. 2021)", "HM3D (Ramakrishnan et al. 2021)",
            "Depth Anything V2 (Yang et al. 2024)", "Qwen3-VL (Qwen Team 2025)", "Voice: Kokoro-82M"]
    for k in frames(n):
        t = k / FPS
        img = canvas()
        paste(img, logo_layer(360), ((S - 360) / 2, 380))
        tracked(img, S / 2, 800, "FORESIGHT", F("Bold", 120), 22, TEXT)
        text(img, (S / 2, 1000), "Zero-shot navigation from a single camera · simulation", F("Regular", 44), MUTED, anchor="mm")
        a = ramp(t, 1.0, 0.6)
        for i, r in enumerate(refs):
            text(img, (S / 2, 1260 + i * 56), r, F("Regular", 32), DIM, a, anchor="mm")
        yield img


# ---------------------------------------------------------------- assembly


class Writer:
    def __init__(self, path: Path, fade=8):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{S}x{S}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.n, self.fade = 0, fade
        self.bg = Image.new("RGB", (S, S), NAVY)
        self.last = None

    def segment(self, frames, n_total, fade_in=True, fade_out=True):
        k = -1
        for k, im in enumerate(frames):
            if k >= n_total:
                break
            im = im.convert("RGB")
            if fade_in and k < self.fade:
                im = Image.blend(self.bg, im, (k + 1) / (self.fade + 1))
            elif fade_out and k >= n_total - self.fade:
                im = Image.blend(self.bg, im, (n_total - k) / (self.fade + 1))
            self.proc.stdin.write(im.tobytes())
            self.last = im
            self.n += 1
        for _ in range(k + 1, n_total):  # a generator that ran short holds its last frame
            self.proc.stdin.write(self.last.tobytes())
            self.n += 1

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def _lay(lines, dur, start=0.5, gap=0.45, anchors=None):
    """Start times (within a segment) for lines spoken in order; anchors force a later start."""
    out, t = {}, start
    for ln in lines:
        t = max(t, (anchors or {}).get(ln, 0.0))
        out[ln] = t
        t += dur[ln] + gap
    return out, t


def render_video(root: Path, out_path: Path, tts_python: str, kokoro_home: str, voice: str, preview=None):
    assets = root / "assets"
    runs = {}
    for p in sorted((root / "pointnav").glob("*.pkl")):
        r = PointnavRun(p)
        runs[r.source] = r
    summ = json.loads(Path("results/runs/pointnav/A/summary.json").read_text())["all"]
    summ_s = json.loads(Path("results/runs/pointnav/S/summary.json").read_text())["all"]
    summ["S"] = {"map:da": summ_s["map:da"]["success"][0] * 100, "map:gt": summ_s["map:gt"]["success"][0] * 100}
    # mirror trap (staged encounter, 2026-10-29 run) and object search (mirror-sofa-bed, FORESIGHT)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
    from run_mirror_trap_demo import stage
    mdir = Path("results/runs/mirror_trap_demo_v2/hm3d-hallway-mirror")
    with open(mdir / "baseline.pkl", "rb") as f:
        mbase = pickle.load(f)
    with open(mdir / "laya.pkl", "rb") as f:
        mlaya = pickle.load(f)
    mirror = {"base": mbase, "laya": mlaya, "st": stage("hm3d-hallway-mirror")}
    with open("results/runs/object_search/eval/mirror-sofa-bed/foresight.pkl", "rb") as f:
        search = pickle.load(f)

    lines = narration(runs, mirror, search, summ)
    vo = Voice(lines, root / "narration", tts_python, kokoro_home, voice)
    dur = vo.dur

    gt = runs["gt"]
    st800 = _stage(str(Path("data/processed/scene_datasets/hm3d/minival/hm3d_minival_basis_pointnav.scene_dataset_config.json")),
                   str(next(Path(f"data/processed/scene_datasets/hm3d/minival/{gt.r['episode']['scene']}").glob("*.basis.glb")).resolve()))

    # thumbnails for the how-it-works card
    v = _Views(st800, size=640)
    t_thumb = max(gt.ep.steps, key=lambda s_: s_["gt_depth_ahead_m"] if s_["t"] > 9 else 0)["t"]
    p0, y0_ = gt.track.at(t_thumb)
    rgb = v.render(p0, y0_)
    v.close()
    j = gt.snap(t_thumb)
    e = gt.r["episode"]
    st_, gl_ = np.asarray(e["start"]), np.asarray(e["goal"])
    c = (st_[[0, 2]] + gl_[[0, 2]]) / 2
    rm = RobotMap(gt.r, ((c[0] - 6.0, c[1] - 6.0), (c[0] + 6.0, c[1] + 6.0)), 290, 290)
    mthumb = rm.draw(gt.r["map"][-1])
    md = ImageDraw.Draw(mthumb)
    tr = gt.path_until(gt.end)
    u, v = rm.to_px(tr[:, 0], tr[:, 2])
    md.line(list(zip(u, v)), fill=BLUE, width=4)
    gu, gv = rm.to_px(gl_[0], gl_[2])
    md.ellipse((gu - 7, gv - 7, gu + 7, gv + 7), fill=RED)
    flag = next(s for s in mlaya.steps if s["event"] == "flag")
    note_words = _wrap(ImageDraw.Draw(canvas()), "“" + " ".join(flag["description"].split()[:40]), F("Regular", 30), 640, 6)
    thumbs = {"rgb": Image.fromarray(rgb), "depth": depth_view(gt.r["depth"][j], 290, 290),
              "map": mthumb,
              "note": [(ln, "Regular", TEXT) for ln in note_words[:5]],
              "laya": [("Is the way ahead a mirror?", "Regular", MUTED),
                       (f"P(mirror) = {flag['p_hazard']:.2f}", "Bold", AMBER),
                       ("> 0.6 → turn back, look again", "Regular", TEXT)]}

    # schedules
    n_end = runs["gt:noise=0.1"].end
    dsched = Sched([(2.4, 0.0, 0.0), (10.0, 0.0, 44.0), (11.5, 44.0, 70.0), (8.5, 70.0, n_end), (99.0, n_end, n_end)])
    l_end = float(mlaya.reached_goal_at)
    msched = Sched([(1.2, 0.0, 0.0), (16.0, 0.0, 17.0), (10.0, 17.0, l_end), (99.0, l_end, l_end)])

    segs = []

    def add(name, make, lines_, seconds_min, start=0.5, anchors=None, tail=1.2, fade=(True, True)):
        cues, end = _lay(lines_, dur, start, anchors=anchors)
        seconds = max(seconds_min, end + tail - 0.45) if lines_ else seconds_min
        segs.append((name, make, cues, seconds, fade))

    add("open", lambda n, c: seg_open(n), [], 4.6)
    add("question", lambda n, c: seg_question(n, c, Image.fromarray(rgb)), ["question"], 10.0, start=0.4)
    add("stack", lambda n, c: seg_stack(n, c, thumbs), ["stack1", "stack2", "stack3", "stack4", "stack5"], 20.0, start=0.8)
    add("homes", lambda n, c: seg_homes(n, c, assets, mosaic_s=5.0), ["homes1", "homes2"], 19.5, start=0.5,
        anchors={"homes2": 14.4})
    add("task_point", lambda n, c: _with_views(st800, S, lambda vw: seg_pointgoal(n, c, vw, gt)), ["task1"], 6.5, start=0.4)
    add("task_search", lambda n, c: _with_views(_stage(HM3D_ANN, "00808-y9hTuugGdiq"), S,
                                                lambda vw: seg_search(n, c, vw, search, [(117.0, 137.5, 3.0), (160.0, 189.5, 4.0)])),
        ["task2"], 14.2, start=0.4)
    arrive_v = dsched.video_at(30.0)
    add("depth", lambda n, c: _with_views(st800, 512, lambda vw: seg_depth(n, c, vw, runs, dsched, assets)),
        ["depth1", "depth2", "depth3"], dsched.lead + sum(p[0] for p in dsched.pieces[:4]) + 1.5, start=0.3,
        anchors={"depth2": dsched.video_at(44.0) - 1.6, "depth3": dsched.video_at(70.0)})
    add("mirror", lambda n, c: seg_mirror(n, c, mirror, msched, assets), ["mirror1", "mirror2", "mirror_crash", "mirror3"],
        msched.lead + 16.0 + 10.0 + 2.0, start=0.3,
        anchors={"mirror2": msched.video_at(2.1), "mirror_crash": msched.video_at(14.8), "mirror3": msched.video_at(17.0)})
    add("results", lambda n, c: seg_results(n, c, summ), ["results1", "results2", "results3"], 16.0, start=0.8)
    add("next", lambda n, c: seg_next(n, c), ["next"], 7.0, start=0.5)
    add("close", lambda n, c: seg_close(n), ["close"], 7.0, start=0.6, tail=2.0)

    t_ = 0.0
    for name, _, cues, seconds, _ in segs:
        print(f"  {t_:6.1f}  {name:12s} {seconds:5.1f} s  " + " ".join(f"{k}@{v:.1f}" for k, v in cues.items()))
        t_ += seconds
    if preview is not None:
        t0 = 0.0
        for name, make, cues, seconds, fade in segs:
            if preview < t0 + seconds:
                n = int(round(seconds * FPS))
                _PREVIEW["k"] = int((preview - t0) * FPS)
                for im in make(n, cues):
                    out = out_path.with_name(f"preview_{preview:06.1f}.png")
                    im.convert("RGB").save(out)
                    print(f"preview {name} @ {preview - t0:.1f} s -> {out}")
                    return
            t0 += seconds
        print("preview time past the end")
        return

    silent = out_path.with_name(out_path.stem + ".silent.mp4")
    w = Writer(silent)
    for name, make, cues, seconds, fade in segs:
        n = int(round(seconds * FPS))
        t0 = w.n / FPS
        vo.cues += [(t0 + t, ln) for ln, t in cues.items()]
        w.segment(make(n, cues), n, *fade)
        print(f"  {name}: {n / FPS:.1f} s (total {w.n / FPS:.1f} s)", flush=True)
    w.close()
    total = w.n / FPS
    wav = out_path.with_name(out_path.stem + ".voice.wav")
    vo.write_mix(wav, total)
    subprocess.run(["ffmpeg", "-y", "-i", str(silent), "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", "-shortest", str(out_path)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    silent.unlink()
    wav.unlink()
    print(f"wrote {out_path} ({total:.1f} s, {S}x{S}, voice-over {len(vo.cues)} lines)")


def _with_views(st, size, fn):
    views = _Views(st, size=size)
    try:
        yield from fn(views)
    finally:
        views.close()
