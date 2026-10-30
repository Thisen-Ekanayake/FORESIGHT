"""Square 1920x1920 narrated video for tools/run_object_search.py ("FORESIGHT, find the sofa. Then the bed.").

Segments: hook (the spoken instruction over the robot's first view) -> the lure (why a mirror is the best-looking
unexplored space, with what Qwen grounded inside it) -> the search (both robots side by side: re-rendered
cameras, an inset of each robot's latest VLM look with its boxes, and a photographic top-down map revealed only
where that robot has mapped, with its remembered objects, frontier, route and walls) -> how the two loops fit ->
the fixed evaluation set's scoreboard -> credits. Kokoro-82M voice-over cued to run events; playback speed is
badged on screen.

Helpers (fonts, boxes, the ffmpeg writer, the voice track, camera re-rendering) are shared with
`foresight.viz.mirror_trap_video`. Every number drawn or spoken comes from the saved runs or eval.json.
"""
from __future__ import annotations

import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from foresight.viz.mirror_trap_video import (AMBER, BG, BLUE, CYAN, DIM, FPS, G, GREEN, MUTED, PANEL, RED, S, TEXT,
                                             Voice, _arrow, _banner, _box, _centered, _dashed, _depth_img, _font,
                                             _place, _spoken_int, _text_block, _Track, _Views, _wrap, _Writer)

PW = (S - 3 * G) // 2
LX, RX = G, 2 * G + PW
PANEL_Y = 200
STRIP_Y, STRIP_H = PANEL_Y + PW + 14, 96
BOT_Y = STRIP_Y + STRIP_H + 14
BOT_H = S - G - BOT_Y
MAP_W, MAP_H = PW, BOT_H  # map panel: the square cutaway (side MAP_W) cropped to the middle rows
INSET = 330
YELLOW = (255, 214, 64)
LABEL_COLORS = {"sofa": (255, 140, 200), "bed": (140, 200, 255), "tv": (200, 170, 255), "refrigerator": (150, 240, 220),
                "toilet": (240, 220, 150)}
NAMES = {"tv": "TV", "refrigerator": "fridge"}


def _name(label):
    return NAMES.get(label, label)


def _tick(ep, t):
    return max(0, int(np.searchsorted(ep.times, t, side="right")) - 1)


class Run:
    """One condition's episode with lookups for the latest logged map / memory / look at any sim time."""

    def __init__(self, pkl: Path, meta: Path):
        with open(pkl, "rb") as f:
            self.ep = pickle.load(f)
        self.meta = json.loads(meta.read_text())
        self.track = _Track(self.ep)
        self.end = self.track.end
        steps = self.ep.steps
        self.map_idx = np.array([i for i, x in enumerate(steps) if "map_score" in x] + [len(steps) - 1])
        self.map_t = self.ep.times[self.map_idx]
        self.looks = [(x["t"], lk) for x in steps for lk in x.get("looks", [])]  # (delivered at, look)
        self.events = steps[-1]["search_events"]
        self.found = steps[-1]["found"]
        self.mirror = [(x["t"], x["event"], x) for x in steps if x.get("event")]
        self.crash = self.ep.contact_at

    def step(self, t):
        return self.ep.steps[_tick(self.ep, t)]

    def mirror_encounter(self, mirror_xz, near_m: float = 1.5, within_s: float = 60.0):
        """-> (sim time, true distance ahead) of the closest approach while within `near_m` of the mirror, or None."""
        if mirror_xz is None:
            return None
        a, b = (np.asarray(p, float) for p in mirror_xz)
        best = None
        for i, x in enumerate(self.ep.steps):
            if x["t"] > within_s:
                break
            p = np.asarray(self.ep.positions[i], float)[[0, 2]]
            u = np.clip((p - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
            if np.linalg.norm(a + u * (b - a) - p) < near_m and (best is None or x["gt_depth_ahead_m"] < best[1]):
                best = (x["t"], x["gt_depth_ahead_m"])
        return best

    def behind_mirror(self, mirror_xz, start_xz):
        """Remembered objects on the far side of the mirror line from the start (seen in the reflection)."""
        if mirror_xz is None:
            return []
        a, b = (np.asarray(p, float) for p in mirror_xz)
        n = np.array([-(b - a)[1], (b - a)[0]])
        side0 = np.sign((np.asarray(start_xz) - a) @ n)
        out = []
        for o in self.ep.steps[-1].get("memory", []):
            q = np.asarray(o["xz"], float)
            u = (q - a) @ (b - a) / ((b - a) @ (b - a))
            if np.sign((q - a) @ n) != side0 and -0.5 < u < 1.5 and abs((q - a) @ n) / np.linalg.norm(n) < 2.0:
                out.append(o)
        return out

    def map_step(self, t):
        k = max(0, int(np.searchsorted(self.map_t, t, side="right")) - 1)
        x = self.ep.steps[self.map_idx[k]]
        if "map_score" not in x:  # final tick: use the last real map
            x = dict(self.ep.steps[self.map_idx[max(0, k - 1)]], **{kk: x[kk] for kk in ("memory", "found") if kk in x})
        return x

    def look(self, t):
        best = None
        for at, lk in self.looks:
            if at > t:
                break
            best = (at, lk)
        return best

    def event_time(self, kind, label=None, nth=0):
        hits = [e[0] for e in self.events if e[1] == kind and (label is None or e[2] == label)]
        return hits[nth] if len(hits) > nth else None

    def mirror_time(self, kind, nth=0):
        hits = [t for t, e, _ in self.mirror if e == kind]
        return hits[nth] if len(hits) > nth else None


class Scene:
    """The shared photographic top-down cutaway and its world <-> pixel mapping."""

    def __init__(self, root: Path):
        self.meta = json.loads((root / "assets.json").read_text())
        self.photo = np.asarray(Image.open(root / "topdown.png").convert("RGB"))
        self._cache = {}

    def to_px(self, x, z, size, crop_h=None):
        c, k = self.meta["center"], size / self.meta["width_m"]
        dy = 0 if crop_h is None else (size - crop_h) / 2
        return size / 2 + (x - c[0]) * k, size / 2 + (z - c[2]) * k - dy

    def base(self, size):
        if size not in self._cache:
            self._cache[size] = np.asarray(Image.fromarray(self.photo).resize((size, size), Image.LANCZOS)).astype(np.float32)
        return self._cache[size]

    def revealed(self, run: Run, t, size, start_xz, crop_h=None):
        """The cutaway lit only where this robot's own map has evidence (explored); the rest dimmed."""
        m = run.map_step(t)
        score = m["map_score"].astype(np.float32) / 30.0
        known = np.abs(score) > 0.25
        n = score.shape[0]
        res = 0.1
        origin = np.asarray(start_xz, float) - n * res / 2
        # sample the grid at each map pixel
        c, w = self.meta["center"], self.meta["width_m"]
        u = (np.arange(size) + 0.5) / size * w + c[0] - w / 2
        v = (np.arange(size) + 0.5) / size * w + c[2] - w / 2
        col = np.clip(((u - origin[0]) / res).astype(int), 0, n - 1)
        row = np.clip(((v - origin[1]) / res).astype(int), 0, n - 1)
        mask = known[row[:, None], col[None, :]].astype(np.float32)
        mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(_blur()), np.float32)[..., None] / 255
        base = self.base(size)
        out = base * (0.22 + 0.78 * mask)
        if crop_h is not None:
            y0 = (size - crop_h) // 2
            out = out[y0:y0 + crop_h]
        return Image.fromarray(out.astype(np.uint8)).convert("RGBA")


def _blur():
    from PIL import ImageFilter
    return ImageFilter.GaussianBlur(3)


# ---------------------------------------------------------------- panels


def _pin(d, x, y, label, solid, faded=False):
    color = LABEL_COLORS.get(label, TEXT)
    if faded:
        color = (110, 110, 110)
    r = 13
    if solid:
        d.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(255, 255, 255), width=3)
    else:
        d.ellipse((x - r, y - r, x + r, y + r), outline=color, width=4)
    d.text((x + 16, y - 16), _name(label), font=_font("Bold", 24), fill=color, stroke_width=3, stroke_fill=(10, 12, 16))


def _map(img, scene: Scene, run: Run, t, xy, color, start_xz, mirror_xz=None):
    size, ch = MAP_W, MAP_H
    canvas = scene.revealed(run, t, size, start_xz, crop_h=ch)
    d = ImageDraw.Draw(canvas)
    px = lambda x, z: scene.to_px(x, z, size, ch)
    st = run.step(t)
    ms = run.map_step(t)
    if mirror_xz is not None:
        a, b = mirror_xz
        d.line([px(*a), px(*b)], fill=CYAN, width=7)
        mx, my = px(*a)
        d.text((mx + 10, my - 6), "mirror", font=_font("Black", 24), fill=CYAN, stroke_width=3, stroke_fill=(10, 12, 16))
    for a, b in st.get("map_walls", []):
        _dashed(d, px(*a), px(*b), AMBER, 7)
    route = [px(p[0], p[1]) for p in st.get("plan_path", [])]
    if len(route) > 1 and st.get("mode") not in ("done", "gave_up", "announce"):
        for a, b in zip(route[:-1:2], route[1::2]):
            d.line([a, b], fill=(255, 255, 255), width=3)
    if st.get("mode") == "explore" and st.get("frontier_xz"):
        fx, fy = px(*st["frontier_xz"])
        d.line([(fx - 12, fy - 12), (fx + 12, fy + 12)], fill=YELLOW, width=5)
        d.line([(fx - 12, fy + 12), (fx + 12, fy - 12)], fill=YELLOW, width=5)
    summary = {}
    for o in ms.get("memory", []):
        confirmed = o["hits"] >= 3 and o["hits"] > o["misses"] and not o["rejected"]
        ignored = o["rejected"] or (o["raw_hits"] > o["hits"] and o["hits"] < 3)  # e.g. seen through a mirror wall
        sm = summary.setdefault(o["label"], [0, 0])
        sm[0 if confirmed else 1] += 0 if ignored else 1
        x, y = px(*o["xz"])
        if 0 <= x < size and 0 <= y < ch:
            _pin(d, x, y, o["label"], confirmed, faded=ignored)
    k = int(np.searchsorted(run.track.t, t, side="right"))
    pos, yaw = run.track.at(t)
    path = [px(p[0], p[2]) for p in run.track.pos[:k]] + [px(pos[0], pos[2])]
    sx, sy = path[0]
    d.ellipse((sx - 10, sy - 10, sx + 10, sy + 10), outline=(255, 255, 255), width=4)
    if len(path) > 1:
        d.line(path, fill=color, width=7, joint="curve")
    cx, cy = path[-1]
    f = np.array([-np.sin(yaw), -np.cos(yaw)])
    n = np.array([-f[1], f[0]])
    tip = np.array([cx, cy]) + f * 28
    d.polygon([tuple(tip), tuple(np.array([cx, cy]) + n * 14 - f * 10), tuple(np.array([cx, cy]) - n * 14 - f * 10)],
              fill=color, outline=(255, 255, 255))
    img.alpha_composite(canvas, xy)
    _box(img, (xy[0] + 10, xy[1] + 10, xy[0] + 400, xy[1] + 52), (8, 10, 14, 205), radius=10)
    d = ImageDraw.Draw(img)
    d.text((xy[0] + 22, xy[1] + 16), "Its own map, as it explores", font=_font("Bold", 24), fill=TEXT)
    if summary:
        parts = [(f"{_name(l)}: {c} sure" + (f", {m} maybe" if m else ""), LABEL_COLORS.get(l, TEXT)) for l, (c, m) in summary.items()]
        w = sum(d.textlength(p, font=_font("Bold", 26)) for p, _ in parts) + 30 * len(parts) + 20
        x0 = xy[0] + MAP_W - 10 - w
        _box(img, (x0, xy[1] + 10, xy[0] + MAP_W - 10, xy[1] + 52), (8, 10, 14, 205), radius=10)
        d = ImageDraw.Draw(img)
        x = x0 + 18
        for ptxt, col in parts:
            d.text((x, xy[1] + 15), ptxt, font=_font("Bold", 26), fill=col)
            x += d.textlength(ptxt, font=_font("Bold", 26)) + 30
    _box(img, (xy[0] + 10, xy[1] + MAP_H - 50, xy[0] + MAP_W - 10, xy[1] + MAP_H - 10), (8, 10, 14, 205), radius=10)
    d = ImageDraw.Draw(img)
    lx, ly, f = xy[0] + 30, xy[1] + MAP_H - 30, _font("Medium", 22)
    d.ellipse((lx - 9, ly - 9, lx + 9, ly + 9), fill=TEXT)
    d.text((lx + 16, ly - 14), "seen 3+ times", font=f, fill=MUTED)
    lx += 200
    d.ellipse((lx - 9, ly - 9, lx + 9, ly + 9), outline=TEXT, width=3)
    d.text((lx + 16, ly - 14), "maybe", font=f, fill=MUTED)
    lx += 120
    d.line([(lx - 9, ly - 9), (lx + 9, ly + 9)], fill=YELLOW, width=4)
    d.line([(lx - 9, ly + 9), (lx + 9, ly - 9)], fill=YELLOW, width=4)
    d.text((lx + 16, ly - 14), "next place to explore", font=f, fill=MUTED)
    lx += 270
    _dashed(d, (lx - 12, ly), (lx + 22, ly), AMBER, 5, dash=8, gap=5)
    d.text((lx + 32, ly - 14), "wall it drew", font=f, fill=MUTED)


def _inset(img, views, run: Run, t, x, y, labels):
    """The robot's latest VLM look: the frame it was about (re-rendered at that pose), its boxes, P(yes)."""
    lk = run.look(t)
    _box(img, (x - 6, y - 6, x + INSET + 6, y + INSET + 74), (8, 10, 14, 225), radius=12)
    d = ImageDraw.Draw(img)
    if lk is None:
        d.text((x + 10, y + 10), "Qwen: first look…", font=_font("Medium", 24), fill=MUTED)
        return
    at, look = lk
    frame = views.render(*run.track.at(look["seen"]))
    im = Image.fromarray(frame).resize((INSET, INSET), Image.LANCZOS).convert("RGBA")
    img.alpha_composite(im, (x, y))
    d = ImageDraw.Draw(img)
    for lab, boxes in look["boxes"].items():
        col = LABEL_COLORS.get(lab, TEXT)
        for b in boxes:
            x1, y1, x2, y2 = [v / 1000 * INSET for v in b]
            d.rectangle((x + x1, y + y1, x + x2, y + y2), outline=col, width=4)
            d.text((x + x1 + 4, y + y1 + 2), _name(lab), font=_font("Bold", 22), fill=col, stroke_width=2, stroke_fill=(0, 0, 0))
    age = t - look["seen"]
    d.text((x + 6, y + INSET + 8), f"Qwen's last look · {age:.1f} s old", font=_font("Medium", 22), fill=MUTED)
    parts = [f"{_name(l)}? {look['present'].get(l, 0):.0%}" for l in labels if l in look.get("present", {})]
    d.text((x + 6, y + INSET + 38), "   ".join(parts), font=_font("Bold", 24), fill=TEXT)


def _first_person(img, views, run, t, x, labels):
    frame = views.render(*run.track.at(t))
    img.alpha_composite(Image.fromarray(frame).resize((PW, PW), Image.LANCZOS).convert("RGBA"), (x, PANEL_Y))
    _inset(img, views, run, t, x + PW - INSET - 20, PANEL_Y + PW - INSET - 94, labels)


def _status(run: Run, t, foresight, near_miss=None):
    """-> (headline, colour) for the strip under a robot's camera, and an optional banner on the camera."""
    st = run.step(t)
    mode, lab = st.get("mode"), st.get("target_label")
    if run.crash is not None and t >= run.crash:
        return "Crashed", RED, ("DROVE INTO THE MIRROR", (205, 40, 50, 230))
    if near_miss is not None and near_miss[0] - 1.5 <= t <= near_miss[0] + 3.0:
        return "Safety layer caught the mirror frame", RED, (f"{near_miss[1]:.1f} m FROM THE MIRROR", (205, 40, 50, 230))
    if mode == "done":
        return "Done", GREEN, None
    if mode == "gave_up":
        return "Gave up: nowhere left to explore", MUTED, None
    if foresight and st.get("verifying"):
        return "Holding for a second look", AMBER, ("MIRROR? DOUBLE-CHECKING", (196, 120, 0, 230))
    if foresight and st.get("hazard"):
        return "Mirror walled off on its map", GREEN, ("MIRROR CONFIRMED: WALLED OFF", (16, 128, 84, 230))
    if mode == "announce":
        last = run.found[[f[0] for f in run.found].index(max(f[0] for f in run.found if f[0] <= t + 1e-6))] if run.found else None
        name = _name(last[1]) if last else _name(lab)
        return f"Found the {name}", GREEN, (f"FOUND THE {name.upper()}", (16, 128, 84, 230))
    if mode == "verify":
        return f"Close-up check: {_name(lab)}?", AMBER, (f"IS THIS THE {_name(lab).upper()}?", (196, 120, 0, 230))
    if mode == "approach":
        return f"Heading to the {_name(lab)}", BLUE, None
    return f"Exploring for the {_name(lab)}", TEXT, None


def _strip(img, run, t, x, foresight, instructions, near_miss=None):
    _box(img, (x, STRIP_Y, x + PW, STRIP_Y + STRIP_H), PANEL + (255,))
    d = ImageDraw.Draw(img)
    head, color, _ = _status(run, t, foresight, near_miss)
    d.text((x + 24, STRIP_Y + 24), head, font=_font("Bold", 40), fill=color)
    # instruction chips: done / current / pending
    st = run.step(t)
    k = instructions.index(st["target_label"]) if st.get("target_label") in instructions else len(instructions)
    found_now = len([f for f in run.found if f[0] <= t])
    cx = x + PW - 24
    for i in reversed(range(len(instructions))):
        lab = _name(instructions[i])
        done = i < found_now
        txt = lab + (": found" if done else "")
        w = d.textlength(txt, font=_font("Bold", 28)) + 28
        fill = (16, 128, 84) if done else ((60, 72, 92) if i == k else (36, 40, 50))
        d.rounded_rectangle((cx - w, STRIP_Y + 26, cx, STRIP_Y + 70), 22, fill=fill)
        d.text((cx - w + 14, STRIP_Y + 31), txt, font=_font("Bold", 28), fill=TEXT)
        cx -= w + 10


def _hud(img, run, t, x, foresight):
    """Overlay at the camera's bottom-left: speed, and for FORESIGHT Laya's P(mirror) and Qwen's latest note."""
    st = run.step(t)
    w = PW - INSET - 60
    h = 300 if foresight else 96
    y = PANEL_Y + PW - h - 20
    _box(img, (x + 18, y, x + 18 + w, y + h), (8, 10, 14, 215), radius=14)
    d = ImageDraw.Draw(img)
    d.text((x + 38, y + 18), "Speed", font=_font("Regular", 26), fill=MUTED)
    d.text((x + 140, y + 12), f"{max(0.0, st['linear_velocity']):.2f} m/s", font=_font("Bold", 38), fill=TEXT)
    if not foresight:
        return
    p = st.get("p_hazard", 0.0)
    d.text((x + 38, y + 72), "Laya: mirror?", font=_font("Regular", 26), fill=MUTED)
    bx0, bx1 = x + 240, x + w - 150
    d.rounded_rectangle((bx0, y + 78, bx1, y + 98), 10, fill=DIM)
    d.rounded_rectangle((bx0, y + 78, bx0 + max(20, int((bx1 - bx0) * p)), y + 98), 10, fill=RED if p > 0.6 else GREEN)
    d.text((bx1 + 14, y + 68), f"{p:.2f}", font=_font("Bold", 34), fill=TEXT)
    d.text((x + 38, y + 112), f"{st.get('laya_infer_ms', 0):.0f} ms per decision, every tick", font=_font("Regular", 22), fill=MUTED)
    note = st.get("description")
    if note:
        _text_block(d, (x + 38, y + 150), "Qwen: “" + " ".join(note.split()) + "”", _font("Italic", 24), TEXT, w - 40, 4)


# ---------------------------------------------------------------- schedule & narration


class Schedule:
    """Piecewise playback speed: video time <-> sim time, with a still start and a held end."""

    def __init__(self, pieces, pre_s, hold_s):
        self.pieces, self.pre_s, self.hold_s = pieces, pre_s, hold_s
        self.v0 = pre_s + np.cumsum([0.0] + [(b - a) / v for a, b, v in pieces])
        self.length = float(self.v0[-1] + hold_s)

    def video(self, t):
        for (a, b, v), v0 in zip(self.pieces, self.v0):
            if t <= b:
                return v0 + (max(t, a) - a) / v
        return float(self.v0[-1])

    def sim(self, tau):
        if tau < self.pre_s:
            return self.pieces[0][0], None
        for (a, b, v), v0, v1 in zip(self.pieces, self.v0[:-1], self.v0[1:]):
            if tau < v1:
                return a + (tau - v0) * v, v
        return self.pieces[-1][1], None


def _pieces(end, slow, fast, slow_speed=1.0):
    """slow: [(t0, t1)] windows at slow_speed, everything else at `fast`."""
    win = sorted((max(0.0, a), min(end, b)) for a, b in slow if b > 0 and a < end)
    merged = []
    for a, b in win:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    out, cur = [], 0.0
    for a, b in merged:
        if a > cur + 0.05:
            out.append((cur, a, fast))
        out.append((max(a, cur), b, slow_speed))
        cur = b
    if end > cur + 0.05:
        out.append((cur, end, fast))
    return out


def eval_summary(rows):
    """Per-condition totals over the evaluation set (the demo episode excluded). Episodes where any condition's run
    was ended by a scan hole (see objectnav_eval) are voided for every condition."""
    void = {r["episode"] for r in rows if r.get("ended_by_scan_hole")}
    by = {"voided": sorted(void)}
    for r in rows:
        if r["episode"].startswith("mirror-sofa-bed") or r["episode"] in void:
            continue
        c = by.setdefault(r["condition"], {"n": 0, "success": 0, "near": 0, "timeout": 0, "spl": 0.0, "crash": 0,
                                           "mirror_crash": 0, "scrapes": []})
        c["n"] += 1
        leg = r["legs"][0]
        c["success"] += int(leg["success"])
        c["near"] += int(leg["declared"] and not r["crashed"] and leg.get("dist_to_nearest_m", 9) <= 1.5)
        c["timeout"] += int(not leg["declared"] and not r["crashed"])
        c["spl"] += leg["spl"]
        c["crash"] += int(r["crashed"])
        c["mirror_crash"] += int(r.get("crash_at_mirror", False))
        c["scrapes"].append(r["navmesh_collisions"])
    for k, c in by.items():
        if k != "voided":
            c["spl"] /= max(c["n"], 1)
    return by


def _finish(run, n_legs):
    return run.found[n_legs - 1][0] if len(run.found) >= n_legs else None


def narration(base: Run, fore: Run, instructions, summary, n_scenes, mirror_xz, start_xz):
    """Voice-over, worded from what the runs actually did (crash or near miss, memory or re-search, who finished)."""
    a, b = _name(instructions[0]), _name(instructions[1]) if len(instructions) > 1 else None
    in_mirror = base.behind_mirror(mirror_xz, start_xz)
    lines = {
        "hook": f"FORESIGHT, find the {a}." + (f" Then go back to the {b}." if b else ""),
        "lure": f"No map. No coordinates. Just one camera. To find the {a}, the robot must explore: head for open space "
                "it hasn't seen yet. But this hallway is a mirror. Depth reads it as open floor, "
                + ("and Qwen even spots furniture inside the reflection." if in_mirror else
                   "so it looks like the best place to explore."),
        "race_intro": "Both robots explore the same way, and use Qwen to spot objects. Only FORESIGHT also has Laya, "
                      "reading Qwen's scene notes for reflections.",
    }
    if base.crash is not None:
        lines["race_base_mirror"] = "The baseline heads for the reflected room, and drives straight into the mirror."
    else:
        nm = base.mirror_encounter(mirror_xz)
        if nm is not None:
            lines["race_base_mirror"] = f"The baseline heads for the reflection too. It stops just {nm[1]:.1f} metres short."
    if fore.mirror_time("confirmed") is not None:
        lines["race_flag"] = "Laya flags a reflection. FORESIGHT stops, and takes a second look."
        lines["race_confirm"] = "Confirmed. A wall goes on its map, and anything seen inside the mirror is ignored."
    lines["race_explore"] = (f"So FORESIGHT explores the real house. A hint from Qwen, about which way the {a} might be, "
                             "nudges where it looks next.")
    f1, f2 = _finish(fore, 1), _finish(fore, 2)
    if f1 is not None:
        lines["race_found1"] = f"{a.capitalize()} seen three times, then checked up close. Found, at {f1:.0f} seconds."
    if b and f2 is not None:
        t_ins = fore.event_time("instruction")
        t_tgt = fore.event_time("target_on_map", instructions[1])
        if t_ins is not None and t_tgt is not None and t_tgt - t_ins < 1.0:
            lines["race_memory"] = f"Now, back to the {b}. It already has one on its map, so it plans straight there."
        else:
            lines["race_memory"] = f"Now, the {b}. It searches again."
        crash = "no crashes" if fore.crash is None else "one crash"
        lines["race_found2"] = f"{b.capitalize()} found. Both objects in {f2:.0f} seconds, with {crash}."
    n = len(instructions)
    if base.crash is None:
        bf = _finish(base, n)
        lines["race_base_done"] = (f"The baseline gets there too, at {bf:.0f} seconds." if bf is not None
                                   else "The baseline runs out of time.")
    lines["how"] = ("Two loops share one camera. The fast loop turns depth into a map, picks where to explore, and keeps "
                    "the robot safe. In the background, Qwen answers: is the object here, where is it, and which way to "
                    "look. Laya reads Qwen's notes every tick, and a reflection flagged twice becomes a wall.")
    bs, fs = summary.get("baseline"), summary.get("foresight")
    if bs and fs:
        times = lambda k: "once" if k == 1 else ("never" if k == 0 else f"{_spoken_int(k).lower()} times")
        lines["score"] = (f"Then we ran {fs['n']} fixed episodes in {_spoken_int(n_scenes).lower()} homes, and the "
                          f"honest answer is: neither robot is good yet. Within one metre, the baseline found the object "
                          f"{times(bs['success'])}, FORESIGHT {times(fs['success'])}; most calls stopped just outside that "
                          f"line, and FORESIGHT's slower looks ran out the clock more often. But the baseline crashed "
                          f"{times(bs['crash'])}, {'every time' if bs['mirror_crash'] == bs['crash'] else str(bs['mirror_crash']) + ' times'} "
                          f"into a mirror. FORESIGHT crashed {times(fs['crash'])}.")
    lines["credits"] = "FORESIGHT. Zero-shot, in simulation, on a laptop GPU. Honest numbers, not a solved problem."
    return lines


# ---------------------------------------------------------------- segments


def hook_frames(rgb, instructions, n):
    big = Image.fromarray(rgb).resize((S, S), Image.LANCZOS).convert("RGBA")
    shade = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    for y in range(S // 2, S):
        sd.line([(0, y), (S, y)], fill=(8, 10, 14, int(235 * (y - S // 2) / (S // 2))))
    for k in range(n):
        t = k / FPS
        img = big.copy()
        img.alpha_composite(shade)
        d = ImageDraw.Draw(img)
        d.text((80, 70), "FORESIGHT", font=_font("Black", 84), fill=TEXT, stroke_width=4, stroke_fill=(10, 12, 16))
        msgs = [f"Find the {_name(instructions[0])}.", f"Then go back to the {_name(instructions[1])}."]
        for i, m in enumerate(msgs):
            if t >= 0.4 + i * 1.6:
                y = 1360 + i * 190
                w = d.textlength(m, font=_font("Bold", 84)) + 90
                _box(img, (80, y, 80 + w, y + 150), (86, 160, 255, 235), radius=40)
                d = ImageDraw.Draw(img)
                d.text((125, y + 26), m, font=_font("Bold", 84), fill=(255, 255, 255))
        d.text((80, 1800), "No map. No coordinates. One RGB camera.", font=_font("Medium", 44), fill=MUTED)
        yield img.convert("RGB")


def lure_frames(ex, base: Run, n, mirror_cols=(0.36, 0.72)):
    """Start frame / depth model / sim GT, and the boxes Qwen put inside the mirror on the baseline's first looks."""
    rgb, pred, gt = ex["rgb"], ex["pred"], ex["gt"]
    h = gt.shape[0]
    cone = (slice(int(.4 * h), int(.9 * h)), slice(int(.25 * h), int(.75 * h)))
    p_ahead, g_ahead = float(np.median(pred[cone])), float(np.median(gt[cone]))
    # boxes from the opening looks (robot still at the start) that sit in the mirror's part of the frame
    in_mirror = []
    for _, lk in base.looks:
        if lk["seen"] > 0.05:  # only the look taken on this very frame (the robot starts turning at once)
            break
        for lab, boxes in lk["boxes"].items():
            in_mirror += [(lab, b) for b in boxes if mirror_cols[0] <= (b[0] + b[2]) / 2000 <= mirror_cols[1]]
    side = 580
    cam = Image.fromarray(rgb).resize((side, side), Image.LANCZOS).convert("RGBA")
    if in_mirror:
        dd = ImageDraw.Draw(cam)
        for lab, b in in_mirror:
            x1, y1, x2, y2 = [v / 1000 * side for v in b]
            col = LABEL_COLORS.get(lab, TEXT)
            dd.rectangle((x1, y1, x2, y2), outline=col, width=5)
            dd.text((x1 + 6, y1 + 4), _name(lab) + "?", font=_font("Bold", 28), fill=col, stroke_width=3, stroke_fill=(0, 0, 0))
    dm = _depth_img(pred, 8.0, side).convert("RGBA")
    gm = _depth_img(gt, 8.0, side).convert("RGBA")
    for k in range(n):
        img = Image.new("RGBA", (S, S), BG)
        d = ImageDraw.Draw(img)
        d.text((60, 50), "Why a mirror lures an explorer", font=_font("Bold", 80), fill=TEXT)
        d.text((60, 150), "The robot's first frame. Exploring means heading for open space it hasn't seen yet.",
               font=_font("Regular", 36), fill=MUTED)
        cam_title = "Camera + Qwen's boxes in the mirror" if in_mirror else "What the camera sees"
        for i, (im, title) in enumerate([(cam, cam_title), (dm, "What the depth model thinks"),
                                         (gm, "What's really there")]):
            x = 60 + i * (side + 40)
            d.text((x, 240), title, font=_font("Bold", 36), fill=TEXT)
            img.alpha_composite(im, (x, 300))
        d = ImageDraw.Draw(img)
        d.text((60, 940), "Straight ahead:", font=_font("Bold", 48), fill=TEXT)
        for j, (lab, v, col) in enumerate([("Depth model", p_ahead, AMBER), ("Really (the mirror)", g_ahead, TEXT)]):
            y = 1030 + j * 110
            d.text((60, y + 8), lab, font=_font("Medium", 40), fill=MUTED)
            w = int(1100 * min(v, 9) / 9)
            d.rounded_rectangle((520, y, 520 + w, y + 70), 14, fill=col)
            d.text((540 + w, y + 6), f"{v:.1f} m", font=_font("Bold", 52), fill=col)
        _text_block(d, (60, 1330), "The reflection looks like a whole new room: the most promising open space in view. "
                    "Qwen even boxes furniture inside it, and lifting those boxes with depth puts them behind the glass.",
                    _font("Regular", 46), TEXT, S - 120, 5)
        yield img.convert("RGB")


def race_frames(scene, views, base, fore, sch, instructions, start_xz, mirror_xz, place):
    labels = list(dict.fromkeys(instructions))
    near = None if base.crash is not None else base.mirror_encounter(mirror_xz)
    for k in range(int(sch.length * FPS)):
        t, speed = sch.sim(k / FPS)
        img = Image.new("RGBA", (S, S), BG)
        d = ImageDraw.Draw(img)
        d.text((LX, 18), "FORESIGHT", font=_font("Black", 66), fill=TEXT)
        d.text((LX + d.textlength("FORESIGHT", font=_font("Black", 66)) + 22, 38), "Object search",
               font=_font("Medium", 46), fill=MUTED)
        d.text((LX, 100), place, font=_font("Regular", 28), fill=TEXT)
        clock = f"t = {t:.1f} s"
        cw = d.textlength(clock, font=_font("Bold", 54))
        d.text((S - G - cw, 30), clock, font=_font("Bold", 54), fill=TEXT)
        if speed is not None and abs(speed - 1.0) > 0.02:
            badge = f"{speed:.0f}× speed" if speed >= 2 else f"{speed:.1f}× speed"
            bw = d.textlength(badge, font=_font("Bold", 32)) + 36
            x1 = S - G - cw - 24
            d.rounded_rectangle((x1 - bw, 38, x1, 90), 26, fill=AMBER if speed > 1 else DIM)
            d.text((x1 - bw + 18, 45), badge, font=_font("Bold", 32), fill=(16, 18, 22) if speed > 1 else TEXT)
        for x, title, sub in [(LX, "BASELINE", "explorer + Qwen spots objects"),
                              (RX, "FORESIGHT", "same, + Laya reads Qwen's notes for reflections")]:
            d.text((x, 146), title, font=_font("Black", 34), fill=TEXT)
            d.text((x + d.textlength(title, font=_font("Black", 34)) + 16, 152), sub, font=_font("Regular", 26), fill=MUTED)
        for run, x, color, fs in [(base, LX, RED, False), (fore, RX, GREEN, True)]:
            tt = min(t, run.end)
            nm = None if fs else near
            _first_person(img, views, run, tt, x, labels)
            _, _, banner = _status(run, tt, fs, nm)
            if banner:
                _banner(img, x, PANEL_Y, PW, banner[0], banner[1])
            _hud(img, run, tt, x, fs)
            _strip(img, run, tt, x, fs, instructions, nm)
            _map(img, scene, run, tt, (x, BOT_Y), color, start_xz, mirror_xz)
        yield img.convert("RGB")


def how_frames(n, look_ms, laya_ms, depth_ms):
    bw, bh = 640, 150
    boxes = {
        "cam": ((960, 330), "Camera (RGB only)", "", TEXT, 0.0),
        "depth": ((480, 590), "Depth Anything V2", f"monocular depth · {depth_ms:.0f} ms", AMBER, 0.8),
        "map": ((480, 840), "Own map + frontiers", "where it has been, what's unexplored", AMBER, 1.6),
        "safety": ((480, 1090), "Route + safety layer", "every control tick", AMBER, 2.4),
        "vlm": ((1440, 590), "Qwen3-VL-2B", f"is it here? where? which way? · {look_ms / 1000:.1f} s", BLUE, 3.4),
        "laya": ((1440, 840), "Laya", f"reads Qwen's note · {laya_ms:.0f} ms", GREEN, 4.8),
        "memory": ((1440, 1090), "Object memory", "3 sightings + a close-up check", RED, 6.0),
        "motor": ((960, 1350), "Motor command", "", TEXT, 2.4),
    }

    def edge(key, side):
        (cx, cy), *_ = boxes[key]
        return {"top": (cx, cy - bh / 2 - 4), "bottom": (cx, cy + bh / 2), "left": (cx - bw / 2 - 4, cy),
                "right": (cx + bw / 2, cy)}[side]

    grey = (110, 118, 130)
    for k in range(n):
        t = k / FPS
        img = Image.new("RGBA", (S, S), BG)
        d = ImageDraw.Draw(img)
        d.text((60, 50), "How FORESIGHT searches", font=_font("Bold", 84), fill=TEXT)
        d.text((60, 150), "Two loops share one camera. Timings measured on a laptop RTX 4060.", font=_font("Regular", 36), fill=MUTED)
        if t >= 0.8:
            d.text((90, 440), "FAST LOOP", font=_font("Black", 32), fill=AMBER)
            d.line([(edge("cam", "left")[0], 330), (480, 330)], fill=grey, width=6)
            _arrow(d, (480, 330), edge("depth", "top"), grey)
        if t >= 1.6:
            _arrow(d, edge("depth", "bottom"), edge("map", "top"), grey)
        if t >= 2.4:
            _arrow(d, edge("map", "bottom"), edge("safety", "top"), grey)
            _arrow(d, edge("safety", "bottom"), (edge("motor", "top")[0] - bw * 0.3, edge("motor", "top")[1]), grey)
        if t >= 3.4:
            d.text((1660, 440), "SLOW LOOP", font=_font("Black", 32), fill=BLUE)
            d.line([(edge("cam", "right")[0], 330), (1440, 330)], fill=grey, width=6)
            _arrow(d, (1440, 330), edge("vlm", "top"), grey)
        if t >= 4.8:
            _arrow(d, edge("vlm", "bottom"), edge("laya", "top"), grey)
        if t >= 6.0:
            _arrow(d, edge("laya", "bottom"), edge("memory", "top"), grey)
            _arrow(d, edge("memory", "left"), (edge("map", "right")[0] + 4, edge("map", "right")[1] + 30), RED)
            d.text((830, 1000), "objects + mirror walls", font=_font("Bold", 28), fill=RED)
        for key, ((cx, cy), title, sub, color, ta) in boxes.items():
            if t < ta:
                continue
            _box(img, (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), PANEL + (255,))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), 16, outline=color, width=4)
            _centered(d, cx - bw / 2, bw, cy - (50 if sub else 24), title, _font("Bold", 44), TEXT)
            if sub:
                _centered(d, cx - bw / 2, bw, cy + 10, sub, _font("Medium", 30), color)
        if t >= 7.0:
            _text_block(d, (60, 1540), "Qwen is slow and over-eager on its own: a yes/no check gates every box, an "
                        "object needs three sightings and a close-up look, and a reflection needs two flags before it "
                        "becomes a wall.", _font("Regular", 44), TEXT, S - 120, 5)
        yield img.convert("RGB")


def score_frames(summary, n_eps_scenes, n):
    b, f = summary["baseline"], summary["foresight"]
    rows = [("Found it (stopped within 1 m)", f"{b['success']}/{b['n']}", f"{f['success']}/{f['n']}"),
            ("Stopped near it (within 1.5 m)", f"{b['near']}/{b['n']}", f"{f['near']}/{f['n']}"),
            ("Ran out of time (180 s)", f"{b['timeout']}/{b['n']}", f"{f['timeout']}/{f['n']}"),
            ("Runs ending in a crash", f"{b['crash']}/{b['n']}", f"{f['crash']}/{f['n']}"),
            ("…of those, into a mirror", f"{b['mirror_crash']}", f"{f['mirror_crash']}")]
    img = Image.new("RGB", (S, S), BG)
    d = ImageDraw.Draw(img)
    d.text((100, 120), "FORESIGHT", font=_font("Black", 60), fill=MUTED)
    d.text((100, 200), "Fixed evaluation set", font=_font("Bold", 92), fill=TEXT)
    void = summary.get("voided", [])
    d.text((100, 320), f"{b['n']} episodes in {n_eps_scenes} homes, picked by a seeded script before any run. "
           "Success = stop within 1 m of the named object.", font=_font("Regular", 34), fill=MUTED)
    if void:
        d.text((100, 370), f"{len(void)} more voided for both: the camera met a hole in the 3D scan "
               f"({', '.join(void)}).", font=_font("Regular", 30), fill=MUTED)
    d.text((1000, 470), "Baseline", font=_font("Bold", 52), fill=RED)
    d.text((1430, 470), "FORESIGHT", font=_font("Bold", 52), fill=GREEN)
    for i, (lab, bv, fv) in enumerate(rows):
        y = 590 + i * 150
        d.rounded_rectangle((90, y - 16, S - 90, y + 116), 18, fill=PANEL)
        d.text((130, y + 22), lab, font=_font("Medium", 48), fill=TEXT)
        d.text((1000, y + 12), bv, font=_font("Black", 68), fill=TEXT)
        d.text((1430, y + 12), fv, font=_font("Black", 68), fill=TEXT)
    _text_block(d, (100, 1360), "Same explorer, same object spotting, same safety layer; FORESIGHT adds Laya reading "
                f"Qwen's scene notes for reflections, which makes each look slower ({summary['look_s']['foresight']:.1f} s "
                f"vs {summary['look_s']['baseline']:.1f} s). Crash = driving forward within 0.3 m of a surface "
                f"(simulator ground truth). Wall scrapes while turning, median per run: baseline "
                f"{np.median(b['scrapes']):.0f}, FORESIGHT {np.median(f['scrapes']):.0f} ticks. Small sample, one "
                "simulator: read the counts, not the percentages.", _font("Regular", 36), MUTED, S - 200, 7)
    for _ in range(n):
        yield img


def credits_frames(n):
    img = Image.new("RGB", (S, S), BG)
    d = ImageDraw.Draw(img)
    d.text((120, 420), "FORESIGHT", font=_font("Black", 130), fill=TEXT)
    d.text((120, 580), "Map-free monocular navigation with an on-board VLM", font=_font("Medium", 50), fill=MUTED)
    d.text((120, 760), "Built with", font=_font("Bold", 40), fill=TEXT)
    for i, line in enumerate(["Habitat-Sim  ·  HM3D (Matterport scans of real homes)",
                              "Depth Anything V2  ·  Qwen3-VL-2B  ·  Laya (Convai Innovations)",
                              "Voice-over: Kokoro-82M"]):
        d.text((120, 830 + i * 64), line, font=_font("Regular", 40), fill=TEXT)
    _text_block(d, (120, 1150), "Zero-shot: no training, no map, no goal coordinates. Simulation only; one laptop GPU "
                "runs every model. Objects are named by the user; success is scored against the scene's human "
                "annotations, which the robot never sees.", _font("Regular", 36), MUTED, S - 240, 5)
    for _ in range(n):
        yield img


# ---------------------------------------------------------------- assembly


def race_plan(base: Run, fore: Run, dur, mirror_xz):
    """Slow (1x) windows around the events the voice-over talks about; the rest fast, sized so the lines fit."""
    nm = base.crash if base.crash is not None else (base.mirror_encounter(mirror_xz) or (None,))[0]
    n = max(len(fore.found), len(base.found), 1)
    ev = {
        "race_flag": fore.mirror_time("flag"), "race_confirm": fore.mirror_time("confirmed"), "race_base_mirror": nm,
        "race_explore": (fore.mirror_time("confirmed") or 10.0) + 4.0,
        "race_found1": _finish(fore, 1), "race_memory": fore.event_time("instruction"), "race_found2": _finish(fore, 2),
        "race_base_done": _finish(base, n) if base.crash is None else None,
    }
    if ev["race_base_done"] is None and base.crash is None:
        ev["race_base_done"] = base.end
    ev = {k: v for k, v in ev.items() if v is not None and k in dur}
    order = sorted(ev, key=lambda k: ev[k])
    end = max(fore.end, base.end)
    slow = [(0.0, 3.0)] + [(ev[k] - 2.0, ev[k] + 5.0) for k in ("race_flag", "race_confirm", "race_base_mirror") if k in ev] \
        + [(ev[k] - 3.0, ev[k] + 2.5) for k in ("race_found1", "race_found2", "race_base_done") if k in ev]
    for slow_speed in (1.0, 0.75, 0.5):  # key moments in slow motion if their lines need the room
        for fast in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0):
            pieces = _pieces(end, slow, fast, slow_speed)
            sch = Schedule(pieces, dur["race_intro"] + 0.9, 0.0)
            cues, last = _place(order, [sch.video(ev[k]) + 0.2 for k in order], dur, gap=0.4)
            late = max(c[0] - sch.video(ev[k]) for c, k in zip(cues, order) if k != "race_explore")  # generic line
            if late <= 4.0 and sch.length < 150:
                hold = max(2.5, last - sch.length + 1.0)
                return Schedule(pieces, sch.pre_s, hold), [(0.4, "race_intro")] + cues
    raise RuntimeError("narration doesn't fit the race")


def render_video(root: Path, episode: dict, eval_rows: list, n_eval_scenes: int, out_path: Path, tts_python: str,
                 kokoro_home: str, voice: str = "af_heart"):
    ep_dir = root / episode["name"]
    base = Run(ep_dir / "baseline.pkl", ep_dir / "baseline_meta.json")
    fore = Run(ep_dir / "foresight.pkl", ep_dir / "foresight_meta.json")
    scene = Scene(ep_dir)
    ex = dict(np.load(ep_dir / "explainer.npz"))
    summary = eval_summary(eval_rows) if eval_rows else {}
    if summary:
        lat = {"baseline": [], "foresight": []}
        for d in root.iterdir():
            for c in lat:
                m = d / f"{c}_meta.json"
                if m.exists() and d.name != episode["name"]:
                    lat[c] += json.loads(m.read_text())["look_latency_s"]
        summary["look_s"] = {c: float(np.median(v)) if v else float("nan") for c, v in lat.items()}
    instructions = episode["instructions"]
    start_xz = np.asarray(episode["start"], float)[[0, 2]]
    mirror_xz = episode.get("mirror_xz")
    vo = Voice(narration(base, fore, instructions, summary, n_eval_scenes, mirror_xz, start_xz), ep_dir / "narration",
               tts_python, kokoro_home, voice)
    dur = vo.dur
    views = _Views(episode, size=768)
    sch, race_cues = race_plan(base, fore, dur, mirror_xz)
    look_ms = float(np.median(fore.meta["look_latency_s"])) * 1000
    depth_ms = float(np.median([s["depth_ms"] for s in fore.ep.steps]))

    def fit(line, minimum):
        return max(minimum, dur[line] + 1.6)

    segs = [
        ("hook", lambda n: hook_frames(ex["rgb"], instructions, n), fit("hook", 4.5), [(0.5, "hook")]),
        ("lure", lambda n: lure_frames(ex, base, n), fit("lure", 10.0), [(0.5, "lure")]),
        ("race", lambda n: race_frames(scene, views, base, fore, sch, instructions, start_xz, mirror_xz,
                                       episode["place"]), sch.length, race_cues),
        ("how", lambda n: how_frames(n, look_ms, fore.meta["laya_infer_ms_median"], depth_ms), fit("how", 10.0),
         [(0.6, "how")]),
    ]
    if "score" in dur:
        segs.append(("score", lambda n: score_frames(summary, n_eval_scenes, n), fit("score", 8.0), [(0.6, "score")]))
    segs.append(("credits", lambda n: credits_frames(n), fit("credits", 5.0), [(0.5, "credits")]))
    silent = out_path.with_name(out_path.stem + ".silent.mp4")
    w = _Writer(silent)
    for name, make, seconds, cues in segs:
        n = int(round(seconds * FPS))
        t0 = w.n / FPS
        vo.cues += [(t0 + t, line) for t, line in cues]
        w.segment(make(n), n)
        print(f"  {name}: {n / FPS:.1f} s (total {w.n / FPS:.1f} s)", flush=True)
    w.close()
    views.close()
    total = w.n / FPS
    wav = out_path.with_name(out_path.stem + ".voice.wav")
    vo.write_mix(wav, total)
    subprocess.run(["ffmpeg", "-y", "-i", str(silent), "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", "-shortest", str(out_path)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    silent.unlink()
    wav.unlink()
    print(f"wrote {out_path} ({total:.1f} s, {S}x{S}, voice-over {len(vo.cues)} lines)")
