"""FORESIGHT promo, "One Camera, No Map": the narrated 1920x1920 LinkedIn video (plan: claude.ai artifact
LZFmCoQ5rUzfAYmhpG3ahY, draft 2, approved 2026-11-21).

Every robot motion is a recorded live run (tools/isaac/g1_live.py --walker sonic: FORESIGHT's decisions, walked by
NVIDIA GEAR-SONIC), re-rendered joint for joint and path traced (tools/isaac/replay_render.py; shots from
tools/isaac/promo/make_shots.py). AR layers are projected through each frame's real camera and hidden behind
geometry using the render's own depth; their content comes from the brain's logs (brain.pkl): the depth it was
given, its map, its plan, its VLM looks. Study numbers (success rates, depth error) come from docs/results/
depth_error_A.md. Head-camera inserts are the frames the robot actually saw (Habitat's render of the same scan).
Rules (the user's): success-led, live, AR labels with one or two numbers, no panels, no file paths on screen.
"""
from __future__ import annotations

import io
import json
import math
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from foresight.viz.ar_overlay import GOOD, HOLO, INK, WARN, Camera, Overlay, font
from foresight.viz.glass_demo_video import (BG, FPS, MIRROR_X, PREVIEW, S, Shot, brand, chip, ease, full, ks,
                                            vignette)
from foresight.viz.showcase_video import logo_layer

FLOOR_Y = 0.0614  # Habitat floor height of the Isaac floor (z = 0) in 00808
CAM_FX_640 = 320.0  # head camera: 640 px, 90 deg horizontal FOV
RED = (255, 110, 96)
TINT = {"depth_gt": GOOD, "depth_da": HOLO, "depth_noise": WARN, "depth_nofs": RED}
NAMES = {"depth_gt": "TRUE DEPTH", "depth_da": "DEPTH ANYTHING", "depth_noise": "10% NOISE", "depth_nofs": "NO CALIBRATION"}
# docs/results/depth_error_A.md (map navigator, set A): success %, depth AbsRel
STUDY = {"depth_gt": (83, 0.00), "depth_da": (79, 1.39), "depth_noise": (5, 0.08), "depth_nofs": (35, 1.42)}


def hab_to_isaac(p):
    """Habitat (x, y, z) -> Isaac (x, -z, y - floor)."""
    p = np.asarray(p, float)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1] - FLOOR_Y], -1)


class Run:
    """A live run: motion.json (root path) + brain.pkl (the brain's per-tick log)."""

    def __init__(self, d: Path):
        m = json.loads((d / "motion.json").read_text())
        self.result = str(m.get("result", ""))
        self.t = np.array([s["t"] for s in m["steps"]])
        self.pos = np.array([s["pos"] for s in m["steps"]])
        self.end = float(self.t[-1])
        self.B = pickle.load(open(d / "brain.pkl", "rb")) if (d / "brain.pkl").exists() else {"ticks": []}
        self.T = self.B["ticks"]
        self.tt = np.array([x["t"] for x in self.T]) if self.T else np.zeros(0)
        self._map = {"k": 0, "pts": np.zeros((0, 3))}
        hit = [w for w in self.result.split() if w.replace(".", "", 1).isdigit()]
        self.event_t = float(hit[0]) if hit else self.end
        self.crashed = self.result.startswith("crash")
        a = m.get("args") or {}
        self.goal = None if a.get("task") == "search" else a.get("goal")
        self.reached = self.result.startswith("reached")

    def at(self, t):
        return self.pos[int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))]

    def path(self, t0, t1, step=0.25):
        ts = np.arange(t0, t1, step)
        return np.array([[*self.at(t)[:2], 0.02] for t in ts]) if len(ts) else np.zeros((0, 3))

    def k(self, t):
        return int(np.clip(np.searchsorted(self.tt, t, side="right") - 1, 0, max(len(self.T) - 1, 0)))

    def tick(self, t):
        return self.T[self.k(t)] if self.T else None

    def events(self, name):
        return [x["t"] for x in self.T if (x.get("diag") or {}).get("event") == name]

    def median(self, key):
        v = [x[key] for x in self.T if x.get(key) is not None]
        return float(np.median(v)) if v else None

    # ------------------------------------------------------------ what the robot believed, in Isaac world coords
    def cloud(self, x, stride=2):
        """The depth the navigator was given at tick x, back-projected into the world (floor-calibrated metres)."""
        d = np.asarray(x["depth_small"], np.float32)[::stride, ::stride]
        s = float((x.get("diag") or {}).get("depth_scale") or 1.0)
        d = d * s
        h, w = d.shape
        f = CAM_FX_640 / 640 * w
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        ok = (d > 0.2) & (d < 8.0)
        cx, cy = (u[ok] - w / 2) / f * d[ok], -(v[ok] - h / 2) / f * d[ok]
        cam = hab_to_isaac(x["position"])
        fx, fz = x["forward_xz"]
        fw = np.array([fx, -fz, 0.0])
        fw /= np.linalg.norm(fw) + 1e-9
        rt = np.array([fw[1], -fw[0], 0.0])
        P = cam + d[ok][:, None] * fw + cx[:, None] * rt + cy[:, None] * np.array([0, 0, 1.0])
        return P

    def map_points(self, t):
        """Obstacle points (0.2-1.8 m high) from every tick up to t, flattened onto the floor: the map as it grows."""
        k = self.k(t)
        if k < self._map["k"]:
            self._map = {"k": 0, "pts": np.zeros((0, 3))}
        new = []
        for x in self.T[self._map["k"]:k + 1]:
            P = self.cloud(x, stride=4)
            P = P[(P[:, 2] > 0.2) & (P[:, 2] < 1.8)]
            new.append(np.c_[np.round(P[:, :2] / 0.08) * 0.08, np.full(len(P), 0.02)])
        if new:
            pts = np.unique(np.concatenate([self._map["pts"]] + new), axis=0)
            self._map = {"k": k + 1, "pts": pts}
        return self._map["pts"]

    def plan(self, t):
        x = self.tick(t)
        pp = (x.get("diag") or {}).get("plan_path") if x else None
        if not pp:
            return None
        P = np.asarray(pp, float)
        return np.c_[P[:, 0], -P[:, 1], np.full(len(P), 0.03)]

    def look(self, t, labels=None):
        """The latest VLM look up to t with a box: (label, world point at the box centre, look time)."""
        best = None
        for x in self.T[: self.k(t) + 1]:
            for key, r in (x.get("looks") or {}).items():
                for lab, boxes in (r.get("boxes") or {}).items():
                    if boxes and (labels is None or lab in labels):
                        best = (float(key), lab, boxes[0])
        if best is None:
            return None
        tk, lab, b = best
        x = self.tick(tk)
        d = np.asarray(x["depth_small"], np.float32) * float((x.get("diag") or {}).get("depth_scale") or 1.0)
        h, w = d.shape
        u, v = (b[0] + b[2]) / 2000 * w, (b[1] + b[3]) / 2000 * h
        r = d[int(v * 0.9):int(v * 1.1) + 1, int(u * 0.9):int(u * 1.1) + 1]
        z = float(np.median(r[r > 0])) if (r > 0).any() else None
        if z is None:
            return None
        f = CAM_FX_640 / 640 * w
        cam = hab_to_isaac(x["position"])
        fx, fz = x["forward_xz"]
        fw = np.array([fx, -fz, 0.0])
        fw /= np.linalg.norm(fw) + 1e-9
        rt = np.array([fw[1], -fw[0], 0.0])
        P = cam + z * fw + (u - w / 2) / f * z * rt - (v - h / 2) / f * z * np.array([0, 0, 1.0])
        return lab, P, tk

    def found(self):
        last = next((x["diag"].get("found") for x in reversed(self.T) if (x.get("diag") or {}).get("found")), None) or []
        return [(float(f[0]), f[1], hab_to_isaac([f[3][0], FLOOR_Y, f[3][1]])) for f in last]

    def pov(self, t):
        """The head-camera frame the robot saw at (or just before) t (kept every other tick)."""
        for i in range(self.k(t), -1, -1):
            if self.T[i].get("rgb_jpg"):
                return Image.open(io.BytesIO(self.T[i]["rgb_jpg"])).convert("RGB"), self.T[i]
        return None, None


def turbo(d, lo=0.3, hi=6.0):
    """Depth map -> RGB (near = warm, far = cool), invalid black."""
    x = np.clip((np.log(np.maximum(d, 1e-3)) - math.log(lo)) / (math.log(hi) - math.log(lo)), 0, 1)
    r = np.clip(1.6 - 2.2 * x, 0, 1)
    g = np.clip(1.2 - np.abs(2.4 * x - 1.2), 0, 1)
    b = np.clip(2.2 * x - 0.5, 0, 1)
    out = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    out[d <= 0] = 0
    return out


def dots(ov: Overlay, P, depth_map, color, alpha=1.0, size=2.4, max_n=9000):
    """World points as glowing dots, hidden behind the render's own geometry (its depth, 1/4 resolution)."""
    if alpha <= 0 or len(P) == 0:
        return
    if len(P) > max_n:
        P = P[np.random.default_rng(0).choice(len(P), max_n, replace=False)]
    xy, z = ov.cam.project(P)
    ok = (z > 0.2) & (xy[:, 0] >= 0) & (xy[:, 0] < ov.cam.w) & (xy[:, 1] >= 0) & (xy[:, 1] < ov.cam.h)
    if depth_map is not None:
        dh, dw = depth_map.shape
        ix = np.clip((xy[:, 0] / ov.cam.w * dw).astype(int), 0, dw - 1)
        iy = np.clip((xy[:, 1] / ov.cam.h * dh).astype(int), 0, dh - 1)
        ok &= z <= depth_map[iy, ix] * 1.04 + 0.05
    r = size * ov.ss
    for x, y in ov._s(xy[ok]):
        ov.sd.ellipse((x - r, y - r, x + r, y + r), fill=color + (int(210 * alpha),))
        ov.gd.ellipse((x - 2 * r, y - 2 * r, x + 2 * r, y + 2 * r), fill=color + (int(70 * alpha),))


def shot_depth(sh: Shot, tv):
    k = int(round(sh.k(tv)))
    p = sh.d / f"{k:05d}_d.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


def speed_chip(img, text, alpha=1.0):
    return chip(img, text, alpha, y=150)


def caption(img, text, alpha=1.0, y=S - 230):
    """A one-line instruction card (the spoken goal) with a waveform, bottom centre."""
    if alpha <= 0:
        return img
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = font("display", 54, "SemiBold")
    tw = d.textlength(text, font=f)
    w = tw + 220
    x0 = (S - w) / 2
    d.rounded_rectangle((x0, y, x0 + w, y + 104), radius=52, fill=(8, 22, 36, int(190 * alpha)),
                        outline=HOLO + (int(220 * alpha),), width=3)
    for i in range(9):  # a small waveform before the words
        hh = 10 + 34 * abs(math.sin(i * 1.7 + 0.35 * (PREVIEW.get("k") or 0)))
        xx = x0 + 44 + i * 12
        d.rounded_rectangle((xx, y + 52 - hh / 2, xx + 6, y + 52 + hh / 2), radius=3, fill=HOLO + (int(255 * alpha),))
    d.text((x0 + 170, y + 20), text, font=f, fill=INK + (int(255 * alpha),))
    img.alpha_composite(layer)
    return img


class Film:
    def __init__(self, root: Path, music_title: str = ""):
        self.root, self.music_title = root, music_title
        P = root / "results/isaac/promo"
        self.sh = {d.name: Shot(d) for d in sorted((P / "film").iterdir()) if (d / "frames.json").exists()} \
            if (P / "film").exists() else {}
        live = P / "live"
        self.R = {d.name: Run(d) for d in sorted(live.iterdir()) if (d / "motion.json").exists()} if live.exists() else {}
        ph = root / "results/isaac/phantom"
        self.phantom = json.loads((ph / "phantom_frames.json").read_text()) if (ph / "phantom_frames.json").exists() else None
        self.ph_dir = ph / "frames"

    def has(self, *names):
        return all(n in self.sh for n in names)

    def missing(self, n, title):
        """A plain card for a segment whose shots aren't rendered yet (previews before the full render)."""
        for k in ks(n):
            im = Image.new("RGBA", (S, S), BG + (255,))
            d = ImageDraw.Draw(im)
            f = font("display", 64, "SemiBold")
            d.text(((S - d.textlength(title, font=f)) / 2, S / 2 - 60), title, font=f, fill=INK + (255,))
            g = font("text", 34)
            msg = "not rendered yet"
            d.text(((S - d.textlength(msg, font=g)) / 2, S / 2 + 30), msg, font=g, fill=(150, 168, 190, 255))
            yield im

    def frame(self, sh: Shot, tv):
        return full(vignette(sh.image(tv))), Overlay(sh.camera(tv)), shot_depth(sh, tv)

    # ================================================================ 1. hook: a room that isn't there
    def seg_hook(self, n, cues):
        if not self.has("hook_dolly") or "mirror_map" not in self.R:
            yield from self.missing(n, "A room that isn't there")
            return
        sh, g = self.sh["hook_dolly"], self.R["mirror_map"]
        a_len = min(sh.length, 6.5)
        b_len = 2.5
        for k in ks(n):
            tv = k / FPS
            if tv < a_len:  # the low dolly into the mirror
                img, ov, dm = self.frame(sh, tv)
                t = sh.sim_t(tv)
                x = g.tick(t)
                rx, ry, yaw = sh.frame(tv)["robot"]
                da = (x.get("diag") or {}).get("depth_ahead_m") if x else None
                if da is not None and math.cos(yaw) > 0.6 and rx < MIRROR_X - 0.2:
                    ay = ry + math.tan(yaw) * (MIRROR_X - rx)
                    ov.label([MIRROR_X, ay, 0.9], "OPEN FLOOR", f"{da:.1f} m", sub="what depth from one camera reads",
                             tone="warn", side="left", alpha=ease((tv - 1.2) / 0.6), lead_px=110, rise_px=150)
                yield brand(ov.compose(img).convert("RGBA"), ease(tv / 0.8))
            elif tv < a_len + b_len:  # the head camera: what it sees, then the depth it reads
                u = (tv - a_len) / b_len
                im, x = g.pov(g.event_t - 1.0)
                if im is None:
                    yield Image.new("RGBA", (S, S), BG + (255,))
                    continue
                im = im.resize((S, S), Image.LANCZOS)
                d = np.asarray(x["depth_small"], np.float32) * float((x.get("diag") or {}).get("depth_scale") or 1.0)
                dep = Image.fromarray(turbo(d)).resize((S, S), Image.BILINEAR)
                cut = int(S * ease(u * 1.6))
                im.paste(dep.crop((0, 0, cut, S)), (0, 0))
                out = im.convert("RGBA")
                ImageDraw.Draw(out).line((cut, 0, cut, S), fill=HOLO + (255,), width=4)
                yield brand(chip(out, "ROBOT'S VIEW · DEPTH FROM PIXELS", ease(u / 0.2)))
            else:  # the phantom room: the depth back-projected, a whole room behind the glass
                yield brand(self._phantom(tv - a_len - b_len))

    def _phantom(self, tv):
        fr = (self.phantom or {}).get("frames") or []
        cand = [f for f in fr if f.get("shown", 0) > 0 and f.get("fill", 0) == 0]
        if not cand:
            return Image.new("RGBA", (S, S), BG + (255,))
        i = cand[min(len(cand) - 1, int(tv * self.phantom.get("fps", 24.0)))]["n"]
        im = Image.open(self.ph_dir / f"{i:05d}.jpg").convert("RGB")
        im = im.resize((int(im.width * S / im.height), S), Image.LANCZOS)
        x0 = (im.width - S) // 2
        return chip(im.crop((x0, 0, x0 + S, S)).convert("RGBA"), "THE ROOM DEPTH SEES · IT ISN'T THERE", 1.0)

    # ================================================================ 2. one camera, two speeds
    def seg_system(self, n, cues):
        if not self.has("sys_front", "sys_shoulder") or "search_v1" not in self.R:
            yield from self.missing(n, "One camera, two speeds")
            return
        r = self.R["search_v1"]
        fast = r.median("tick_ms")
        looks = [x for x in r.T if x.get("looks")]
        slow = None
        if len(looks) > 2:
            ts = sorted({float(k) for x in looks for k in x["looks"]})
            slow = float(np.median(np.diff(ts))) if len(ts) > 2 else None
        a = self.sh["sys_front"]
        for k in ks(n):
            tv = k / FPS
            sh, ts_ = (a, tv) if tv < a.length else (self.sh["sys_shoulder"], tv - a.length)
            img, ov, dm = self.frame(sh, ts_)
            t = sh.sim_t(ts_)
            x = r.tick(t)
            p = r.at(t)
            if x is not None:
                dots(ov, r.cloud(x), dm, HOLO, alpha=ease((tv - 1.0) / 0.8))  # the depth it reads, this frame
                if tv > 5.0:
                    dots(ov, r.map_points(t), dm, GOOD, alpha=ease((tv - 5.0) / 1.0) * 0.8, size=1.8, max_n=6000)
                pl = r.plan(t)
                if pl is not None and tv > 9.0:
                    ov.route(pl, tone="info", width=7, alpha=ease((tv - 9.0) / 0.8), pulse=(tv * 0.4) % 1.0)
            if tv > 1.5:
                ov.label([p[0], p[1], 1.45], "FAST", f"{1000 / max(fast or 100, 100):.0f} Hz" if fast else None,
                         sub="depth + map + safety, every frame", tone="info", side="left",
                         alpha=ease((tv - 1.5) / 0.6), lead_px=110, rise_px=90)
            if tv > 13.0:
                lk = r.look(t)
                ov.label([p[0], p[1], 1.6], "SLOW", f"~{slow:.0f} s" if slow else None, sub="Qwen3-VL-2B looks",
                         tone="good", side="right", alpha=ease((tv - 13.0) / 0.6), lead_px=110, rise_px=120)
                if lk is not None:
                    lab, P, _ = lk
                    ov.label(P, lab.upper(), None, sub="seen by the vision model", tone="good",
                             alpha=ease((tv - 14.0) / 0.6), lead_px=80, rise_px=70)
            yield brand(ov.compose(img).convert("RGBA"))

    # ================================================================ 3. same robot, different depth
    def seg_depth(self, n, cues):
        need = ["depth_gt", "depth_da", "depth_noise", "depth_nofs"]
        if not self.has("depth_crane") or any(q not in self.R for q in need):
            yield from self.missing(n, "Same robot, different depth")
            return
        crane = self.sh["depth_crane"]
        grid_len = 7.0
        close = [s for s in ("depth_nofs_crash", "depth_noise_stall") if s in self.sh]
        lens = [crane.length, grid_len] + [self.sh[s].length for s in close]
        starts = np.cumsum([0.0] + lens)
        speed = crane.f[-1]["t"] / max(crane.length, 1e-6) if crane.n else 1.0
        for k in ks(n):
            tv = k / FPS
            i = int(np.searchsorted(starts, tv, side="right") - 1)
            ts_ = tv - starts[min(i, len(starts) - 1)]
            if i == 0:  # crane, time-lapse: four robots, one start
                img, ov, dm = self.frame(crane, ts_)
                f = crane.frame(ts_)
                tg = f.get("t_ghosts") or [f["t"]] * 4
                for j, nm in enumerate(need):
                    rr = self.R[nm]
                    t = tg[j] if j < len(tg) else f["t"]
                    p = rr.at(t)
                    if t >= rr.event_t - 0.05 and (rr.crashed or rr.reached):
                        title = "COLLISION" if rr.crashed else "GOAL"
                        tone = "warn" if rr.crashed else "good"
                    elif t >= rr.end - 0.05:  # out of time: frozen in place (noise) or lost (no calibration)
                        title, tone = ("STALLED", "warn") if nm == "depth_noise" else ("NEVER ARRIVES", "warn")
                    else:
                        title, tone = NAMES[nm], "info"
                    ov.label([p[0], p[1], 1.35], title, None, sub=NAMES[nm].lower() if title != NAMES[nm] else None,
                             tone=tone, side="left" if j % 2 else "right", alpha=ease((ts_ - 0.6) / 0.6),
                             lead_px=60, rise_px=40 + 30 * j, scale=0.85)
                im = brand(ov.compose(img).convert("RGBA"))
                yield speed_chip(im, f"TIME-LAPSE ×{speed:.0f}", ease(ts_ / 0.6))
            elif i == 1:  # what each robot was given: its depth map, side by side, at the same moment
                yield brand(self._depth_grid(need, ts_ / grid_len))
            else:
                sh = self.sh[close[i - 2]]
                img, ov, dm = self.frame(sh, ts_)
                nm = "depth_nofs" if "nofs" in sh.d.name else "depth_noise"
                rr = self.R[nm]
                t = sh.sim_t(ts_)
                p = rr.at(t)
                x = rr.tick(t)
                if nm == "depth_noise" and x is not None:  # the phantom obstacles noise puts on the floor
                    dots(ov, rr.cloud(x), dm, WARN, alpha=ease((ts_ - 0.8) / 0.6) * 0.9)
                    ov.label([p[0], p[1], 1.4], NAMES[nm], f"error {STUDY[nm][1]:.2f}", sub="walls everywhere",
                             tone="warn", alpha=ease((ts_ - 1.0) / 0.6), lead_px=80, rise_px=70)
                else:  # no calibration: a crash, or (as on a-00808-04 with SONIC) lost short of the goal
                    hit = t >= rr.event_t and rr.crashed
                    gd = float(np.hypot(*(np.array(rr.goal) - p[:2]))) if rr.goal is not None else None
                    ov.label([p[0], p[1], 1.4], "COLLISION" if hit else NAMES[nm],
                             f"{rr.event_t:.1f} s" if hit else (f"{gd:.1f} m short" if gd is not None else None),
                             sub="depth 2× off in scale, never calibrated", tone="warn",
                             alpha=ease((ts_ - 0.6) / 0.6), lead_px=80, rise_px=70)
                    if rr.goal is not None and not hit:
                        ov.ring([rr.goal[0], rr.goal[1], 0.02], 0.4, "good", t=tv)
                yield brand(ov.compose(img).convert("RGBA"))

    def _depth_grid(self, names, u):
        """2x2: each robot's head-camera frame with the depth it was given wiping across, name and study numbers."""
        out = Image.new("RGBA", (S, S), BG + (255,))
        half = S // 2
        d = ImageDraw.Draw(out)
        for j, nm in enumerate(names):
            rr = self.R[nm]
            t = min(rr.end - 0.5, 6.0 + 4.0 * u)
            im, x = rr.pov(t)
            if im is None:
                continue
            tile = im.resize((half - 12, half - 12), Image.LANCZOS)
            dep = np.asarray(x["depth_small"], np.float32) * float((x.get("diag") or {}).get("depth_scale") or 1.0)
            dt = Image.fromarray(turbo(dep)).resize(tile.size, Image.BILINEAR)
            cut = int(tile.width * ease((u - 0.1) / 0.5))
            tile.paste(dt.crop((0, 0, cut, tile.height)), (0, 0))
            ox, oy = (j % 2) * half + 6, (j // 2) * half + 6
            out.paste(tile, (ox, oy))
            col = TINT[nm]
            d.rounded_rectangle((ox + 18, oy + 18, ox + 18 + 430, oy + 18 + 112), radius=18, fill=(8, 22, 36, 200),
                                outline=col + (255,), width=3)
            d.text((ox + 38, oy + 30), NAMES[nm], font=font("display", 34, "SemiBold"), fill=INK + (255,))
            sr, ab = STUDY[nm]
            d.text((ox + 38, oy + 76), f"error {ab:.2f} · {sr}% reach the goal", font=font("text", 26, "Medium"),
                   fill=col + (255,))
        return out

    # ================================================================ 4. the mirror trap
    def seg_mirror(self, n, cues):
        if not self.has("mirror_duel", "mirror_check", "mirror_exit") or any(q not in self.R for q in ("mirror_v1", "mirror_map")):
            yield from self.missing(n, "The mirror trap")
            return
        hero, ghost = self.R["mirror_v1"], self.R["mirror_map"]
        A, B, C = self.sh["mirror_duel"], self.sh["mirror_check"], self.sh["mirror_exit"]
        desc = next((x["diag"]["description"] for x in hero.T if (x.get("diag") or {}).get("description")
                     and "reflection" in x["diag"]["description"].lower()), None)
        quote = None
        if desc:
            s0 = desc.split("\n")[0]
            i = s0.lower().find("appears to be")
            quote = ("…" + s0[i:].split(";")[0].split("—")[0].rstrip(", .") + "…") if i >= 0 else s0[:110] + "…"
        confirm_t = (hero.events("confirmed") or [None])[0]
        verify_t = next((x["t"] for x in hero.T if (x.get("diag") or {}).get("verifying")), None)
        goal = np.array([2.4, -1.1])
        for k in ks(n):
            tv = k / FPS
            if tv < A.length:
                img, ov, dm = self.frame(A, tv)
                f = A.frame(tv)
                t = A.sim_t(tv)
                tg = f["t_ghost"] + (t - f["t"])
                gp = ghost.at(tg)
                ov.label([gp[0], gp[1], 1.15], "DEPTH ONLY", None, sub="same start, same goal", tone="info",
                         alpha=ease((tv - 0.5) / 0.6), lead_px=70, rise_px=60)
                if ghost.crashed and tg >= ghost.event_t:
                    ov.label([MIRROR_X, gp[1], 1.0], "COLLISION", f"{ghost.event_t:.1f} s", tone="warn",
                             alpha=ease((tg - ghost.event_t) / 0.3), lead_px=80, rise_px=90)
            elif tv < A.length + B.length:
                ts_ = tv - A.length
                img, ov, dm = self.frame(B, ts_)
                t = B.sim_t(ts_)
                h = hero.at(t)
                x = hero.tick(t)
                walls = [[(a[0], -a[1]), (b[0], -b[1])] for a, b in (x or {}).get("walls", [])]
                for (a, b) in walls:
                    ov.route([[a[0], a[1], 0.03], [b[0], b[1], 0.03]], tone="warn", width=9, alpha=0.95, fade_tail=False)
                if quote:
                    ov.thought([h[0], h[1], 1.4], quote, alpha=ease((ts_ - 0.6) / 0.6) * (1 - ease((ts_ - 8.0) / 0.6)),
                               max_w_px=600)
                if verify_t is not None and t >= verify_t and (confirm_t is None or t < confirm_t):
                    ov.label([MIRROR_X, h[1], 1.2], "MIRROR?", None, sub="checking again…", tone="warn",
                             alpha=ease((t - verify_t) / 0.6), lead_px=90, rise_px=110)
                if confirm_t is not None and t >= confirm_t and walls:
                    (a, b) = walls[0]
                    ov.label([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.05], "MIRROR", "wall added", tone="warn",
                             alpha=ease((t - confirm_t) / 1.0), lead_px=90, rise_px=110)
            else:
                ts_ = tv - A.length - B.length
                img, ov, dm = self.frame(C, ts_)
                t = C.sim_t(ts_)
                h = hero.at(t)
                ov.route(hero.path(t + 0.3, min(t + 10.0, hero.end)), tone="good", width=7, alpha=0.9,
                         pulse=(tv * 0.35) % 1.0)
                ov.ring([goal[0], goal[1], 0.02], 0.4, "good", t=tv)
                done = hero.reached and t >= hero.event_t - 0.3
                ov.label([goal[0], goal[1], 0.45], "GOAL", "reached" if done else f"{np.hypot(*(goal - h[:2])):.1f} m",
                         sub="0 collisions" if done else "behind the mirror", tone="good", lead_px=70, rise_px=60)
            im = brand(ov.compose(img).convert("RGBA"))
            yield chip(im, "REAL HOME · 3D-SCANNED", ease(tv / 0.8) * (1 - ease((tv - 3.0) / 0.6)))

    # ================================================================ 5. "find the sofa" (s2-00808-sofa-1)
    def seg_search(self, n, cues):
        if not self.has("search_lapse") or "search_v1" not in self.R:
            yield from self.missing(n, "Find the sofa")
            return
        r = self.R["search_v1"]
        L = self.sh["search_lapse"]
        found = r.found()
        speed = (L.f[-1]["t"] - L.f[0]["t"]) / max(L.length, 1e-6) if L.n > 1 else 1.0
        instr = (r.B.get("header") or {}).get("instructions") or ["sofa"]
        words = "Find the " + ", then the ".join(instr) + "."
        for k in ks(n):
            tv = k / FPS
            tl = min(tv, L.length - 1e-3)
            img, ov, dm = self.frame(L, tl)
            t = L.sim_t(tl)
            x = r.tick(t)
            lk = r.look(t, labels=instr)
            if lk is not None and not any(t >= tf for tf, _, _ in found):  # sighted, not yet confirmed
                lab, P, tk = lk
                ov.label(P, lab.upper(), "seen", sub="the vision model, this look", tone="info",
                         alpha=ease((t - tk) / 0.6), lead_px=80, rise_px=80)
            for tf, lab, P in found:
                if t >= tf:
                    ov.ring([P[0], P[1], 0.02], 0.45, "good", t=tv)
                    ov.label([P[0], P[1], 0.8], lab.upper(), "found", sub=f"{tf:.0f} s · no map, no coordinates",
                             tone="good", alpha=ease((t - tf) / 0.8), lead_px=80, rise_px=90)
            if x is not None:
                dots(ov, r.map_points(t), dm, GOOD, alpha=0.55, size=1.6, max_n=5000)
            im = brand(ov.compose(img).convert("RGBA"))
            if speed > 1.5:
                im = speed_chip(im, f"TIME-LAPSE ×{speed:.0f}", 1.0)
            # the spoken goal, as on-screen words with a waveform (the user's choice, 2026-11-21)
            yield caption(im, f"“{words}”", alpha=ease(tv / 0.5) * (1 - ease((tv - 5.5) / 0.6)))

    # ================================================================ 6. end card
    def seg_end(self, n, cues):
        pull = self.sh.get("end_pull")
        credits = ["Walking: NVIDIA GEAR-SONIC (NVIDIA Open Model License) · Unitree G1",
                   "NVIDIA Isaac Sim · Isaac Lab · HM3D (Ramakrishnan et al. 2021)",
                   "Depth Anything V2 · Qwen3-VL · Laya · Voice: Kokoro-82M",
                   f"Music: “{self.music_title}” by Kevin MacLeod (incompetech.com), CC BY 4.0"]
        pull_len = pull.length if pull is not None else 0.0
        for k in ks(n):
            tv = k / FPS
            if tv < pull_len:
                img, ov, dm = self.frame(pull, tv)
                yield brand(ov.compose(img).convert("RGBA"))
                continue
            te = tv - pull_len
            im = Image.new("RGBA", (S, S), BG + (255,))
            if pull is not None and te < 1.0:  # dissolve from the last frame
                last = full(vignette(pull.image(pull.length - 1e-3))).convert("RGBA")
                im = Image.blend(last, im, ease(te / 1.0))
            a = ease((te - 0.4) / 0.8)
            lg = logo_layer(300).convert("RGBA")
            lg.putalpha(lg.getchannel("A").point(lambda v: int(v * a)))
            im.alpha_composite(lg, ((S - 300) // 2, 520))
            d = ImageDraw.Draw(im)
            ft = font("display", 112, "SemiBold")
            d.text(((S - d.textlength("FORESIGHT", font=ft)) / 2, 870), "FORESIGHT", font=ft, fill=(240, 246, 255, int(255 * a)))
            fs = font("text", 44, "Medium")
            tag = "One camera. No map. On board."
            d.text(((S - d.textlength(tag, font=fs)) / 2, 1030), tag, font=fs, fill=HOLO + (int(255 * ease((te - 1.0) / 0.8)),))
            fc = font("text", 28, "Regular")
            for i, c in enumerate(credits):
                d.text(((S - d.textlength(c, font=fc)) / 2, 1330 + i * 48), c, font=fc,
                       fill=(150, 168, 190, int(255 * ease((te - 1.6) / 0.8))))
            yield im
