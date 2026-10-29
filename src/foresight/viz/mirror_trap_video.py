"""Square 1920x1920 narrated demo video for tools/run_mirror_trap_demo.py.

Segments: hook -> why a camera is fooled (depth heatmaps) -> the race (both runs side by side, each with its own
photographic top-down map, playback speed badged on screen) -> slowed replay of the first flag -> how the loops
fit together -> decision-latency chart -> credits. A Kokoro-82M voice-over (tools/kokoro_tts.py) is cued to
events in the runs, and segments stretch to fit their lines.

Smooth motion: runs are logged at the control rate (~10 Hz), so every video frame re-renders both cameras in
Habitat at the pose interpolated to that frame's sim time rather than repeating the last logged frame.

Every number drawn or spoken comes from the saved runs (per-tick logs, laya_metrics.json, assets) or the Qwen
direct-decision benchmark passed in. The narration's descriptive words (e.g. which surfaces the false alarms
were about) were written for the 2026-10-29 run. Frames are piped straight into ffmpeg.
"""
from __future__ import annotations

import functools
import json
import os
import pickle
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

S, FPS, FADE = 1920, 30, 8
SR = 24000  # Kokoro's sample rate
BG, PANEL, TEXT, MUTED, DIM = (12, 15, 20), (26, 30, 38), (236, 238, 241), (150, 157, 166), (60, 66, 78)
RED, GREEN, AMBER, BLUE, CYAN = (255, 92, 97), (52, 211, 140), (255, 180, 40), (86, 160, 255), (120, 220, 255)
G = 16
PW = (S - 3 * G) // 2          # first-person panel side
LX, RX = G, 2 * G + PW
PANEL_Y = 214
TL_Y, TL_H = PANEL_Y + PW + 16, 214
BOT_Y = TL_Y + TL_H + 16
BOT_H = S - G - BOT_Y
MAP = 480
RECENT_S = 2.5  # how long a dismissed / no-change result stays on screen (sim s)
REPO_ROOT = Path(__file__).resolve().parents[3]


@functools.lru_cache(maxsize=None)
def _font(weight, size):
    return ImageFont.truetype(f"/usr/share/fonts/TTF/Roboto-{weight}.ttf", size)


def _box(img, xy, fill, radius=16):
    x0, y0, x1, y1 = (int(v) for v in xy)
    layer = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle((0, 0, x1 - x0 - 1, y1 - y0 - 1), radius, fill=fill)
    img.alpha_composite(layer, (x0, y0))


def _bar(d, x, y, w, h, frac, color):
    d.rounded_rectangle((x, y, x + w, y + h), h // 2, fill=DIM)
    if frac > 0:
        d.rounded_rectangle((x, y, x + max(h, int(w * min(frac, 1.0))), y + h), h // 2, fill=color)


def _wrap(d, text, font, max_w, max_lines):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        if d.textlength(trial, font=font) <= max_w:
            cur = trial
            continue
        lines.append(cur)
        cur = wd
        if len(lines) == max_lines:
            break
    if len(lines) < max_lines and cur:
        lines.append(cur)
    if len(lines) == max_lines and " ".join(lines) != " ".join(words):
        last = lines[-1]
        while last and d.textlength(last + "…", font=font) > max_w:
            last = last[:-1]
        lines[-1] = last.rstrip() + "…"
    return lines


def _text_block(d, xy, text, font, fill, max_w, max_lines, leading=1.3):
    x, y = xy
    for k, line in enumerate(_wrap(d, text, font, max_w, max_lines)):
        d.text((x, y + k * font.size * leading), line, font=font, fill=fill)


def _first_sentence(text):
    text = " ".join(text.split())
    cut = text.find("; ")
    cut = cut if 0 < cut < 200 else text.find(". ")
    return text[: cut] + "." if 0 < cut < 300 else text


def _centered(d, x0, w, y, msg, font, fill):
    d.text((x0 + (w - d.textlength(msg, font=font)) / 2, y), msg, font=font, fill=fill)


def _arrow(d, a, b, color, width=6, head=22):
    a, b = np.array(a, float), np.array(b, float)
    v = (b - a) / max(np.linalg.norm(b - a), 1e-6)
    tip_base = b - v * head
    d.line([tuple(a), tuple(tip_base)], fill=color, width=width)
    n = np.array([-v[1], v[0]])
    d.polygon([tuple(b), tuple(tip_base + n * head * 0.55), tuple(tip_base - n * head * 0.55)], fill=color)


def _dashed(d, a, b, color, width, dash=14, gap=9):
    a, b = np.array(a, float), np.array(b, float)
    length = np.linalg.norm(b - a)
    v = (b - a) / max(length, 1e-6)
    s = 0.0
    while s < length:
        d.line([tuple(a + v * s), tuple(a + v * min(s + dash, length))], fill=color, width=width)
        s += dash + gap


def _turbo(x):
    """Google's turbo colormap polynomial approximation; x in [0, 1] (0 = blue, 1 = red)."""
    x = np.clip(x, 0, 1)
    r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))))
    g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))))
    b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))))
    return (np.clip(np.stack([r, g, b], -1), 0, 1) * 255).astype(np.uint8)


def _depth_img(depth, dmax, size):
    rgb = _turbo(1.0 - np.clip(depth, 0, dmax) / dmax)  # near = red, far = blue
    return Image.fromarray(rgb).resize((size, size), Image.BILINEAR)


# ---------------------------------------------------------------- data


class _Track:
    """An episode's pose as a function of sim time (linear position, unwrapped-yaw interpolation)."""

    def __init__(self, ep):
        dt_last = ep.steps[-1]["dt"]
        self.t = np.append(ep.times, ep.times[-1] + dt_last)
        self.pos = np.asarray(ep.positions, float)
        f = np.array([s["forward_xz"] for s in ep.steps])
        yaw = np.arctan2(-f[:, 0], -f[:, 1])  # forward = (-sin yaw, -cos yaw) in (x, z)
        yaw = np.append(yaw, yaw[-1] - ep.steps[-1]["angular_velocity"] * dt_last)  # + = turn right = yaw down
        self.yaw = np.unwrap(yaw)
        self.end = self.t[-1]

    def at(self, t):
        t = float(np.clip(t, 0.0, self.end))
        p = np.array([np.interp(t, self.t, self.pos[:, k]) for k in range(3)])
        return p, float(np.interp(t, self.t, self.yaw))


class _Views:
    """Re-renders the robot's camera (same sensor as the runs) at any pose."""

    def __init__(self, st, size=1024):
        import habitat_sim
        from foresight.sim.sensors import make_sim_config

        self._hs = habitat_sim
        self.sim = habitat_sim.Simulator(make_sim_config(st["dataset"], st["scene"], size, size, sensor_uuids=("rgb",)))
        self.agent = self.sim.get_agent(0)

    def render(self, pos, yaw):
        from habitat_sim.utils.common import quat_from_angle_axis

        self.agent.set_state(self._hs.AgentState(position=pos, rotation=quat_from_angle_axis(yaw, np.array([0, 1.0, 0]))))
        return np.asarray(self.sim.get_sensor_observations()["rgb"][..., :3])

    def close(self):
        self.sim.close()


class StageRun:
    def __init__(self, root: Path, st: dict):
        out = root / st["name"]
        self.st = st
        with open(out / "baseline.pkl", "rb") as f:
            self.base = pickle.load(f)
        with open(out / "laya.pkl", "rb") as f:
            self.laya = pickle.load(f)
        self.metrics = json.loads((out / "laya_metrics.json").read_text())
        self.meta = json.loads((out / "assets.json").read_text())
        self.topdown = Image.open(out / "topdown.png").convert("RGB")
        self.explainer = dict(np.load(out / "explainer.npz"))
        self.base_track, self.laya_track = _Track(self.base), _Track(self.laya)
        self.t_end = float(self.laya.reached_goal_at if self.laya.reached_goal_at is not None else self.laya_track.end)
        self.events = [(s["t"], s["event"], s) for s in self.laya.steps if s["event"]]
        self._map_cache = {}

    def map_base(self, size):
        if size not in self._map_cache:
            self._map_cache[size] = self.topdown.resize((size, size), Image.LANCZOS).convert("RGBA")
        return self._map_cache[size]

    def to_map(self, p, size):
        c, k = self.meta["center"], size / self.meta["width_m"]
        return size / 2 + (p[0] - c[0]) * k, size / 2 + (p[2] - c[2]) * k

    def first(self, kind):
        return next((s for t, e, s in self.events if e == kind), None)

    def false_flags(self):
        """Flags raised after the mirror was walled off: none of them added a wall."""
        t_wall = next(s["t"] for s in self.laya.steps if s["wall_added"])
        return [s for t, e, s in self.events if e == "flag" and t > t_wall]

    def recent(self, t, kinds=("dismissed", "confirmed")):
        for te, e, s in reversed(self.events):
            if te <= t:
                return (e, s) if e in kinds and t - te < RECENT_S else None
        return None


def _tick_at(ep, t):
    return max(0, int(np.searchsorted(ep.times, t, side="right")) - 1)


def _ended(ep, t):
    return t >= ep.times[-1] + ep.steps[-1]["dt"]


# ---------------------------------------------------------------- narration


def narration(run: StageRun, bench_ms: float) -> dict:
    """Voice-over lines; numbers come from the runs, wording was written for the 2026-10-29 run."""
    ex = run.explainer
    h = ex["gt"].shape[0]
    cone = (slice(int(.4 * h), int(.9 * h)), slice(int(.25 * h), int(.75 * h)))
    pred, gt = float(np.median(ex["pred"][cone])), float(np.median(ex["gt"][cone]))
    flag = run.first("flag")
    lat = float(np.median(run.metrics["vlm_describe_s"]))
    laya_ms = run.metrics["laya_infer_ms_median"]
    hits = int(run.laya.collided.sum())
    return {
        "hook": "Is that an open hallway? Or a mirror?",
        "problem": f"This is the robot's first view. Depth Anything V2 reads about {pred:.0f} metres of open floor ahead. "
                   f"It's really a mirror, {gt:.1f} metres away. Trust depth alone, and you drive right into it.",
        "race_intro": "Same start for both robots, and the goal is behind the mirror. Only FORESIGHT has Qwen and Laya.",
        "race_flag": f"{_spoken_int(flag['t'])} seconds in, Qwen's note calls the hallway a reflection. Laya flags it, "
                     "so FORESIGHT turns back to check.",
        "race_confirm": "Confirmed. A wall goes on its own map, and it plans the real way round.",
        "race_crash": "The depth-only robot trusts the reflection, and drives into the mirror.",
        "race_route": "Now FORESIGHT drives out of the bedroom, along the corridor, toward the goal.",
        "race_false": f"Along the way, Qwen raises {_spoken_int(len(run.false_flags())).lower()} false alarms, on blank "
                      "walls and a TV. Second looks, and a check on the map, keep it moving.",
        "race_goal": f"FORESIGHT reaches the goal at {run.t_end:.0f} seconds, with "
                     + ("zero collisions." if hits == 0 else f"{hits} collisions."),
        "replay": f"Slowed down: one look from Qwen takes {lat:.1f} seconds. Laya re-reads the latest note every tick, in "
                  f"{laya_ms:.0f} milliseconds, and one flag sends the robot back for a second look.",
        "how": "Two loops share one camera. The fast loop turns depth into a map and a route. In the background, Qwen "
               "describes the scene, and Laya reads it. A reflection flagged twice becomes a wall on the map.",
        "latency": f"Qwen answering the questions itself takes {bench_ms:.0f} milliseconds. Laya takes {laya_ms:.0f}.",
        "credits": "FORESIGHT. One staged mirror, in one home: a demo, not a benchmark.",
    }


def _spoken_int(x):
    words = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]
    n = int(round(x))
    return words[n] if n < len(words) else str(n)


class Voice:
    """Synthesises the lines with Kokoro in its own interpreter (cached on the exact text), then mixes the cued
    clips into one track."""

    def __init__(self, lines: dict, cache: Path, python: str, hf_home: str, voice: str):
        cache.mkdir(parents=True, exist_ok=True)
        spec = [{"id": k, "text": v} for k, v in lines.items()]
        key = json.dumps({"lines": spec, "voice": voice}, sort_keys=True)
        if not ((cache / "durations.json").exists() and (cache / "key.json").exists()
                and (cache / "key.json").read_text() == key):
            (cache / "lines.json").write_text(json.dumps(spec, indent=1))
            env = dict(os.environ, HF_HOME=hf_home, HF_HUB_OFFLINE="1")
            subprocess.run([python, str(REPO_ROOT / "tools" / "kokoro_tts.py"), str(cache / "lines.json"), str(cache),
                            "--voice", voice], check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            (cache / "key.json").write_text(key)
        self.dur = json.loads((cache / "durations.json").read_text())
        self.cache = cache
        self.cues = []  # (absolute video time s, line id)

    def clip(self, line_id):
        with wave.open(str(self.cache / f"{line_id}.wav")) as w:
            return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768

    def write_mix(self, path: Path, total_s: float):
        mix = np.zeros(int(total_s * SR) + SR, np.float32)
        for t, line_id in self.cues:
            x = self.clip(line_id)
            i = int(t * SR)
            mix[i:i + len(x)] += x[: max(0, len(mix) - i)]
        mix *= 0.89 / max(1e-6, float(np.abs(mix).max()))
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((mix[: int(total_s * SR)] * 32767).astype(np.int16).tobytes())


def _place(lines, anchors, dur, gap=0.35):
    """Lay lines out in order, each at its anchor (video s) or right after the previous one."""
    out, end = [], -1e9
    for line_id, anchor in zip(lines, anchors):
        start = max(anchor, end + gap)
        out.append((start, line_id))
        end = start + dur[line_id]
    return out, end


# ---------------------------------------------------------------- shared pieces


def _header(d, subtitle, line2, t=None, speed=None):
    d.text((LX, 22), "FORESIGHT", font=_font("Black", 72), fill=TEXT)
    d.text((LX + d.textlength("FORESIGHT", font=_font("Black", 72)) + 24, 44), subtitle, font=_font("Medium", 50), fill=MUTED)
    if line2:
        d.text((LX, 112), line2, font=_font("Regular", 32), fill=TEXT)
    if t is not None:
        clock = f"t = {t:.1f} s"
        cw = d.textlength(clock, font=_font("Bold", 56))
        d.text((S - G - cw, 36), clock, font=_font("Bold", 56), fill=TEXT)
        if speed is not None and abs(speed - 1.0) > 0.02:
            badge = f"{speed:.1f}× speed".replace(".0×", "×")
            bw = d.textlength(badge, font=_font("Bold", 34)) + 36
            x1 = S - G - cw - 24
            d.rounded_rectangle((x1 - bw, 44, x1, 98), 27, fill=AMBER if speed > 1 else DIM)
            d.text((x1 - bw + 18, 51), badge, font=_font("Bold", 34), fill=(16, 18, 22) if speed > 1 else TEXT)


def _banner(img, x, y, w, msg, color):
    _box(img, (x + 18, y + 18, x + w - 18, y + 122), color)
    d = ImageDraw.Draw(img)
    size = 58
    while size > 30 and d.textlength(msg, font=_font("Black", size)) > w - 80:
        size -= 2
    _centered(d, x, w, y + 70 - size * 0.62, msg, _font("Black", size), (255, 255, 255))


def _first_person(img, frame, x, y, size, step, moving=True):
    fp = Image.fromarray(frame).resize((size, size), Image.LANCZOS).convert("RGBA")
    img.alpha_composite(fp, (x, y))
    _box(img, (x + 18, y + size - 150, x + 640, y + size - 18), (8, 10, 14, 205))
    d = ImageDraw.Draw(img)
    d.text((x + 40, y + size - 136), "Depth estimate", font=_font("Regular", 28), fill=MUTED)
    d.text((x + 300, y + size - 142), f"{step['depth_ahead_m']:.1f} m clear ahead", font=_font("Bold", 36), fill=AMBER)
    d.text((x + 40, y + size - 78), "Really", font=_font("Regular", 28), fill=MUTED)
    d.text((x + 300, y + size - 84), f"{step['gt_depth_ahead_m']:.2f} m ahead", font=_font("Bold", 36), fill=TEXT)


def _map(img, run, ep, track, t, xy, size, color, foresight=False):
    """Photographic top-down cutaway: this run's path so far, current heading, start, goal, mirror and, for
    FORESIGHT, the walls Laya put on its map and its planned route."""
    canvas = run.map_base(size).copy()
    d = ImageDraw.Draw(canvas)
    st = run.st
    step = ep.steps[_tick_at(ep, t)]
    fwd = st["fwd"]
    perp = np.array([-fwd[2], 0, fwd[0]]) * st["trap_width_m"] / 2
    d.line([run.to_map(st["trap_point"] - perp, size), run.to_map(st["trap_point"] + perp, size)], fill=CYAN, width=9)
    tx, ty = run.to_map(st["trap_point"] - perp, size)
    d.text((tx + 10, ty - 6), st["trap"], font=_font("Black", 26), fill=CYAN, stroke_width=3, stroke_fill=(10, 12, 16))
    if foresight:
        for k, (a, b) in enumerate(step["map_walls"]):
            pa, pb = run.to_map((a[0], 0, a[1]), size), run.to_map((b[0], 0, b[1]), size)
            _dashed(d, pa, pb, AMBER, 7)
            if k == 0:
                d.text((pb[0] - 70, pb[1] + 6), "Laya wall", font=_font("Bold", 24), fill=AMBER,
                       stroke_width=3, stroke_fill=(10, 12, 16))
        route = [run.to_map((p[0], 0, p[1]), size) for p in step["plan_path"]]
        if len(route) > 1 and not (ep.reached_goal_at is not None and t >= ep.reached_goal_at):
            for a, b in zip(route[:-1:2], route[1::2]):
                d.line([a, b], fill=(255, 255, 255), width=3)
    gx, gy = run.to_map(st["goal"], size)
    d.ellipse((gx - 13, gy - 13, gx + 13, gy + 13), fill=RED, outline=(255, 255, 255), width=3)
    gl = gx + 16 if gx < size - 90 else gx - 16 - d.textlength("goal", font=_font("Bold", 28))
    d.text((gl, gy - 18), "goal", font=_font("Bold", 28), fill=(255, 255, 255), stroke_width=3, stroke_fill=(10, 12, 16))
    k = int(np.searchsorted(track.t, t, side="right"))
    pos, yaw = track.at(t)
    path = [run.to_map(p, size) for p in track.pos[:k]] + [run.to_map(pos, size)]
    sx, sy = path[0]
    d.ellipse((sx - 10, sy - 10, sx + 10, sy + 10), outline=(255, 255, 255), width=4)
    if len(path) > 1:
        d.line(path, fill=color, width=8, joint="curve")
    cx, cy = path[-1]
    f = np.array([-np.sin(yaw), -np.cos(yaw)])
    n = np.array([-f[1], f[0]])
    tip = np.array([cx, cy]) + f * 30
    d.polygon([tuple(tip), tuple(np.array([cx, cy]) + n * 15 - f * 10), tuple(np.array([cx, cy]) - n * 15 - f * 10)],
              fill=color, outline=(255, 255, 255))
    img.alpha_composite(canvas, xy)
    _box(img, (xy[0] + 10, xy[1] + 10, xy[0] + 190, xy[1] + 52), (8, 10, 14, 200), radius=10)
    ImageDraw.Draw(img).text((xy[0] + 22, xy[1] + 16), "Top-down", font=_font("Bold", 26), fill=TEXT)


def _timeline(img, run, t, xy, w, h):
    x0, y0 = xy
    _box(img, (x0, y0, x0 + w, y0 + h), PANEL + (255,))
    d = ImageDraw.Draw(img)
    d.text((x0 + 28, y0 + 14), "Decisions over time", font=_font("Bold", 30), fill=TEXT)
    d.text((x0 + 330, y0 + 20), "gridlines every 5 s", font=_font("Regular", 22), fill=MUTED)
    lx0, lx1 = x0 + 250, x0 + w - 36
    t_end = run.t_end

    def tx(tt):
        return lx0 + (lx1 - lx0) * min(tt, t_end) / t_end

    rows = [("VLM looks", y0 + 62), ("Laya", y0 + 112), ("Depth-only", y0 + 162)]
    for k in range(0, int(t_end) + 1, 5):
        d.line([(tx(k), y0 + 56), (tx(k), y0 + h - 12)], fill=(40, 45, 55), width=1)
    for label, y in rows:
        d.text((x0 + 28, y), label, font=_font("Medium", 28), fill=MUTED)
    done = 0
    for s, e in run.metrics["vlm_jobs"]:
        if s > t or s >= t_end:
            break
        d.rounded_rectangle((tx(s) + 1, rows[0][1] + 2, max(tx(s) + 4, tx(min(e, t)) - 1), rows[0][1] + 34), 6, fill=BLUE)
        done += int(e <= t)
    n_laya = 0
    for s in run.laya.steps:
        if s["t"] > t:
            break
        n_laya += 1
        d.line([(tx(s["t"]), rows[1][1] + 2), (tx(s["t"]), rows[1][1] + 34)], fill=RED if s["p_hazard"] > 0.6 else GREEN, width=2)
    for s in run.base.steps:
        if s["t"] > t:
            break
        d.line([(tx(s["t"]), rows[2][1] + 2), (tx(s["t"]), rows[2][1] + 34)], fill=(120, 126, 136), width=2)
    d.line([(tx(t), y0 + 56), (tx(t), y0 + h - 10)], fill=(255, 255, 255), width=3)
    summary = f"Laya {n_laya} decisions  ·  VLM {done} looks finished"
    d.text((x0 + w - 36 - d.textlength(summary, font=_font("Bold", 28)), y0 + 14), summary, font=_font("Bold", 28), fill=TEXT)


def _hud_left(img, run, step, t, xy, w, h):
    x, y = xy
    _box(img, (x, y, x + w, y + h), PANEL + (255,))
    d = ImageDraw.Draw(img)
    ep = run.base
    speed = 0.0 if _ended(ep, t) else step["linear_velocity"]
    d.text((x + 28, y + 22), "Speed", font=_font("Regular", 26), fill=MUTED)
    d.text((x + 28, y + 52), f"{speed:.2f} m/s", font=_font("Bold", 58), fill=TEXT)
    d.text((x + 28, y + 150), "Status", font=_font("Regular", 26), fill=MUTED)
    if ep.contact_at is not None and t >= ep.contact_at:
        status, color = f"Drove into the {run.st['trap']} at t = {ep.contact_at:.1f} s. The reflection looked like open floor.", RED
    elif ep.reached_goal_at is not None and t >= ep.reached_goal_at:
        status, color = f"Reached the goal at t = {ep.reached_goal_at:.1f} s.", GREEN
    elif step["looking_around"]:
        status, color = "Looking around to map the room.", TEXT
    else:
        status, color = "Its map shows an open hallway. It has no idea it's a reflection.", TEXT
    _text_block(d, (x + 28, y + 184), status, _font("Medium", 32), color, w - 56, 6)
    d.text((x + 28, y + h - 46), f"depth only  ·  {step['depth_ms']:.0f} ms / decision", font=_font("Regular", 24), fill=MUTED)


def _foresight_status(run, step, t):
    ep = run.laya
    if ep.reached_goal_at is not None and t >= ep.reached_goal_at:
        return f"At the goal, t = {ep.reached_goal_at:.1f} s.", GREEN
    if step["verifying"]:
        if step["verify_aligned"]:
            return "Holding still for the VLM's second look.", AMBER
        return "Laya flagged a mirror. Turning back for a second look.", AMBER
    if step["hazard"]:
        return "Flagged twice: wall added to its map. Replanning.", GREEN
    recent = run.recent(t)
    if recent and recent[0] == "dismissed":
        return "Second look disagreed: false alarm dismissed.", CYAN
    if recent and recent[0] == "confirmed" and not recent[1]["wall_added"]:
        return "Flagged twice, but the map shows no phantom opening there: no change.", CYAN
    if step["looking_around"]:
        return "Looking around to map the room.", TEXT
    return "Following its route on its own map.", TEXT


def _hud_right(img, run, step, t, xy, w, h):
    x, y = xy
    _box(img, (x, y, x + w, y + h), PANEL + (255,))
    d = ImageDraw.Draw(img)
    lat = float(np.median(run.metrics["vlm_describe_s"]))
    d.text((x + 28, y + 20), "VLM", font=_font("Bold", 32), fill=BLUE)
    d.text((x + 110, y + 24), f"looking…  {step['vlm_progress'] * lat:3.1f} / {lat:.1f} s", font=_font("Medium", 26), fill=TEXT)
    _bar(d, x + 28, y + 66, w - 56, 14, step["vlm_progress"], BLUE)
    p = step["p_hazard"]
    d.text((x + 28, y + 96), "Laya", font=_font("Bold", 32), fill=GREEN)
    d.text((x + 120, y + 100), f"P({run.st['trap']}) {p:.2f}", font=_font("Medium", 26), fill=TEXT)
    d.text((x + w - 28 - d.textlength(f"{step['laya_infer_ms']:.0f} ms", font=_font("Bold", 30)), y + 96),
           f"{step['laya_infer_ms']:.0f} ms", font=_font("Bold", 30), fill=GREEN)
    _bar(d, x + 28, y + 142, w - 56, 14, p, RED if p > 0.6 else GREEN)
    status, color = _foresight_status(run, step, t)
    _text_block(d, (x + 28, y + 172), status, _font("Bold", 26), color, w - 56, 2, 1.2)
    if step["description"]:
        d.text((x + 28, y + 246), f"Latest VLM note ({t - step['description_seen_at']:.1f} s old):",
               font=_font("Regular", 22), fill=MUTED)
        _text_block(d, (x + 28, y + 278), "“" + _first_sentence(step["description"]) + "”",
                    _font("Italic", 25), TEXT, w - 56, 6, 1.25)
    else:
        d.text((x + 28, y + 246), "No VLM note yet: the first look takes ~2 s.", font=_font("Regular", 22), fill=MUTED)


# ---------------------------------------------------------------- segments


class _Schedule:
    """Piecewise-constant playback speed: video time <-> sim time, with a still start (pre_s) and end (hold_s)."""

    def __init__(self, pieces, pre_s, hold_s):
        self.pieces = pieces  # [(sim_t0, sim_t1, speed)]
        self.pre_s, self.hold_s = pre_s, hold_s
        self.v0 = pre_s + np.cumsum([0.0] + [(b - a) / v for a, b, v in pieces])
        self.length = float(self.v0[-1] + hold_s)

    def video(self, t):
        for (a, b, v), v0 in zip(self.pieces, self.v0):
            if t <= b:
                return v0 + (max(t, a) - a) / v
        return float(self.v0[-1])

    def sim(self, tau):
        """-> (sim time, playback speed, or None while the picture is held still)."""
        if tau < self.pre_s:
            return self.pieces[0][0], None
        for (a, b, v), v0, v1 in zip(self.pieces, self.v0[:-1], self.v0[1:]):
            if tau < v1:
                return a + (tau - v0) * v, v
        return self.pieces[-1][1], None


def race_plan(run, dur):
    """Playback speeds and voice cues for the race, fitted to the narration: the intro is spoken over the still
    start, the opening (flag, double-check, baseline crash) plays at up to 1x, the long drive at up to 3x, the
    arrival at 1x."""
    flag, confirm = run.first("flag")["t"], run.first("confirmed")["t"]
    crash = run.base.contact_at or run.base.times[-1]
    t_a = min(run.t_end, crash + 1.5)
    t_b = max(t_a, run.t_end - 3.0)
    moving = next(s["t"] for s in run.laya.steps if s["t"] > confirm and s["linear_velocity"] > 0.05)
    first_false = run.false_flags()[0]["t"]
    pre = dur["race_intro"] + 0.9
    p1 = (["race_flag", "race_confirm", "race_crash"], [flag, confirm, crash])
    p2 = (["race_route", "race_false"], [max(t_a, moving), first_false])
    for v1 in np.arange(1.0, 0.49, -0.05):
        for v2 in (3.0, 2.5, 2.0, 1.5):
            sch = _Schedule([(0.0, t_a, v1), (t_a, t_b, v2), (t_b, run.t_end, 1.0)], pre, 0.0)
            cues1, end1 = _place(p1[0], [sch.video(a) + 0.2 for a in p1[1]], dur)
            cues2, end2 = _place(p2[0], [max(sch.video(a), end1 + 0.5) for a in p2[1]], dur, gap=0.5)
            late = max(c[0] - sch.video(a) for c, a in zip(cues1, p1[1]))
            if late <= 2.5 and end2 <= sch.video(t_b) + 0.5:
                goal_at = sch.video(run.t_end) + 0.2
                sch = _Schedule(sch.pieces, pre, max(2.5, dur["race_goal"] + 1.2))
                return sch, [(0.4, "race_intro")] + cues1 + cues2 + [(goal_at, "race_goal")]
    raise RuntimeError("narration doesn't fit the race even at 0.5x / 1.5x")


def race_frames(run, views, sch, subtitle, line2):
    st = run.st
    n = int(sch.length * FPS)
    for k in range(n):
        t, speed = sch.sim(k / FPS)
        img = Image.new("RGBA", (S, S), BG)
        d = ImageDraw.Draw(img)
        _header(d, subtitle, line2, t, speed)
        for x, title, sub in [(LX, "DEPTH-ONLY BASELINE", "monocular depth · own map · planner"),
                              (RX, "FORESIGHT", "same, + Qwen3-VL-2B looks · Laya decides")]:
            d.text((x, 162), title, font=_font("Black", 36), fill=TEXT)
            d.text((x + d.textlength(title, font=_font("Black", 36)) + 18, 168), sub, font=_font("Regular", 28), fill=MUTED)
        bi, li = _tick_at(run.base, t), _tick_at(run.laya, t)
        bs, ls = run.base.steps[bi], run.laya.steps[li]
        _first_person(img, views.render(*run.base_track.at(t)), LX, PANEL_Y, PW, bs)
        _first_person(img, views.render(*run.laya_track.at(t)), RX, PANEL_Y, PW, ls)
        if run.base.contact_at is not None and t >= run.base.contact_at:
            _banner(img, LX, PANEL_Y, PW, f"DROVE INTO THE {st['trap'].upper()}", (205, 40, 50, 230))
        recent = run.recent(t)
        if run.laya.reached_goal_at is not None and t >= run.laya.reached_goal_at:
            _banner(img, RX, PANEL_Y, PW, "REACHED THE GOAL", (16, 128, 84, 230))
        elif ls["verifying"]:
            _banner(img, RX, PANEL_Y, PW, f"{st['trap'].upper()}? DOUBLE-CHECKING", (196, 120, 0, 230))
        elif ls["hazard"]:
            _banner(img, RX, PANEL_Y, PW, f"{st['trap'].upper()} CONFIRMED: ROUTING AROUND", (16, 128, 84, 230))
        elif recent and recent[0] == "dismissed":
            _banner(img, RX, PANEL_Y, PW, "FALSE ALARM DISMISSED", (60, 72, 92, 230))
        _timeline(img, run, t, (LX, TL_Y), S - 2 * G, TL_H)
        hud_w = PW - MAP - 16
        _map(img, run, run.base, run.base_track, t, (LX, BOT_Y + (BOT_H - MAP) // 2), MAP, RED)
        _map(img, run, run.laya, run.laya_track, t, (RX, BOT_Y + (BOT_H - MAP) // 2), MAP, GREEN, foresight=True)
        _hud_left(img, run, bs, t, (LX + MAP + 16, BOT_Y), hud_w, BOT_H)
        _hud_right(img, run, ls, t, (RX + MAP + 16, BOT_Y), hud_w, BOT_H)
        yield img.convert("RGB")


def hook_frames(run, n):
    big = Image.fromarray(run.explainer["rgb"]).resize((2112, 2112), Image.LANCZOS)
    shade = Image.new("RGBA", (S, S // 2), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    for y in range(S // 2):
        sd.line([(0, y), (S, y)], fill=(8, 10, 14, int(235 * y / (S // 2))))
    for k in range(n):
        z = 1.0 + 0.05 * k / max(n - 1, 1)  # slow push-in
        side = int(2112 / z * (S / 2112) * 1.1)
        off = (2112 - side) // 2
        img = big.crop((off, off, off + side, off + side)).resize((S, S), Image.BILINEAR).convert("RGBA")
        img.alpha_composite(shade, (0, S // 2))
        _box(img, (0, 0, S, 150), (8, 10, 14, 170), radius=0)
        d = ImageDraw.Draw(img)
        d.text((G * 3, 36), "FORESIGHT", font=_font("Black", 72), fill=TEXT)
        d.text((G * 3, 1500), "What a camera-only robot sees.", font=_font("Medium", 56), fill=(210, 214, 220))
        d.text((G * 3, 1580), run.st["hook"], font=_font("Black", 120), fill=(255, 255, 255))
        yield img.convert("RGB")


def problem_frames(run, n):
    ex = run.explainer
    h = ex["gt"].shape[0]
    cone = (slice(int(.4 * h), int(.9 * h)), slice(int(.25 * h), int(.75 * h)))
    pred_ahead, gt_ahead = float(np.median(ex["pred"][cone])), float(np.median(ex["gt"][cone]))
    dmax, P = 8.0, 600
    panels = [("What the camera sees", Image.fromarray(ex["rgb"]).resize((P, P), Image.LANCZOS)),
              ("What the depth model thinks", _depth_img(ex["pred"], dmax, P)),
              ("What's really there", _depth_img(ex["gt"], dmax, P))]
    xs = [30, 660, 1290]
    reveal = [0.0, 2.0, 5.5, 5.5, 8.5]  # roughly in step with the voice-over
    for k in range(n):
        t = k / FPS
        img = Image.new("RGBA", (S, S), BG)
        d = ImageDraw.Draw(img)
        d.text((60, 50), "Why a camera gets fooled", font=_font("Bold", 84), fill=TEXT)
        d.text((60, 150), "The robot's first frame, before it moves.", font=_font("Regular", 36), fill=MUTED)
        for i, (label, im) in enumerate(panels):
            if t >= reveal[i]:
                d.text((xs[i], 236), label, font=_font("Bold", 36), fill=TEXT)
                img.alpha_composite(im.convert("RGBA"), (xs[i], 290))
        if t >= reveal[1]:
            legend = Image.fromarray(_turbo(np.linspace(1, 0, 1230)[None, :].repeat(26, 0)))
            img.alpha_composite(legend.convert("RGBA"), (660, 912))
            d.text((660, 948), "near", font=_font("Regular", 26), fill=MUTED)
            d.text((1890 - d.textlength("far (8 m+)", font=_font("Regular", 26)), 948), "far (8 m+)", font=_font("Regular", 26), fill=MUTED)
        if t >= reveal[3]:
            d.text((60, 1040), "Straight ahead:", font=_font("Medium", 46), fill=TEXT)
            scale = 1250 / max(dmax, pred_ahead)
            for j, (label, v, color) in enumerate([("Depth model", pred_ahead, AMBER), ("Really (the mirror)", gt_ahead, TEXT)]):
                y = 1130 + j * 110
                d.text((60, y + 6), label, font=_font("Medium", 38), fill=MUTED)
                d.rounded_rectangle((520, y, 520 + int(scale * v), y + 64), 14, fill=color)
                d.text((540 + int(scale * v), y + 4), f"{v:.1f} m", font=_font("Bold", 50), fill=color)
        if t >= reveal[4]:
            _text_block(d, (60, 1420), "The mirror reflects the hallway, so a single camera sees a corridor that isn't there. "
                        "A depth-only robot drives straight into it.", _font("Regular", 46), TEXT, S - 120, 4)
        yield img.convert("RGB")


def replay_plan(run, dur, speed=0.4, post=1.0):
    """From the moment the VLM starts the look that gets flagged, to just after the flag."""
    flag = run.first("flag")
    t_flag = flag["t"]
    t0, t1 = flag["description_seen_at"], t_flag + post
    play_s = (t1 - t0) / speed
    freeze_s = max(3.2, dur["replay"] + 1.6 - play_s)
    return dict(speed=speed, t0=t0, t1=t1, freeze_s=freeze_s, length=play_s + freeze_s)


def replay_frames(run, views, plan):
    speed, t0, t1, freeze_s = plan["speed"], plan["t0"], plan["t1"], plan["freeze_s"]
    flag, confirm = run.first("flag"), run.first("confirmed")
    t_flag = flag["t"]
    lat_s = flag["t"] - flag["description_seen_at"]
    cards = [
        (flag["description_seen_at"], BLUE, f"t = {flag['description_seen_at']:.1f} s",
         "Qwen starts looking at the mirror view, in the background, while the robot keeps turning to map the room."),
        (t_flag, BLUE, f"t = {t_flag:.1f} s", f"{lat_s:.1f} s later its note arrives: “{_first_sentence(flag['description'])}”"),
        (t_flag, GREEN, f"+ {flag['laya_infer_ms']:.0f} ms",
         f"Laya reads the note: P({run.st['trap']}) = {flag['p_hazard']:.2f}. One forward pass."),
        (t_flag, RED, "same tick", f"One flag isn't enough: stop, turn back, look again. Flagged again at t = "
                                   f"{confirm['t']:.1f} s, so the mirror becomes a wall on the robot's map."),
    ]
    look_s = float(np.median(run.metrics["vlm_describe_s"]))
    per_look = int(round(look_s / float(np.median([s["dt"] for s in run.laya.steps]))))
    n_play = int((t1 - t0) / speed * FPS)
    freeze_at = int((t_flag - t0) / speed * FPS)
    n_freeze = int(freeze_s * FPS)
    frames = list(range(freeze_at)) + [freeze_at] * n_freeze + list(range(freeze_at, n_play))
    big = 1180
    inset = Image.fromarray(views.render(*run.laya_track.at(flag["description_seen_at"]))).resize((380, 380), Image.LANCZOS)
    for idx, k in enumerate(frames):
        paused = freeze_at <= idx < freeze_at + n_freeze
        t = t_flag if paused else t0 + k / FPS * speed
        if idx < freeze_at:
            shown = 1
        elif paused:
            shown = 2 + min(2, int((idx - freeze_at) / (n_freeze * 0.22)))
        else:
            shown = 4
        img = Image.new("RGBA", (S, S), BG)
        d = ImageDraw.Draw(img)
        _header(d, f"Replay at {speed:.1f}×: the moment it decides", None, t)
        step = run.laya.steps[_tick_at(run.laya, t)]
        _first_person(img, views.render(*run.laya_track.at(t)), LX, 150, big, step)
        if step["verifying"] or (paused and shown == 4):
            _banner(img, LX, 150, big, f"{run.st['trap'].upper()}? DOUBLE-CHECKING", (196, 120, 0, 230))
        if idx >= freeze_at:  # the note is about the view the robot had when the look started, not this one
            ix, iy = LX + big - 20 - 380, 150 + 200
            _box(img, (ix - 10, iy - 52, ix + 390, iy + 390), (8, 10, 14, 230), radius=12)
            img.paste(inset, (ix, iy))
            ImageDraw.Draw(img).text((ix + 4, iy - 44), f"The view it flagged (t = {flag['description_seen_at']:.1f} s)",
                                     font=_font("Bold", 26), fill=TEXT)
        if paused:
            _box(img, (LX + big - 230, 150 + big - 90, LX + big - 18, 150 + big - 20), (8, 10, 14, 220), radius=12)
            ImageDraw.Draw(img).text((LX + big - 210, 150 + big - 82), "PAUSED", font=_font("Black", 44), fill=AMBER)
        ex0, ew = LX + big + 16, S - G - (LX + big + 16)
        y = 150
        for j, (te, color, stamp, text) in enumerate(cards):
            on = j < shown
            _box(img, (ex0, y, ex0 + ew, y + 280), PANEL + (255,) if on else (18, 21, 27, 255))
            d = ImageDraw.Draw(img)
            d.text((ex0 + 24, y + 18), stamp, font=_font("Bold", 34), fill=color if on else DIM)
            _text_block(d, (ex0 + 24, y + 68), text, _font("Medium", 28), TEXT if on else DIM, ew - 48, 6, 1.25)
            y += 296
        # zoomed timeline: every Laya decision drawn at its true compute width
        zy, zh = 150 + big + 24, S - G - (150 + big + 24)
        _box(img, (LX, zy, S - G, zy + zh), PANEL + (255,))
        d = ImageDraw.Draw(img)
        d.text((LX + 28, zy + 18), f"Zoomed in: {t0:.1f}–{t1:.1f} s, everything drawn to scale", font=_font("Bold", 32), fill=TEXT)
        lx0, lx1 = LX + 200, S - G - 40
        px_per_s = (lx1 - lx0) / (t1 - t0)

        def tx(tt):
            return lx0 + (tt - t0) * px_per_s

        d.text((LX + 28, zy + 112), "VLM", font=_font("Bold", 36), fill=BLUE)
        d.text((LX + 28, zy + 252), "Laya", font=_font("Bold", 36), fill=GREEN)
        for s_, e_ in run.metrics["vlm_jobs"]:
            if e_ < t0 or s_ > min(t, t1):
                continue
            a, b = tx(max(s_, t0)), tx(min(e_, t, t1))
            if b > a:
                d.rounded_rectangle((a, zy + 90, b, zy + 180), 12, fill=BLUE)
                lbl = f"one look = {(e_ - s_) * 1000:,.0f} ms"
                if b - a > d.textlength(lbl, font=_font("Bold", 32)) + 20:
                    d.text((a + 16, zy + 114), lbl, font=_font("Bold", 32), fill=(10, 20, 40))
        for s in run.laya.steps:
            if s["t"] < t0 or s["t"] > min(t, t1):
                continue
            a = tx(s["t"])
            d.rectangle((a, zy + 230, a + max(2, s["laya_infer_ms"] / 1000 * px_per_s), zy + 320),
                        fill=RED if s["p_hazard"] > 0.6 else GREEN)
        d.text((lx0, zy + 340), f"Each bar is one Laya decision, drawn {run.metrics['laya_infer_ms_median']:.0f} ms wide; "
               "one per control tick (~0.1 s). Red = mirror flagged.", font=_font("Regular", 28), fill=MUTED)
        d.text((LX + 28, zy + 420), f"While the VLM takes one look ({look_s:.1f} s), Laya makes ~{per_look} decisions.",
               font=_font("Bold", 44), fill=TEXT)
        d.line([(tx(t), zy + 80), (tx(t), zy + 326)], fill=(255, 255, 255), width=3)
        yield img.convert("RGB")


def how_frames(run, n):
    lat = float(np.median(run.metrics["vlm_describe_s"]))
    laya_ms = run.metrics["laya_infer_ms_median"]
    depth_ms = float(np.median([s["depth_ms"] for s in run.base.steps]))
    bw, bh = 640, 150
    boxes = {  # centre, title, subtitle, colour, reveal time (s)
        "cam": ((960, 330), "Camera (RGB only)", "", TEXT, 0.0),
        "depth": ((480, 590), "Depth Anything V2", f"monocular depth · {depth_ms:.0f} ms", AMBER, 0.8),
        "map": ((480, 840), "Own map + route", "built from depth as it drives", AMBER, 1.6),
        "safety": ((480, 1090), "Safety layer", "stops before obstacles", AMBER, 2.4),
        "vlm": ((1440, 590), "Qwen3-VL-2B", f"looks, writes a note · {lat:.1f} s", BLUE, 3.4),
        "laya": ((1440, 840), "Laya", f"reads the latest note · {laya_ms:.0f} ms", GREEN, 4.6),
        "rule": ((1440, 1090), "Rule", "flagged twice: wall on the map", RED, 6.0),
        "motor": ((960, 1350), "Motor command", "every control tick", TEXT, 2.4),
    }

    def edge(key, side):
        (cx, cy), *_ = boxes[key]
        return {"top": (cx, cy - bh / 2 - 4), "bottom": (cx, cy + bh / 2), "left": (cx - bw / 2 - 4, cy),
                "right": (cx + bw / 2, cy)}[side]

    for k in range(n):
        t = k / FPS
        img = Image.new("RGBA", (S, S), BG)
        d = ImageDraw.Draw(img)
        d.text((60, 50), "How FORESIGHT decides", font=_font("Bold", 84), fill=TEXT)
        d.text((60, 150), "Two loops share one camera. Timings measured on a laptop RTX 4060.", font=_font("Regular", 36), fill=MUTED)
        grey = (110, 118, 130)
        if t >= 0.8:
            for j, line in enumerate(["FAST LOOP", "every tick"]):
                f = _font("Black" if j == 0 else "Medium", 32)
                d.text((140 - d.textlength(line, font=f) / 2 + 0, 432 + j * 40), line, font=f, fill=AMBER)
            d.line([(edge("cam", "left")[0], 330), (480, 330)], fill=grey, width=6)
            _arrow(d, (480, 330), edge("depth", "top"), grey)
        if t >= 1.6:
            _arrow(d, edge("depth", "bottom"), edge("map", "top"), grey)
        if t >= 2.4:
            _arrow(d, edge("map", "bottom"), edge("safety", "top"), grey)
            _arrow(d, edge("safety", "bottom"), (edge("motor", "top")[0] - bw * 0.3, edge("motor", "top")[1]), grey)
        if t >= 3.4:
            for j, line in enumerate(["SLOW LOOP", "in the background"]):
                f = _font("Black" if j == 0 else "Medium", 32)
                d.text((1780 - d.textlength(line, font=f) / 2, 432 + j * 40), line, font=f, fill=BLUE)
            d.line([(edge("cam", "right")[0], 330), (1440, 330)], fill=grey, width=6)
            _arrow(d, (1440, 330), edge("vlm", "top"), grey)
        if t >= 4.6:
            _arrow(d, edge("vlm", "bottom"), edge("laya", "top"), grey)
        if t >= 6.0:
            _arrow(d, edge("laya", "bottom"), edge("rule", "top"), grey)
            _arrow(d, edge("rule", "left"), (edge("map", "right")[0] + 4, edge("map", "right")[1] + 30), RED)
            d.text((930, 880), "adds a wall", font=_font("Bold", 28), fill=RED)
        for key, ((cx, cy), title, sub, color, ta) in boxes.items():
            if t < ta:
                continue
            _box(img, (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), PANEL + (255,))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), 16, outline=color, width=4)
            _centered(d, cx - bw / 2, bw, cy - (50 if sub else 24), title, _font("Bold", 44), TEXT)
            if sub:
                _centered(d, cx - bw / 2, bw, cy + 10, sub, _font("Medium", 32), color)
        if t >= 7.0:
            _text_block(d, (60, 1560), "Laya never sees pixels. It turns the VLM's latest note into calibrated "
                        "probabilities, fast enough to run on every control tick.", _font("Regular", 46), TEXT, S - 120, 4)
        yield img.convert("RGB")


def latency_frames(run, bench, n):
    rows = [("Qwen3-VL-2B: describe the scene", float(np.median(run.metrics["vlm_describe_s"])) * 1000, MUTED),
            ("Qwen3-VL-2B: answer the questions itself", bench, BLUE),
            ("Laya: answer from the description", run.metrics["laya_infer_ms_median"], GREEN)]
    vmax = max(v for _, v, _ in rows)
    n_dec = len(run.laya.steps)
    for k in range(n):
        grow = min(1.0, k / (FPS * 1.0))
        img = Image.new("RGB", (S, S), BG)
        d = ImageDraw.Draw(img)
        d.text((120, 150), "FORESIGHT", font=_font("Black", 64), fill=MUTED)
        d.text((120, 240), "Time to make one decision", font=_font("Bold", 92), fill=TEXT)
        d.text((120, 360), "Same robot, same laptop GPU (RTX 4060)", font=_font("Regular", 40), fill=MUTED)
        for i, (label, v, color) in enumerate(rows):
            y = 520 + i * 230
            d.text((120, y), label, font=_font("Medium", 48), fill=TEXT)
            w = max(18, int(1380 * v / vmax * grow))
            d.rounded_rectangle((120, y + 76, 120 + w, y + 146), 16, fill=color)
            d.text((150 + w, y + 78), f"{v * grow:,.0f} ms", font=_font("Bold", 58), fill=color)
        _text_block(d, (120, 1250), "The VLM refreshes what the robot \"sees\" about every 2 s. Laya re-decided on every "
                    f"control tick: {n_dec} decisions in {run.t_end:.0f} s, each a single forward pass with calibrated "
                    "probabilities.", _font("Regular", 46), TEXT, S - 240, 4)
        yield img


def credits_frames(run, n):
    img = Image.new("RGB", (S, S), BG)
    d = ImageDraw.Draw(img)
    d.text((120, 420), "FORESIGHT", font=_font("Black", 130), fill=TEXT)
    d.text((120, 580), "Map-free monocular navigation with an on-board VLM", font=_font("Medium", 50), fill=MUTED)
    d.text((120, 760), "Built with", font=_font("Bold", 40), fill=TEXT)
    for i, line in enumerate(["Habitat-Sim  ·  HM3D (Matterport scans of real homes)",
                              "Depth Anything V2  ·  Qwen3-VL-2B  ·  Laya (Convai Innovations)",
                              "Voice-over: Kokoro-82M"]):
        d.text((120, 830 + i * 64), line, font=_font("Regular", 40), fill=TEXT)
    _text_block(d, (120, 1150), f"Zero-shot prototype in simulation ({run.st['place']}). One staged mirror in one home: "
                "a demo, not a benchmark. Laya raises the flag; simple rules (flag twice, then a wall on the robot's "
                "own map) decide what to do.", _font("Regular", 36), MUTED, S - 240, 5)
    for _ in range(n):
        yield img


# ---------------------------------------------------------------- assembly


class _Writer:
    def __init__(self, path: Path):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{S}x{S}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.n = 0
        self.bg = Image.new("RGB", (S, S), BG)

    def segment(self, frames, n_total):
        for k, im in enumerate(frames):
            if k >= n_total:
                break
            if k < FADE:
                im = Image.blend(self.bg, im, (k + 1) / (FADE + 1))
            elif k >= n_total - FADE:
                im = Image.blend(self.bg, im, (n_total - k) / (FADE + 1))
            self.proc.stdin.write(im.tobytes())
            self.n += 1

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def render_video(root: Path, stages: dict, bench_ms: float, out_path: Path, tts_python: str, kokoro_home: str,
                 voice: str = "af_heart"):
    run = StageRun(root, stages["hm3d-hallway-mirror"])
    vo = Voice(narration(run, bench_ms), root / "narration", tts_python, kokoro_home, voice)
    dur = vo.dur
    views = _Views(run.st)
    race_sch, race_cues = race_plan(run, dur)
    rp = replay_plan(run, dur)

    def fit(line, minimum):  # seconds for a still-ish segment carrying one line
        return max(minimum, dur[line] + 1.6)

    segs = [  # name, frames, seconds, [(t within segment, line)]
        ("hook", lambda n: hook_frames(run, n), fit("hook", 3.0), [(0.5, "hook")]),
        ("problem", lambda n: problem_frames(run, n), fit("problem", 9.0), [(0.5, "problem")]),
        ("race", lambda n: race_frames(run, views, race_sch, "Mirror-trap test",
                                       "Real home (Matterport scan) in Habitat-Sim. Same start for both; the goal is "
                                       "behind the mirror."), race_sch.length, race_cues),
        ("replay", lambda n: replay_frames(run, views, rp), rp["length"], [(0.4, "replay")]),
        ("how", lambda n: how_frames(run, n), fit("how", 9.5), [(0.6, "how")]),
        ("latency", lambda n: latency_frames(run, bench_ms, n), fit("latency", 7.0), [(0.6, "latency")]),
        ("credits", lambda n: credits_frames(run, n), fit("credits", 5.0), [(0.5, "credits")]),
    ]
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
