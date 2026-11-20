"""FORESIGHT · Through the Glass: the narrated 1920x1920 demo of the live Unitree G1 runs in Isaac Sim.

Every robot motion on screen is a recorded live run (tools/isaac/g1_live.py: FORESIGHT's decisions, NVIDIA's G1
walking policy), re-rendered joint for joint, path traced (tools/isaac/replay_render.py). Labels are AR overlays
projected through each frame's real camera (foresight.viz.ar_overlay); their numbers come from the runs' own logs.
No file names, no panels (the user's standing rules for demos).
"""
from __future__ import annotations

import json
import math
import pickle
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from foresight.viz.ar_overlay import GOOD, HOLO, INK, WARN, Camera, Overlay, font
from foresight.viz.mirror_trap_video import SR, Voice
from foresight.viz.showcase_video import logo_layer

S, FPS = 1920, 30
BG = (8, 14, 24)
MIRROR_X = -0.07


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


class Shot:
    """A rendered shot: frames at `fps` per video second, with each frame's camera and the robot's pose."""

    def __init__(self, d: Path):
        self.d = d
        m = json.loads((d / "frames.json").read_text())
        self.fps, self.res, self.ap = m["fps"], m["res"], m["aperture"]
        self.f = m["frames"]
        self.n = len(self.f)
        self.length = self.n / self.fps

    def k(self, tv):
        return float(np.clip(tv * self.fps, 0, self.n - 1))

    def image(self, tv):
        k = self.k(tv)
        a, u = int(k), k - int(k)
        im = Image.open(self.d / f"{a:05d}.jpg").convert("RGB")
        if u > 0.02 and a + 1 < self.n:
            im = Image.blend(im, Image.open(self.d / f"{a + 1:05d}.jpg").convert("RGB"), u)
        return im.resize((S, S), Image.LANCZOS)

    def frame(self, tv):
        return self.f[int(round(self.k(tv)))]

    def sim_t(self, tv):
        k = self.k(tv)
        a = int(k)
        b = min(a + 1, self.n - 1)
        return self.f[a]["t"] + (self.f[b]["t"] - self.f[a]["t"]) * (k - a)

    def camera(self, tv):
        k = self.k(tv)
        a = int(k)
        b = min(a + 1, self.n - 1)
        u = k - a
        lerp = lambda key: (1 - u) * np.array(self.f[a][key]) + u * np.array(self.f[b][key])  # noqa: E731
        return Camera.from_focal(lerp("eye"), lerp("target"), self.f[a]["focal"], self.ap, S, S)


class Motion:
    def __init__(self, path):
        m = json.loads(Path(path).read_text())
        self.t = np.array([s["t"] for s in m["steps"]])
        self.pos = np.array([s["pos"] for s in m["steps"]])

    def at(self, t):
        return self.pos[int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))]

    def path(self, t0, t1, step=0.25):
        ts = np.arange(t0, t1, step)
        return np.array([[*self.at(t)[:2], 0.02] for t in ts]) if len(ts) else np.zeros((0, 3))


class Brain:
    def __init__(self, path):
        self.L = pickle.load(open(path, "rb"))
        self.T = self.L["ticks"]
        self.t = np.array([x["t"] for x in self.T])

    def at(self, t):
        return self.T[int(np.clip(np.searchsorted(self.t, t, side="right") - 1, 0, len(self.T) - 1))]

    def events(self):
        return [(x["t"], x["diag"].get("event")) for x in self.T if x["diag"].get("event")]

    def walls_isaac(self, t):
        """Walls on the map at sim time t, in Isaac xy (Habitat xz -> (x, -z))."""
        return [[(a[0], -a[1]), (b[0], -b[1])] for a, b in self.at(t)["walls"]]


def full(img):
    out = Image.new("RGBA", (S, S), BG + (255,))
    out.paste(img, (0, 0))
    return out


_VIG = {}


def vignette(img, strength=0.55):
    """Darken the edges a little (cinematic)."""
    if "m" not in _VIG:
        w = Image.new("L", (S // 8, S // 8), 0)
        ImageDraw.Draw(w).ellipse((-S // 32, -S // 32, S // 8 + S // 32, S // 8 + S // 32), fill=255)
        _VIG["m"] = w.filter(ImageFilter.GaussianBlur(S // 60)).resize((S, S), Image.BILINEAR)
        _VIG["k"] = Image.new("RGB", (S, S), (0, 0, 0))
    return Image.composite(img, Image.blend(img, _VIG["k"], strength), _VIG["m"])


PREVIEW = {"k": None}


def ks(n):
    """Frame indices of a segment: all, or just the previewed one."""
    return [PREVIEW["k"]] if PREVIEW["k"] is not None and PREVIEW["k"] < n else ([] if PREVIEW["k"] is not None else range(n))


def brand(img, alpha=1.0):
    im = img.convert("RGBA")
    paste = logo_layer(58).convert("RGBA")
    if alpha < 1:
        paste.putalpha(paste.getchannel("A").point(lambda v: int(v * alpha)))
    im.alpha_composite(paste, (48, 44))
    ImageDraw.Draw(im).text((120, 52), "FORESIGHT", font=font("display", 36, "SemiBold"), fill=(240, 246, 255, int(235 * alpha)))
    return im


def chip(img, text, alpha=1.0, y=150):
    """Small place chip under the brand (e.g. REAL HOME · 3D-SCANNED)."""
    d = ImageDraw.Draw(img)
    f = font("text", 30, "SemiBold")
    w = d.textlength(text, font=f) + 44
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ld.rounded_rectangle((48, y, 48 + w, y + 56), radius=28, fill=(8, 22, 36, int(170 * alpha)),
                         outline=HOLO + (int(200 * alpha),), width=2)
    ld.text((48 + 22, y + 12), text, font=f, fill=INK + (int(255 * alpha),))
    img.alpha_composite(layer)
    return img


class Film:
    def __init__(self, root: Path, rate: float = 0.8, music_title: str = ""):
        self.root, self.rate, self.music_title = root, rate, music_title
        fr = root / "results/isaac/film"
        self.sh = {}
        for g, names in (("home", ("h_duel", "h_check", "h_exit", "h_pov_ghost")), ("office", ("o_orbit", "o_walk", "o_pov"))):
            for n in names:
                for cand in (fr / g / n, fr / g / f"{n}_v1"):  # a re-shoot in progress falls back to its first version
                    if (cand / "frames.json").exists():
                        self.sh[n] = Shot(cand)
                        break
        live = root / "results/isaac/live"
        self.m_hero = Motion(live / "home_v1b/motion.json")
        self.m_ghost = Motion(live / "home_map_hab/motion.json")
        self.m_office = Motion(live / "office_map/motion.json")
        self.b_hero = Brain(live / "home_v1b/brain.pkl")
        self.b_ghost = Brain(live / "home_map_hab/brain.pkl")
        self.office_vlm = json.loads((fr / "office_vlm.json").read_text()) if (fr / "office_vlm.json").exists() else {}
        self.hero_end = float(self.m_hero.t[-1])
        self.crash_t = 28.8

    # ------------------------------------------------------------------ segments (generators of RGBA frames)
    def seg_open(self, n, cues):
        sh = self.sh["o_pov"]
        v = self.office_vlm
        for k in ks(n):
            tv = k / FPS * self.rate
            img = full(vignette(sh.image(tv * 0.9)))
            ov = Overlay(sh.camera(tv * 0.9))
            a = ease((tv - 1.4) / 0.6)
            if v:
                ov.label([-11.0, 40.3, 1.45], "GLASS", f"{v['p_glass']:.2f}", sub="the vision model, from this frame",
                         tone="warn", alpha=a, lead_px=90, rise_px=60)
            im = ov.compose(img, corner="SIMULATED IN NVIDIA ISAAC SIM").convert("RGBA")
            yield brand(im, ease((tv - 0.3) / 0.8))

    def seg_office(self, n, cues):
        a_len = self.sh["o_orbit"].length
        for k in ks(n):
            tv = k / FPS * self.rate
            if tv < a_len:
                sh, ts = self.sh["o_orbit"], tv
            else:
                sh, ts = self.sh["o_walk"], tv - a_len
            img = full(vignette(sh.image(ts)))
            ov = Overlay(sh.camera(ts))
            t = sh.sim_t(ts)
            p = self.m_office.at(t)
            if sh is self.sh["o_walk"]:
                ov.route(self.m_office.path(t + 0.3, min(t + 9.0, 54.0)), tone="info", width=7, alpha=0.9,
                         pulse=(tv * 0.35) % 1.0)
                ov.ring([-8.0, 32.5, 0.02], 0.4, "good", alpha=ease((ts - 1.0) / 0.8), t=tv)
                if ts > 1.0:
                    ov.label([-8.0, 32.5, 0.45], "GOAL", f"{np.hypot(-8.0 - p[0], 32.5 - p[1]):.1f} m", tone="good",
                             alpha=ease((ts - 1.0) / 0.8))
            else:
                ov.label([p[0] + 0.05, p[1], 1.32], "ONE CAMERA", None, sub="no depth sensor · no map given",
                         tone="info", side="left", alpha=ease((ts - 1.0) / 0.7), lead_px=100, rise_px=70)
                ov.label([-11.0, 38.6, 1.6], "GLASS WALL", None, tone="warn", alpha=ease((ts - 4.0) / 0.7),
                         lead_px=80, rise_px=50)
            im = ov.compose(img).convert("RGBA")
            im = brand(im)
            yield chip(im, "NVIDIA OFFICE", ease(tv / 0.8))

    def seg_pov(self, n, cues):
        sh = self.sh["h_pov_ghost"]
        for k in ks(n):
            tv = k / FPS * self.rate
            img = full(vignette(sh.image(tv)))
            ov = Overlay(sh.camera(tv))
            t = sh.sim_t(tv)
            b = self.b_ghost.at(t)
            da, gt = b["diag"].get("depth_ahead_m"), b.get("gt_ahead_m")
            rx, ry, yaw = sh.frame(tv)["robot"]
            cx, cy = rx + 0.1 * math.cos(yaw), ry + 0.1 * math.sin(yaw)
            facing = math.cos(yaw) > 0.6 and cx < MIRROR_X - 0.15
            if da is not None and gt is not None and tv > 0.8 and facing:
                ay = cy + math.tan(yaw) * (MIRROR_X - cx)  # where the line of sight meets the mirror
                ov.label([MIRROR_X, ay - 0.15, 1.0], "MIRROR", f"{gt:.1f} m", sub=f"depth from pixels reads {da:.1f} m",
                         tone="warn", side="left", alpha=ease((tv - 0.8) / 0.6), lead_px=110, rise_px=180)
            im = ov.compose(img).convert("RGBA")
            im = brand(im)
            yield chip(im, "REAL HOME · 3D-SCANNED", ease(tv / 0.8))

    def seg_duel(self, n, cues):
        sh = self.sh["h_duel"]
        for k in ks(n):
            tv = k / FPS * self.rate
            img = full(vignette(sh.image(tv)))
            ov = Overlay(sh.camera(tv))
            f = sh.frame(tv)
            t, tg = sh.sim_t(tv), f["t_ghost"] + (sh.sim_t(tv) - f["t"])
            g = self.m_ghost.at(tg)
            h = self.m_hero.at(t)
            ov.label([g[0], g[1], 1.15], "DEPTH ONLY", None, sub="same start, same goal", tone="info", side="right",
                     alpha=ease((tv - 0.5) / 0.6), lead_px=70, rise_px=60)
            if tg >= self.crash_t:
                ov.label([MIRROR_X, g[1], 1.0], "COLLISION", f"{self.crash_t:.1f} s", tone="warn",
                         alpha=ease((tg - self.crash_t) / 0.3), lead_px=80, rise_px=90)
            yield brand(ov.compose(img).convert("RGBA"))

    def seg_check(self, n, cues):
        sh = self.sh["h_check"]
        desc = next((x["diag"]["description"] for x in self.b_hero.T if x["diag"].get("description")
                     and "reflection" in x["diag"]["description"].lower()), None)
        quote = None
        if desc:
            s = desc.split("\n")[0]
            i = s.lower().find("appears to be")
            quote = ("…" + s[i:].split(";")[0].split("—")[0].rstrip(", .") + "…") if i >= 0 else s[:110] + "…"
        confirm_t = next((t for t, e in self.b_hero.events() if e == "confirmed"), None)
        for k in ks(n):
            tv = k / FPS * self.rate
            img = full(vignette(sh.image(tv)))
            ov = Overlay(sh.camera(tv))
            t = sh.sim_t(tv)
            h = self.m_hero.at(t)
            for (a, b) in self.b_hero.walls_isaac(t):
                ov.route([[a[0], a[1], 0.03], [b[0], b[1], 0.03]], tone="warn", width=9, alpha=0.95, fade_tail=False)
            if quote:
                ov.thought([h[0], h[1], 1.4], quote, alpha=ease((tv - 0.6) / 0.6) * (1 - ease((tv - 9.0) / 0.6)),
                           max_w_px=600)
            walls = self.b_hero.walls_isaac(t)
            if confirm_t is not None and t >= confirm_t and walls:
                (a, b) = walls[0]
                ov.label([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.05], "MIRROR AHEAD", "walled off", tone="warn",
                         alpha=ease((t - confirm_t) / 1.5), lead_px=90, rise_px=110)
            yield brand(ov.compose(img).convert("RGBA"))

    def seg_exit(self, n, cues):
        sh = self.sh["h_exit"]
        goal = np.array([2.4, -1.1])
        for k in ks(n):
            tv = k / FPS * self.rate
            img = full(vignette(sh.image(tv)))
            ov = Overlay(sh.camera(tv))
            t = sh.sim_t(tv)
            h = self.m_hero.at(t)
            ov.route(self.m_hero.path(t + 0.3, min(t + 10.0, self.hero_end)), tone="good", width=7, alpha=0.9,
                     pulse=(tv * 0.35) % 1.0)
            ov.ring([goal[0], goal[1], 0.02], 0.4, "good", t=tv)
            d = float(np.hypot(*(goal - h[:2])))
            done = t >= self.hero_end - 0.5
            ov.label([goal[0], goal[1], 0.45], "GOAL", "reached" if done else f"{d:.1f} m",
                     sub="0 collisions" if done else "behind the mirror", tone="good", lead_px=70, rise_px=60)
            yield brand(ov.compose(img).convert("RGBA"))

    def seg_end(self, n, cues):
        credits = ["NVIDIA Isaac Sim · Isaac Lab · Unitree G1 walking policy", "HM3D (Ramakrishnan et al. 2021)",
                   "Depth Anything V2 · Qwen3-VL · Laya", "Voice: Kokoro-82M",
                   f"Music: “{self.music_title}” by Kevin MacLeod (incompetech.com), CC BY 4.0"]
        for k in ks(n):
            tv = k / FPS
            im = Image.new("RGBA", (S, S), BG + (255,))
            a = ease(tv / 0.8)
            lg = logo_layer(300).convert("RGBA")
            lg.putalpha(lg.getchannel("A").point(lambda v: int(v * a)))
            im.alpha_composite(lg, ((S - 300) // 2, 520))
            d = ImageDraw.Draw(im)
            ft = font("display", 112, "SemiBold")
            w = d.textlength("FORESIGHT", font=ft)
            d.text(((S - w) / 2, 870), "FORESIGHT", font=ft, fill=(240, 246, 255, int(255 * a)))
            fs = font("text", 44, "Medium")
            tag = "One camera. No map. It sees the mirror."
            w = d.textlength(tag, font=fs)
            d.text(((S - w) / 2, 1030), tag, font=fs, fill=HOLO + (int(255 * ease((tv - 0.6) / 0.8)),))
            fc = font("text", 28, "Regular")
            for i, c in enumerate(credits):
                w = d.textlength(c, font=fc)
                d.text(((S - w) / 2, 1330 + i * 48), c, font=fc, fill=(150, 168, 190, int(255 * ease((tv - 1.2) / 0.8))))
            yield im


def mix_music(voice_wav: Path, music: Path, out_wav: Path, total_s: float, cues_s):
    """Music bed under the voice: -16 dB, ducked a further 8 dB while a line plays, 2 s fades."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(music), "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    m = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    n = int(total_s * SR)
    m = np.resize(m, n) if len(m) < n else m[:n]
    with wave.open(str(voice_wav)) as w:
        v = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    v = np.pad(v, (0, max(0, n - len(v))))[:n]
    env = np.ones(n) * 10 ** (-16 / 20)
    talk = np.zeros(n)
    for a, b in cues_s:
        talk[int(a * SR):int(b * SR)] = 1.0
    k = int(0.4 * SR)
    talk = np.convolve(talk, np.ones(k) / k, mode="same")
    env *= 10 ** (-8 * talk / 20)
    fade = int(2 * SR)
    env[:fade] *= np.linspace(0, 1, fade)
    env[-fade:] *= np.linspace(1, 0, fade)
    mix = v + m * env
    mix *= 0.9 / max(1e-6, float(np.abs(mix).max()))
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())
