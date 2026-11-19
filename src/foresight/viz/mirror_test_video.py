"""FORESIGHT · The Mirror Test: a narrated square 1920x1920 demo rendered in NVIDIA Isaac Sim (path traced).

A Unitree H1 humanoid walks up to the full-height hallway mirror of HM3D home 00808 and moves in front of it
(tools/isaac/mirror_test.py: scripted, not a closed loop). Its head-camera frames go through the two models FORESIGHT
navigates with, afterwards (tools/isaac/mirror_test_perception.py): Depth Anything V2 and Qwen3-VL-2B. Then the
Phantom Room (tools/isaac/phantom_room.py): DA V2's depth of the mirror as a point cloud in a cut-away of the house,
and the same points after the mirror fill.

Segments: open -> hook (cinematic chase) -> test (chase + head camera + depth + VLM readouts, phase by phase) ->
phantom room -> findings -> close. Every number on screen or in the voice-over comes from perception.json /
phantom_frames.json (and docs/results/foresight_B.md for the one quoted evaluation result, labelled as such).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from foresight.viz.mirror_trap_video import Voice, _wrap
from foresight.viz.showcase_video import (AMBER, BLUE, CYAN, DIM, FPS, G, GREEN, MUTED, NAVY, PANEL, RED, TEXT, S, Sched,
                                          Writer, _lay, box, canvas, chip, corner_brand, depth_view, ease, frames,
                                          logo_layer, paste, ramp, speed_badge, text, tracked, F, _PREVIEW)

PHASE_CAPTION = {
    "approach": "Test 1 · walk in: does it come to meet me?",
    "look": "Looking",
    "turn": "Test 2 · turn: does it turn with me?",
    "step": "Watching",
    "close": "Close up: the mirror fills the view",
}


class Run:
    def __init__(self, root: Path, chase_alt: Path | None = None, alt_until: float = 5.6):
        """chase_alt: a second render of the same scripted walk-up (tools/isaac/mirror_test.py --chase corner) whose
        chase frames replace the main render's before `alt_until` (the main side camera starts inside a wall)."""
        self.root = root
        self.alt, self.alt_until = None, alt_until
        if chase_alt is not None and (chase_alt / "frames.json").exists():
            self.alt = chase_alt
            self.alt_t = np.array([f["t"] for f in json.loads((chase_alt / "frames.json").read_text())["frames"]])
        self.meta = json.loads((root / "frames.json").read_text())
        self.fr = self.meta["frames"]
        self.per = {r["i"]: r for r in json.loads((root / "perception.json").read_text())["frames"]}
        self.t = np.array([f["t"] for f in self.fr])
        self.vlm = [r for r in self.per.values() if "p_mirror" in r]
        self.vt = np.array([r["t"] for r in self.vlm])
        self.growth = growth_series(self)

    def idx(self, t):
        return int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))

    def img(self, kind, i):
        return Image.open(self.root / kind / f"{i:05d}.jpg").convert("RGB")

    def img_at(self, kind, t):
        """Frame at sim time t, blending the two rendered frames around it (smooth slow motion)."""
        root, ts = self.root, self.t
        if kind == "chase" and self.alt is not None and t < self.alt_until:
            root, ts = self.alt, self.alt_t
        j = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 1))
        ld = lambda k: Image.open(root / kind / f"{k:05d}.jpg").convert("RGB")
        if j + 1 >= len(ts):
            return ld(j)
        u = float(np.clip((t - ts[j]) / max(ts[j + 1] - ts[j], 1e-6), 0, 1))
        a = ld(j)
        return a if u < 0.02 else Image.blend(a, ld(j + 1), u)

    def depth(self, i):
        return np.load(self.root / "da" / f"{i:05d}.npy").astype(np.float32)

    def gt(self, i):
        return np.load(self.root / "eye_depth" / f"{i:05d}.npy").astype(np.float32)

    def mask(self, i):
        return np.asarray(Image.open(self.root / "eye_mirror" / f"{i:05d}.png")) > 127

    def vlm_at(self, t):
        """Latest VLM readout at or before t (what the robot would know by then)."""
        k = int(np.searchsorted(self.vt, t, side="right")) - 1
        return self.vlm[max(k, 0)]

    def phase_span(self, ph):
        ts = [f["t"] for f in self.fr if f["phase"] == ph]
        return (min(ts), max(ts)) if ts else (0.0, 0.0)


MIRROR_X = -0.07  # the hallway mirror plane (Isaac ground-truth depth on the mirror pixels)


def growth_series(run, clip_px=950):
    """Walk-up self-motion test. Qwen's robot box height on each approach look (until the box reaches the image
    bottom) against two predictions scaled from the first look: the robot's own reflection (distance 2 x the
    distance to the mirror, so it closes in twice as fast) and a robot standing still where the reflection first was."""
    out = []
    for r in run.vlm:
        if r["phase"] != "approach" or not r.get("box_robot") or r["box_robot"][0][3] >= clip_px:
            continue
        c = run.fr[r["i"]]["cam_pos"][0]
        out.append({"t": r["t"], "c": c, "h_obs": r["box_robot"][0][3] - r["box_robot"][0][1]})
    c0, h0 = out[0]["c"], out[0]["h_obs"]
    x_still = 2 * MIRROR_X - c0
    for o in out:
        o["h_refl"] = h0 * (MIRROR_X - c0) / (MIRROR_X - o["c"])
        o["h_still"] = h0 * (x_still - c0) / (x_still - o["c"])
    return out


def cover(im, w, h):
    W, H = im.size
    s = max(w / W, h / H)
    im = im.resize((int(W * s + 0.5), int(H * s + 0.5)), Image.LANCZOS)
    x0, y0 = (im.width - w) // 2, (im.height - h) // 2
    return im.crop((x0, y0, x0 + w, y0 + h))


def caption_band(img, y, s, alpha=1.0, size=44):
    w = ImageDraw.Draw(img).textlength(s, font=F("SemiBold", size)) + 64
    box(img, ((S - w) / 2, y, (S + w) / 2, y + size + 40), (10, 14, 28, int(200 * alpha)), radius=(size + 40) // 2)
    text(img, (S / 2, y + (size + 40) / 2), s, F("SemiBold", size), TEXT, alpha, anchor="mm")


def bar(img, x, y, w, label, p, color, sub=None):
    text(img, (x, y), label, F("SemiBold", 30), TEXT)
    text(img, (x + w, y), f"{p:.2f}", F("Bold", 30), color, anchor="ra")
    box(img, (x, y + 46, x + w, y + 64), (255, 255, 255, 40), radius=9)
    if p > 0.005:
        box(img, (x, y + 46, x + max(18, w * p), y + 64), color + (255,), radius=9)
    if sub:
        text(img, (x, y + 76), sub, F("Regular", 24), MUTED)


# ------------------------------------------------------------------------------------------------- segments


def seg_open(n, cues):
    for k in frames(n):
        t = k / FPS
        img = canvas()
        a = ramp(t, 0.2, 0.8)
        paste(img, logo_layer(300, progress=ease(t / 1.6)), ((S - 300) / 2, 470), a)
        tracked(img, S / 2, 830, "FORESIGHT", F("Bold", 120), 22, TEXT, a)
        text(img, (S / 2, 1040), "The Mirror Test", F("SemiBold", 76), TEXT, ramp(t, 1.0, 0.7), anchor="mm")
        text(img, (S / 2, 1140), "Ray-traced in NVIDIA Isaac Sim · a humanoid meets its reflection", F("Regular", 40),
             MUTED, ramp(t, 1.5, 0.7), anchor="mm")
        yield img


def seg_hook(n, cues, run: Run, sched: Sched):
    """Full-bleed cinematic chase view, title lines over it."""
    for k in frames(n):
        tv = k / FPS
        ts = sched.sim(tv)
        i = run.idx(ts)
        img = canvas()
        paste(img, cover(run.img_at("chase", ts), S, S), (0, 0))
        # vignette band top and bottom for legibility
        grad = Image.new("L", (1, 512))
        grad.putdata([int(210 * (1 - y / 511) ** 1.6) for y in range(512)])
        top = Image.new("RGBA", (S, 512), NAVY + (255,))
        top.putalpha(grad.resize((S, 512)))
        img.alpha_composite(top, (0, 0))
        bot = Image.new("RGBA", (S, 512), NAVY + (255,))
        bot.putalpha(grad.resize((S, 512)).transpose(Image.FLIP_TOP_BOTTOM))
        img.alpha_composite(bot, (0, S - 512))
        corner_brand(img)
        text(img, (G * 3, 150), "In 1970, a mirror was put in front of chimpanzees.", F("SemiBold", 54), TEXT,
             ramp(tv, cues.get("hook1", 0.3), 0.6))
        text(img, (G * 3, 228), "Would they recognise themselves?", F("Regular", 48), MUTED, ramp(tv, cues.get("hook1", 0.3) + 1.6, 0.6))
        caption_band(img, S - 190, "Unitree H1 · HM3D home 00808 · path traced in Isaac Sim 6.0.1", ramp(tv, 0.6, 0.6), size=36)
        speed_badge(img, S - G * 3, 70, sched.speed(tv))
        yield img


def depth_panel(run: Run, i, w, h):
    d = run.depth(i)
    im = depth_view(d, w, h, dmax=8.0).convert("RGBA")
    m = run.mask(i)
    if m.any():
        mm = Image.fromarray((m * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
        edge = mm.filter(__import__("PIL.ImageFilter", fromlist=["FIND_EDGES"]).FIND_EDGES)
        ov = Image.new("RGBA", (w, h), (255, 255, 255, 0))
        ov.putalpha(edge.point(lambda v: 255 if v > 0 else 0))
        im.alpha_composite(ov)
    return im


def seg_test(n, cues, run: Run, sched: Sched, ph_cues):
    chase_h = 1080
    y_chase = 262
    y_row = y_chase + chase_h + 28
    ph = S - y_row - 40
    pw = ph
    x_eye, x_dep = G * 3, G * 3 + pw + 24
    x_vlm = x_dep + pw + 24
    vw = S - G * 3 - x_vlm
    for k in frames(n):
        tv = k / FPS
        ts = sched.sim(tv)
        i = run.idx(ts)
        f, r = run.fr[i], run.per[i]
        q = run.vlm_at(ts)
        img = canvas()
        corner_brand(img)
        text(img, (G * 3, 132), "The Mirror Test", F("Bold", 66), TEXT)
        text(img, (G * 3, 214), "Isaac Sim, path traced · perception: Depth Anything V2 + Qwen3-VL-2B, run on these frames",
             F("Regular", 32), MUTED)
        text(img, (S - G * 3, 70), f"t = {ts:.1f} s", F("Bold", 44), TEXT, anchor="ra")
        speed_badge(img, S - G * 3, 132, sched.speed(tv))
        paste(img, cover(run.img_at("chase", ts), S - G * 6, chase_h), (G * 3, y_chase))
        caption_band(img, y_chase + chase_h - 100, PHASE_CAPTION.get(f["phase"], ""), size=40)
        # head camera
        box(img, (x_eye - 6, y_row - 6, x_eye + pw + 6, y_row + ph + 6), PANEL + (255,), radius=14)
        paste(img, cover(run.img_at("eye", ts), pw, ph), (x_eye, y_row))
        chip(img, x_eye + 14, y_row + 14, "Head camera", (12, 16, 30), F("Bold", 28))
        # DA depth
        box(img, (x_dep - 6, y_row - 6, x_dep + pw + 6, y_row + ph + 6), PANEL + (255,), radius=14)
        paste(img, depth_panel(run, i, pw, ph), (x_dep, y_row))
        chip(img, x_dep + 14, y_row + 14, "DA V2 depth", (12, 16, 30), F("Bold", 28))
        if "mirror_da_m" in r:
            box(img, (x_dep, y_row + ph - 116, x_dep + pw, y_row + ph), (10, 14, 28, 215), radius=0)
            text(img, (x_dep + 18, y_row + ph - 104), f"mirror read at {r['mirror_da_m']:.1f} m", F("Bold", 32), RED)
            text(img, (x_dep + 18, y_row + ph - 58), f"the glass is at {r['mirror_gt_m']:.1f} m", F("Regular", 30), TEXT)
        # VLM panel
        box(img, (x_vlm, y_row - 6, x_vlm + vw, y_row + ph + 6), PANEL + (255,), radius=14)
        text(img, (x_vlm + 28, y_row + 18), "Qwen3-VL-2B asks itself", F("Bold", 32), TEXT)
        bw = vw - 56
        bar(img, x_vlm + 28, y_row + 66, bw, "Is there a mirror?", q["p_mirror"], AMBER)
        bar(img, x_vlm + 28, y_row + 136, bw, "Is there a robot?", q["p_robot"], BLUE)
        bar(img, x_vlm + 28, y_row + 206, bw, "Is that robot me, in a mirror?", q["p_self"], GREEN)
        # walk-up test: how fast the "other robot" grows in the image (Qwen's box height) vs the two predictions
        sy = y_row + 286
        text(img, (x_vlm + 28, sy), "Walking in: how fast does it grow? (box height)", F("SemiBold", 26), MUTED)
        sx0, sx1, sy0, sy1 = x_vlm + 28, x_vlm + vw - 28, sy + 40, y_row + ph - 18
        box(img, (sx0, sy0, sx1, sy1), (12, 16, 30, 255), radius=10)
        d = ImageDraw.Draw(img)
        g = run.growth
        ga, gb = g[0]["t"], g[-1]["t"]
        hmax = max(max(r["h_obs"], r["h_refl"]) for r in g) * 1.04
        hmin = min(min(r["h_obs"], r["h_still"]) for r in g) * 0.92

        def X(tt):
            return sx0 + 14 + (sx1 - sx0 - 28) * (tt - ga) / max(gb - ga, 1e-6)

        def Yh(h):
            return sy1 - 12 - (sy1 - sy0 - 52) * (h - hmin) / (hmax - hmin)

        upto = [r for r in g if r["t"] <= ts]
        if len(upto) > 1:
            d.line([(X(r["t"]), Yh(r["h_still"])) for r in upto], fill=(220, 220, 230), width=4)
            d.line([(X(r["t"]), Yh(r["h_refl"])) for r in upto], fill=GREEN, width=4)
        for r in upto:
            xx, yy = X(r["t"]), Yh(r["h_obs"])
            d.ellipse((xx - 6, yy - 6, xx + 6, yy + 6), fill=AMBER)
        lx = sx0 + 14
        for lab, colr in (("● Qwen's box", AMBER), ("— my reflection", GREEN), ("— a robot standing still", (220, 220, 230))):
            text(img, (lx, sy0 + 10), lab, F("SemiBold", 22), colr)
            lx += d.textlength(lab, font=F("SemiBold", 22)) + 22
        yield img


def seg_phantom(n, cues, root: Path, meta, run: Run):
    fps = meta["fps"]
    nf = len(meta["frames"])
    y0 = 262
    h = S - y0 - 230
    for k in frames(n):
        tv = k / FPS
        j = int(np.clip(round(tv * fps), 0, nf - 1))
        fr = meta["frames"][j]
        img = canvas()
        corner_brand(img)
        text(img, (G * 3, 132), "The Phantom Room", F("Bold", 66), TEXT)
        text(img, (G * 3, 214), "What Depth Anything V2 believes is behind the mirror · head-camera depth, back-projected",
             F("Regular", 32), MUTED)
        paste(img, cover(Image.open(root / "frames" / f"{j:05d}.jpg").convert("RGB"), S - G * 6, h), (G * 3, y0))
        u = fr["fill"]
        yl = y0 + h + 40
        x = G * 3
        x += chip(img, x, yl, "real room · DA V2 depth", (60, 70, 100), F("SemiBold", 28)) + 18
        if u < 0.999:
            x += chip(img, x, yl, "mirror read as a room behind the wall", RED, F("SemiBold", 28)) + 18
        if u > 0.001:
            chip(img, x, yl, "after the fill: on the glass", (30, 150, 190), F("SemiBold", 28))
        yield img


def seg_findings(n, cues, rows):
    for k in frames(n):
        t = k / FPS
        img = canvas()
        corner_brand(img)
        text(img, (G * 3, 132), "What the robot learned", F("Bold", 66), TEXT)
        text(img, (G * 3, 214), "One scripted encounter, perception run afterwards on the rendered frames", F("Regular", 32), MUTED)
        y = 330
        for j, (big, color, head, sub) in enumerate(rows):
            a = ramp(t, cues.get(f"find{j + 1}", 0.5 + j), 0.6)
            box(img, (G * 3, y, S - G * 3, y + 300), PANEL + (int(255 * a),), radius=22)
            dd = ImageDraw.Draw(img)
            size = 104
            while size > 48 and dd.textlength(big, font=F("Bold", size)) > 580:
                size -= 4
            text(img, (G * 3 + 40, y + 150), big, F("Bold", size), color, a, anchor="lm")
            x_t = G * 3 + 680
            for li, ln in enumerate(_wrap(dd, head, F("SemiBold", 42), S - G * 3 - 40 - x_t, 2)):
                text(img, (x_t, y + 50 + li * 54), ln, F("SemiBold", 42), TEXT, a)
            for li, ln in enumerate(_wrap(dd, sub, F("Regular", 30), S - G * 3 - 40 - x_t, 3)):
                text(img, (x_t, y + 172 + li * 38), ln, F("Regular", 30), MUTED, a)
            y += 340
        yield img


def seg_close(n, cues):
    refs = ["NVIDIA Isaac Sim 6.0.1 (RTX path tracing) · Unitree H1 walking policy", "HM3D (Ramakrishnan et al. 2021)",
            "Depth Anything V2 (Yang et al. 2024) · Qwen3-VL (Qwen Team 2025)",
            "Mirror self-recognition test: Gallup, Science 1970", "Voice: Kokoro-82M"]
    for k in frames(n):
        t = k / FPS
        img = canvas()
        paste(img, logo_layer(320), ((S - 320) / 2, 400))
        tracked(img, S / 2, 790, "FORESIGHT", F("Bold", 120), 22, TEXT)
        text(img, (S / 2, 990), "Map-free navigation from a single camera · simulation", F("Regular", 44), MUTED, anchor="mm")
        a = ramp(t, 0.8, 0.6)
        for i, r in enumerate(refs):
            text(img, (S / 2, 1220 + i * 56), r, F("Regular", 32), DIM, a, anchor="mm")
        yield img


# ------------------------------------------------------------------------------------------------- assembly


def render(run_dir: Path, phantom_dir: Path, out_path: Path, lines: dict, rows, tts_python, kokoro_home, voice,
           preview=None, hook_sched=None, test_sched=None, chase_alt=None):
    import subprocess
    run = Run(run_dir, chase_alt)
    pmeta = json.loads((phantom_dir / "phantom_frames.json").read_text())
    vo = Voice(lines, out_path.parent / "narration", tts_python, kokoro_home, voice)
    dur = vo.dur
    segs = []

    def add(name, make, ls, seconds_min, start=0.5, anchors=None, tail=1.0, fade=(True, True)):
        cues, end = _lay(ls, dur, start, anchors=anchors)
        seconds = max(seconds_min, end + tail - 0.45) if ls else seconds_min
        segs.append((name, make, cues, seconds, fade))

    add("open", lambda n, c: seg_open(n, c), [], 4.2)
    add("hook", lambda n, c: seg_hook(n, c, run, hook_sched), ["hook1", "hook2"], hook_sched.length, start=0.4)
    ts = test_sched
    add("test", lambda n, c: seg_test(n, c, run, ts, []),
        ["test1", "test2", "test3", "test4", "test5"], ts.length, start=0.4,
        anchors={"test2": ts.video_at(1.4), "test3": ts.video_at(run.phase_span("turn")[0]),
                 "test4": ts.video_at(run.phase_span("step")[0]), "test5": ts.video_at(run.phase_span("close")[0])})
    p_len = len(pmeta["frames"]) / pmeta["fps"]
    add("phantom", lambda n, c: seg_phantom(n, c, phantom_dir, pmeta, run), ["ph1", "ph2", "ph3"], p_len, start=0.5,
        anchors={"ph2": pmeta["t_reveal"][1] - 0.5, "ph3": pmeta["t_fill"][0]})
    add("findings", lambda n, c: seg_findings(n, c, rows), ["find1", "find2", "find3", "find4"], 12.0, start=0.6)
    add("close", lambda n, c: seg_close(n, c), ["close"], 6.0, start=0.6, tail=1.8)

    t_ = 0.0
    for name, _, cues, seconds, _ in segs:
        print(f"  {t_:6.1f}  {name:9s} {seconds:5.1f} s  " + " ".join(f"{k}@{v:.1f}" for k, v in cues.items()))
        t_ += seconds
    if preview is not None:
        t0 = 0.0
        for name, make, cues, seconds, fade in segs:
            if preview < t0 + seconds:
                _PREVIEW["k"] = int((preview - t0) * FPS)
                for im in make(int(round(seconds * FPS)), cues):
                    p = out_path.with_name(f"preview_{preview:06.1f}.png")
                    im.convert("RGB").save(p)
                    print(f"preview {name} @ {preview - t0:.1f} s -> {p}")
                    _PREVIEW["k"] = None
                    return
            t0 += seconds
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
    print(f"wrote {out_path} ({total:.1f} s)")
