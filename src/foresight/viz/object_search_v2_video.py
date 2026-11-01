"""Square 1920x1920 narrated video for FORESIGHT v2 (fast/slow mirror handling) on object search.

Segments: hook (the spoken instruction) -> recap (what v1 traded: fewer mirror crashes for slow looks, from the first
evaluation set) -> how v2 works (a cheap mirror check on every look; the slow scene note + Laya only for a mirror in
the robot's way) -> the search (baseline vs v2 side by side, same panels as the v1 video) -> the same v2 run replayed
in NVIDIA Isaac Sim (`foresight.viz.isaac_replay`) -> the fresh evaluation set's scoreboard (baseline, v1, v2) ->
credits. Kokoro-82M voice-over, worded from what the runs did; every number comes from saved runs or eval.json.

Panels, schedule and voice helpers are shared with `foresight.viz.object_search_video`.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from foresight.viz import object_search_video as v1
from foresight.viz.isaac_replay import IsaacReplay
from foresight.viz.mirror_trap_video import (AMBER, BG, BLUE, DIM, FPS, GREEN, MUTED, PANEL, RED, S, TEXT, Voice,
                                             _arrow, _banner, _box, _centered, _place, _spoken_int, _text_block,
                                             _Views, _Writer, _font)
from foresight.viz.object_search_video import (BOT_Y, INSET, LX, PANEL_Y, PW, RX, Run, Scene, Schedule, _finish,
                                               _first_person, _map, _name, _pieces, _strip, eval_summary)

V2_COND = "foresight_fs"
COND_NAMES = {"baseline": "Baseline", "foresight": "FORESIGHT v1", V2_COND: "FORESIGHT v2"}
NOTE = ("Same explorer, object spotting, close-in arrival and safety layer for every robot; the versions differ only in "
        "when Qwen's slow scene note runs. Crash = driving forward within 0.3 m of a surface (simulator ground truth). "
        "Small samples, one simulator: read the counts, not the percentages.")


# ---------------------------------------------------------------- panels


def _status_v2(run, t, near_miss=None):
    head, color, banner = v1._status(run, t, True, near_miss)
    st = run.step(t)
    if st.get("verifying") and not (run.crash is not None and t >= run.crash):
        return "Mirror in its path: stopped, asking for a scene note", AMBER, ("MIRROR IN ITS PATH? CHECKING", (196, 120, 0, 230))
    if st.get("mode") == "close_in":
        return f"Confirmed: closing in on the {_name(st.get('target_label'))}", GREEN, None
    return head, color, banner


def _fast_check(run, t):
    """Latest look's P(yes) to "is there a mirror?" (asked on every look)."""
    lk = run.look(t)
    if lk is None:
        return None
    return lk[1].get("present", {}).get("mirror")


def _hud_v2(img, run, t, x):
    st = run.step(t)
    w = PW - INSET - 60
    h = 330
    y = PANEL_Y + PW - h - 20
    _box(img, (x + 18, y, x + 18 + w, y + h), (8, 10, 14, 215), radius=14)
    d = ImageDraw.Draw(img)
    d.text((x + 38, y + 18), "Speed", font=_font("Regular", 26), fill=MUTED)
    d.text((x + 140, y + 12), f"{max(0.0, st['linear_velocity']):.2f} m/s", font=_font("Bold", 38), fill=TEXT)
    p = _fast_check(run, t)
    d.text((x + 38, y + 72), "Every look: mirror?", font=_font("Regular", 26), fill=MUTED)
    bx0, bx1 = x + 300, x + w - 120
    d.rounded_rectangle((bx0, y + 78, bx1, y + 98), 10, fill=DIM)
    if p is not None:
        d.rounded_rectangle((bx0, y + 78, bx0 + max(20, int((bx1 - bx0) * p)), y + 98), 10, fill=AMBER if p > 0.5 else GREEN)
        d.text((bx1 + 14, y + 68), f"{p:.0%}", font=_font("Bold", 32), fill=TEXT)
    checks = [(tt, e, xx) for tt, e, xx in run.mirror if tt <= t and e in ("confirmed", "dismissed")]
    if st.get("verifying"):
        d.text((x + 38, y + 118), "Slow check: Qwen writes a note…", font=_font("Bold", 28), fill=AMBER)
    elif checks:
        tt, e, xx = checks[-1]
        col = RED if e == "confirmed" else GREEN
        verdict = "mirror, walled off" if e == "confirmed" else "no mirror, go on"
        d.text((x + 38, y + 118), f"Slow check: Laya {xx['p_hazard']:.2f}, {verdict}", font=_font("Bold", 26), fill=col)
    else:
        d.text((x + 38, y + 118), "No mirror in its path yet: no slow check", font=_font("Regular", 26), fill=MUTED)
    note = st.get("description")
    if note:
        _text_block(d, (x + 38, y + 160), "Qwen: “" + " ".join(note.split()) + "”", _font("Italic", 24), TEXT, w - 40, 5)


def _hud_base(img, run, t, x):
    v1._hud(img, run, t, x, False)


def race_frames(scene, views, base, fore, sch, instructions, start_xz, mirror_xz, place):
    labels = list(dict.fromkeys(instructions))
    near = None if base.crash is not None else base.mirror_encounter(mirror_xz)
    for k in range(int(sch.length * FPS)):
        t, speed = sch.sim(k / FPS)
        img = Image.new("RGBA", (S, S), BG)
        d = ImageDraw.Draw(img)
        d.text((LX, 18), "FORESIGHT", font=_font("Black", 66), fill=TEXT)
        d.text((LX + d.textlength("FORESIGHT", font=_font("Black", 66)) + 22, 38), "Object search · v2.1",
               font=_font("Medium", 46), fill=MUTED)
        d.text((LX, 100), place, font=_font("Regular", 28), fill=TEXT)
        clock = f"t = {t:.1f} s"
        cw = d.textlength(clock, font=_font("Bold", 54))
        d.text((S - 16 - cw, 30), clock, font=_font("Bold", 54), fill=TEXT)
        if speed is not None and abs(speed - 1.0) > 0.02:
            badge = f"{speed:.0f}× speed" if speed >= 2 else f"{speed:.1f}× speed"
            bw = d.textlength(badge, font=_font("Bold", 32)) + 36
            x1 = S - 16 - cw - 24
            d.rounded_rectangle((x1 - bw, 38, x1, 90), 26, fill=AMBER if speed > 1 else DIM)
            d.text((x1 - bw + 18, 45), badge, font=_font("Bold", 32), fill=(16, 18, 22) if speed > 1 else TEXT)
        for x, title, sub in [(LX, "BASELINE", "explorer + Qwen spots objects"),
                              (RX, "FORESIGHT v2.1", "same, + mirror check; Laya when one's in the way")]:
            d.text((x, 146), title, font=_font("Black", 34), fill=TEXT)
            d.text((x + d.textlength(title, font=_font("Black", 34)) + 16, 152), sub, font=_font("Regular", 26), fill=MUTED)
        for run, x, color, fs in [(base, LX, RED, False), (fore, RX, GREEN, True)]:
            tt = min(t, run.end)
            nm = None if fs else near
            _first_person(img, views, run, tt, x, labels)
            _, _, banner = _status_v2(run, tt) if fs else v1._status(run, tt, False, nm)
            if banner:
                _banner(img, x, PANEL_Y, PW, banner[0], banner[1])
            (_hud_v2 if fs else _hud_base)(img, run, tt, x)
            _strip(img, run, tt, x, fs, instructions, nm)
            if fs:  # the strip's headline, v2 wording
                head, col, _ = _status_v2(run, tt)
                _box(img, (x + 8, v1.STRIP_Y + 8, x + PW * 0.62, v1.STRIP_Y + v1.STRIP_H - 8), PANEL + (255,), radius=10)
                ImageDraw.Draw(img).text((x + 24, v1.STRIP_Y + 28), head, font=_font("Bold", 34), fill=col)
            _map(img, scene, run, tt, (x, BOT_Y), color, start_xz, mirror_xz)
        yield img.convert("RGB")


# ---------------------------------------------------------------- slides


def recap_frames(old, n):
    """The first evaluation set's result that motivated v2 (baseline vs v1)."""
    b, f = old["baseline"], old["foresight"]
    for k in range(n):
        t = k / FPS
        img = Image.new("RGBA", (S, S), BG)
        d = ImageDraw.Draw(img)
        d.text((100, 110), "Last time", font=_font("Bold", 92), fill=TEXT)
        d.text((100, 230), f"FORESIGHT v1 vs a baseline, {b['n']} fixed episodes in real scanned homes (Habitat-Sim)",
               font=_font("Regular", 36), fill=MUTED)
        rows = [("Runs ending in a crash", f"{b['crash']}", f"{f['crash']}", GREEN),
                ("Ran out of time", f"{b['timeout']}", f"{f['timeout']}", RED),
                ("Seconds per look", f"{old['look_s']['baseline']:.1f}", f"{old['look_s']['foresight']:.1f}", RED)]
        d.text((1080, 380), "Baseline", font=_font("Bold", 48), fill=MUTED)
        d.text((1450, 380), "v1", font=_font("Bold", 48), fill=MUTED)
        for i, (lab, bv, fv, col) in enumerate(rows):
            if t < 0.6 + i * 1.6:
                continue
            y = 480 + i * 170
            d.rounded_rectangle((90, y - 16, S - 90, y + 126), 18, fill=PANEL)
            d.text((130, y + 26), lab, font=_font("Medium", 52), fill=TEXT)
            d.text((1080, y + 12), bv, font=_font("Black", 76), fill=TEXT)
            d.text((1450, y + 12), fv, font=_font("Black", 76), fill=col)
        if t >= 5.4:
            _text_block(d, (100, 1080), "v1 asked Qwen for a full scene description on every look, so Laya could check it "
                        "for mirrors. It stopped the mirror crashes, but each look took seconds, and the robot often ran "
                        "out of time.", _font("Regular", 46), TEXT, S - 200, 5)
        if t >= 8.0:
            _box(img, (90, 1440, S - 90, 1640), (86, 160, 255, 60), radius=24)
            d = ImageDraw.Draw(img)
            _text_block(d, (130, 1480), "v2: pay for the slow description only when a mirror is actually in the "
                        "robot's way.", _font("Bold", 52), TEXT, S - 260, 3)
        yield img.convert("RGB")


def how_frames(n, fast_ms, slow_ms, laya_ms, depth_ms, slow_frac):
    bw, bh = 640, 150
    boxes = {
        "cam": ((960, 330), "Camera (RGB only)", "", TEXT, 0.0),
        "depth": ((480, 590), "Depth Anything V2", f"depth · map · safety · {depth_ms:.0f} ms", AMBER, 0.6),
        "fast": ((1440, 590), "Qwen3-VL, every look", f"object? where? mirror? · {fast_ms / 1000:.1f} s", BLUE, 1.6),
        "cand": ((1440, 860), "Mirror box + depth", "a candidate wall, if it reads too deep", BLUE, 3.0),
        "gate": ((960, 1120), "Is it in the robot's way?", "route or heading crosses it", AMBER, 4.6),
        "slow": ((480, 1380), "Stop · Qwen scene note", f"only now · {slow_ms / 1000:.1f} s", RED, 6.2),
        "laya": ((1440, 1380), "Laya decides", f"mirror? · {laya_ms:.0f} ms · wall or dismiss", GREEN, 7.4),
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
        d.text((60, 50), "How v2 handles mirrors", font=_font("Bold", 84), fill=TEXT)
        d.text((60, 150), "Fast check always, slow check on demand. Timings measured on a laptop RTX 4060.",
               font=_font("Regular", 36), fill=MUTED)
        if t >= 0.6:
            d.line([(edge("cam", "left")[0], 330), (480, 330)], fill=grey, width=6)
            _arrow(d, (480, 330), edge("depth", "top"), grey)
        if t >= 1.6:
            d.line([(edge("cam", "right")[0], 330), (1440, 330)], fill=grey, width=6)
            _arrow(d, (1440, 330), edge("fast", "top"), grey)
        if t >= 3.0:
            _arrow(d, edge("fast", "bottom"), edge("cand", "top"), grey)
        if t >= 4.6:
            _arrow(d, edge("cand", "bottom"), (edge("gate", "right")[0] - 40, edge("gate", "top")[1]), grey)
            _arrow(d, edge("depth", "bottom"), (edge("gate", "left")[0] + 40, edge("gate", "top")[1]), grey)
        if t >= 6.2:
            _arrow(d, (edge("gate", "left")[0] + 60, edge("gate", "bottom")[1]), edge("slow", "top"), grey)
            d.text((560, 1215), "yes", font=_font("Bold", 30), fill=AMBER)
        if t >= 7.4:
            _arrow(d, edge("slow", "right"), edge("laya", "left"), grey)
        for key, ((cx, cy), title, sub, color, ta) in boxes.items():
            if t < ta:
                continue
            _box(img, (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), PANEL + (255,))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), 16, outline=color, width=4)
            _centered(d, cx - bw / 2, bw, cy - (50 if sub else 24), title, _font("Bold", 42), TEXT)
            if sub:
                _centered(d, cx - bw / 2, bw, cy + 10, sub, _font("Medium", 28), color)
        if t >= 8.6:
            _text_block(d, (60, 1560), f"On the evaluation runs, {slow_frac:.0%} of v2's looks needed the slow note. "
                        "The two signals that turn a mirror into a wall are the same as v1's double check: a mirror the "
                        "fast check boxed, and Laya reading Qwen's description of that view.",
                        _font("Regular", 40), TEXT, S - 120, 5)
        yield img.convert("RGB")


def isaac_frames(replay: IsaacReplay, plan, n):
    """plan: Schedule over the replay's sim time; frame k of the replay is at replay.frames[k]['t']."""
    ts = np.array([f["t"] for f in replay.frames])
    for k in range(n):
        t, speed = plan.sim(k / FPS)
        i = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
        yield replay.frame(i, speed=speed or 1.0)


def score_frames(summary, conds, title, lines, note, n, callout=None, names=None):
    names = names or COND_NAMES
    cols = {c: (820 + i * 380 if len(conds) == 3 else 900 + i * 480) for i, c in enumerate(conds)}
    rows = [("Found it (within 1 m)", lambda c: f"{c['success']}/{c['n']}"),
            ("Path efficiency (SPL)", lambda c: f"{c['spl']:.2f}"),
            ("Ran out of time", lambda c: f"{c['timeout']}"),
            ("Runs ending in a crash", lambda c: f"{c['crash']}"),
            ("…into a mirror", lambda c: f"{c['mirror_crash']}"),
            ("Seconds per look (median)", None)]
    img = Image.new("RGB", (S, S), BG)
    d = ImageDraw.Draw(img)
    d.text((100, 100), title, font=_font("Bold", 84), fill=TEXT)
    for i, ln in enumerate(lines):
        d.text((100, 214 + i * 46), ln, font=_font("Regular", 34), fill=MUTED)
    colors = {"baseline": RED, "foresight": AMBER, V2_COND: GREEN}
    for c, x in cols.items():
        _centered(d, x - 60, 340, 420, names[c], _font("Bold", 44), colors[c])
    for i, (lab, fn) in enumerate(rows):
        y = 510 + i * 130
        d.rounded_rectangle((90, y - 14, S - 90, y + 100), 18, fill=PANEL)
        d.text((130, y + 18), lab, font=_font("Medium", 44), fill=TEXT)
        for c, x in cols.items():
            v = f"{summary['look_s'][c]:.1f}" if fn is None else fn(summary[c])
            _centered(d, x - 60, 340, y + 6, v, _font("Black", 62), TEXT)
    y = 1320
    if callout:
        img_rgba = img.convert("RGBA")
        _box(img_rgba, (90, y, S - 90, y + 200), (255, 120, 80, 50), radius=24)
        img = img_rgba.convert("RGB")
        d = ImageDraw.Draw(img)
        _text_block(d, (130, y + 34), callout, _font("Bold", 42), TEXT, S - 260, 3)
        y += 240
    _text_block(d, (100, y + 10), note, _font("Regular", 34), MUTED, S - 200, 6)
    for _ in range(n):
        yield img


def credits_frames(n):
    img = Image.new("RGB", (S, S), BG)
    d = ImageDraw.Draw(img)
    d.text((120, 400), "FORESIGHT", font=_font("Black", 130), fill=TEXT)
    d.text((120, 560), "Map-free monocular navigation with an on-board VLM", font=_font("Medium", 50), fill=MUTED)
    d.text((120, 740), "Built with", font=_font("Bold", 40), fill=TEXT)
    for i, line in enumerate(["Habitat-Sim  ·  HM3D (Matterport scans of real homes)",
                              "Depth Anything V2  ·  Qwen3-VL-2B  ·  Laya (Convai Innovations)",
                              "NVIDIA Isaac Sim (RTX, PhysX)  ·  Unitree Go2 model and walking policy from Isaac Sim",
                              "Voice-over: Kokoro-82M"]):
        d.text((120, 810 + i * 64), line, font=_font("Regular", 40), fill=TEXT)
    _text_block(d, (120, 1150), "Zero-shot: no training, no map, no goal coordinates. Every decision was made in "
                "Habitat-Sim on one laptop GPU; the Isaac Sim segment replays that route with a simulated Go2. "
                "Simulation only.", _font("Regular", 36), MUTED, S - 240, 5)
    for _ in range(n):
        yield img


# ---------------------------------------------------------------- narration & assembly


def _times(k):
    return "once" if k == 1 else ("never" if k == 0 else f"{_spoken_int(k).lower()} times")


def narration(base: Run, fore: Run, instructions, s2, s3, old, mirror_xz):
    """s2: second set (baseline, v1, v2); s3: third set (baseline, v2.1); old: first set (baseline, v1)."""
    a, b = _name(instructions[0]), _name(instructions[1]) if len(instructions) > 1 else None
    ob, of = old["baseline"], old["foresight"]
    lines = {
        "hook": f"FORESIGHT, find the {a}." + (f" Then go back to the {b}." if b else ""),
        "recap": (f"Last time, FORESIGHT cut mirror crashes from {_spoken_int(ob['crash']).lower()} to "
                  f"{_spoken_int(of['crash']).lower()}. But it described every frame, at {old['look_s']['foresight']:.1f} "
                  "seconds a look, so it kept running out of time. Version two pays for that only when it matters."),
        "how": ("Now, every look asks Qwen a quick yes or no: is there a mirror? If so, Qwen boxes it, and depth shows "
                "whether it reads too deep, like a phantom room. Only if the robot's route runs into it does the robot "
                "stop, and ask for the slow scene note. Laya reads it, and decides: wall it off, or carry on."),
        "race_intro": "Same house, same instruction. The baseline on the left, FORESIGHT on the right.",
    }
    if base.crash is not None:
        lines["race_base_mirror"] = "The baseline heads for the reflected room, and drives straight into the mirror."
    else:
        nm = base.mirror_encounter(mirror_xz)
        if nm is not None:
            lines["race_base_mirror"] = f"The baseline heads for the reflection too. Its safety layer stops it {nm[1]:.1f} metres short."
    if fore.mirror_time("suspect") is not None:
        lines["race_flag"] = "A mirror, and the route runs into it. Only now does FORESIGHT stop for the slow check."
    first_c, first_d = fore.mirror_time("confirmed"), fore.mirror_time("dismissed")
    if first_c is not None:
        lines["race_confirm"] = "Laya: a reflection. It becomes a wall on the map, and anything seen inside it is ignored."
    if first_d is not None and (first_c is None or first_d < first_c):
        lines["race_dismiss"] = "Here the slow check says no mirror, so it carries on."
    f1, f2 = _finish(fore, 1), _finish(fore, 2)
    if f1 is not None:
        lines["race_found1"] = f"{a.capitalize()} seen three times, checked up close, then it closes in. Found, at {f1:.0f} seconds."
    if b and f2 is not None:
        t_ins, t_tgt = fore.event_time("instruction"), fore.event_time("target_on_map", instructions[1])
        lines["race_memory"] = (f"Now, back to the {b}. It already has one on its map, so it heads straight there."
                                if t_ins is not None and t_tgt is not None and t_tgt - t_ins < 1.0 else f"Now, the {b}. It searches again.")
        lines["race_found2"] = f"{b.capitalize()} found. Both objects in {f2:.0f} seconds" + (", no crashes." if fore.crash is None else ".")
    elif b and f1 is not None and fore.crash is None:
        rej = fore.event_time("rejected", instructions[1])
        lines["race_end"] = ((f"Its close-up check rejects what it took for the {b}, and the clock runs out. "
                              "No crash, but only one of two." if rej is not None else
                              f"It runs out of time before reaching the {b}. No crash, but only one of two."))
    n = len(instructions)
    if base.crash is None:
        bf = _finish(base, n)
        lines["race_base_done"] = (f"The baseline finishes at {bf:.0f} seconds." if bf is not None else "The baseline runs out of time.")
    lines["isaac"] = ("Here is the same run, replayed in NVIDIA Isaac Sim. The scanned home is ray traced, so the mirror "
                      "now reflects for real, and a Unitree Go2 walks FORESIGHT's route, its legs simulated by PhysX.")
    b2, v1s, v2s = (s2.get(c) for c in ("baseline", "foresight", V2_COND))
    if b2 and v1s and v2s:
        speed = s2["look_s"]["foresight"] / max(s2["look_s"][V2_COND], 1e-6)
        lines["score2"] = (f"First, {v2s['n']} new episodes, fixed before any run. Version two found as many objects as "
                           f"version one, {_spoken_int(v2s['success']).lower()} of {_spoken_int(v2s['n']).lower()}, "
                           f"with looks {_spoken_int(round(speed)).lower()} times faster. But it hit a closet mirror. "
                           "Replaying those frames showed two bugs: an old wall at the wrong angle hid the mirror, and Qwen "
                           "boxed things inside the reflection instead of the glass. Both fixed, in version two point one.")
    b3, v3 = s3.get("baseline"), s3.get(V2_COND)
    if b3 and v3:
        lines["score3"] = (f"So, a third set: {_spoken_int(v3['n']).lower()} more episodes nobody had looked at. Found within "
                           f"a metre: baseline {_times(b3['success'])}, FORESIGHT {_times(v3['success'])}. Crashes: baseline "
                           f"{_spoken_int(b3['crash']).lower()}, FORESIGHT {_spoken_int(v3['crash']).lower()}."
                           + (" No mirror crashes now, but in a home with eighteen mirrors it stops to check too often, and "
                              "runs out of time. Fewer, closer checks are next." if v3["timeout"] > b3["timeout"] else ""))
    lines["credits"] = "FORESIGHT. Zero-shot, in simulation, on a laptop GPU. Honest numbers, not a solved problem."
    return lines


def race_plan(base: Run, fore: Run, dur, mirror_xz):
    nm = base.crash if base.crash is not None else (base.mirror_encounter(mirror_xz) or (None,))[0]
    n = max(len(fore.found), len(base.found), 1)
    ev = {"race_flag": fore.mirror_time("suspect"), "race_confirm": fore.mirror_time("confirmed"),
          "race_dismiss": fore.mirror_time("dismissed"), "race_base_mirror": nm,
          "race_found1": _finish(fore, 1), "race_memory": fore.event_time("instruction"), "race_found2": _finish(fore, 2),
          "race_end": fore.end if "race_end" in dur else None,
          "race_base_done": (_finish(base, n) or base.end) if base.crash is None else None}
    ev = {k: v for k, v in ev.items() if v is not None and k in dur}
    order = sorted(ev, key=lambda k: ev[k])
    end = max(fore.end, base.end)
    slow = [(0.0, 3.0)] + [(ev[k] - 2.0, ev[k] + 5.0) for k in ("race_flag", "race_confirm", "race_base_mirror") if k in ev] \
        + [(ev[k] - 3.0, ev[k] + 2.5) for k in ("race_found1", "race_found2", "race_base_done") if k in ev]
    for slow_speed in (1.0, 0.75, 0.5):
        for fast in (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0):
            pieces = _pieces(end, slow, fast, slow_speed)
            sch = Schedule(pieces, dur["race_intro"] + 0.9, 0.0)
            cues, last = _place(order, [sch.video(ev[k]) + 0.2 for k in order], dur, gap=0.4)
            late = max([c[0] - sch.video(ev[k]) for c, k in zip(cues, order)] or [0.0])
            if late <= 4.0 and sch.length < 140:
                return Schedule(pieces, sch.pre_s, max(2.5, last - sch.length + 1.0)), [(0.4, "race_intro")] + cues
    raise RuntimeError("narration doesn't fit the race")


def isaac_plan(replay: IsaacReplay, fore: Run, dur, max_len_s: float = 40.0):
    """Highlights of the replay: real time around the first confirmed mirror and the finds, fast elsewhere."""
    end = replay.frames[-1]["t"]
    keys = [x for x in (fore.mirror_time("confirmed"), _finish(fore, 1), _finish(fore, 2)) if x is not None and x < end]
    slow = [(0.0, 4.0)] + [(k - 3.0, k + 3.0) for k in keys]
    for fast in (4.0, 6.0, 8.0, 10.0, 12.0, 16.0):
        pieces = _pieces(end, slow, fast, 1.0)
        sch = Schedule(pieces, 0.0, 1.5)
        if sch.length <= max(max_len_s, dur["isaac"] + 2.0):
            return sch
    return sch


def render_video(root: Path, episode: dict, rows3: list, rows2: list, old_rows: list, out_path: Path, tts_python: str,
                 kokoro_home: str, voice: str = "af_heart", isaac_dir: Path = None, isaac_run: Path = None,
                 root2: Path = None, old_root: Path = None, scenes=(4, 4)):
    """root: the third set's runs (baseline + v2.1, incl. the demo episode shown); root2: the second set's (baseline,
    v1, v2); old_root: the first set's (baseline, v1)."""
    ep_dir = root / episode["name"]
    base = Run(ep_dir / "baseline.pkl", ep_dir / "baseline_meta.json")
    fore = Run(ep_dir / f"{V2_COND}.pkl", ep_dir / f"{V2_COND}_meta.json")
    scene = Scene(ep_dir)
    ex = dict(np.load(ep_dir / "explainer.npz"))

    def summarize(rows, folder, conds):
        s = eval_summary(rows)
        lat = {c: [] for c in conds}
        n_looks = {c: [0, 0] for c in conds}
        for d in folder.iterdir():
            if not d.is_dir() or d.name == episode["name"]:
                continue
            for c in conds:
                m = d / f"{c}_meta.json"
                if m.exists():
                    meta = json.loads(m.read_text())
                    lat[c] += meta["look_latency_s"]
                    n_looks[c][0] += len(meta["look_latency_s"])
                    n_looks[c][1] += meta.get("slow_looks") or 0
        s["look_s"] = {c: float(np.median(v)) if v else float("nan") for c, v in lat.items()}
        s["slow_frac"] = {c: (b / a if a else 0.0) for c, (a, b) in n_looks.items()}
        return s

    s3 = summarize(rows3, root, ["baseline", V2_COND])
    s2 = summarize(rows2, root2, ["baseline", "foresight", V2_COND])
    old = summarize(old_rows, old_root, ["baseline", "foresight"])
    instructions = episode["instructions"]
    start_xz = np.asarray(episode["start"], float)[[0, 2]]
    mirror_xz = episode.get("mirror_xz")
    lines = narration(base, fore, instructions, s2, s3, old, mirror_xz)
    replay = None
    if isaac_dir is not None and (Path(isaac_dir) / "frames.json").exists():
        replay = IsaacReplay(isaac_dir, isaac_run, ep_dir / "topdown.png", ep_dir / "assets.json", instructions)
    else:
        lines.pop("isaac")
    vo = Voice(lines, ep_dir / "narration_v2", tts_python, kokoro_home, voice)
    dur = vo.dur
    views = _Views(episode, size=768)
    sch, race_cues = race_plan(base, fore, dur, mirror_xz)
    fast_ms = float(np.median([sum(lk["ms"].values()) for _, lk in fore.looks if "describe" not in lk.get("ms", {})] or [0]))
    slow_ms = float(np.median([lk["ms"]["describe"] for _, lk in fore.looks if "describe" in lk.get("ms", {})] or [0]))
    laya = [x["laya_infer_ms"] for x in fore.ep.steps if x.get("laya_called")]
    laya_ms = float(np.median(laya)) if laya else 25.0
    depth_ms = float(np.median([s["depth_ms"] for s in fore.ep.steps]))

    def fit(line, minimum):
        return max(minimum, dur[line] + 1.6)

    segs = [("hook", lambda n: v1.hook_frames(ex["rgb"], instructions, n), fit("hook", 4.5), [(0.5, "hook")]),
            ("recap", lambda n: recap_frames(old, n), fit("recap", 11.0), [(0.6, "recap")]),
            ("how", lambda n: how_frames(n, fast_ms, slow_ms, laya_ms, depth_ms, s3["slow_frac"][V2_COND]),
             fit("how", 11.0), [(0.6, "how")]),
            ("race", lambda n: race_frames(scene, views, base, fore, sch, instructions, start_xz, mirror_xz,
                                           episode["place"]), sch.length, race_cues)]
    if replay is not None:
        iplan = isaac_plan(replay, fore, dur)
        segs.append(("isaac", lambda n: isaac_frames(replay, iplan, n), iplan.length, [(0.8, "isaac")]))
    if "score2" in dur:
        segs.append(("score2", lambda n: score_frames(
            s2, ["baseline", "foresight", V2_COND], "Second evaluation set",
            [f"{s2[V2_COND]['n']} new episodes in {scenes[0]} homes, drawn by a seeded script and committed before any run.",
             "Success = stop within 1 m of the named object (scene annotations the robot never sees)."],
            NOTE, n, callout="v2's mirror crash, and its crash on the demo start, were the same closet mirror. Two bugs "
                             "found by replaying those frames; fixed in v2.1."), fit("score2", 9.0), [(0.6, "score2")]))
    if "score3" in dur:
        segs.append(("score3", lambda n: score_frames(
            s3, ["baseline", V2_COND], "Third set: v2.1",
            [f"{s3[V2_COND]['n']} more episodes in {scenes[1]} homes, drawn and committed before any run, after the fixes.",
             "Success = stop within 1 m of the named object."], NOTE, n,
            callout=("No mirror crashes, but in 00808 (18 annotated mirrors) v2.1 stopped for 9-17 slow checks per run, "
                     "holding still 35-77 s of 180: fewer, closer checks are next." if s3[V2_COND]["timeout"] > s3["baseline"]["timeout"] else None),
            names=dict(COND_NAMES, **{V2_COND: "FORESIGHT v2.1"})), fit("score3", 8.0), [(0.6, "score3")]))
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
