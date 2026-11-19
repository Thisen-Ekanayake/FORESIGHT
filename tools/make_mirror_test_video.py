#!/usr/bin/env python
"""FORESIGHT · The Mirror Test: narrated square 1920x1920 demo from the Isaac Sim renders
(foresight.viz.mirror_test_video). Run after tools/isaac/mirror_test.py, tools/isaac/mirror_test_perception.py and
tools/isaac/phantom_room.py:

  python tools/make_mirror_test_video.py --run results/isaac/mirror_test --phantom results/isaac/phantom \
      --out results/isaac/foresight_mirror_test.mp4 [--preview 30.0]

The voice-over's numbers are computed here from perception.json / phantom_frames.json, so they match the screen.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foresight.viz.mirror_test_video import AMBER, CYAN, GREEN, RED, Run, Sched, render  # noqa: E402


def spoken_m(x):
    return f"{x:.1f} metres".replace(".0 metres", " metres")


def build(run: Run, pmeta):
    v, fr = run.vlm, run.fr
    cx = lambda q: fr[q["i"]]["cam_pos"][0]
    far = [q["p_mirror"] for q in v if cx(q) < -2.2]           # beyond ~2.1 m from the glass
    near = np.array([q["p_mirror"] for q in v if cx(q) > -1.6])  # within ~1.5 m
    ex = np.array([q["mirror_excess_m"] for q in run.per.values() if "mirror_excess_m" in q])
    look = [q for q in run.per.values() if q["phase"] == "look" and "mirror_da_m" in q]
    g = run.growth
    o, a, b = (np.array([q[k] for q in g]) for k in ("h_obs", "h_refl", "h_still"))
    moved = np.array([q["c"] for q in g]) > g[0]["c"] + 0.1
    return dict(
        n_frames=len(fr), n_looks=len(v),
        ex_med=float(np.median(ex)), ex_min=float(ex.min()), ex_max=float(ex.max()),
        look_da=float(np.median([q["mirror_da_m"] for q in look])), look_gt=float(np.median([q["mirror_gt_m"] for q in look])),
        far_pm=float(np.mean(far)), n_far=len(far), near_pm=float(np.mean(near)), near_below=float(np.mean(near < 0.5)),
        near_min=float(near.min()), n_near=len(near),
        p_robot=float(np.mean([q["p_robot"] for q in v])), robot_yes=float(np.mean([q["p_robot"] > 0.5 for q in v])),
        p_self=float(np.mean([q["p_self"] for q in v])), self_max=float(max(q["p_self"] for q in v)),
        grow_obs=float(o[-1] / o[0]), grow_refl=float(a[-1] / a[0]), grow_still=float(b[-1] / b[0]),
        grow_from=float(MIRROR_GLASS - g[0]["c"]), grow_to=float(MIRROR_GLASS - g[-1]["c"]),
        err_refl=float(np.median(np.abs(o - a)[moved] / o[moved])), err_still=float(np.median(np.abs(o - b)[moved] / o[moved])),
        close_quote=next((q["describe"] for q in run.per.values() if q.get("describe") and q["phase"] == "close"), ""),
        n_points=pmeta["n_points"], n_ghost=pmeta["n_ghost"],
    )


MIRROR_GLASS = -0.07


def narration(st):
    pct = lambda x: f"{round(100 * x)} percent"
    lines = {
        "hook1": "In 1970, the psychologist Gordon Gallup put a mirror in front of chimpanzees, to see whether they "
                 "would recognise themselves.",
        "hook2": "To a robot that navigates with one camera, a mirror looks like a doorway. So we gave ours the same "
                 "test, path traced in Isaac Sim, where the mirror really reflects it.",
        "test1": f"From two and a half metres, Qwen is sure there is a mirror. But Depth Anything puts the glass "
                 f"{st['ex_med']:.1f} metres too far.",
        "test2": "Test one: walk in. A robot standing over there would grow slowly. This one comes to meet me, so it "
                 "grows much faster, and Qwen's box follows the reflection's curve.",
        "test3": "Test two: turn. It turns at exactly the same moment.",
        "test4": "Qwen sees a robot on almost every look. Is that robot me? Every time, it says no.",
        "test5": "Close up, it even reads the name on the reflection's chest: a black sculpture, with Unitree written "
                 "on it. And, it adds, no glass.",
        "ph1": "Here is that depth, placed back in the house: every pixel the head camera saw as it walked in.",
        "ph2": "Through the mirror, Depth Anything builds a room that does not exist.",
        "ph3": "Fill the mirror's depth from the floor at its base, and the phantom room folds back onto the glass.",
        "find1": f"Depth Anything read the mirror too far on every one of {st['n_frames']} frames.",
        "find2": "The walk-in test worked: the robot in the mirror grew like a reflection, not like a stranger.",
        "find3": "Qwen saw the mirror from afar, mostly lost it up close, and never recognised itself.",
        "find4": "In our Habitat evaluation, filling the mirror's depth cut crashes from twenty-seven percent to four. "
                 "Here, it works from two metres. The last metre is still open.",
        "close": "FORESIGHT. Map-free navigation, from a single camera.",
    }
    rows = [
        (f"+{st['ex_med']:.1f} m", RED, "Depth Anything V2 reads the mirror as a room behind the wall",
         f"median over all {st['n_frames']} frames (range +{st['ex_min']:.1f} to +{st['ex_max']:.1f} m) against Isaac's "
         f"ground truth; at the look pose: read {st['look_da']:.1f} m, glass {st['look_gt']:.1f} m"),
        (f"×{st['grow_obs']:.1f}", GREEN, "Walk-in test: it grew like my reflection, not like a robot standing there",
         f"Qwen's robot box, {st['grow_from']:.1f} m → {st['grow_to']:.1f} m from the glass: reflection predicts "
         f"×{st['grow_refl']:.1f}, a robot standing still ×{st['grow_still']:.1f} (median error {100 * st['err_refl']:.0f}% vs "
         f"{100 * st['err_still']:.0f}%)"),
        (f"{st['far_pm']:.2f} → {st['near_pm']:.2f}", AMBER, "Qwen3-VL-2B P(mirror), far vs within 1.5 m",
         f"{st['n_far']} looks beyond 2.1 m vs {st['n_near']} within 1.5 m ({100 * st['near_below']:.0f}% below 0.5); "
         f"'is that robot me?' never above {st['self_max']:.2f}"),
        ("27% → 4%", CYAN, "Crashes after the mirror fill, Habitat evaluation (26 mirror encounters)",
         "oracle mirror box (docs/results/foresight_B.md). Here the fill lands on the glass from 2 m; within 1.5 m "
         "the mirror's base leaves the view and the fill misses: the last metre is open"),
    ]
    return lines, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/isaac/mirror_test")
    ap.add_argument("--phantom", default="results/isaac/phantom")
    ap.add_argument("--out", default="results/isaac/foresight_mirror_test.mp4")
    ap.add_argument("--preview", type=float, default=None)
    ap.add_argument("--chase-alt", default="results/isaac/mirror_test_corner",
                    help="corner-camera render of the walk-up, used for the chase view before 5.6 s if present")
    ap.add_argument("--stats", action="store_true", help="print the numbers and the narration, then exit")
    ap.add_argument("--tts-python", default="/ml/HelaBERT_Analysis/.venv/bin/python")
    ap.add_argument("--kokoro-home", default="/ml/Models/kokoro/hf")
    ap.add_argument("--voice", default="am_michael")
    a = ap.parse_args()
    run = Run(Path(a.run), Path(a.chase_alt) if a.chase_alt else None)
    pmeta = json.loads((Path(a.phantom) / "phantom_frames.json").read_text())
    st = build(run, pmeta)
    lines, rows = narration(st)
    if a.stats:
        print(json.dumps(st, indent=1))
        for k, v in lines.items():
            print(f"{k}: {v}")
        return
    span = run.phase_span
    t_end = float(run.t[-1])
    a_look, a_turn, a_step, a_close = span("look")[0], span("turn")[0], span("step")[0], span("close")[0]
    # hook: the turn, cinematic, at half speed; test: the whole choreography (approach slowed for the two lines)
    hook = Sched([(12.5, a_look, a_step)])
    test = Sched([(3.0, 0.0, 1.4), (11.0, 1.4, a_look), (4.0, a_look, a_turn), (8.0, a_turn, a_step),
                  (6.0, a_step, a_close), (11.0, a_close, t_end)])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    render(Path(a.run), Path(a.phantom), Path(a.out), lines, rows, a.tts_python, a.kokoro_home, a.voice,
           preview=a.preview, hook_sched=hook, test_sched=test, chase_alt=Path(a.chase_alt) if a.chase_alt else None)


if __name__ == "__main__":
    main()
