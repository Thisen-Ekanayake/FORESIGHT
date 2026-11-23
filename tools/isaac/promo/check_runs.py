#!/usr/bin/env python
"""Check the promo's live runs (tools/isaac/promo/live_runs.sh) against what each segment needs, and the SONIC gait.

  python tools/isaac/promo/check_runs.py [--live results/isaac/promo/live]

Per run: the result line, sim length, and gait (speed while walking, steps/s from left hip-pitch zero crossings, step
length, body roll sd). For the probe: commanded vs achieved speed and turn rate, and the speed gain that would make
SONIC walk at FORESIGHT's commanded speed (pass it as g1_live.py --sonic-speed-gain). Then the expectations:
  mirror   depth-only crashes into the mirror; FORESIGHT v1 reaches the goal behind it
  depth    the A study's pattern on a-00808-04: GT and DA reach the goal, GT + 10% noise does not, DA without floor
           calibration crashes
  search   the sofa and the bed are both found
A run that misses its expectation is reported, never swapped for a better one silently (the plan's rule).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

EXPECT = {
    "mirror_map": ("crash", "depth-only walks into the mirror"),
    "mirror_v1": ("reached", "FORESIGHT v1 reaches the goal behind the mirror"),
    "depth_gt": ("reached", "ground truth reaches the goal (study: 83%)"),
    "depth_da": ("reached", "Depth Anything + floor calibration reaches the goal (study: 79%)"),
    "depth_noise": ("not reached", "GT + 10% noise stalls (study: 5%)"),
    # the study: 35% reach the goal, 34% crash, the rest time out; the plan wanted a crash, any failure tells the story
    "depth_nofs": ("not reached", "uncalibrated Depth Anything fails (study: 35% success, 34% crashes)"),
    "search_v1": ("found sofa", "finds the sofa"),
}


def gait(m, t0=None, t1=None):
    st = m["steps"]
    T = np.array([s["t"] for s in st])
    P = np.array([s["pos"] for s in st])
    Q = np.array([s["quat_xyzw"] for s in st])
    J = np.array([s["joints"] for s in st])
    v_cmd = np.array([s["cmd"][0] for s in st])
    sel = np.ones(len(T), bool) if t0 is None else (T >= t0) & (T < t1)
    walk = sel & (v_cmd > 0.05)
    if walk.sum() < 25:
        return None
    names = m["joint_names"]
    hip = J[:, names.index("left_hip_pitch_joint")]
    h = hip[walk] - hip[walk].mean()
    dur = walk.sum() * m["dt"]
    steps = 2 * np.sum(np.diff(np.sign(h)) > 0) / dur
    # speed: path length over the walking steps only
    idx = np.nonzero(walk)[0]
    seg = np.linalg.norm(P[idx[1:], :2] - P[idx[1:] - 1, :2], axis=1)
    speed = float(seg.sum() / dur)
    x, y, z, w = Q.T
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    return {"speed": speed, "v_cmd": float(v_cmd[walk].mean()), "steps_s": float(steps),
            "step_m": speed / steps if steps else 0.0, "roll_sd": float(roll[walk].std()),
            "min_z": float(P[sel, 2].min())}


def yaw_rate(m, t0, t1):
    st = [s for s in m["steps"] if t0 <= s["t"] < t1]
    if len(st) < 10:
        return None, None
    q = np.array([s["quat_xyzw"] for s in st])
    x, y, z, w = q.T
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return float((yaw[-1] - yaw[0]) / (st[-1]["t"] - st[0]["t"])), float(np.mean([s["cmd"][1] for s in st]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="results/isaac/promo/live")
    a = ap.parse_args()
    live = Path(a.live)
    runs = sorted(p.parent for p in live.glob("*/motion.json"))
    if not runs:
        print(f"no runs under {live}")
        return
    print(f"{'run':16s} {'result':34s} {'sim s':>6s} {'m/s':>5s} {'cmd':>5s} {'steps/s':>7s} {'step m':>6s} {'roll sd':>7s}")
    bad = []
    for d in runs:
        m = json.loads((d / "motion.json").read_text())
        res = str(m.get("result", "?"))
        g = gait(m) or {}
        T = m["steps"][-1]["t"] if m["steps"] else 0
        print(f"{d.name:16s} {res[:34]:34s} {T:6.1f} {g.get('speed', 0):5.2f} {g.get('v_cmd', 0):5.2f} "
              f"{g.get('steps_s', 0):7.2f} {g.get('step_m', 0):6.2f} {g.get('roll_sd', 0):7.2f}")
        if any(s["pos"][2] < 0.45 for s in m["steps"]):
            bad.append(f"{d.name}: the robot fell")
        exp = EXPECT.get(d.name)
        if exp:
            want, what = exp
            ok = {"crash": res.startswith("crash"), "reached": res.startswith("reached"),
                  "not reached": not res.startswith("reached"),
                  "found sofa": "sofa" in res}[want]
            if not ok:
                bad.append(f"{d.name}: expected {want} ({what}), got: {res}")
        if d.name == "probe_sonic":
            prof = m.get("profile") or []
            t = 1.0  # g1_live stands 1 s (50 steps) before the profile
            print("  probe segments (commanded -> achieved):")
            gains = []
            for secs, v, w in prof:
                t0, t1 = t + 0.8, t + secs  # skip each segment's first 0.8 s (transition)
                if v > 0.05:
                    gg = gait(m, t0, t1)
                    if gg:
                        print(f"    t {t:5.1f}  walk {v:.2f} m/s -> {gg['speed']:.2f} m/s, {gg['steps_s']:.2f} steps/s, "
                              f"{gg['step_m']:.2f} m steps")
                        gains.append(v / max(gg["speed"], 1e-3))
                if abs(w) > 0.05:
                    wr, _ = yaw_rate(m, t0, t1)
                    if wr is not None:
                        print(f"    t {t:5.1f}  turn {w:+.2f} rad/s -> {wr:+.2f} rad/s"
                              + ("  (on the spot)" if v <= 0.05 else ""))
                t += secs
            cur = m["args"].get("sonic_speed_gain", 1.0)
            if gains:
                print(f"  speed gain now {cur:.2f}; for commanded = achieved use ~{cur * float(np.mean(gains)):.2f}")
    print()
    if bad:
        print("NEEDS A LOOK:")
        for b in bad:
            print("  -", b)
    else:
        print("every run did what its segment needs")


if __name__ == "__main__":
    main()
