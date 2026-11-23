#!/usr/bin/env python
"""FORESIGHT promo, pass 2 setup: the shot list for tools/isaac/replay_render.py, timed from the live runs' own events.

  python tools/isaac/promo/make_shots.py [--live results/isaac/promo/live] [--out results/isaac/promo]

Writes <out>/shots.json (every path-traced shot of the video) and <out>/shots_stills.json (one still per shot at its
key moment, for the look check before the full render), and prints the frame count and render-time estimate.

Walking is shown at real speed (1.0) everywhere except the two marked time-lapses (the four-robot crane and the
object search), which the video labels with their speed-up. Absolute camera positions for the home are the ones
hand-tuned for Through the Glass (same start and goal); the depth episode's are first guesses to tune on the stills.
"""
import argparse
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np

FPS = 30
TINT = {"depth_gt": [0.35, 0.95, 0.6], "depth_da": [0.35, 0.8, 1.0], "depth_noise": [1.0, 0.75, 0.3],
        "depth_nofs": [1.0, 0.4, 0.35]}


class Run:
    def __init__(self, d: Path):
        self.d = d
        m = json.loads((d / "motion.json").read_text())
        self.m = m
        self.t = np.array([s["t"] for s in m["steps"]])
        self.pos = np.array([s["pos"] for s in m["steps"]])
        q = np.array([s["quat_xyzw"] for s in m["steps"]])
        x, y, z, w = q.T
        self.yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self.end = float(self.t[-1])
        self.result = str(m.get("result", ""))
        hit = re.search(r"(crash|reached|fell) at ([\d.]+)", self.result)
        self.event_t = float(hit.group(2)) if hit else None
        self.crash_t = self.event_t if self.result.startswith("crash") else None
        self.brain = None
        if (d / "brain.pkl").exists():
            self.brain = pickle.load(open(d / "brain.pkl", "rb"))

    def at(self, t):
        k = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        return self.pos[k], float(self.yaw[k])

    def path(self):
        return str(self.d / "motion.json")

    def events(self, name):
        if not self.brain:
            return []
        return [x["t"] for x in self.brain["ticks"] if (x.get("diag") or {}).get("event") == name]

    def first(self, key):
        if not self.brain:
            return None
        return next((x["t"] for x in self.brain["ticks"] if (x.get("diag") or {}).get(key)), None)

    def found(self):
        """[(t, label)] of the object search's confirmed finds."""
        if not self.brain:
            return []
        last = next((x["diag"].get("found") for x in reversed(self.brain["ticks"]) if (x.get("diag") or {}).get("found")), [])
        return [(float(f[0]), f[1]) for f in (last or [])]


def follow(back=2.0, side=0.5, height=0.6, ahead=1.2, focal=16, smooth=0.06):
    return {"mode": "follow", "back": back, "side": side, "height": height, "ahead": ahead, "focal": focal, "smooth": smooth}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="results/isaac/promo/live")
    ap.add_argument("--out", default="results/isaac/promo")
    ap.add_argument("--s-per-frame", type=float, default=24.6, help="measured: 1440^2, 64 spp, deterministic")
    a = ap.parse_args()
    live, out = Path(a.live), Path(a.out)
    R = {d.name: Run(d) for d in sorted(live.iterdir()) if (d / "motion.json").exists()}
    need = ["mirror_map", "mirror_v1", "depth_gt", "depth_da", "depth_noise", "depth_nofs", "search_v1"]
    missing = [n for n in need if n not in R]
    if missing:
        raise SystemExit(f"missing live runs: {missing} (tools/isaac/promo/live_runs.sh)")
    shots, keys = [], {}

    def add(name, key_t, **s):
        shots.append(dict(name=name, **s))
        keys[name] = key_t

    # ---- hook: a low dolly behind the depth-only robot, down the hallway into the mirror (it sees its reflection)
    g = R["mirror_map"]
    c = g.crash_t or g.end
    add("hook_dolly", c - 2.0, hero=g.path(), t0=max(1.0, c - 9.0), t1=min(g.end, c + 1.5), speed=1.0,
        cam=follow(back=2.1, side=0.3, height=0.35, ahead=1.6, focal=16))

    # ---- system: the object-search run (FORESIGHT v1: Depth Anything, its map and plan, and live Qwen3-VL looks,
    # so every AR layer, including the VLM label, is the run's own): its first half; the search segment shows the
    # second half (spotting and reaching the sofa), so no stretch of the run appears twice
    d = R["search_v1"]
    split = min(24.0, d.end / 2)
    add("sys_front", 6.0, hero=d.path(), t0=1.0, t1=12.0, speed=1.0,
        cam={"mode": "rig", "offset": [2.4, 1.0, 0.25], "look": [0.3, 0.0, -0.1], "focal": 18, "smooth": 0.06})
    add("sys_shoulder", 18.0, hero=d.path(), t0=12.0, t1=split, speed=1.0,
        cam=follow(back=1.8, side=0.7, height=0.95, ahead=1.6, focal=15))

    # ---- depth centrepiece: four coloured robots from one start (time-lapse), then two real-time close-ups
    runs4 = ["depth_gt", "depth_da", "depth_noise", "depth_nofs"]
    T4 = max(R[n].end for n in runs4)
    s0, _ = R["depth_gt"].at(0.0)
    s1, _ = R["depth_gt"].at(R["depth_gt"].end)
    mid = (np.array(s0) + np.array(s1)) / 2
    away = np.array(s0[:2]) - mid[:2]
    away = away / (np.linalg.norm(away) + 1e-9)
    eye = [float(s0[0] + 1.2 * away[0]), float(s0[1] + 1.2 * away[1]), 2.15]  # behind the start, under the ceiling
    crane_s = 14.0
    add("depth_crane", 0.4 * T4, hero=None, t0=0.0, t1=T4, speed=T4 / crane_s,
        ghosts=[{"motion": R[n].path(), "tint": TINT[n], "opacity": 0.5, "offset": 0.0} for n in runs4],
        cam={"mode": "fixed", "eye": eye, "target": [float(mid[0]), float(mid[1]), 0.6], "track": 0.25, "focal": 14,
             "smooth": 0.05}, timelapse=True)
    n_ = R["depth_nofs"]
    cn = n_.crash_t or n_.end
    add("depth_nofs_crash", cn - 1.0, hero=n_.path(), t0=max(1.0, cn - 6.0), t1=min(n_.end, cn + 1.5), speed=1.0,
        cam=follow(back=2.0, side=-0.6, height=0.7, ahead=1.4, focal=16))
    z = R["depth_noise"]
    tz = min(z.end - 1.0, 25.0)  # stalled by then (the study: it creeps and stops in front of phantom obstacles)
    pz, _ = z.at(tz)
    add("depth_noise_stall", tz, hero=z.path(), t0=tz - 4.0, t1=tz + 4.0, speed=1.0,
        cam={"mode": "orbit", "center": [float(pz[0]), float(pz[1]), 0.9], "radius": 1.9, "height": 0.6,
             "deg0": 200.0, "deg1": 260.0, "focal": 16})

    # ---- mirror: the duel (same start, same goal), the second look, the way round (camera spots from 2026-11-20)
    h = R["mirror_v1"]
    add("mirror_duel", c - 1.0, hero=h.path(), ghost=g.path(), t0=max(1.0, c - 10.0), t1=c + 1.5, speed=1.0,
        cam={"mode": "fixed", "eye": [-3.12, 0.72, 1.65], "target": [-0.6, -0.4, 1.0], "track": 0.35, "focal": 15,
             "smooth": 0.08})
    conf = (h.events("confirmed") or [None])[0]
    ver = h.first("verifying")
    tc = conf if conf is not None else (ver if ver is not None else min(h.end, 40.0))
    add("mirror_check", tc, hero=h.path(), t0=max(1.0, tc - 6.0), t1=min(h.end, tc + 5.0), speed=1.0,
        cam={"mode": "fixed", "eye": [-0.7, -1.15, 1.55], "target": [-2.8, 0.1, 0.9], "track": 0.5, "focal": 14,
             "smooth": 0.06})
    add("mirror_exit", h.end - 3.0, hero=h.path(), t0=max(1.0, h.end - 11.0), t1=h.end, speed=1.0,
        cam=follow(back=1.9, side=0.5, height=0.6, ahead=1.2, focal=14))

    # ---- search: the second half of the search at real speed (spotting the sofa, walking over), then the pull-back
    s = R["search_v1"]
    tf = next((t for t, lab in s.found() if lab == "sofa"), s.end)
    add("search_lapse", tf - 2.0, hero=s.path(), t0=max(split, s.end - 22.0), t1=s.end, speed=1.0,
        cam=follow(back=2.6, side=0.6, height=1.3, ahead=1.0, focal=14, smooth=0.04))
    pe, ye = s.at(s.end)
    fwd = np.array([math.cos(ye), math.sin(ye)])
    e0 = np.array(pe[:2]) + 1.4 * fwd
    add("end_pull", s.end, hero=s.path(), t0=s.end - 1.4, t1=s.end, speed=0.35,
        cam={"mode": "fixed", "eye": [float(e0[0]), float(e0[1]), 1.35],
             "eye_move": [float(1.2 * fwd[0]), float(1.2 * fwd[1]), 0.5],
             "target": [float(pe[0]), float(pe[1]), 0.9], "track": 0.0, "focal": 18})

    out.mkdir(parents=True, exist_ok=True)
    (out / "shots.json").write_text(json.dumps(shots, indent=1))
    stills = []
    for sh in shots:
        k = keys[sh["name"]]
        t = float(np.clip(k, sh["t0"], sh["t1"]))
        u = (t - sh["t0"]) / max(sh["t1"] - sh["t0"], 1e-6)
        st = dict(sh, t0=t, t1=t + 0.05, name=sh["name"])
        if "eye_move" in sh["cam"]:  # the still shows the camera where it is at that moment
            st["cam"] = dict(sh["cam"], eye=[e + u * m for e, m in zip(sh["cam"]["eye"], sh["cam"]["eye_move"])],
                             eye_move=[0, 0, 0])
        if sh["cam"]["mode"] == "orbit":
            st["cam"] = dict(sh["cam"], deg0=sh["cam"]["deg0"] + u * (sh["cam"]["deg1"] - sh["cam"]["deg0"]))
            st["cam"]["deg1"] = st["cam"]["deg0"]
        stills.append(st)
    (out / "shots_stills.json").write_text(json.dumps(stills, indent=1))
    total = 0
    print(f"{'shot':18s} {'sim t':>13s} {'speed':>6s} {'video s':>7s} {'frames':>6s}")
    for sh in shots:
        n = int(round((sh["t1"] - sh["t0"]) / sh["speed"] * FPS))
        total += n
        print(f"{sh['name']:18s} {sh['t0']:6.1f}-{sh['t1']:6.1f} {sh['speed']:6.2f} {n / FPS:7.1f} {n:6d}"
              + ("  time-lapse" if sh.get("timelapse") else ""))
    print(f"total {total} frames = {total / FPS:.0f} s of path-traced footage; at {a.s_per_frame:.1f} s/frame "
          f"~{total * a.s_per_frame / 3600:.1f} h (half that if 32 spp passes the stills check)")
    print(f"runs: " + ", ".join(f"{k}: {v.result}" for k, v in R.items()))
    print(f"wrote {out / 'shots.json'} and {out / 'shots_stills.json'}")


if __name__ == "__main__":
    main()
