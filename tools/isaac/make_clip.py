#!/usr/bin/env python
"""Standalone 1920x1920 clip of an Isaac Sim replay (`tools/isaac/render_replay.py`), composed with
`foresight.viz.isaac_replay` (chase view, Go2 head camera, the run's state, route minimap). Silent; constant speed.

  python tools/isaac/make_clip.py results/isaac/replay_v21 results/isaac/v21_demo_run.json \
      results/runs/object_search/set3/mirror-sofa-bed results/isaac/foresight_isaac_replay.mp4 --speed 2
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from foresight.viz.isaac_replay import IsaacReplay  # noqa: E402
from foresight.viz.mirror_trap_video import FPS, S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay")
    ap.add_argument("run_json")
    ap.add_argument("episode_dir", help="has topdown.png + assets.json for the minimap")
    ap.add_argument("out")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--instructions", nargs="+", default=["sofa", "bed"])
    args = ap.parse_args()
    ep = Path(args.episode_dir)
    r = IsaacReplay(args.replay, args.run_json, ep / "topdown.png", ep / "assets.json", args.instructions)
    ts = np.array([f["t"] for f in r.frames])
    n = int((ts[-1] - ts[0]) / args.speed * FPS)
    proc = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{S}x{S}", "-r", str(FPS), "-i", "-",
                             "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags",
                             "+faststart", args.out], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for k in range(n):
        i = int(np.clip(np.searchsorted(ts, ts[0] + k / FPS * args.speed), 0, len(ts) - 1))
        proc.stdin.write(r.frame(i, speed=args.speed).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(f"wrote {args.out} ({n / FPS:.1f} s at {args.speed:g}x)")


if __name__ == "__main__":
    main()
