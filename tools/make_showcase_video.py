#!/usr/bin/env python
"""FORESIGHT showcase video: a narrated square 1920x1920 overview of the project, built like a platform promo
(title cards, captioned fly-throughs of the HM3D homes, a task montage, then the depth-error finding as the
centrepiece: one point-goal episode driven four times with different depth, and the result bars).

  --phase record   re-run the showcase point-goal episode (set A) with each depth source, keeping what the video
                   needs and the evaluation drops: the full per-tick log, the depth the navigator was given and its
                   self-built map. Same loop, navigator, crash rule and noise seed as tools/run_pointnav_eval.py.
  --phase assets   cutaway height maps of the homes (for the tilted 3D floor plans), fly-through camera paths
  --phase render   compose the video (foresight.viz.showcase_video; Kokoro voice-over)

Every number drawn or spoken comes from the recorded runs or docs/results/depth_error_A.md's summary
(results/runs/pointnav/A/summary.json); the four recorded runs are one example episode, labelled as such.
"""
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

EPISODE = "a-00800-00"  # GT, DA and GT x2.5 all arrive in 44 s in the A sweep; GT + 10% noise makes no progress
SOURCES = {"gt": "Ground-truth depth", "da": "Depth Anything V2", "gt:scale=2.5": "Depth read 2.5× too far",
           "gt:noise=0.1": "10% depth noise"}


def _args_ns(**kw):
    base = dict(max_linear_mps=0.35, no_floor_scale=False, threshold=0.6, vlm_tokens=96, set="A")
    base.update(kw)
    return argparse.Namespace(**base)


def record(args, out: Path):
    import run_pointnav_eval as rpe
    from foresight.perception.depth.sources import make_source
    from foresight.sim.timed_episode import run_timed_episode
    from make_pointnav_episodes import ALL10, scene_ref

    e = next(x for x in json.loads(Path(rpe.SETS["A"]).read_text())["episodes"] if x["name"] == args.episode)
    start, goal = np.asarray(e["start"], float), np.asarray(e["goal"], float)
    seed = int(hashlib.sha1(e["name"].encode()).hexdigest()[:8], 16)  # the evaluation's noise seed
    out.mkdir(parents=True, exist_ok=True)
    for spec in args.sources:
        dst = out / f"{rpe.slug(spec)}.pkl"
        if dst.exists():
            continue
        est = rpe.model("depth") if spec.split(":")[0] == "da" else None
        source = make_source(spec, est)
        decide, nav = rpe.build("map", e, _args_ns(), start[[0, 2]], goal[[0, 2]])
        snaps = {"t": [], "depth": [], "map": [], "path": []}

        def rec_decide(rgb, depth, bearing, t, pose):
            r = decide(rgb, depth, bearing, t, pose)
            snaps["t"].append(t)
            snaps["depth"].append(depth[::4, ::4].astype(np.float16))  # what the navigator was given, 160 px
            snaps["map"].append(np.clip(nav.map.score * 30, -127, 127).astype(np.int8))
            snaps["path"].append(np.asarray(nav.last_path, np.float32))
            return r

        ep = run_timed_episode(dataset_config=str(ALL10), scene_id=scene_ref(ALL10, e["scene"]), decide=rec_decide,
                               depth_estimator=source, start_position=start, start_yaw_rad=math.radians(e["yaw_deg"]),
                               goal_position=goal, max_sim_s=e["max_sim_s"], width=640, height=640, model_size=None,
                               stop_radius_m=0.36, stop_on_contact_m=0.3, keep_frames=False, seed=seed)
        m = nav.map
        with open(dst, "wb") as f:
            pickle.dump({"episode": e, "source": spec, "label": SOURCES.get(spec, spec), "ep": ep,
                         "snap_t": np.asarray(snaps["t"]), "depth": np.stack(snaps["depth"]),
                         "map": np.stack(snaps["map"]), "path": snaps["path"],
                         "map_origin": np.asarray(m.origin), "map_res": float(m.res), "map_n": int(m.n)}, f)
        end = "reached" if ep.reached_goal_at is not None else ("CRASH" if ep.contact_at is not None else "timeout")
        print(f"{spec:14s} {end} at {ep.reached_goal_at or ep.times[-1]:.1f} s, {len(ep.steps)} ticks", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["record", "assets", "render", "all"], default="all")
    p.add_argument("--out", default="results/runs/showcase")
    p.add_argument("--episode", default=EPISODE)
    p.add_argument("--sources", nargs="+", default=list(SOURCES))
    p.add_argument("--video", default=None, help="default: <out>/foresight_showcase.mp4")
    p.add_argument("--tts-python", default="/ml/HelaBERT_Analysis/.venv/bin/python")
    p.add_argument("--kokoro-home", default="/ml/Models/kokoro/hf")
    p.add_argument("--voice", default="am_michael")
    p.add_argument("--preview", type=float, default=None, help="render only this video second as a PNG")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.phase in ("record", "all"):
        record(args, out / "pointnav")
    if args.phase in ("assets", "all"):
        from foresight.viz.showcase_assets import build_assets
        build_assets(out / "assets", out / "pointnav", args.episode)
    if args.phase in ("render", "all"):
        from foresight.viz.showcase_video import render_video
        render_video(out, Path(args.video or out / "foresight_showcase.mp4"), args.tts_python, args.kokoro_home,
                     args.voice, preview=args.preview)


if __name__ == "__main__":
    main()
