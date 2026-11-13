#!/usr/bin/env python
"""How good must a mirror detector be? The v2.3 mirror guard driven by a simulated detector instead of the VLM.

Same runner, episodes, navigator, guard geometry (wedges, near_m / walled_m / gap_s) and records as
`run_pointnav_eval.py --conditions foresight --guard v23`, but each look's mirror boxes come from the semantic
annotation (connected mirror/glass regions of the frame, >= --min-area of it), each kept with probability --recall,
plus one false box (random bearing, 10-30% of the width) with probability --fp per look. The slow check always
confirms (a stub Laya returns P = 1), so the detector alone decides which walls are built: --recall 1 --fp 0 is an
oracle detector, the ceiling of any detector feeding this guard.

Timing stays latency-honest: no model runs, so each look is charged --look-s of sim time (the v2.3 dev runs'
median look latency, 0.17 s) and each slow-check description --describe-s (an assumption: a 96-token Qwen3-VL-2B
description on the RTX 4060 takes about 3 s).

  python tools/run_detector_ablation.py --set E --recall 1 --fp 0 --out results/runs/pointnav/det/E
--guard v24 runs the v2.4 guard (walls from boxes at the image border extended). Outputs
<out>/r<recall>_fp<fp>/foresight_<guard>/<depth>/<episode>.json, read by summarize_pointnav.py / guard_precision.py.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_pointnav_eval as R  # noqa: E402


class _Answer:
    def __init__(self):
        self.answers = {"mirror": {"probabilities": {"A": 1.0}}}
        self.infer_ms = self.roundtrip_ms = 0.0


class ConfirmAll:
    """Stands in for Laya: every slow check confirms, so walls are exactly the detector's candidates."""
    def predict(self, state, questions):
        return _Answer()

    def close(self):
        pass


def boxes_from_semantic(sem, ids, min_area):
    """Connected mirror/glass regions of an instance-id image -> Qwen-style boxes [x1, y1, x2, y2] in 0-1000."""
    from scipy.ndimage import find_objects, label
    m = np.isin(sem[::4, ::4], ids)
    if not m.any():
        return []
    lab, n = label(m)
    h, w = m.shape
    out = []
    for i, sl in enumerate(find_objects(lab), start=1):
        if (lab[sl] == i).sum() < min_area * h * w:
            continue
        ys, xs = sl
        out.append([xs.start / w * 1000, ys.start / h * 1000, xs.stop / w * 1000, ys.stop / h * 1000])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", choices=[s for s in R.SETS if s[0] in "SE"], required=True)
    p.add_argument("--recall", type=float, default=1.0)
    p.add_argument("--fp", type=float, default=0.0, help="probability of one false box per look")
    p.add_argument("--min-area", type=float, default=0.01, help="smallest region boxed, as a share of the frame")
    p.add_argument("--look-s", type=float, default=0.17)
    p.add_argument("--describe-s", type=float, default=3.0)
    p.add_argument("--guard", choices=["v23", "v24"], default="v23")
    p.add_argument("--depth", default="da")
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--episodes", nargs="*", default=None)
    p.add_argument("--max-sim-s", type=float, default=None, help="cap sim time (smoke tests); default per episode")
    a = p.parse_args()

    from foresight.planning.fast_slow import FastSlowMirrorGuard
    from foresight.planning.map_navigator import MapNavigator
    from foresight.planning.object_search import PerceptionWorker
    from foresight.perception.depth.sources import reflective_ids
    import foresight.sim.timed_episode as TE

    orig_run = TE.run_timed_episode
    TE.run_timed_episode = lambda *x, **k: orig_run(*x, **dict(k, oracle_semantic=True))  # pose gets the id image
    orig_build = R.build

    def build(cond, e, args, start_xz, goal_xz):
        if cond != "foresight":
            return orig_build(cond, e, args, start_xz, goal_xz)
        st = {"sem": None, "ids": None, "boxes": [], "rng": np.random.default_rng(
            int(hashlib.sha1((e["name"] + f"{a.recall}{a.fp}").encode()).hexdigest()[:8], 16))}
        worker_ref = []

        def present(rgb, lab):
            worker_ref[0].min_latency_s = a.look_s
            bx = [b for b in boxes_from_semantic(st["sem"], st["ids"], a.min_area) if st["rng"].random() < a.recall]
            if st["rng"].random() < a.fp:
                wdt = st["rng"].uniform(100, 300)
                x1 = st["rng"].uniform(0, 1000 - wdt)
                bx.append([x1, 300.0, x1 + wdt, 700.0])
            st["boxes"] = bx
            return 1.0 if bx else 0.0

        def describe(rgb):
            worker_ref[0].min_latency_s = a.describe_s
            return "a mirror ahead"

        when = []
        worker = PerceptionWorker(lambda rgb, lab: st["boxes"], labels=lambda: [], describe=describe, present=present,
                                  extra_labels=["mirror"], describe_when=lambda: when[0](), min_latency_s=a.look_s)
        worker_ref.append(worker)
        nav = MapNavigator(worker=worker, **_nav_kw(e, args, start_xz, goal_xz))
        guard = FastSlowMirrorGuard(ConfirmAll(), worker, nav, threshold=0.6, near_m=1.5, walled_m=0.8, gap_s=5.0,
                                    gap_candidates=False, edge_extend_m=R.EDGE_EXTEND_M if a.guard == "v24" else 0.0)
        when.append(guard.want_note)

        def decide(rgb, depth, bearing, t, pose):
            if st["ids"] is None:
                st["ids"] = reflective_ids(pose["oracle_sim"])
            st["sem"] = pose.pop("oracle_semantic")
            pose.pop("oracle_sim", None)
            lin, ang, d = guard.step(rgb, depth, bearing, t, pose)
            return lin, ang, R._lean(d)
        return decide, guard
    R.build = build

    args = argparse.Namespace(set=a.set, size=640, success_m=0.36, contact_m=0.3, max_linear_mps=0.35, max_sim_s=a.max_sim_s,
                              no_floor_scale=False, robust_scan=False, threshold=0.6, guard=a.guard, vlm_tokens=96)
    eps = R.json.loads(Path(R.SETS[a.set]).read_text())["episodes"]
    if a.episodes:
        eps = [e for e in eps if e["name"] in set(a.episodes)]
    eps = eps[a.shard::a.nshards]
    out = Path(a.out) / f"r{a.recall:g}_fp{a.fp:g}" / f"foresight_{a.guard}" / R.slug(a.depth)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(eps)} episodes -> {out}", flush=True)
    for e in eps:
        f = out / f"{e['name']}.json"
        if f.exists():
            continue
        try:
            R.run_one(args, e, "foresight", a.depth, f)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"FAILED {e['name']}: {exc}", flush=True)


def _nav_kw(e, args, start_xz, goal_xz):
    """MapNavigator settings exactly as run_pointnav_eval.build makes them."""
    from foresight.safety.reactive import ReactiveSafetyLayer
    center = (start_xz + goal_xz) / 2
    size = float(max(16.0, 2 * np.linalg.norm(goal_xz - start_xz) + 12.0))
    return dict(goal_xz=goal_xz, map_center_xz=center, map_size_m=size, max_linear_mps=args.max_linear_mps,
                max_angular_radps=np.deg2rad(40.0),
                safety=ReactiveSafetyLayer(stop_distance_m=0.4, slow_distance_m=0.8, clearance_quantile=0.02),
                floor_scale=not args.no_floor_scale, escape=True, robust_scan=args.robust_scan)


if __name__ == "__main__":
    main()
