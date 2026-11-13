#!/usr/bin/env python
"""Point-goal evaluation: navigation condition x depth source x episode, latency-honest, scored on the navmesh.

The base for the depth-error study (A), the stratified benchmark (C) and the VLM ablation (B) in PLAN.md. Every run
uses the same loop (`foresight.sim.timed_episode`, sim time advances by measured compute), the same goal bearing
(GPS+compass: the robot knows its own position and the goal's), the same crash rule, and differs only in:

  conditions   heuristic   depth-only potential field (milestone #7), shared control law + safety layer
               map         `MapNavigator`: map-free occupancy map from depth, Dijkstra, carrot (no VLM)
               foresight   `map` + `FastSlowMirrorGuard`: Qwen3-VL yes/no mirror check + mirror box every look, the
                           slow note + Laya only for a candidate mirror in the robot's way (FORESIGHT v2.1)
  depth        da | gt | gt:<errors> | mix:<a>@<mask>=<b>   (`foresight.perception.depth.sources`)

Success: the robot comes within --success-m of the goal (it knows both positions) before the time budget; a crash
(driving forward with < --contact-m true clearance) ends the run. SPL and progress use geodesic distances. Per frame,
the depth the navigator was given is compared with ground truth (RMSE, AbsRel, delta1, scale-aligned RMSE) and the
frame's difficulty logged (texture energy, textureless share, reflective share on annotated homes): scoring only.

  python tools/run_pointnav_eval.py --set A --conditions map heuristic --depth gt da --out results/runs/pointnav/pilot
  python tools/run_pointnav_eval.py ... --shard 0 --nshards 3     # three workers in parallel, disjoint jobs
Outputs <out>/<condition>/<depth-slug>/<episode>.json (+ .npz trace); existing outputs are skipped (resumable).
"""
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Evaluation sets (drawn with the clearance rule, committed before any run) and the first draws, kept as dev sets for
# developing/tuning: the pilot and any fix found on it run there, never on A/S.
EDGE_EXTEND_M = 1.0  # v2.4: a wall from a mirror box touching the image border is extended this far on that side
SETS = {"A": "experiments/configs/pointnav/episodes_A_3003.json", "S": "experiments/configs/pointnav/episodes_S_3004.json",
        "A_dev": "experiments/configs/pointnav/episodes_A_3001.json", "S_dev": "experiments/configs/pointnav/episodes_S_3002.json",
        # mirror/glass encounters for deliverable C (tools/make_encounter_episodes.py): start facing a pane, goal through it
        "E": "experiments/configs/pointnav/episodes_E_3005.json",
        # dev encounters (same panes, other starts; one per pane): recall check while tuning the mirror guard only
        "E_dev": "experiments/configs/pointnav/episodes_E_9001.json"}
_MODELS = {}


def slug(depth: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", depth.replace(",", "+").replace("@", "-at-").replace("=", ""))


def model(name):
    if name not in _MODELS:
        if name == "depth":
            from foresight.perception.depth.depth_anything import ZeroShotDepthEstimator
            _MODELS[name] = ZeroShotDepthEstimator()
        elif name == "vlm":
            from foresight.perception.vlm.qwen_vl import Qwen3VLSceneReasoner
            vlm = Qwen3VLSceneReasoner(model_name="models/Qwen3-VL-2B-Instruct")
            vlm.describe(np.zeros((64, 64, 3), np.uint8), max_new_tokens=4)  # warmup
            _MODELS[name] = vlm
        elif name == "laya":
            from foresight.planning.laya_client import LayaClient
            laya = LayaClient(model="models/laya")
            laya.predict("warmup", {"q": {"type": "noul", "instructions": "warmup"}})
            _MODELS[name] = laya
    return _MODELS[name]


def frame_stats_fn(semantic: bool):
    from foresight.perception.depth.sources import depth_error, reflective_ids, texture_energy
    state = {"ids": None}

    def stats(rgb, depth, obs):
        e = depth_error(depth, obs["depth"])
        te = texture_energy(rgb[::2, ::2], window=8) * 2.0  # half resolution, gradients rescaled to full-res units
        out = {"err_rmse": None if e is None else e.rmse, "err_absrel": None if e is None else e.absrel,
               "err_delta1": None if e is None else e.delta1, "err_rmse_aligned": None if e is None else e.rmse_aligned,
               "err_scale": None if e is None else e.scale, "tex_energy": float(np.median(te)),
               "textureless_frac": float((te < 2.0).mean())}
        if semantic and "semantic" in obs and state["ids"] is not None:
            out["reflective_frac"] = float(np.isin(obs["semantic"][::4, ::4], state["ids"]).mean())
        return out

    def bind(sim):
        if semantic:
            state["ids"] = reflective_ids(sim)
    return stats, bind


def build(cond, e, args, start_xz, goal_xz):
    """-> decide(rgb, depth, bearing, t, pose), controller (for meta)."""
    from foresight.planning.depth_heuristic import DepthOnlyHeuristicController
    from foresight.planning.local_map import FloorScale
    from foresight.planning.map_navigator import MapNavigator
    from foresight.safety.reactive import ReactiveSafetyLayer

    # Margin over the 0.3 m crash rule, as in object search.
    safety = lambda: ReactiveSafetyLayer(stop_distance_m=0.4, slow_distance_m=0.8, clearance_quantile=0.02)
    v, w = args.max_linear_mps, np.deg2rad(40.0)
    if cond == "heuristic":
        ctl = DepthOnlyHeuristicController(max_linear_mps=v, max_angular_radps=w, safety=safety())
        fs = FloorScale()

        def decide(rgb, depth, bearing, t, pose):
            s = fs.update(depth) if not args.no_floor_scale else 1.0
            r = ctl.step(depth * s, bearing)
            return r.linear_velocity, r.angular_velocity, {"depth_scale": s, "safety_level": r.safety.level,
                                                            "min_clearance_m": r.safety.min_clearance_m}
        return decide, ctl
    center = (start_xz + goal_xz) / 2
    size = float(max(16.0, 2 * np.linalg.norm(goal_xz - start_xz) + 12.0))
    nav_kw = dict(goal_xz=goal_xz, map_center_xz=center, map_size_m=size, max_linear_mps=v, max_angular_radps=w,
                  safety=safety(), floor_scale=not args.no_floor_scale, escape=True, robust_scan=args.robust_scan)
    if cond == "map":
        nav = MapNavigator(**nav_kw)

        def decide(rgb, depth, bearing, t, pose):
            lin, ang, d = nav.step(depth, pose, t)
            return lin, ang, _lean(d)
        return decide, nav
    if cond == "foresight":
        from foresight.perception.vlm.qwen_vl import GROUNDING_NAMES, PRESENT_PROMPT, REFLECTION_AWARE_PROMPT
        from foresight.planning.fast_slow import FastSlowMirrorGuard
        from foresight.planning.object_search import PerceptionWorker
        vlm = model("vlm")
        laya = model("laya") if args.guard != "fill" else None
        present = lambda rgb, lab: vlm.option_probs(rgb, PRESENT_PROMPT.format(name=GROUNDING_NAMES.get(lab, lab)), ["yes", "no"])[0]
        when = []
        worker = PerceptionWorker(lambda rgb, lab: vlm.locate(rgb, lab), labels=lambda: [],
                                  describe=lambda rgb: vlm.describe(rgb, prompt=REFLECTION_AWARE_PROMPT, max_new_tokens=args.vlm_tokens),
                                  present=present, extra_labels=["mirror"], describe_when=lambda: when[0]())
        nav = MapNavigator(worker=worker, **nav_kw)
        if args.guard == "fill":  # FORESIGHT v3: mirror boxes re-fill the depth, no slow check, no walls
            from foresight.planning.fast_slow import BoxFillGuard
            guard = BoxFillGuard(worker, nav, hold_s=args.fill_hold_s)
            when.append(guard.want_note)

            def decide(rgb, depth, bearing, t, pose):
                lin, ang, d = guard.step(rgb, depth, bearing, t, pose)
                return lin, ang, _lean(d)
            return decide, guard
        v22 = dict(near_m=1.5, walled_m=0.8, gap_s=5.0) if args.guard in ("v22", "v23", "v24") else {}  # see the guard's docstring
        if args.guard in ("v23", "v24"):
            v22["gap_candidates"] = False
        if args.guard == "v24":
            v22["edge_extend_m"] = EDGE_EXTEND_M
        guard = FastSlowMirrorGuard(laya, worker, nav, threshold=args.threshold, **v22)
        when.append(guard.want_note)

        def decide(rgb, depth, bearing, t, pose):
            lin, ang, d = guard.step(rgb, depth, bearing, t, pose)
            return lin, ang, _lean(d)
        return decide, guard
    raise ValueError(cond)


_KEEP = ("depth_scale", "depth_ahead_m", "safety_level", "min_clearance_m", "looking_around", "escaping", "background_ms",
         "p_hazard", "verifying", "event", "wall_added", "laya_called", "laya_infer_ms", "vlm_progress")


def _lean(d):
    """Per-tick diagnostics without the map arrays (a 300x300 int8 map per tick is ~100 MB a run)."""
    return {k: d[k] for k in _KEEP if k in d}


def run_one(args, e, cond, depth_spec, out_json: Path):
    from foresight.perception.depth.sources import make_source, needs_semantic
    from foresight.sim.timed_episode import run_timed_episode
    from make_pointnav_episodes import ALL10, ANNOTATED, scene_ref
    import habitat_sim

    dataset = ANNOTATED if args.set[0] in "SE" else ALL10
    semantic = args.set[0] in "SE"  # annotated homes: log reflective share; reflective-masked sources need it
    if needs_semantic(depth_spec) and not semantic:
        raise ValueError(f"{depth_spec} needs the annotated set S")
    est = model("depth") if "da" in re.split(r"[:@=,]", depth_spec) else None
    source = make_source(depth_spec, est)
    start, goal = np.asarray(e["start"], float), np.asarray(e["goal"], float)
    decide, ctl = build(cond, e, args, start[[0, 2]], goal[[0, 2]])
    stats, bind = frame_stats_fn(semantic)
    seed = int(hashlib.sha1(e["name"].encode()).hexdigest()[:8], 16)  # same injected noise for every condition

    def on_end(sim, ep):
        pf = sim.pathfinder

        def geo(p):
            path = habitat_sim.ShortestPath()
            path.requested_start, path.requested_end = np.asarray(p, np.float32), np.asarray(goal, np.float32)
            return float(path.geodesic_distance) if pf.find_path(path) else float("nan")
        g0 = e["geodesic_m"]
        g_end = geo(ep.positions[-1])
        length = float(np.linalg.norm(np.diff(ep.positions[:, [0, 2]], axis=0), axis=1).sum())
        out = {"geodesic_end_m": g_end, "path_m": length}
        if cond == "foresight" and semantic:
            # wall precision: a confirmed wall is right if its midpoint is within 0.6 m of a mirror/glass object
            from foresight.perception.depth.sources import REFLECTIVE_CATEGORIES
            refl = [o for o in sim.semantic_scene.objects if o is not None and o.category is not None
                    and o.category.name().lower().strip() in REFLECTIVE_CATEGORIES]
            walls = ctl.nav.map.walls
            hits = 0
            for w0, w1 in walls:
                m = (np.asarray(w0) + np.asarray(w1)) / 2
                p3 = np.array([m[0], start[1] + 1.0, m[1]], np.float32)
                hits += any(o.obb.distance(p3) < 0.6 for o in refl)
            for w in ctl.wall_log:
                m = (np.asarray(w["a"]) + np.asarray(w["b"])) / 2
                p3 = np.array([m[0], start[1] + 1.0, m[1]], np.float32)
                w["dist_reflective_m"] = round(float(min([o.obb.distance(p3) for o in refl] or [99.0])), 2)
                w["on_reflective"] = w["dist_reflective_m"] < 0.6
            out.update(walls_on_reflective=hits, walls_total=len(walls), wall_log=ctl.wall_log)
        return out

    # bind the reflective ids once the simulator exists: run_timed_episode resets the source with it; piggyback on that
    orig_reset = source.reset

    def reset(sim=None, seed=0):
        bind(sim)
        orig_reset(sim, seed)
    source.reset = reset

    t0 = time.time()
    ep = run_timed_episode(dataset_config=str(dataset), scene_id=scene_ref(dataset, e["scene"]), decide=decide,
                           depth_estimator=source, start_position=start, start_yaw_rad=math.radians(e["yaw_deg"]),
                           goal_position=goal, max_sim_s=args.max_sim_s or e["max_sim_s"], width=args.size, height=args.size,
                           model_size=None, stop_radius_m=args.success_m, stop_on_contact_m=args.contact_m,
                           keep_frames=False, semantic=semantic, frame_stats=stats, on_end=on_end, seed=seed)
    wall = time.time() - t0
    s = ep.steps
    success = ep.reached_goal_at is not None
    g0, x = e["geodesic_m"], ep.extra
    spl = float(success) * g0 / max(x["path_m"], g0)
    progress = float(np.clip(1.0 - x["geodesic_end_m"] / g0, 0.0, 1.0)) if math.isfinite(x["geodesic_end_m"]) else 0.0

    def agg(k, f=np.nanmean):
        vals = np.array([st[k] for st in s if st.get(k) is not None], float)
        return float(f(vals)) if vals.size else None
    crash = ep.contact_at is not None
    last = s[-15:] if crash else []  # the ~1.5 s before a crash: what the robot was seeing
    rec = {"episode": e["name"], "scene": e["scene"], "condition": cond, "depth": depth_spec, "set": args.set,
           "stratum": e.get("stratum"), "reflective_path_mean": e.get("reflective_mean"),
           "success": success, "spl": spl, "progress": progress, "crash": crash, "contact_at": ep.contact_at,
           # GT depth ~0 straight ahead = a hole in the scanned mesh, not an obstacle: voided for every condition
           "scan_hole": bool(crash and min(st["gt_depth_ahead_m"] for st in s[-3:]) < 0.05),
           "reached_at": ep.reached_goal_at, "sim_s": float(ep.times[-1] + s[-1]["dt"]) if s else 0.0,
           "timeout": not success and not crash, "geodesic_m": g0, "geodesic_end_m": x["geodesic_end_m"], "path_m": x["path_m"],
           "navmesh_collisions": int(ep.collided.sum()), "ticks": len(s), "wall_s": wall,
           "tick_ms_median": agg("tick_compute_ms", np.nanmedian), "depth_ms_median": agg("depth_ms", np.nanmedian),
           "err_rmse": agg("err_rmse"), "err_absrel": agg("err_absrel"), "err_delta1": agg("err_delta1"),
           "err_rmse_aligned": agg("err_rmse_aligned"), "err_scale": agg("err_scale", np.nanmedian),
           "depth_scale_final": s[-1].get("depth_scale") if s else None,
           "tex_energy": agg("tex_energy", np.nanmedian), "textureless_frac": agg("textureless_frac"),
           "reflective_frac": agg("reflective_frac"), "reflective_frac_max": agg("reflective_frac", np.nanmax),
           "pre_crash_reflective": (float(np.nanmean([st.get("reflective_frac", np.nan) for st in last])) if last and semantic else None),
           "pre_crash_textureless": (float(np.nanmean([st["textureless_frac"] for st in last])) if last else None),
           "pre_crash_err_absrel": (float(np.nanmean([st["err_absrel"] or np.nan for st in last])) if last else None),
           "floor_scale": not args.no_floor_scale, "robust_scan": args.robust_scan, "success_m": args.success_m, "contact_m": args.contact_m}
    if cond == "foresight":
        rec.update(guard=args.guard, walls_on_reflective=x.get("walls_on_reflective"), wall_log=x.get("wall_log"), slow_looks=ctl.n_slow_looks, mirror_events=[list(map(lambda v: v if not isinstance(v, float) else round(v, 2), ev))
                                                                  for ev in ctl.events],
                   walls=len(ctl.nav.map.walls), filled_ticks=getattr(ctl, "n_filled", None), look_latency_s_median=float(np.median(ctl.worker.latencies_s or [0.0])))
    out_json.write_text(json.dumps(rec, indent=1, default=float))
    np.savez_compressed(out_json.with_suffix(".npz"), t=ep.times, pos=ep.positions,
                        linear=np.array([st["linear_velocity"] for st in s]),
                        absrel=np.array([st["err_absrel"] if st["err_absrel"] is not None else np.nan for st in s]),
                        refl=np.array([st.get("reflective_frac", np.nan) for st in s]),
                        texless=np.array([st["textureless_frac"] for st in s]),
                        gt_ahead=np.array([st["gt_depth_ahead_m"] for st in s]),
                        depth_ahead=np.array([st.get("depth_ahead_m", np.nan) for st in s]))
    print(f"{e['name']:12s} {cond:9s} {depth_spec:28s} {'OK ' if success else ('CRASH' if crash else 'time ')} "
          f"spl {spl:.2f} prog {progress:.2f} sim {rec['sim_s']:.0f}s wall {wall:.0f}s absrel {rec['err_absrel'] or 0:.2f}", flush=True)
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", choices=list(SETS), required=True)
    p.add_argument("--episodes", nargs="*", default=None, help="names (default: the whole set)")
    p.add_argument("--limit", type=int, default=None, help="first N episodes per home (pilots)")
    p.add_argument("--conditions", nargs="+", default=["map"], choices=["heuristic", "map", "foresight"])
    p.add_argument("--depth", nargs="+", default=["da"])
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--size", type=int, default=640, help="square render = model resolution")
    p.add_argument("--success-m", type=float, default=0.36)
    p.add_argument("--contact-m", type=float, default=0.3)
    p.add_argument("--max-linear-mps", type=float, default=0.35)
    p.add_argument("--max-sim-s", type=float, default=None)
    p.add_argument("--no-floor-scale", action="store_true", help="use depth as given (no floor-plane self-calibration)")
    p.add_argument("--robust-scan", action="store_true", help="map navigators: noise-robust depth scan (median + support)")
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--fill-hold-s", type=float, default=3.0, help="--guard fill: how long a confident mirror box is used")
    p.add_argument("--guard", choices=["v21", "v22", "v23", "v24", "fill"], default="v22", help="foresight: mirror guard version")
    p.add_argument("--vlm-tokens", type=int, default=96)
    args = p.parse_args()
    eps = json.loads(Path(SETS[args.set]).read_text())["episodes"]
    if args.episodes:
        eps = [e for e in eps if e["name"] in set(args.episodes)]
    if args.limit:
        seen = {}
        eps = [e for e in eps if seen.setdefault(e["scene"], []).append(e) or len(seen[e["scene"]]) <= args.limit]
    jobs = [(e, c, d) for e in eps for c in args.conditions for d in args.depth]
    jobs = jobs[args.shard::args.nshards]
    tag = "_".join(t for t, on in (("nofs", args.no_floor_scale), ("robust", args.robust_scan),
                                   (args.guard + (f"{args.fill_hold_s:g}" if args.guard == "fill" else ""), args.guard != "v22" and "foresight" in args.conditions)) if on)
    print(f"{len(jobs)} jobs (shard {args.shard}/{args.nshards})", flush=True)
    for e, c, d in jobs:
        out = Path(args.out) / (c + (f"_{tag}" if tag else "")) / slug(d)
        out.mkdir(parents=True, exist_ok=True)
        f = out / f"{e['name']}.json"
        if f.exists():
            continue
        try:
            run_one(args, e, c, d, f)
        except Exception as exc:  # one bad episode must not kill a long sweep; logged, retried on the next pass
            import traceback
            traceback.print_exc()
            print(f"FAILED {e['name']} {c} {d}: {exc}", flush=True)
    if "laya" in _MODELS:
        _MODELS["laya"].close()


if __name__ == "__main__":
    main()
