#!/usr/bin/env python
"""Seeded point-goal evaluation sets for the depth-error study (A), the stratified benchmark (C) and the VLM ablation (B).

Written once and committed before any run on them (as for object search). Two sets:

  A  all 10 HM3D minival homes (navmesh only, no semantics needed): uniform depth-error sweeps.
  S  the 4 homes with semantic annotations: every episode carries its reflective exposure, the share of mirror/glass
     pixels seen along the shortest path (rendered at 0.5 m steps facing along it), and the set is stratified so half
     its episodes are the most exposed candidates of their home. Structured (reflective) errors and the mirror/glass
     stratum need these.

Episodes: start and goal navigable on the same floor (|dy| < 0.25 m), geodesic 3-12 m, geodesic/euclidean >= 1.05,
start and goal at least --min-clearance m from the navmesh edge, random start yaw; time budget 30 s + 3x the geodesic
at 0.35 m/s. The clearance rule came after the pilot on the first sets (A seed 3001, S seed 3002, now dev sets): with
ground-truth depth, starts ~0.1-0.2 m from a wall left the robot facing it for the whole episode, and starts on a
wall-hugging navmesh edge "crashed" before moving.

  python tools/make_pointnav_episodes.py --set A --seed 3001
  python tools/make_pointnav_episodes.py --set S --seed 3002
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import habitat_sim  # noqa: E402
from habitat_sim.utils.common import quat_from_angle_axis  # noqa: E402

ROOT = Path("data/processed/scene_datasets/hm3d/minival")
ALL10 = ROOT / "hm3d_minival_basis_pointnav.scene_dataset_config.json"
ANNOTATED = ROOT / "hm3d_annotated_minival_basis.scene_dataset_config.json"
SEMANTIC_SCENES = ["00800-TEEsavR23oF", "00802-wcojb4TFT35", "00803-k1cupFYWXJ6", "00808-y9hTuugGdiq"]
OUT = Path("experiments/configs/pointnav")


def scenes_all10():
    return sorted(d.name for d in ROOT.iterdir() if d.is_dir() and list(d.glob("*.basis.glb")))


def scene_ref(dataset, scene):
    """scene_id for habitat-sim: the annotated config knows scenes by name; the all-10 config by mesh path."""
    if dataset == ANNOTATED:
        return scene
    return str(next((ROOT / scene).glob("*.basis.glb")).resolve())


def make_sim(dataset, scene, semantic=False, res=160):
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file, cfg.scene_id = str(dataset), scene_ref(dataset, scene)
    cfg.load_semantic_mesh = semantic
    specs = []
    # habitat-sim 0.3.3 renders HM3D semantics as all zeros without a colour sensor beside them (checked 2026-11-02)
    for uuid, kind in ((("rgb", habitat_sim.SensorType.COLOR), ("semantic", habitat_sim.SensorType.SEMANTIC)) if semantic else ()):
        s = habitat_sim.CameraSensorSpec()
        s.uuid, s.sensor_type, s.resolution, s.position = uuid, kind, [res, res], [0.0, 1.5, 0.0]
        specs.append(s)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = specs
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))


def sample(sim, rng, n_want, max_tries=6000, min_clearance=0.0):
    pf = sim.pathfinder
    out = []
    for _ in range(max_tries):
        if len(out) >= n_want:
            break
        a, b = pf.get_random_navigable_point(), pf.get_random_navigable_point()
        if abs(a[1] - b[1]) > 0.25:
            continue
        if min_clearance > 0 and min(pf.distance_to_closest_obstacle(a, 2.0), pf.distance_to_closest_obstacle(b, 2.0)) < min_clearance:
            continue
        path = habitat_sim.ShortestPath()
        path.requested_start, path.requested_end = a, b
        if not pf.find_path(path) or not math.isfinite(path.geodesic_distance):
            continue
        g = float(path.geodesic_distance)
        e = float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
        if not (3.0 <= g <= 12.0) or g / max(e, 1e-6) < 1.05:
            continue
        out.append({"start": [float(x) for x in a], "goal": [float(x) for x in b],
                    "yaw_deg": float(rng.uniform(-180, 180)), "geodesic_m": round(g, 3),
                    "points": [[float(x) for x in p] for p in path.points]})
    return out


def exposure(sim, ep, refl_ids, step_m=0.5):
    """Mean / max share of reflective pixels along the shortest path, looking along it."""
    pts = np.asarray(ep["points"], float)
    fracs = []
    agent = sim.get_agent(0)
    for p, q in zip(pts[:-1], pts[1:]):
        seg = q - p
        L = float(np.linalg.norm(seg[[0, 2]]))
        if L < 1e-3:
            continue
        yaw = math.atan2(-seg[0], -seg[2])  # habitat: forward is -z at yaw 0
        for s in np.arange(0.0, L, step_m):
            st = habitat_sim.AgentState()
            st.position = p + seg * (s / L)
            st.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
            agent.set_state(st)
            sem = sim.get_sensor_observations()["semantic"]
            fracs.append(float(np.isin(sem, refl_ids).mean()))
    return (float(np.mean(fracs)), float(np.max(fracs))) if fracs else (0.0, 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", choices=["A", "S"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--per-scene", type=int, default=None, help="A: 12, S: 16")
    p.add_argument("--candidates", type=int, default=80, help="S: candidates per home before stratifying")
    p.add_argument("--min-clearance", type=float, default=0.3, help="start/goal distance to the navmesh edge, m")
    args = p.parse_args()
    from foresight.perception.depth.sources import REFLECTIVE_CATEGORIES, reflective_ids

    rng = np.random.default_rng(args.seed)
    episodes = []
    if args.set == "A":
        n = args.per_scene or 12
        dataset, scenes = ALL10, scenes_all10()
    else:
        n = args.per_scene or 16
        dataset, scenes = ANNOTATED, SEMANTIC_SCENES
    for scene in scenes:
        sim = make_sim(dataset, scene, semantic=args.set == "S")
        try:
            sim.pathfinder.seed(int(rng.integers(1 << 31)))
            if args.set == "A":
                eps = sample(sim, rng, n, min_clearance=args.min_clearance)
            else:
                cands = sample(sim, rng, args.candidates, min_clearance=args.min_clearance)
                ids = reflective_ids(sim)
                for c in cands:
                    c["reflective_mean"], c["reflective_max"] = (round(x, 4) for x in exposure(sim, c, ids))
                by_exp = sorted(cands, key=lambda c: -c["reflective_mean"])
                top = by_exp[: n // 2]
                rest = [c for c in cands if not any(c is t for t in top)]
                pick = rng.choice(len(rest), size=min(n - len(top), len(rest)), replace=False)
                eps = top + [rest[i] for i in sorted(pick)]
                for c in top:
                    c["stratum"] = "exposed"
                for c in eps:
                    c.setdefault("stratum", "random")
        finally:
            sim.close()
        for i, e in enumerate(eps):
            e.pop("points", None)
            e["name"] = f"{args.set.lower()}-{scene[:5]}-{i:02d}"
            e["scene"] = scene
            e["max_sim_s"] = round(30.0 + 3.0 * e["geodesic_m"] / 0.35, 1)
            episodes.append(e)
        print(f"{scene}: {len(eps)} episodes" + (f", exposed mean {np.mean([e['reflective_mean'] for e in eps if e['stratum'] == 'exposed']):.3f}"
                                                 if args.set == "S" else ""), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"episodes_{args.set}_{args.seed}.json"
    meta = {"set": args.set, "seed": args.seed, "dataset": str(dataset), "n": len(episodes),
            "reflective_categories": list(REFLECTIVE_CATEGORIES) if args.set == "S" else None,
            "rules": f"same floor |dy|<0.25 m, geodesic 3-12 m, geodesic/euclid >= 1.05, clearance >= {args.min_clearance} m, random yaw; "
                     "max_sim_s = 30 + 3 * geodesic / 0.35",
            "episodes": episodes}
    out.write_text(json.dumps(meta, indent=1))
    print(f"-> {out} ({len(episodes)} episodes)")


if __name__ == "__main__":
    main()
