#!/usr/bin/env python
"""Encounter episodes for the mirror/glass stratum of deliverable C (set E): the robot starts facing a mirror or glass
pane and its goal lies straight through it.

Deliverable A showed that random point-goal episodes dilute reflective failures to nothing: injected reflective error
left the map navigator at the ground-truth rate even on mirror-exposed paths (docs/results/depth_error_A.md). The
failure is a tail event (the 00808 hallway mirror crashed the depth-only robots in every object-search set), so it has
to be measured where it happens. Per large reflective pane in the 4 annotated homes:

  pane     HM3D category in REFLECTIVE_CATEGORIES, horizontal extent >= 0.5 m, vertical extent >= 0.5 m, spanning the
           camera band (0.4-1.6 m above the floor the episode is on); its thin horizontal axis is the wall normal
  start    navigable, 1.5-4.5 m from the pane, within 45 deg of its normal, clearance >= 0.3 m; yaw facing the pane's centre
           +- 15 deg; kept only if the rendered start view shows the pane in its centre (>= 8% of the central region;
           the semantic image shows the nearest surface, so an occluded pane fails)
  goal     navigable, same floor, clearance >= 0.3 m, geodesic 3-14 m, and the straight line start -> goal crosses the
           pane: the shortest-looking way is through the mirror, as in the staged hallway-mirror trap
  <= 4 encounters per pane, starts >= 1 m apart. Written once, committed before any run.

  python tools/make_encounter_episodes.py --seed 3005
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import habitat_sim  # noqa: E402
from habitat_sim.utils.common import quat_from_angle_axis  # noqa: E402
from make_pointnav_episodes import ANNOTATED, OUT, SEMANTIC_SCENES, make_sim  # noqa: E402

from foresight.perception.depth.sources import REFLECTIVE_CATEGORIES  # noqa: E402


def panes(sim, floor_y):
    out = []
    for o in sim.semantic_scene.objects:
        if o is None or o.category is None or o.category.name().lower().strip() not in REFLECTIVE_CATEGORIES:
            continue
        c, s = np.array(o.obb.center, float), np.array(o.obb.sizes, float)
        lo, hi = c[1] - s[1] / 2 - floor_y, c[1] + s[1] / 2 - floor_y
        if s[1] < 0.5 or hi < 0.4 or lo > 1.6 or max(s[0], s[2]) < 0.5:
            continue
        if s[0] <= s[2]:  # thin along x: the pane runs along z, normal is x
            a, b, n = np.array([c[0], c[2] - s[2] / 2]), np.array([c[0], c[2] + s[2] / 2]), np.array([1.0, 0.0])
        else:
            a, b, n = np.array([c[0] - s[0] / 2, c[2]]), np.array([c[0] + s[0] / 2, c[2]]), np.array([0.0, 1.0])
        out.append({"id": int(o.semantic_id), "category": o.category.name().lower().strip(), "a": a, "b": b, "n": n,
                    "center": c, "width": float(np.linalg.norm(b - a))})
    return out


def crosses(p, q, a, b):
    def orient(x, y, z):
        return np.sign((y[0] - x[0]) * (z[1] - x[1]) - (y[1] - x[1]) * (z[0] - x[0]))
    return orient(p, q, a) != orient(p, q, b) and orient(a, b, p) != orient(a, b, q)


def geodesic(pf, a, b):
    path = habitat_sim.ShortestPath()
    path.requested_start, path.requested_end = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(path.geodesic_distance) if pf.find_path(path) else float("inf")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--per-pane", type=int, default=4)
    p.add_argument("--points", type=int, default=8000, help="navigable points sampled per home")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    episodes = []
    for scene in SEMANTIC_SCENES:
        sim = make_sim(ANNOTATED, scene, semantic=True, res=256)
        try:
            pf = sim.pathfinder
            pf.seed(int(rng.integers(1 << 31)))
            pts = [np.asarray(pf.get_random_navigable_point(), float) for _ in range(args.points)]
            pts = [q for q in pts if pf.distance_to_closest_obstacle(q.astype(np.float32), 2.0) >= 0.3]
            # candidate floor heights in 0.1 m bins, most-populated first: real floors claim their panes before the stair
            # and ramp bins do, and each pane is processed once (it can sit in the camera band of several bins)
            bins = {}
            for q in pts:
                bins[round(q[1], 1)] = bins.get(round(q[1], 1), 0) + 1
            floors = sorted(bins, key=lambda y: -bins[y])
            done_panes = set()
            agent = sim.get_agent(0)
            n_scene = 0
            for fy in floors:
                on_floor = [q for q in pts if abs(q[1] - fy) < 0.25]
                for pane in panes(sim, fy):
                    if pane["id"] in done_panes:
                        continue
                    done_panes.add(pane["id"])
                    a, b, n, mid = pane["a"], pane["b"], pane["n"], (pane["a"] + pane["b"]) / 2
                    starts = []
                    order = rng.permutation(len(on_floor))
                    for i in order:
                        s = on_floor[i]
                        v = mid - s[[0, 2]]
                        d = float(np.linalg.norm(v))
                        if not 1.5 <= d <= 4.5 or abs(v @ n) / d < math.cos(math.radians(45)):
                            continue
                        if any(np.linalg.norm(s[[0, 2]] - t["start_xz"]) < 1.0 for t in starts):
                            continue
                        yaw = math.atan2(-v[0], -v[1]) + math.radians(rng.uniform(-15, 15))
                        st = habitat_sim.AgentState()
                        st.position = s
                        st.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
                        agent.set_state(st)
                        sem = sim.get_sensor_observations()["semantic"]
                        h, w = sem.shape
                        frac = float((sem[int(.3 * h):int(.7 * h), int(.25 * w):int(.75 * w)] == pane["id"]).mean())
                        if frac < 0.08:
                            continue  # not in view, or occluded by something nearer
                        goals = []
                        for g in on_floor[:1500]:
                            if not crosses(s[[0, 2]], g[[0, 2]], a, b):
                                continue
                            gd = geodesic(pf, s, g)
                            if 3.0 <= gd <= 14.0:
                                goals.append((g, gd))
                        if not goals:
                            continue
                        g, gd = goals[int(rng.integers(len(goals)))]
                        starts.append({"start_xz": s[[0, 2]], "start": s.tolist(), "goal": g.tolist(), "yaw_deg": math.degrees(yaw),
                                       "geodesic_m": round(gd, 3), "pane_id": pane["id"], "pane_category": pane["category"],
                                       "pane_width_m": round(pane["width"], 2), "start_dist_m": round(d, 2),
                                       "pane_view_frac": round(frac, 3)})
                        if len(starts) >= args.per_pane:
                            break
                    for e in starts:
                        e.pop("start_xz")
                        e["name"] = f"e-{scene[:5]}-{pane['category'].replace(' ', '')}{pane['id']}-{len([x for x in episodes if x['pane_id'] == pane['id'] and x['scene'] == scene])}"
                        e["scene"] = scene
                        e["stratum"] = "encounter"
                        e["max_sim_s"] = round(30.0 + 3.0 * e["geodesic_m"] / 0.35, 1)
                        episodes.append(e)
                        n_scene += 1
            print(f"{scene}: {n_scene} encounters", flush=True)
        finally:
            sim.close()
    out = OUT / f"episodes_E_{args.seed}.json"
    cats = {}
    for e in episodes:
        cats[e["pane_category"]] = cats.get(e["pane_category"], 0) + 1
    out.write_text(json.dumps({"set": "E", "seed": args.seed, "dataset": str(ANNOTATED), "n": len(episodes),
                               "by_category": cats, "rules": __doc__.split("to be measured where it happens.")[1].split("python tools")[0].strip(),
                               "episodes": episodes}, indent=1))
    print(f"-> {out} ({len(episodes)} encounters: {cats})")


if __name__ == "__main__":
    main()
