"""Ground-truth scoring for object search, from the scene's semantic annotations (never shown to agents).

Success follows the ObjectNav convention: the agent declares "found" while within `success_m` (1.0 m,
horizontal) of any annotated instance of the named category on its own floor. SPL uses the shortest
navmesh path from the leg's start to any navigable point that would have counted as success.

HM3D note: in habitat-sim 0.3.3, `SemanticObject.aabb` comes back stretched to include the world origin
for most objects (min or max pinned at 0), so instances are taken from `obb` (centre/sizes/distance are right).
"""
from __future__ import annotations

from dataclasses import dataclass

import habitat_sim
import numpy as np

CATEGORY_NAMES = {
    "sofa": {"sofa", "couch"},
    "bed": {"bed"},
    "tv": {"tv", "television"},
    "refrigerator": {"refrigerator", "fridge"},
    "toilet": {"toilet"},
    "plant": {"plant", "potted plant"},
    "chair": {"chair"},
    "sink": {"sink"},
    "mirror": {"mirror", "closet mirror wall"},
}


@dataclass
class Instance:
    semantic_id: int
    category: str
    obb: object
    center: np.ndarray


def instances(sim: habitat_sim.Simulator, label: str, floor_y: float, max_above_m: float = 2.3):
    """Annotated instances of `label` whose centre is between the floor at `floor_y` and `max_above_m` above it."""
    names = CATEGORY_NAMES[label]
    out = []
    for o in sim.semantic_scene.objects:
        if o is None or o.category is None or o.category.name().lower().strip() not in names:
            continue
        c = np.asarray(o.obb.center, float)
        if -0.3 < c[1] - floor_y < max_above_m:
            out.append(Instance(o.semantic_id, label, o.obb, c))
    return out


def distance_to(inst: Instance, pos) -> float:
    """Horizontal distance from `pos` to the instance's oriented box (evaluated at the box's mid height)."""
    p = np.array([pos[0], inst.center[1], pos[2]], np.float32)
    return float(inst.obb.distance(p))


def nearest(sim, label, pos, floor_y):
    insts = instances(sim, label, floor_y)
    if not insts:
        return None, float("inf")
    d = [distance_to(i, pos) for i in insts]
    k = int(np.argmin(d))
    return insts[k], d[k]


def success_points(sim, label, floor_y, success_m: float = 1.0, step: float = 0.2):
    """Navigable points on this floor within success_m of an instance (the targets for the shortest path)."""
    pts = []
    for inst in instances(sim, label, floor_y):
        half = float(np.max(np.asarray(inst.obb.sizes)) / 2 + success_m)
        xs = np.arange(inst.center[0] - half, inst.center[0] + half + 1e-6, step)
        zs = np.arange(inst.center[2] - half, inst.center[2] + half + 1e-6, step)
        for x in xs:
            for z in zs:
                p = np.array([x, floor_y, z], np.float32)
                if not sim.pathfinder.is_navigable(p, 0.5):
                    continue
                q = np.asarray(sim.pathfinder.snap_point(p), np.float32)
                if abs(q[1] - floor_y) < 0.3 and distance_to(inst, q) <= success_m:
                    pts.append(q)
    return pts


def shortest_path_m(sim, start, label, floor_y, success_m: float = 1.0) -> float:
    ends = success_points(sim, label, floor_y, success_m)
    if not ends:
        return float("inf")
    if any(np.hypot(*(np.asarray(e)[[0, 2]] - np.asarray(start)[[0, 2]])) < 1e-3 for e in ends):
        return 0.0
    mp = habitat_sim.MultiGoalShortestPath()
    mp.requested_start = np.asarray(start, np.float32)
    mp.requested_ends = ends
    return float(mp.geodesic_distance) if sim.pathfinder.find_path(mp) else float("inf")


def score_episode(dataset_config: str, scene_id: str, ep, instructions, success_m: float = 1.0) -> dict:
    """Per-leg success / SPL / path length for a finished `TimedEpisode` of an `ObjectSearchNavigator`."""
    from foresight.sim.sensors import make_sim_config

    last = ep.steps[-1] if ep.steps else {}
    found = last.get("found", [])
    times = np.asarray(ep.times)
    pos = np.asarray(ep.positions)
    floor_y = float(ep.start[1])
    # Instance OBBs come out zero-sized at the origin unless the semantic mesh is loaded *with* an RGB sensor
    # (habitat-sim 0.3.3: a semantic-only config also gives zeros).
    sim = habitat_sim.Simulator(make_sim_config(dataset_config, scene_id, 64, 64, sensor_uuids=("rgb", "semantic")))
    try:
        legs, k0, start = [], 0, pos[0]
        for i, label in enumerate(instructions):
            leg = {"label": label, "declared": False, "success": False, "spl": 0.0}
            shortest = shortest_path_m(sim, start, label, floor_y, success_m)
            leg["shortest_m"] = shortest
            if i < len(found):
                t_found = found[i][0]
                k1 = int(np.searchsorted(times, t_found, side="right"))  # positions[k] is after tick k-1
                path = float(np.sum(np.linalg.norm(np.diff(pos[k0:k1 + 1][:, [0, 2]], axis=0), axis=1)))
                at = pos[k1]
                inst, d = nearest(sim, label, at, floor_y)
                ok = d <= success_m
                leg.update(declared=True, t_found=float(t_found), path_m=path, dist_to_nearest_m=d,
                           nearest_id=None if inst is None else inst.semantic_id, success=bool(ok),
                           spl=float(ok) * shortest / max(path, shortest) if np.isfinite(shortest) else 0.0)
                k0, start = k1, at
            else:
                legs.append(leg)
                break
            legs.append(leg)
        crashed = ep.contact_at is not None
        if crashed:  # a crash ends the run: anything after it can't count
            for leg in legs:
                if leg.get("t_found") is not None and leg["t_found"] > ep.contact_at:
                    leg.update(success=False, spl=0.0)
        return {"legs": legs, "crashed": crashed, "contact_at": ep.contact_at,
                "navmesh_collisions": int(np.sum(ep.collided)),
                "sim_s": float(times[-1] + ep.steps[-1]["dt"]) if len(times) else 0.0,
                "end_mode": last.get("mode")}
    finally:
        sim.close()
