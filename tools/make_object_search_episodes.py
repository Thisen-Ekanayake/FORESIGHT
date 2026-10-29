#!/usr/bin/env python
"""Generate the fixed object-search evaluation set (seeded; written once, before any evaluation run).

Per scene/floor: random navigable start, random yaw, a target category with an annotated instance on that floor,
3-12 m away by the shortest path to a success point (within 1 m of an instance), and not visible from the start
(< 0.5% of pixels). Plus the 00808 mirror-pose episodes (start facing the hallway mirror). Output:
experiments/configs/object_search/episodes.json, read by tools/run_object_search.py --episodes eval.
"""
import json
import math
import sys
from pathlib import Path

import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foresight.sim.objectnav_eval import CATEGORY_NAMES, instances, shortest_path_m  # noqa: E402
from foresight.sim.sensors import make_sim_config  # noqa: E402

HM3D = "data/processed/scene_datasets/hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json"
# (scene, floor y, episodes): the floors with the most navigable area / target categories (checked 2026-10-30)
PLAN = [("00808-y9hTuugGdiq", 0.06, 4), ("00800-TEEsavR23oF", 0.2, 2), ("00800-TEEsavR23oF", 3.2, 2),
        ("00802-wcojb4TFT35", 0.0, 4), ("00803-k1cupFYWXJ6", 6.5, 2)]
LABELS = ["sofa", "bed", "tv", "refrigerator", "toilet"]
MIRROR_START = dict(start=[-2.592, 0.061, 0.290], yaw_deg=-89.1)
OUT = Path("experiments/configs/object_search/episodes.json")


def main():
    rng = np.random.default_rng(2026)
    eps = []
    for scene, floor, n in PLAN:
        sim = habitat_sim.Simulator(make_sim_config(HM3D, scene, 320, 320, sensor_uuids=("rgb", "depth", "semantic")))
        agent = sim.get_agent(0)
        labels = [l for l in LABELS if instances(sim, l, floor)]
        got = 0
        for _ in range(3000):
            if got >= n:
                break
            p = np.asarray(sim.pathfinder.get_random_navigable_point(), np.float32)
            if abs(p[1] - floor) > 0.35:
                continue
            lab = labels[int(rng.integers(len(labels)))]
            d = shortest_path_m(sim, p, lab, float(p[1]))
            if not 3.0 < d < 12.0:
                continue
            yaw = float(rng.uniform(-math.pi, math.pi))
            agent.set_state(habitat_sim.AgentState(position=p, rotation=quat_from_angle_axis(yaw, np.array([0, 1.0, 0]))))
            obs = sim.get_sensor_observations()
            if np.median(obs["depth"]) < 0.8:
                continue
            ids = [i.semantic_id for i in instances(sim, lab, float(p[1]))]
            if np.isin(obs["semantic"], ids).mean() > 0.005:
                continue
            got += 1
            eps.append(dict(name=f"{scene[:5]}-{lab}-{got}", scene=scene, start=[round(float(x), 3) for x in p],
                            yaw_deg=round(math.degrees(yaw), 1), instructions=[lab], max_sim_s=180.0,
                            shortest_m=round(d, 2)))
            print(eps[-1], flush=True)
        sim.close()
    for lab in ["tv", "refrigerator"]:
        eps.append(dict(name=f"00808-mirror-{lab}", scene="00808-y9hTuugGdiq", **MIRROR_START,
                        instructions=[lab], max_sim_s=180.0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"seed": 2026, "note": "fixed before any evaluation run; do not edit by hand",
                               "episodes": eps}, indent=1))
    print(f"{len(eps)} episodes -> {OUT}")


if __name__ == "__main__":
    main()
