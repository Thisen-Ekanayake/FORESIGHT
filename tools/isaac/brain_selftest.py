#!/usr/bin/env python
"""Self-test of the brain protocol without Isaac: a Habitat point-goal episode whose decisions come from
tools/isaac/foresight_brain.py over the socket (same loop as foresight.sim.timed_episode).

  python tools/isaac/foresight_brain.py &      # in another shell
  python tools/isaac/brain_selftest.py --cond map --episode e-00808-...  [--max-sim-s 40]
"""
import argparse
import json
import math
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import wire  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="map")
    ap.add_argument("--episode", default=None)
    ap.add_argument("--set", default="E_3005")
    ap.add_argument("--max-sim-s", type=float, default=30.0)
    a = ap.parse_args()
    import habitat_sim
    import quaternion
    from habitat_sim.utils.common import quat_from_angle_axis
    from make_pointnav_episodes import ANNOTATED, scene_ref
    from foresight.sim.pose import goal_bearing_rad
    from foresight.sim.sensors import make_sim_config
    from foresight.sim.velocity_step import integrate_velocity_step

    eps = json.load(open(ROOT / f"experiments/configs/pointnav/episodes_{a.set}.json"))["episodes"]
    e = next(x for x in eps if x["name"] == a.episode) if a.episode else eps[0]
    sim = habitat_sim.Simulator(make_sim_config(str(ANNOTATED), scene_ref(ANNOTATED, e["scene"]), 640, 640,
                                                sensor_uuids=("rgb", "depth")))
    agent = sim.get_agent(0)
    st = habitat_sim.AgentState()
    st.position = np.asarray(sim.pathfinder.snap_point(np.asarray(e["start"], np.float32)))
    st.rotation = quat_from_angle_axis(math.radians(e["yaw_deg"]), np.array([0.0, 1.0, 0.0]))
    agent.set_state(st)
    goal = np.asarray(e["goal"], float)
    s = socket.create_connection(("127.0.0.1", wire.PORT))
    wire.send(s, {"type": "reset", "cond": a.cond, "start_xz": list(np.asarray(e["start"])[[0, 2]]),
                  "goal_xz": list(goal[[0, 2]]), "cam_height_m": 1.5, "hazard_labels": ["mirror"], "robust_scan": False,
                  "log_path": str(ROOT / f"results/isaac/brain_selftest/{a.cond}.pkl")})
    wire.recv(s)
    t, result = 0.0, "timeout"
    while t < a.max_sim_s:
        ag = agent.get_state()
        if np.hypot(*(goal - ag.position)[[0, 2]]) < 0.5:
            result = f"reached at {t:.1f} s"
            break
        fwd = quaternion.rotate_vectors(ag.rotation, np.array([0.0, 0.0, -1.0]))
        obs = sim.get_sensor_observations()
        rgb = np.ascontiguousarray(obs["rgb"][..., :3])
        h, w = obs["depth"].shape
        gt_ahead = float(np.median(obs["depth"][int(.4 * h):int(.9 * h), int(.25 * w):int(.75 * w)]))
        wire.send(s, {"type": "tick", "t": t, "position": [float(v) for v in ag.position],
                      "forward_xz": [float(fwd[0]), float(fwd[2])],
                      "bearing": float(goal_bearing_rad(ag.position, ag.rotation, goal)), "h": 640, "w": 640},
                  rgb.tobytes())
        r, _ = wire.recv(s)
        dt = max(0.1, r["tick_ms"] / 1000)
        integrate_velocity_step(sim, agent, r["linear"], r["angular"], dt)
        t += dt
        if int(t * 10) % 50 == 0:
            print(f"  t {t:5.1f} lin {r['linear']:.2f} ang {r['angular']:+.2f} tick {r['tick_ms']:.0f} ms "
                  f"{r['safety']} {r['event']}", flush=True)
        if gt_ahead < 0.3 and r["linear"] > 0.02:
            result = f"crash at {t:.1f} s"
            break
    wire.send(s, {"type": "end", "result": result})
    wire.recv(s)
    print(e["name"], a.cond, result)


if __name__ == "__main__":
    main()
