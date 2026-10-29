"""Generic closed-loop episode runner shared by the action fusion controller (#6) and the
depth-only heuristic baseline (#7).

Both drive a bare habitat_sim.Simulator the same way — same start/goal sampling, same zero-shot
depth estimator, same kinematic velocity integration, same termination logic — and differ only
in the `decide` callable that turns (rgb, depth, goal_bearing) into a velocity command. Sharing
this loop (rather than two near-identical copies) is what makes the eventual VLM-vs-baseline
ablation (Project_FORESIGHT deliverable B) a fair one: given the same seed, both conditions see
literally the same start/goal/scene and the same integration step, so any difference in outcome
traces back to the heading source, not to incidental differences in how the episode was run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis

from foresight.perception.depth.depth_anything import ZeroShotDepthEstimator
from foresight.sim.navigate import _sample_start_goal
from foresight.sim.pose import goal_bearing_rad
from foresight.sim.sensors import make_sim_config
from foresight.sim.velocity_step import integrate_velocity_step

# rgb (H,W,3) uint8, depth (H,W) float32 meters, goal_bearing_rad float
# -> (linear_velocity m/s, angular_velocity rad/s, extra per-tick diagnostics dict)
DecideFn = Callable[[np.ndarray, np.ndarray, float], tuple]


@dataclass
class ClosedLoopEpisode:
    rgb_frames: list  # (H, W, 3) uint8 per control tick
    positions: np.ndarray  # (T, 3) float32, includes the start pose
    start: np.ndarray
    goal: np.ndarray
    planned_path: np.ndarray
    geodesic_distance: float
    success: bool
    collided_steps: int
    steps: list  # per-tick diagnostics dicts
    topdown: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    meters_per_pixel: float = field(default=0.05)


def run_closed_loop_episode(
    dataset_config: str,
    scene_id: str,
    decide: DecideFn,
    depth_estimator: ZeroShotDepthEstimator,
    width: int = 640,
    height: int = 480,
    seed: int = 0,
    min_path_dist: float = 5.0,
    goal_radius: float = 0.3,
    max_steps: int = 30,
    time_step: float = 1.0,
    meters_per_pixel: float = 0.05,
    max_sample_tries: int = 200,
    start_position: Optional[np.ndarray] = None,
    start_yaw_rad: float = 0.0,
    goal_position: Optional[np.ndarray] = None,
) -> ClosedLoopEpisode:
    """start_position/goal_position: pin a hand-picked episode (e.g. a staged mirror encounter)
    instead of sampling one from `seed`; both must be given together. start_yaw_rad rotates the
    agent about +Y (0 = facing -Z, habitat_sim's default forward)."""
    if (start_position is None) != (goal_position is None):
        raise ValueError("start_position and goal_position must be given together")
    cfg = make_sim_config(dataset_config, scene_id, width, height, sensor_uuids=("rgb",))
    sim = habitat_sim.Simulator(cfg)
    try:
        sim.pathfinder.seed(seed)
        agent = sim.get_agent(0)

        if start_position is None:
            start, goal, planned_path, geo_dist = _sample_start_goal(
                sim.pathfinder, min_path_dist, max_sample_tries
            )
        else:
            start = np.asarray(sim.pathfinder.snap_point(start_position), np.float32)
            goal = np.asarray(sim.pathfinder.snap_point(goal_position), np.float32)
            path = habitat_sim.ShortestPath()
            path.requested_start, path.requested_end = start, goal
            if not (sim.pathfinder.find_path(path) and np.isfinite(path.geodesic_distance)):
                raise RuntimeError(f"no navmesh path from {start} to {goal}")
            planned_path = np.asarray(path.points, np.float32)
            geo_dist = float(path.geodesic_distance)
        state = habitat_sim.AgentState()
        state.position = start
        state.rotation = quat_from_angle_axis(start_yaw_rad, np.array([0.0, 1.0, 0.0]))
        agent.set_state(state)

        rgb_frames, positions, step_log = [], [np.asarray(start, np.float32)], []
        success, collided_steps = False, 0

        for _ in range(max_steps):
            agent_state = agent.get_state()
            dist_to_goal = float(np.linalg.norm(np.asarray(agent_state.position) - goal))
            if dist_to_goal < goal_radius:
                success = True
                break

            obs = sim.get_sensor_observations()
            rgb = np.array(obs["rgb"][..., :3], dtype=np.uint8)
            depth = depth_estimator.estimate(rgb)
            bearing = goal_bearing_rad(agent_state.position, agent_state.rotation, goal)

            linear_velocity, angular_velocity, diagnostics = decide(rgb, depth, bearing)
            collided = integrate_velocity_step(sim, agent, linear_velocity, angular_velocity, time_step)

            rgb_frames.append(rgb)
            positions.append(np.asarray(agent.get_state().position, np.float32))
            collided_steps += int(collided)
            step_log.append(
                {
                    "dist_to_goal_m": dist_to_goal,
                    "goal_bearing_rad": bearing,
                    "linear_velocity": linear_velocity,
                    "angular_velocity": angular_velocity,
                    "collided": collided,
                    **diagnostics,
                }
            )

        bmin, bmax = sim.pathfinder.get_bounds()
        topdown = sim.pathfinder.get_topdown_view(meters_per_pixel, float(start[1]))

        return ClosedLoopEpisode(
            rgb_frames=rgb_frames,
            positions=np.stack(positions).astype(np.float32),
            start=start,
            goal=goal,
            planned_path=planned_path,
            geodesic_distance=geo_dist,
            success=success,
            collided_steps=collided_steps,
            steps=step_log,
            topdown=np.asarray(topdown, dtype=bool),
            bounds_min=np.asarray(bmin, np.float32),
            bounds_max=np.asarray(bmax, np.float32),
            meters_per_pixel=meters_per_pixel,
        )
    finally:
        sim.close()
