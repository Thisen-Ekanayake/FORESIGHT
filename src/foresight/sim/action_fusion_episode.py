"""Closed-loop episode runner for the action fusion controller (PROGRESS.md milestone #6).

Drives a bare habitat_sim.Simulator step-by-step: at each control tick, render RGB, run the
zero-shot depth estimator, compute the goal bearing, run one `ActionFusionController.step`
(which internally budgets its VLM heading queries to 0.5-2 Hz and applies the reactive safety
layer every tick), and integrate the resulting velocity command via
`foresight.sim.velocity_step.integrate_velocity_step`. No habitat-lab task/RL machinery
involved, matching `foresight.sim.navigate`'s standalone style.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import habitat_sim
import numpy as np

from foresight.perception.depth.depth_anything import DEFAULT_MODEL as DEFAULT_DEPTH_MODEL
from foresight.perception.depth.depth_anything import ZeroShotDepthEstimator
from foresight.perception.vlm.qwen_vl import DEFAULT_MODEL as DEFAULT_VLM_MODEL
from foresight.perception.vlm.qwen_vl import Qwen3VLSceneReasoner
from foresight.planning.fusion import ActionFusionController
from foresight.sim.navigate import _sample_start_goal
from foresight.sim.pose import goal_bearing_rad
from foresight.sim.sensors import make_sim_config
from foresight.sim.velocity_step import integrate_velocity_step


@dataclass
class ActionFusionEpisode:
    rgb_frames: list  # (H, W, 3) uint8 per control tick
    positions: np.ndarray  # (T, 3) float32
    start: np.ndarray
    goal: np.ndarray
    planned_path: np.ndarray
    geodesic_distance: float
    success: bool
    collided_steps: int
    steps: list  # per-tick diagnostics dicts (see run_action_fusion_episode)
    topdown: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    meters_per_pixel: float = field(default=0.05)


def run_action_fusion_episode(
    dataset_config: str,
    scene_id: str,
    width: int = 640,
    height: int = 480,
    seed: int = 0,
    min_path_dist: float = 5.0,
    goal_radius: float = 0.3,
    max_steps: int = 30,
    time_step: float = 1.0,
    vlm_hz: float = 1.0,
    max_linear_mps: float = 0.25,
    max_angular_degps: float = 10.0,
    depth_model: str = DEFAULT_DEPTH_MODEL,
    vlm_model: str = DEFAULT_VLM_MODEL,
    meters_per_pixel: float = 0.05,
    max_sample_tries: int = 200,
) -> ActionFusionEpisode:
    cfg = make_sim_config(dataset_config, scene_id, width, height, sensor_uuids=("rgb",))
    sim = habitat_sim.Simulator(cfg)
    try:
        sim.pathfinder.seed(seed)
        agent = sim.get_agent(0)

        start, goal, planned_path, geo_dist = _sample_start_goal(
            sim.pathfinder, min_path_dist, max_sample_tries
        )
        state = habitat_sim.AgentState()
        state.position = start
        agent.set_state(state)

        depth_estimator = ZeroShotDepthEstimator(model_name=depth_model)
        vlm = Qwen3VLSceneReasoner(model_name=vlm_model)
        controller = ActionFusionController(
            propose_heading=vlm.propose_heading,
            vlm_hz=vlm_hz,
            max_linear_mps=max_linear_mps,
            max_angular_radps=np.deg2rad(max_angular_degps),
        )

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

            # Wall-clock, not simulated time: the VLM query budget is a real compute-time
            # constraint, so query cadence must reflect how long inference actually takes.
            result = controller.step(rgb, depth, bearing, now=time.monotonic())
            collided = integrate_velocity_step(
                sim, agent, result.linear_velocity, result.angular_velocity, time_step
            )

            rgb_frames.append(rgb)
            positions.append(np.asarray(agent.get_state().position, np.float32))
            collided_steps += int(collided)
            step_log.append(
                {
                    "dist_to_goal_m": dist_to_goal,
                    "goal_bearing_rad": bearing,
                    "fused_heading_rad": result.fused_heading_rad,
                    "vlm_heading_rad": result.vlm_heading_rad,
                    "vlm_queried": result.vlm_queried,
                    "linear_velocity": result.linear_velocity,
                    "angular_velocity": result.angular_velocity,
                    "safety_level": result.safety.level,
                    "min_clearance_m": result.safety.min_clearance_m,
                    "collided": collided,
                }
            )

        bmin, bmax = sim.pathfinder.get_bounds()
        topdown = sim.pathfinder.get_topdown_view(meters_per_pixel, float(start[1]))

        return ActionFusionEpisode(
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
