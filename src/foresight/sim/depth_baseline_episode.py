"""Closed-loop episode runner for the depth-only heuristic baseline (PROGRESS.md milestone #7).

Thin wrapper over `foresight.sim.closed_loop_episode.run_closed_loop_episode` — the same shared
loop `action_fusion_episode.py` uses, so a given seed/scene produces the exact same start, goal,
and per-tick depth estimate as the VLM-driven episode. Only the `decide` closure differs: a
`DepthOnlyHeuristicController` instead of `ActionFusionController`, i.e. no VLM in the loop.
"""
from __future__ import annotations

import numpy as np

from foresight.perception.depth.depth_anything import DEFAULT_MODEL as DEFAULT_DEPTH_MODEL
from foresight.perception.depth.depth_anything import ZeroShotDepthEstimator
from foresight.planning.depth_heuristic import DepthOnlyHeuristicController
from foresight.sim.closed_loop_episode import ClosedLoopEpisode, run_closed_loop_episode

DepthBaselineEpisode = ClosedLoopEpisode


def run_depth_baseline_episode(
    dataset_config: str,
    scene_id: str,
    width: int = 640,
    height: int = 480,
    seed: int = 0,
    min_path_dist: float = 5.0,
    goal_radius: float = 0.3,
    max_steps: int = 30,
    time_step: float = 1.0,
    max_linear_mps: float = 0.25,
    max_angular_degps: float = 10.0,
    influence_range_m: float = 2.0,
    repulsion_gain: float = 1.0,
    depth_model: str = DEFAULT_DEPTH_MODEL,
    meters_per_pixel: float = 0.05,
    max_sample_tries: int = 200,
) -> ClosedLoopEpisode:
    depth_estimator = ZeroShotDepthEstimator(model_name=depth_model)
    controller = DepthOnlyHeuristicController(
        max_linear_mps=max_linear_mps,
        max_angular_radps=np.deg2rad(max_angular_degps),
        influence_range_m=influence_range_m,
        repulsion_gain=repulsion_gain,
    )

    def decide(rgb, depth, bearing):
        result = controller.step(depth, bearing)
        diagnostics = {
            "proposed_heading_rad": result.proposed_heading_rad,
            "safety_level": result.safety.level,
            "min_clearance_m": result.safety.min_clearance_m,
        }
        return result.linear_velocity, result.angular_velocity, diagnostics

    return run_closed_loop_episode(
        dataset_config=dataset_config,
        scene_id=scene_id,
        decide=decide,
        depth_estimator=depth_estimator,
        width=width,
        height=height,
        seed=seed,
        min_path_dist=min_path_dist,
        goal_radius=goal_radius,
        max_steps=max_steps,
        time_step=time_step,
        meters_per_pixel=meters_per_pixel,
        max_sample_tries=max_sample_tries,
    )
