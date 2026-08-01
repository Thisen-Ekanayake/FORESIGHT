"""Depth-only potential-field heuristic planner — the control condition for the VLM ablation.

Project_FORESIGHT.pdf §1 (baseline planner) / §3 (deliverable B): "a depth-only potential-field
/ heuristic planner that consumes the same pseudo-depth but no VLM ... isolates the marginal
value of the VLM." Shares the same downstream heading->velocity control law
(`foresight.planning.fusion.heading_to_velocity`) and the same `ReactiveSafetyLayer` as
`ActionFusionController` (milestone #6) — the only thing this baseline swaps out is *how the
heading is proposed*: pure geometry over the depth map instead of a VLM call, so the eventual
ablation isolates the VLM's contribution rather than incidental control-law differences.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from foresight.planning.fusion import heading_to_velocity
from foresight.safety.reactive import ReactiveSafetyLayer, SafetyResult


def propose_heading_from_depth(
    depth: np.ndarray,
    goal_bearing_rad: float,
    hfov_rad: float = np.deg2rad(90.0),
    num_bins: int = 9,
    influence_range_m: float = 2.0,
    repulsion_gain: float = 1.0,
    row_frac: tuple[float, float] = (0.4, 0.9),
) -> float:
    """Potential-field heading: an attractive pull toward the goal bearing, plus a repulsive
    push away from every depth-map angular bin whose mean depth is inside `influence_range_m`
    (closer bins push harder, via a 1/d - 1/d_max falloff). Bins run left-to-right across
    `hfov_rad`, matching the camera's horizontal field of view (habitat_sim default: 90 deg).

    No VLM, no learned model, no history — a pure function of the current depth frame and goal
    bearing. Returns radians relative to straight-ahead (positive = right), clipped to the
    camera's +/- hfov/2 so it stays within what the depth map could have justified.
    """
    h, w = depth.shape
    r0, r1 = int(row_frac[0] * h), int(row_frac[1] * h)
    band = depth[r0:r1]

    fx, fy = np.cos(goal_bearing_rad), np.sin(goal_bearing_rad)

    edges = np.linspace(0, w, num_bins + 1).astype(int)
    for i in range(num_bins):
        cell = band[:, edges[i] : edges[i + 1]]
        valid = cell[np.isfinite(cell) & (cell > 0)]
        if valid.size == 0:
            continue
        d = float(valid.mean())
        if d >= influence_range_m:
            continue
        bin_angle = -hfov_rad / 2 + (i + 0.5) / num_bins * hfov_rad
        magnitude = repulsion_gain * (1.0 / d - 1.0 / influence_range_m)
        fx -= magnitude * np.cos(bin_angle)
        fy -= magnitude * np.sin(bin_angle)

    heading = float(np.arctan2(fy, fx))
    return float(np.clip(heading, -hfov_rad / 2, hfov_rad / 2))


@dataclass
class DepthHeuristicResult:
    linear_velocity: float
    angular_velocity: float
    proposed_heading_rad: float
    safety: SafetyResult


class DepthOnlyHeuristicController:
    """No VLM, so no fast-slow split is needed — the potential field is cheap enough to
    recompute every control tick, unlike `ActionFusionController`'s budgeted VLM query.
    """

    def __init__(
        self,
        max_linear_mps: float = 0.25,
        max_angular_radps: float = np.deg2rad(10.0),
        heading_gain: float = 1.5,
        hfov_rad: float = np.deg2rad(90.0),
        num_bins: int = 9,
        influence_range_m: float = 2.0,
        repulsion_gain: float = 1.0,
        safety: Optional[ReactiveSafetyLayer] = None,
    ):
        self.max_linear_mps = max_linear_mps
        self.max_angular_radps = max_angular_radps
        self.heading_gain = heading_gain
        self.hfov_rad = hfov_rad
        self.num_bins = num_bins
        self.influence_range_m = influence_range_m
        self.repulsion_gain = repulsion_gain
        self.safety = safety or ReactiveSafetyLayer()

    def step(self, depth: np.ndarray, goal_bearing_rad: float) -> DepthHeuristicResult:
        proposed = propose_heading_from_depth(
            depth,
            goal_bearing_rad,
            hfov_rad=self.hfov_rad,
            num_bins=self.num_bins,
            influence_range_m=self.influence_range_m,
            repulsion_gain=self.repulsion_gain,
        )
        linear_velocity, angular_velocity = heading_to_velocity(
            proposed, self.max_linear_mps, self.max_angular_radps, self.heading_gain
        )
        safety = self.safety.apply(depth, linear_velocity, angular_velocity)
        return DepthHeuristicResult(
            linear_velocity=safety.linear_velocity,
            angular_velocity=safety.angular_velocity,
            proposed_heading_rad=proposed,
            safety=safety,
        )
