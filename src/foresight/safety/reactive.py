"""Reactive safety layer: a per-frame obstacle guard over the (monocular, zero-shot) depth map.

This is the fast half of Project_FORESIGHT's fast-slow architecture (spec'd 10-30 Hz — in
practice, "called every control step", since it has no internal timing budget of its own).
It sits downstream of the action fusion controller (`foresight.planning.fusion`) and either
passes its commanded velocity through unchanged or overrides it when the depth map shows an
obstacle inside the safety margins.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyResult:
    linear_velocity: float
    angular_velocity: float
    min_clearance_m: float
    level: str  # "clear" | "slow" | "stop"


def _mean_valid(region: np.ndarray) -> float:
    valid = region[np.isfinite(region) & (region > 0)]
    return float(valid.mean()) if valid.size else 0.0


class ReactiveSafetyLayer:
    """Looks at a forward collision cone (central columns, lower-middle rows — where a
    frontal obstacle projects in a forward-facing camera) for the closest valid depth:

      - closer than `stop_distance_m`: hard override — stop and turn toward whichever image
        half (left/right) has more average clearance.
      - closer than `slow_distance_m`: linearly scale down the commanded linear velocity.
      - otherwise: pass the commanded velocity through unchanged.
    """

    def __init__(
        self,
        stop_distance_m: float = 0.4,
        slow_distance_m: float = 1.2,
        turn_radps: float = np.deg2rad(20.0),
        row_frac: tuple[float, float] = (0.4, 0.9),
        col_frac: tuple[float, float] = (0.25, 0.75),
    ):
        if stop_distance_m >= slow_distance_m:
            raise ValueError("stop_distance_m must be < slow_distance_m")
        self.stop_distance_m = stop_distance_m
        self.slow_distance_m = slow_distance_m
        self.turn_radps = turn_radps
        self.row_frac = row_frac
        self.col_frac = col_frac

    def _cone(self, depth: np.ndarray) -> np.ndarray:
        h, w = depth.shape
        r0, r1 = int(self.row_frac[0] * h), int(self.row_frac[1] * h)
        c0, c1 = int(self.col_frac[0] * w), int(self.col_frac[1] * w)
        return depth[r0:r1, c0:c1]

    def apply(self, depth: np.ndarray, linear_velocity: float, angular_velocity: float) -> SafetyResult:
        cone = self._cone(depth)
        valid = cone[np.isfinite(cone) & (cone > 0)]
        min_clearance = float(valid.min()) if valid.size else float("inf")

        if min_clearance < self.stop_distance_m:
            w = depth.shape[1]
            left_clear = _mean_valid(depth[:, : w // 2])
            right_clear = _mean_valid(depth[:, w // 2 :])
            turn = self.turn_radps if left_clear >= right_clear else -self.turn_radps
            return SafetyResult(0.0, turn, min_clearance, "stop")

        if min_clearance < self.slow_distance_m:
            scale = (min_clearance - self.stop_distance_m) / (self.slow_distance_m - self.stop_distance_m)
            return SafetyResult(linear_velocity * scale, angular_velocity, min_clearance, "slow")

        return SafetyResult(linear_velocity, angular_velocity, min_clearance, "clear")
