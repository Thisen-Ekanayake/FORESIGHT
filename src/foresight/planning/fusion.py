"""Action fusion controller: Project_FORESIGHT's fast-slow architecture (§2).

A compact VLM proposes a heading at a budgeted 0.5-2 Hz (the "slow" planner); every control
step (spec'd 10-30 Hz) fuses that (possibly stale) heading with the current goal bearing into
a velocity command, then passes it through a `ReactiveSafetyLayer` reading the per-frame depth
map. Model calls (VLM heading proposal) are injected as a callable so the control law here can
be unit-tested with synthetic inputs, without loading the VLM.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from foresight.safety.reactive import ReactiveSafetyLayer, SafetyResult


def wrap_pi(angle: float) -> float:
    """Wrap an angle (radians) into (-pi, pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class FusionResult:
    linear_velocity: float
    angular_velocity: float
    fused_heading_rad: float
    vlm_heading_rad: Optional[float]
    vlm_queried: bool
    safety: SafetyResult


class ActionFusionController:
    def __init__(
        self,
        propose_heading: Callable[[np.ndarray], Optional[float]],
        vlm_hz: float = 1.0,
        vlm_weight: float = 0.5,
        max_linear_mps: float = 0.25,
        max_angular_radps: float = np.deg2rad(10.0),
        heading_gain: float = 1.5,
        safety: Optional[ReactiveSafetyLayer] = None,
    ):
        """
        propose_heading: rgb (H,W,3) uint8 -> heading in radians relative to straight-ahead
            (positive = right), or None if no clear proposal (e.g. VLM reports the way blocked).
        vlm_hz: planner query budget, held to the spec'd 0.5-2 Hz range.
        vlm_weight: fusion weight given to the (cached) VLM heading vs. the goal bearing,
            combined as a circular weighted average so a stale heading behind the agent
            doesn't dominate on wrap-around.
        heading_gain: proportional gain, fused heading error (rad) -> angular velocity (rad/s).
        """
        if not 0.5 <= vlm_hz <= 2.0:
            raise ValueError(f"vlm_hz={vlm_hz} outside the spec'd 0.5-2 Hz planner budget")
        self._propose_heading = propose_heading
        self.vlm_period_s = 1.0 / vlm_hz
        self.vlm_weight = vlm_weight
        self.max_linear_mps = max_linear_mps
        self.max_angular_radps = max_angular_radps
        self.heading_gain = heading_gain
        self.safety = safety or ReactiveSafetyLayer()

        self._vlm_heading_rad: Optional[float] = None
        self._last_vlm_time: float = -np.inf

    def reset(self) -> None:
        self._vlm_heading_rad = None
        self._last_vlm_time = -np.inf

    def step(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        goal_bearing_rad: float,
        now: Optional[float] = None,
    ) -> FusionResult:
        now = time.monotonic() if now is None else now

        vlm_queried = (now - self._last_vlm_time) >= self.vlm_period_s
        if vlm_queried:
            self._vlm_heading_rad = self._propose_heading(rgb)
            self._last_vlm_time = now

        if self._vlm_heading_rad is None:
            fused_heading = wrap_pi(goal_bearing_rad)
        else:
            w = self.vlm_weight
            fused_heading = float(
                np.arctan2(
                    (1 - w) * np.sin(goal_bearing_rad) + w * np.sin(self._vlm_heading_rad),
                    (1 - w) * np.cos(goal_bearing_rad) + w * np.cos(self._vlm_heading_rad),
                )
            )

        heading_error = wrap_pi(fused_heading)
        angular_velocity = float(
            np.clip(self.heading_gain * heading_error, -self.max_angular_radps, self.max_angular_radps)
        )
        # Slow down for sharp turns so the agent doesn't clip corners while rotating in place.
        turn_frac = min(1.0, abs(heading_error) / (np.pi / 2))
        linear_velocity = self.max_linear_mps * (1.0 - turn_frac)

        safety = self.safety.apply(depth, linear_velocity, angular_velocity)

        return FusionResult(
            linear_velocity=safety.linear_velocity,
            angular_velocity=safety.angular_velocity,
            fused_heading_rad=fused_heading,
            vlm_heading_rad=self._vlm_heading_rad,
            vlm_queried=vlm_queried,
            safety=safety,
        )
