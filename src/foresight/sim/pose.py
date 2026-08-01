"""Agent-relative pose math shared by the closed-loop action-fusion episode runner."""
from __future__ import annotations

import numpy as np
import quaternion


def goal_bearing_rad(position: np.ndarray, rotation: "quaternion.quaternion", goal: np.ndarray) -> float:
    """Bearing to `goal` in the agent's local frame: 0 = straight ahead, positive = to the right.

    Matches the sign convention of `foresight.perception.vlm.qwen_vl.propose_heading` and
    `foresight.planning.fusion.ActionFusionController`. Computed in the horizontal (x-z) plane,
    ignoring height (habitat_sim: +Y up, local forward = -Z, local right = +X).
    """
    forward = quaternion.rotate_vectors(rotation, np.array([0.0, 0.0, -1.0]))
    right = quaternion.rotate_vectors(rotation, np.array([1.0, 0.0, 0.0]))
    to_goal = np.asarray(goal, np.float64) - np.asarray(position, np.float64)
    to_goal[1] = 0.0
    return float(np.arctan2(np.dot(to_goal, right), np.dot(to_goal, forward)))
