"""Composite PointGoal reward measure: rewards geodesic progress toward the goal while penalizing
collisions and non-smooth control, to produce stable, smooth continuous-velocity navigation.

`RLTaskEnv.get_reward` already adds `slack_reward` every step and `success_reward` on success, so this
measure must contribute ONLY the shaped terms:

    r_measure = progress - w_col * collided - w_ang * |angular_vel| - w_jerk * (|dlin| + |dang|)

where progress = -(new_geodesic_dist - prev_geodesic_dist), and the velocity terms read the continuous
`velocity_control` action (linear/angular in [-1, 1]). w_ang suppresses spin-in-place; w_jerk suppresses
abrupt velocity changes (jerk) -> smooth trajectories.

Importing this module registers both the measure (runtime registry) and its Hydra structured config,
so `tools/train_pointnav.py` / `record_pointnav_demo.py` import it before composing the config.
"""
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from habitat.config.default_structured_configs import MeasurementConfig
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.tasks.nav.nav import DistanceToGoal
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@registry.register_measure
class PointNavSmoothReward(Measure):
    cls_uuid: str = "pointnav_smooth_reward"

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._previous_distance: Optional[float] = None
        self._prev_action: Optional[Tuple[float, float]] = None
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, [DistanceToGoal.cls_uuid])
        self._previous_distance = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._prev_action = None
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    @staticmethod
    def _velocities(action) -> Tuple[Optional[float], Optional[float]]:
        """Pull (linear, angular) out of the velocity_control action dict; None if unavailable (e.g. reset)."""
        if not isinstance(action, dict):
            return None, None
        args = action.get("action_args") or {}
        lin, ang = args.get("linear_velocity"), args.get("angular_velocity")
        if lin is None or ang is None:
            return None, None
        return float(np.asarray(lin).reshape(-1)[0]), float(np.asarray(ang).reshape(-1)[0])

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, action=None, **kwargs: Any):
        distance = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        reward = -(distance - self._previous_distance)  # geodesic progress toward goal
        self._previous_distance = distance

        if getattr(self._sim, "previous_step_collided", False):
            reward -= self._config.collision_penalty

        lin, ang = self._velocities(action)
        if ang is not None:
            reward -= self._config.angular_penalty * abs(ang)
            if self._prev_action is not None:
                reward -= self._config.jerk_penalty * (
                    abs(lin - self._prev_action[0]) + abs(ang - self._prev_action[1])
                )
            self._prev_action = (lin, ang)

        self._metric = reward


@dataclass
class PointNavSmoothRewardMeasurementConfig(MeasurementConfig):
    """Schema only — the weights are MISSING here on purpose so the values live in the YAML config
    (`habitat.task.measurements.pointnav_smooth_reward` in experiments/configs/rl/pointnav_continuous.yaml)
    rather than being duplicated in code. Composing without them fails loudly."""

    type: str = "PointNavSmoothReward"
    collision_penalty: float = MISSING  # per colliding step
    angular_penalty: float = MISSING    # per unit |angular_velocity| (action in [-1, 1])
    jerk_penalty: float = MISSING       # per unit L1 change in (linear, angular) between steps


ConfigStore.instance().store(
    package="habitat.task.measurements.pointnav_smooth_reward",
    group="habitat/task/measurements",
    name="pointnav_smooth_reward",
    node=PointNavSmoothRewardMeasurementConfig,
)
