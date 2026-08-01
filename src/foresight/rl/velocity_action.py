"""Continuous velocity task-action that tolerates array-shaped velocity arguments.

habitat-baselines' gym wrapper (`continuous_vector_action_to_hab_dict`) passes each velocity as a shape-(1,)
array — a slice of the flat Gaussian-policy output. The stock habitat-lab `VelocityAction.step` then builds
`np.array([0.0, 0.0, -linear_velocity])`, which raises "setting an array element with a sequence
(inhomogeneous shape)" under NumPy >= 1.24 (this env has 1.26). This subclass squeezes both args to Python
floats first, then defers to the stock implementation (kinematic `VelocityControl` integration + navmesh
sliding). Referenced via `habitat.task.actions.velocity_control.type: SmoothVelocityAction` in the config.
"""
from typing import Any

import numpy as np
from habitat.core.registry import registry
from habitat.tasks.nav.nav import VelocityAction


@registry.register_task_action
class SmoothVelocityAction(VelocityAction):
    def step(self, *args: Any, linear_velocity, angular_velocity, **kwargs: Any):
        linear_velocity = float(np.asarray(linear_velocity).reshape(-1)[0])
        angular_velocity = float(np.asarray(angular_velocity).reshape(-1)[0])
        return super().step(
            *args, linear_velocity=linear_velocity, angular_velocity=angular_velocity, **kwargs
        )
