"""Continuous-velocity agent integration outside habitat-lab's task/action machinery.

Mirrors `habitat.tasks.nav.nav.VelocityAction.step` (kinematic `VelocityControl` integration +
navmesh sliding) so a bare `habitat_sim.Simulator` can be driven by (linear_velocity,
angular_velocity) commands without a full habitat-lab Env/task — used by the action-fusion
episode runner, which drives the sim directly to keep the VLM/depth/fusion/safety stack
decoupled from the RL config machinery under `foresight.rl`.
"""
from __future__ import annotations

import habitat_sim
import magnum as mn
import numpy as np


def integrate_velocity_step(
    sim: habitat_sim.Simulator,
    agent: habitat_sim.Agent,
    linear_velocity: float,
    angular_velocity: float,
    time_step: float,
    allow_sliding: bool = True,
) -> bool:
    """Advance `agent` by one control step at the given body-frame velocities.

    linear_velocity: m/s, forward-positive. angular_velocity: rad/s, positive = turn right
    (matches `foresight.planning.fusion`'s convention). Returns whether the navmesh sliding
    filter shortened the requested move (i.e. a collision occurred).
    """
    vel_control = habitat_sim.physics.VelocityControl()
    vel_control.controlling_lin_vel = True
    vel_control.controlling_ang_vel = True
    vel_control.lin_vel_is_local = True
    vel_control.ang_vel_is_local = True
    vel_control.linear_velocity = np.array([0.0, 0.0, -linear_velocity])
    vel_control.angular_velocity = np.array([0.0, -angular_velocity, 0.0])

    agent_state = agent.get_state()
    rotation = agent_state.rotation
    agent_mn_quat = mn.Quaternion(rotation.imag, rotation.real)
    current_rigid_state = habitat_sim.RigidState(agent_mn_quat, agent_state.position)

    goal_rigid_state = vel_control.integrate_transform(time_step, current_rigid_state)

    step_fn = sim.pathfinder.try_step if allow_sliding else sim.pathfinder.try_step_no_sliding
    final_position = step_fn(agent_state.position, goal_rigid_state.translation)
    final_rotation = [*goal_rigid_state.rotation.vector, goal_rigid_state.rotation.scalar]

    delta_requested = goal_rigid_state.translation - agent_state.position
    delta_actual = final_position - agent_state.position
    collided = (np.dot(delta_actual, delta_actual) + 1e-5) < np.dot(delta_requested, delta_requested)

    new_state = habitat_sim.AgentState()
    new_state.position = final_position
    new_state.rotation = final_rotation
    agent.set_state(new_state, reset_sensors=False)
    return bool(collided)
