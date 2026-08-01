# `src/foresight/sim/velocity_step.py`

Continuous-velocity agent integration outside habitat-lab's task/action machinery. Library code — no file I/O, no CLI.

Mirrors `habitat.tasks.nav.nav.VelocityAction.step` (the same kinematic `VelocityControl` integration + navmesh sliding used by the RL stack under [`foresight.rl`](../../../rl/pointnav.md)) so a bare `habitat_sim.Simulator` can be driven by `(linear_velocity, angular_velocity)` commands without pulling in a full habitat-lab `Env`/task. Written for the closed-loop episode runner ([`foresight.sim.closed_loop_episode`](closed_loop_episode.md)), which deliberately stays independent of the RL config machinery so the VLM/depth/fusion/safety stack (milestones #6-7) isn't coupled to `habitat-baselines`.

## `integrate_velocity_step(sim, agent, linear_velocity, angular_velocity, time_step, allow_sliding=True) -> bool`

Advances `agent` by one control step at the given body-frame velocities:

- `linear_velocity` — m/s, forward-positive.
- `angular_velocity` — rad/s, **positive = turn right** (matches [`foresight.planning.fusion`](../planning/fusion.md)'s convention). Internally negated before being handed to `habitat_sim.physics.VelocityControl`, because habitat_sim's own `+Y`-axis angular-velocity convention is the opposite (positive `+Y` = turn left) — verified empirically (see below) rather than assumed from docs.
- `time_step` — seconds of simulated motion to integrate for one call (independent of how long the call actually took in wall-clock time).
- `allow_sliding` — `True` uses `pathfinder.try_step` (slide along walls on collision, matching normal habitat-lab behavior); `False` uses `try_step_no_sliding` (hard stop on collision).

Returns whether the navmesh sliding filter shortened the requested move — i.e. whether a collision occurred (`dist_moved_after_filter < dist_moved_before_filter`, same collision-detection heuristic as `habitat.tasks.nav.nav.VelocityAction`).

## Sign convention, verified

The `+Y`-axis rotation direction isn't obvious from the API, so it was checked directly against a bare `habitat_sim.physics.VelocityControl` before wiring this up: integrating a `+1.0 rad/s` `+Y` angular velocity for 1s from identity rotation rotates the local-forward vector `[0,0,-1]` toward **negative x** (left), confirming `+Y` = turn left in habitat_sim's convention — hence the negation in this module to expose the "positive = right" convention the rest of the stack uses.
