# `src/foresight/sim/pose.py`

Agent-relative pose math shared by the closed-loop episode runner. Library code — a single pure function, no file I/O, no CLI.

## `goal_bearing_rad(position, rotation, goal) -> float`

Bearing to `goal` in the agent's local frame: `0` = straight ahead, **positive = to the right** — matching the sign convention used throughout the planning/safety stack ([`foresight.perception.vlm.qwen_vl.propose_heading`](../perception/vlm/qwen_vl.md), [`foresight.planning.fusion.ActionFusionController`](../planning/fusion.md)).

- `position` — `(3,)` world position (habitat_sim convention: `+Y` up).
- `rotation` — a `quaternion.quaternion` (habitat_sim's `AgentState.rotation` type), local→world orientation.
- `goal` — `(3,)` world position of the target.

Implementation: rotates the local forward (`[0, 0, -1]`, habitat_sim's forward axis) and local right (`[1, 0, 0]`) vectors into world space via `quaternion.rotate_vectors`, projects the world-space vector to the goal onto the horizontal (x-z) plane (zeroes the `y` component — height difference doesn't affect bearing), and returns `atan2(dot(to_goal, right), dot(to_goal, forward))`.

Used every control tick by [`foresight.sim.closed_loop_episode.run_closed_loop_episode`](closed_loop_episode.md) to compute the `goal_bearing_rad` argument passed into both the VLM-fusion and depth-only controllers.
