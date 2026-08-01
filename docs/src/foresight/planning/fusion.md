# `src/foresight/planning/fusion.py`

The action fusion controller (PROGRESS.md milestone #6) — Project_FORESIGHT.pdf §2's fast-slow architecture. Library code — no file I/O, no CLI, no `habitat_sim` import (model calls are injected as a callable, so the control law is unit-testable with synthetic inputs); driven by [`foresight.sim.action_fusion_episode`](../sim/action_fusion_episode.md) inside a real simulator loop, rendered by [`tools/run_action_fusion_demo.py`](../../../tools/run_action_fusion_demo.md).

## Sign convention (shared across the whole planning/safety stack)

All headings and bearings in this module — and in [`foresight.perception.vlm.qwen_vl.propose_heading`](../perception/vlm/qwen_vl.md) and [`foresight.sim.pose.goal_bearing_rad`](../sim/pose.md) — are **radians relative to straight-ahead, positive = right**. Angular velocity follows the same convention (positive = turning right); [`foresight.sim.velocity_step.integrate_velocity_step`](../sim/velocity_step.md) negates it internally to match `habitat_sim`'s own (opposite) `+Y`-axis convention, so nothing above the sim boundary needs to know that quirk exists.

## `wrap_pi(angle) -> float`

Wraps a radian angle into `(-pi, pi]`. Used everywhere a heading difference is computed, so a target behind the agent doesn't produce a >180° turn command.

## `heading_to_velocity(heading_rad, max_linear_mps, max_angular_radps, heading_gain) -> (linear_velocity, angular_velocity)`

The proportional heading-to-motion law shared by **every** planner in this package — both `ActionFusionController` here and [`DepthOnlyHeuristicController`](depth_heuristic.md) (milestone #7, the VLM ablation's control condition) call this same function. Factored out deliberately: Project_FORESIGHT deliverable B (VLM vs. depth-only ablation) needs to isolate the heading *source*, not incidental differences in how a heading becomes a velocity command.

- Angular velocity: `clip(heading_gain * wrap_pi(heading_rad), -max_angular_radps, max_angular_radps)` — plain P-control.
- Linear velocity: `max_linear_mps * (1 - turn_frac)`, where `turn_frac = min(1, |heading_error| / (pi/2))` — full speed straight ahead, linearly throttled down to zero by the time the heading error reaches 90°, so the agent doesn't clip corners while rotating in place.

## `ActionFusionController`

```python
controller = ActionFusionController(propose_heading=vlm.propose_heading, vlm_hz=1.0)
result = controller.step(rgb, depth, goal_bearing_rad, now=time.monotonic())
```

- **Constructor** — `propose_heading` is injected (usually [`Qwen3VLSceneReasoner.propose_heading`](../perception/vlm/qwen_vl.md)) rather than hardcoded, purely for testability (see below). `vlm_hz` (default 1.0) must be in `[0.5, 2.0]` — the spec'd planner budget; the constructor raises `ValueError` outside that range. `vlm_weight` (default 0.5) is the circular-average weight given to the cached VLM heading vs. the goal bearing. `safety` defaults to a fresh [`ReactiveSafetyLayer()`](../safety/reactive.md) if not supplied.
- **`step(rgb, depth, goal_bearing_rad, now=None)`** — one control tick:
  1. If `now - last_vlm_time >= 1/vlm_hz`, call `propose_heading(rgb)` and cache the result (which may be `None`, meaning "no proposal this cycle"). `now` defaults to `time.monotonic()` if not passed — **callers should pass real wall-clock time**, not a simulated clock, since the VLM budget is a real compute-time constraint (see the closed-loop episode runner's docs for a bug this caused).
  2. Fuse: if no cached VLM heading, fall back to the goal bearing alone; otherwise combine the two via a **circular weighted average** (`atan2` of the weighted sine/cosine sums, not a linear average) — this avoids the wrap-around failure mode where a stale heading pointing behind the agent would otherwise cancel or fight the goal bearing incorrectly near ±180°.
  3. Convert the fused heading to a velocity command via `heading_to_velocity`.
  4. Pass the command through `self.safety.apply(depth, ...)` and return its (possibly overridden) result.
- **`reset()`** — clears the cached heading and last-query time (e.g. at the start of a new episode).
- **`FusionResult`** dataclass — `linear_velocity`, `angular_velocity` (post-safety), `fused_heading_rad`, `vlm_heading_rad` (the cached value, may be `None`), `vlm_queried` (bool, whether the VLM was actually called this tick), `safety` (the full `SafetyResult`, so callers/logging can see whether/why the safety layer intervened).

## Why the model call is injected, not hardcoded

`ActionFusionController` takes zero `numpy`/model-loading dependencies beyond `numpy` itself — no `torch`, no `transformers`. This let the fusion/caching/circular-average logic be verified with synthetic depth arrays and a fake `propose_heading` closure (no GPU, no model load) before ever touching the real VLM:

```python
calls = []
def fake_propose(rgb):
    calls.append(1)
    return np.deg2rad(30)
ctrl = ActionFusionController(propose_heading=fake_propose, vlm_hz=1.0)
ctrl.step(rgb, clear_depth, goal_bearing_rad=0.0, now=0.0)   # queries (first call)
ctrl.step(rgb, clear_depth, goal_bearing_rad=0.0, now=0.2)   # cached, no query
ctrl.step(rgb, near_obstacle_depth, goal_bearing_rad=0.0, now=0.3)  # safety overrides to stop
```
