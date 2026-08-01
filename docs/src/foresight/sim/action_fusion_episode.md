# `src/foresight/sim/action_fusion_episode.py`

Closed-loop episode runner for the action fusion controller (PROGRESS.md milestone #6). Library code — no CLI; consumed by [`tools/run_action_fusion_demo.py`](../../../tools/run_action_fusion_demo.md).

A thin wrapper over [`foresight.sim.closed_loop_episode.run_closed_loop_episode`](closed_loop_episode.md): loads the depth and VLM models, builds an [`ActionFusionController`](../planning/fusion.md), and adapts its `.step()` into the shared runner's `decide(rgb, depth, bearing) -> (linear, angular, diagnostics)` interface.

## `ActionFusionEpisode`

Re-export of `foresight.sim.closed_loop_episode.ClosedLoopEpisode` — kept as a separate name so callers/tests can import the episode type from this module without needing to know it's shared with the depth-only baseline.

## `run_action_fusion_episode(dataset_config, scene_id, ..., vlm_hz=1.0, max_linear_mps=0.25, max_angular_degps=10.0, depth_model=..., vlm_model=..., ...) -> ActionFusionEpisode`

- Constructs a [`ZeroShotDepthEstimator`](../perception/depth/depth_anything.md) and a [`Qwen3VLSceneReasoner`](../perception/vlm/qwen_vl.md) (both real model loads — this is not mockable at this layer, unlike `ActionFusionController` itself).
- Builds the `decide` closure: calls `controller.step(rgb, depth, bearing, now=time.monotonic())` — **wall-clock time, not simulated time** — and packs `fused_heading_rad`, `vlm_heading_rad`, `vlm_queried`, `safety_level`, `min_clearance_m` into the diagnostics dict that ends up in each tick's logged step.
- Delegates everything else (start/goal sampling, per-tick rendering, integration, termination) to `run_closed_loop_episode`.

## Bug found and fixed: simulated time vs. wall-clock time for the VLM gate

The first version passed a simulated clock (incrementing by `time_step` each tick) into `controller.step`'s `now` argument. Because `time_step` (1.0 s) happened to equal the VLM period (`1/vlm_hz`, also 1.0 s at the default budget), the VLM was queried on **every** tick regardless of the configured budget — invisible as a bug because the two numbers happened to coincide. Fixed by switching to `time.monotonic()`: the VLM query budget is a real compute-time constraint (the model genuinely takes that long to run), not a simulated one, so the gate has to reflect actual elapsed wall time between calls. Re-verified: at an achieved ~2.4 Hz real control rate, a 1 Hz VLM budget produced 2 queries over 15 ticks, correctly spaced (not one every tick).

## Verified

End-to-end on GPU in `00800-TEEsavR23oF`, real depth + VLM models in the loop: 15-tick episode, 0 collisions, agent advances toward the goal along a sensible path (confirmed visually in the rendered demo frames and the traced top-down path).
