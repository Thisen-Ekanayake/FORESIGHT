# `src/foresight/sim/depth_baseline_episode.py`

Closed-loop episode runner for the depth-only heuristic baseline (PROGRESS.md milestone #7). Library code — no CLI; consumed by [`tools/run_depth_baseline_demo.py`](../../../tools/run_depth_baseline_demo.md).

A thin wrapper over [`foresight.sim.closed_loop_episode.run_closed_loop_episode`](closed_loop_episode.md) — the same shared loop [`action_fusion_episode.py`](action_fusion_episode.md) uses, so a given seed/scene produces the exact same start, goal, and per-tick depth estimate as the VLM-driven episode. Only the `decide` closure differs: a [`DepthOnlyHeuristicController`](../planning/depth_heuristic.md) instead of an `ActionFusionController` — no VLM in the loop, no `time.monotonic()` gating (nothing to budget).

## `DepthBaselineEpisode`

Re-export of `foresight.sim.closed_loop_episode.ClosedLoopEpisode`, mirroring `action_fusion_episode.py`'s `ActionFusionEpisode` alias.

## `run_depth_baseline_episode(dataset_config, scene_id, ..., max_linear_mps=0.25, max_angular_degps=10.0, influence_range_m=2.0, repulsion_gain=1.0, depth_model=..., ...) -> DepthBaselineEpisode`

- Constructs a `ZeroShotDepthEstimator` (no VLM — this is the point of the baseline) and a `DepthOnlyHeuristicController`.
- `decide` closure: calls `controller.step(depth, bearing)` and packs `proposed_heading_rad`, `safety_level`, `min_clearance_m` into the diagnostics dict (no `vlm_queried`/`vlm_heading_rad` fields, since there's no VLM).
- Same start/goal sampling, integration, and termination as `run_action_fusion_episode`, via the shared `run_closed_loop_episode`.

## To compare against the VLM condition

Run this and `run_action_fusion_episode` with the **same `dataset_config`/`scene_id`/`seed`**, which reproduces the same start/goal pair and the same per-tick depth (only the heading source differs) — see [`tools/run_depth_baseline_demo.py`](../../../tools/run_depth_baseline_demo.md)'s usage note. Verified finding from doing exactly this (`00800-TEEsavR23oF`, seed 0): the depth-only baseline collided 6 times in 15 ticks vs. 0 for the VLM-driven run on the identical episode — see [`foresight.planning.depth_heuristic`](../planning/depth_heuristic.md)'s docs for the root cause.
