# `src/foresight/planning/depth_heuristic.py`

The depth-only heuristic planner (PROGRESS.md milestone #7) — the control condition for the VLM ablation (Project_FORESIGHT deliverable B). Library code — no file I/O, no CLI; driven by [`foresight.sim.depth_baseline_episode`](../sim/depth_baseline_episode.md), rendered by [`tools/run_depth_baseline_demo.py`](../../../tools/run_depth_baseline_demo.md).

Per Project_FORESIGHT.pdf §1: "a depth-only potential-field / heuristic planner that consumes the same pseudo-depth but no VLM ... isolates the marginal value of the VLM." Deliberately shares as much machinery as possible with [`ActionFusionController`](fusion.md) — the same `heading_to_velocity` control law and the same [`ReactiveSafetyLayer`](../safety/reactive.md) — so the only thing this swaps out relative to milestone #6 is *how the heading is proposed*.

## `propose_heading_from_depth(depth, goal_bearing_rad, hfov_rad=90°, num_bins=9, influence_range_m=2.0, repulsion_gain=1.0, row_frac=(0.4, 0.9)) -> float`

A pure function of one depth frame and the goal bearing — no VLM, no learned model, no history, recomputed fresh every call. Classic potential-field heading:

1. Take a horizontal band of the depth map (`row_frac` of image height — same band convention as `ReactiveSafetyLayer`'s collision cone), split it into `num_bins` angular columns spanning the camera's horizontal FOV (`hfov_rad`, default 90° = `habitat_sim`'s default).
2. Start with an **attractive** force vector pointing at `goal_bearing_rad`.
3. For each bin whose mean valid depth is closer than `influence_range_m`, subtract a **repulsive** force pointing back along that bin's angle, magnitude `repulsion_gain * (1/d - 1/influence_range_m)` (closer bins push harder; a bin at exactly `influence_range_m` contributes nothing, matching the reactive safety layer's `slow_distance_m` motivation of a smooth falloff rather than a hard cutoff).
4. Return `atan2` of the summed force vector, clipped to `±hfov_rad/2` — the heading is never asked to point somewhere the depth map couldn't have justified (outside the camera's own FOV).

## `DepthHeuristicResult`

Dataclass: `linear_velocity`, `angular_velocity` (both post-safety), `proposed_heading_rad`, `safety` (the full `SafetyResult`).

## `DepthOnlyHeuristicController`

```python
controller = DepthOnlyHeuristicController()
result = controller.step(depth, goal_bearing_rad)
```

No fast-slow split, unlike `ActionFusionController` — there's no expensive model call to budget, so the potential field is recomputed every control tick. `step()` is a straight pipeline: `propose_heading_from_depth` → `heading_to_velocity` → `ReactiveSafetyLayer.apply`. Constructor accepts the same `max_linear_mps` / `max_angular_radps` / `heading_gain` as `ActionFusionController` plus the potential-field parameters (`hfov_rad`, `num_bins`, `influence_range_m`, `repulsion_gain`) and an optional `ReactiveSafetyLayer` instance.

## Verified finding: real, uncorrected baseline weakness

Run end-to-end via `tools/run_depth_baseline_demo.py` on the same seed/scene as the milestone #6 demo (`00800-TEEsavR23oF`): no VLM calls means a much faster loop (~3.4 Hz vs. ~2.4 Hz for the VLM-driven controller), but **6 collisions in 15 ticks** vs. **0** for the VLM-driven run on the identical episode. Root cause: the potential field only sees a forward horizontal band (`row_frac`), so it doesn't register a doorway edge the agent clips while turning — a genuine limitation of the pure-geometry approach, left as-is (not patched) because it's exactly the kind of failure mode the VLM ablation (deliverable B) is meant to surface, not something to engineer away in the control condition.
