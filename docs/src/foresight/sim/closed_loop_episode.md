# `src/foresight/sim/closed_loop_episode.py`

The generic closed-loop episode runner shared by the action fusion controller (milestone #6) and the depth-only heuristic baseline (milestone #7). Library code — no CLI; wrapped by [`foresight.sim.action_fusion_episode`](action_fusion_episode.md) and [`foresight.sim.depth_baseline_episode`](depth_baseline_episode.md), which supply the `decide` closure and the CLI-facing defaults.

Both conditions drive a bare `habitat_sim.Simulator` identically — same start/goal sampling, same zero-shot depth estimator, same kinematic velocity integration, same termination logic — differing **only** in the `decide` callable that turns `(rgb, depth, goal_bearing)` into a velocity command. This sharing is deliberate: given the same seed, both conditions see the exact same start/goal/scene and the same integration step, so the eventual VLM-vs-baseline ablation (Project_FORESIGHT deliverable B) attributes any outcome difference to the heading source, not to incidental differences in how the episode was run.

## `DecideFn`

```python
DecideFn = Callable[[np.ndarray, np.ndarray, float], tuple]
# (rgb (H,W,3) uint8, depth (H,W) float32 meters, goal_bearing_rad float)
#   -> (linear_velocity m/s, angular_velocity rad/s, diagnostics dict)
```

The `diagnostics` dict is opaque to this module — it's merged into each tick's logged step (see below), so each condition can log whatever's relevant to it (e.g. `vlm_queried` for the fusion controller, `proposed_heading_rad` for the depth-only one) without this shared runner needing to know about either.

## `ClosedLoopEpisode`

Dataclass: `rgb_frames` (list, one per control tick), `positions` (`(T, 3)` float32, **includes the start pose**, so `len(positions) == len(rgb_frames) + 1`), `start`, `goal`, `planned_path` (geodesic waypoints, for reference/plotting only — not followed), `geodesic_distance`, `success`, `collided_steps` (count across the episode), `steps` (list of per-tick diagnostics dicts), `topdown`/`bounds_min`/`bounds_max`/`meters_per_pixel` (navmesh raster, for [`foresight.viz.render.render_topdown_panel`](../viz/render.md)).

## `run_closed_loop_episode(dataset_config, scene_id, decide, depth_estimator, ...) -> ClosedLoopEpisode`

1. Builds an RGB-only sim config via [`foresight.sim.sensors.make_sim_config`](sensors.md) (`sensor_uuids=("rgb",)` — no GT depth or semantic pass, matching the "no depth sensor" constraint) and constructs `habitat_sim.Simulator`.
2. Samples a start/goal pair via `foresight.sim.navigate._sample_start_goal` (shared with [`foresight.sim.navigate`](navigate.md)'s demo_1 episode runner — same sampling logic, `min_path_dist`/`max_sample_tries` control it identically).
3. Per tick, until `max_steps` or the agent is within `goal_radius` of the goal (`success = True`, loop breaks):
   - Render RGB, run `depth_estimator.estimate(rgb)`, compute `goal_bearing_rad` via [`foresight.sim.pose.goal_bearing_rad`](pose.md).
   - Call `decide(rgb, depth, bearing)` → `(linear_velocity, angular_velocity, diagnostics)`.
   - Integrate via [`foresight.sim.velocity_step.integrate_velocity_step`](velocity_step.md) at the fixed `time_step` (default 1.0 s of simulated motion per tick, independent of how long the tick actually took to compute).
   - Log `dist_to_goal_m`, `goal_bearing_rad`, `linear_velocity`, `angular_velocity`, `collided`, plus whatever `diagnostics` returned.
4. Always closes the simulator (`try`/`finally`).

`depth_estimator` is passed in (not constructed internally) so the two wrapper modules can each own model loading/lifetime — this runner has no `perception`-package dependency beyond the `ZeroShotDepthEstimator` type hint.
