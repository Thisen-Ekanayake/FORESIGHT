# `src/foresight/sim/navigate.py`

Drives a Habitat-Sim agent along the navmesh shortest path between two points, capturing first-person RGB — the demo_1 episode runner. Library code — no CLI; consumed by `tools/record_demo.py`, and its start/goal sampling helper is reused by [`foresight.sim.closed_loop_episode`](closed_loop_episode.md) (milestones #6-7).

No learned policy involved: the agent follows Habitat's built-in greedy geodesic follower over the navmesh — the honest stand-in for a trained/reasoning navigator until the action fusion controller ([`foresight.sim.action_fusion_episode`](action_fusion_episode.md), milestone #6) and depth-only baseline ([`foresight.sim.depth_baseline_episode`](depth_baseline_episode.md), milestone #7) existed. They now do; this module still exists as the geodesic-shortest-path reference/demo_1 recorder and as the source of the shared start/goal sampling logic.

## Two-phase motion, for smoothness

1. **Plan** — run the discrete follower (move 0.25 m / turn 10° per action), collecting keyframe poses only via `agent.act` (no rendering — cheap).
2. **Render** — resample those keyframes into a continuous pose track (lerp position + slerp rotation) of an arbitrary frame count, then render RGB at each interpolated pose. This decouples the clip's smoothness/duration from the follower's coarse per-action step size.

## `Episode`

Dataclass holding everything needed to render the demo, decoupled from the (by-then-closed) simulator: `rgb_frames`, `positions` (`(T, 3)` float32, one per rendered/interpolated frame), `start`, `goal`, `planned_path` (geodesic waypoints), `geodesic_distance`, `success` (follower reached the goal radius before `max_steps`), `n_keyframes` (discrete follower steps before interpolation, for reporting), `topdown`/`bounds_min`/`bounds_max`/`meters_per_pixel` (navmesh raster for [`foresight.viz.render`](../viz/render.md)).

## `_sample_start_goal(pathfinder, min_dist, max_tries) -> (start, goal, path_points, geodesic_distance)`

Samples two navigable, geodesically connected points at least `min_dist` apart along the navmesh, using the pathfinder's own RNG (seed it first). Raises `RuntimeError` if no valid pair is found within `max_tries` (small or fragmented navmesh). **Shared** (imported directly, underscore-prefixed but reused across modules) by [`foresight.sim.closed_loop_episode.run_closed_loop_episode`](closed_loop_episode.md) — the same sampling logic backs demo_1, the action fusion controller, and the depth-only baseline, so all three demos are directly comparable in how their episodes are generated.

## `_plan_poses(follower, agent, goal, max_steps)` / `_resample_track(positions, rotations, n_frames)`

Internal helpers for the two-phase motion described above. `_resample_track` is also reused by `tools/record_pointnav_demo.py` (see [`docs/rl/pointnav.md`](../../../rl/pointnav.md)) to smooth a trained RL policy's rollout the same way.

## `run_shortest_path_episode(dataset_config, scene_id, width=640, height=480, seed=0, min_path_dist=5.0, goal_radius=0.25, max_steps=500, meters_per_pixel=0.05, max_sample_tries=200, n_frames=None) -> Episode`

Samples a start/goal, walks the navmesh shortest path via `habitat_sim.nav.GreedyGeodesicFollower`, and renders first-person RGB along the (optionally interpolated) path. RGB-only sim config (`sensor_uuids=("rgb",)`) — depth/semantic passes are skipped since the demo doesn't use them. `n_frames`, if given, interpolates the discrete path to exactly that many smooth frames (set to `duration_seconds * fps` for a fixed-length clip); otherwise one frame per follower step. Always closes the simulator (`try`/`finally`).
