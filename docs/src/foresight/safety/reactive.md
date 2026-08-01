# `src/foresight/safety/reactive.py`

The reactive safety layer (part of PROGRESS.md milestone #6) — the fast half of Project_FORESIGHT.pdf §2's fast-slow architecture (spec'd 10-30 Hz; in practice it has no timing budget of its own, it just runs every time its caller calls it). Library code — no file I/O, no CLI, no `habitat_sim`/model dependency; a pure function of a depth array and a commanded velocity. Used downstream by both [`ActionFusionController`](../planning/fusion.md) (milestone #6) and [`DepthOnlyHeuristicController`](../planning/depth_heuristic.md) (milestone #7) — deliberately the *same* instance type in both, so the VLM-vs-baseline ablation isolates the heading source, not the safety behavior.

## `SafetyResult`

Dataclass: `linear_velocity`, `angular_velocity` (the — possibly overridden — command to actually execute), `min_clearance_m` (closest valid depth found in the collision cone), `level` (`"clear"` | `"slow"` | `"stop"`, for logging/diagnostics).

## `ReactiveSafetyLayer`

```python
safety = ReactiveSafetyLayer(stop_distance_m=0.4, slow_distance_m=1.2)
result = safety.apply(depth, linear_velocity, angular_velocity)
```

- **Collision cone** — a rectangular crop of the depth map: `row_frac` (default `(0.4, 0.9)`, lower-middle rows — where a frontal obstacle projects in a forward-facing camera) × `col_frac` (default `(0.25, 0.75)`, central columns). `min_clearance_m` is the minimum valid (`isfinite` and `> 0`) depth inside that crop; `inf` if the crop has no valid pixels.
- **`apply(depth, linear_velocity, angular_velocity) -> SafetyResult`** — three-tier response:
  1. `min_clearance_m < stop_distance_m` (default 0.4 m): **hard override**. Zero the linear velocity; turn (`± turn_radps`, default 20°/s) toward whichever image half (left/right) has more average clearance (`_mean_valid` over each half of the full frame, not just the cone).
  2. `min_clearance_m < slow_distance_m` (default 1.2 m): **scale down**. Linear velocity multiplied by `(min_clearance_m - stop_distance_m) / (slow_distance_m - stop_distance_m)` — 0 at the stop boundary, 1 at the slow boundary; angular velocity passes through unchanged.
  3. Otherwise: **pass through** unchanged — the commanded `(linear_velocity, angular_velocity)` is returned as-is.
- Constructor raises `ValueError` if `stop_distance_m >= slow_distance_m` (the two-tier logic assumes a strict ordering).

## Depth source

The depth map passed in is whatever the caller supplies — in practice always [`ZeroShotDepthEstimator`](../perception/depth/depth_anything.md)'s zero-shot monocular estimate (milestone #4), not simulator ground truth, matching Project_FORESIGHT.pdf's constraint that "the reactive safety layer can reason in real distances" using only the monocular pipeline, no depth sensor.

## Verified

Unit-tested with synthetic depth arrays before any model was involved: a uniform 5 m "clear" depth passes a command through unchanged; a close (0.2 m) patch centered in the cone triggers `"stop"` with the linear velocity zeroed and a turn toward the clearer half. See [`foresight.planning.fusion`](../planning/fusion.md)'s docs for the exact test snippet.
