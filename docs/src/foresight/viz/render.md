# `src/foresight/viz/render.py`

Shared demo-video compositing: a top-down navmesh map with the traced agent path, side by side with the first-person feed. Library code — no CLI; used by every demo recorder in the repo: `tools/record_demo.py` (demo_1), `tools/record_pointnav_demo.py` (see [`docs/rl/pointnav.md`](../../../rl/pointnav.md)), [`tools/run_action_fusion_demo.py`](../../../tools/run_action_fusion_demo.md), and [`tools/run_depth_baseline_demo.py`](../../../tools/run_depth_baseline_demo.md).

## Coordinate convention

Verified against `habitat_sim` 0.3.3's `get_topdown_view`: the navmesh raster is indexed `[row, col]` with `row = (z - z_min) / mpp` and `col = (x - x_min) / mpp`. Both the raster and every path/marker overlay go through the same `_world_xz_to_grid` mapping, so they register exactly — this was verified empirically, not assumed from docs, and is the reason every demo recorder passes `bounds_min` and `meters_per_pixel` straight through from the sim rather than recomputing them.

## Palette

Module-level constants: `FREE`/`BLOCKED` (navmesh floor/wall), `PLANNED` (full geodesic route, thin, underneath), `TRAVELED` (path actually walked, thick, on top), `START_C`/`GOAL_C` (green/red markers), `CURRENT_C` (white current-position dot with a `TRAVELED`-colored ring), `PANEL_BG`, `CAPTION_H`.

## `render_topdown_panel(topdown, bounds_min, mpp, planned_path, traveled, start, goal, panel_w, panel_h) -> Image.Image`

Draws the navmesh raster (nearest-neighbour resized to fit `(panel_w, panel_h)`, preserving aspect and letterboxed — walls stay crisp, no interpolation blur) plus the planned route (thin line), the traveled path so far (thick line), start/goal markers, and a ringed marker at the current (last `traveled`) position. `planned_path`/`traveled` can be `None` or `<2` points (e.g. before the agent has moved) — both are skipped gracefully rather than erroring.

## `compose_frame(rgb, topdown_panel, left_caption, right_caption) -> Image.Image`

Stacks the first-person RGB (left) and the top-down panel (right) side by side, each with a caption strip below. The first-person feed is scaled to the top-down panel's height (`LANCZOS`, for a crisp downscale from a high-res capture) rather than the reverse, so a high-resolution capture stays large and sharp instead of being shrunk down to match a smaller top-down panel. Final canvas is cropped by at most 1px per dimension if needed — H.264 requires even width/height.

## Fonts

`_load_font(size)` tries a short list of hardcoded DejaVu Sans paths (system fonts + one habitat-env-local path) and falls back to `ImageFont.load_default()` if none exist. Not configurable — if captions look wrong on a different machine, check whether one of the hardcoded paths resolves there.
