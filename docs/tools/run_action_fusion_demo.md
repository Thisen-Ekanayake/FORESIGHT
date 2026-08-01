# `tools/run_action_fusion_demo.py`

Records a demo of the action fusion controller (PROGRESS.md milestone #6) driving an agent closed-loop in Habitat-Sim: zero-shot depth + a Qwen3-VL-2B heading proposal (budgeted 0.5-2 Hz) fused with the goal bearing every control tick, wrapped by a per-tick reactive safety layer. Wraps [`foresight.sim.action_fusion_episode.run_action_fusion_episode`](../src/foresight/sim/action_fusion_episode.md); rendering reuses [`foresight.viz.render`](../src/foresight/viz/render.md) — the same first-person/top-down composite layout as demo_1 (`tools/record_demo.py`).

Unlike demo_1/the RL demo, **no smoothing/interpolation**: each rendered frame is one real control tick, so the clip's pace directly reflects the actually-achieved control rate on this hardware — see Project_FORESIGHT.pdf's note to "report the operating point actually achieved (model size, device, achieved Hz) rather than assuming one."

## Usage

```bash
DISPLAY=:1 /home/thisen-ekanayake/miniforge3/envs/habitat/bin/python tools/run_action_fusion_demo.py \
  --scene 00800-TEEsavR23oF --max-steps 30
```

Run from the repo root. `DISPLAY=:1` is required (real Habitat-Sim rendering, unlike `describe_scene.py`/`estimate_depth.py`). Expect ~10-15s of model-load time up front (depth + VLM), then roughly 0.5-1s per control tick once warm — 15-30 steps is a reasonable demo length.

| Flag | Default | Notes |
|---|---|---|
| `--dataset-config` | HM3D minival annotated config | |
| `--scene` | `00800-TEEsavR23oF` | Any HM3D minival scene ID — semantics aren't used by this stack |
| `--out` | `results/runs/action_fusion_demo` | Output root; scene ID is appended |
| `--seed` | `0` | Also seeds the pathfinder's start/goal sampling |
| `--width` / `--height` | `640` / `480` | First-person capture resolution |
| `--playback-fps` | `2` | Video fps — 1 rendered frame = 1 control tick, so this sets clip pacing directly |
| `--mpp` | `0.05` | Top-down meters per pixel |
| `--panel` | `700` | Top-down panel size / body height, px |
| `--min-path-dist` | `4.0` | Minimum start→goal geodesic distance, m |
| `--goal-radius` | `0.3` | Episode ends in success once within this distance of the goal |
| `--max-steps` | `30` | Max control ticks before the episode is cut off (not necessarily a failure — see `metrics.json`) |
| `--time-step` | `1.0` | Simulated seconds integrated per control tick (independent of wall-clock tick duration) |
| `--vlm-hz` | `1.0` | VLM heading-query budget; must be in `[0.5, 2.0]` per spec |
| `--max-linear-mps` | `0.25` | |
| `--max-angular-degps` | `10.0` | |
| `--keep-frames` | off | Keep the intermediate per-tick PNGs alongside the video |

## Output — `<out>/<scene>/`

| File | Contents |
|---|---|
| `action_fusion_demo.mp4` | First-person (left) + top-down (right) composite, captioned per-tick with safety level, `vlm_queried`, and clearance |
| `metrics.json` | `success`, `collided_steps`, `control_ticks`, `geodesic_distance_m`, `final_dist_to_goal_m`, `wall_time_s`, `achieved_hz` (real control-loop rate, not assumed), `vlm_queries`, `vlm_hz_budget`, and the full per-tick `steps` log (dist-to-goal, bearings, headings, velocities, safety level/clearance, collision, `vlm_queried`) |
| `frames/` | Per-tick PNGs, only if `--keep-frames` |

## Verified

`00800-TEEsavR23oF`, seed 0, 15 ticks: 0 collisions, agent advances toward the goal along a sensible path (visually confirmed in the rendered frames and the traced top-down path). The VLM-query throttle was confirmed live: at an achieved ~2.4 Hz control rate against a 1 Hz budget, the VLM was queried 2 times over 15 ticks, correctly spaced — not once per tick, which an earlier bug (fixed; see the episode-runner docs) had caused.

## Comparing against the depth-only baseline

Run [`tools/run_depth_baseline_demo.py`](run_depth_baseline_demo.md) with the same `--scene`/`--seed` for a directly comparable episode (identical start/goal/depth per tick, only the heading source differs) — this is the manual precursor to the VLM ablation (Project_FORESIGHT deliverable B).
