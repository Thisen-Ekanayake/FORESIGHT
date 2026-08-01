# `tools/run_depth_baseline_demo.py`

Records a demo of the depth-only heuristic baseline (PROGRESS.md milestone #7) — the control condition for the VLM ablation (Project_FORESIGHT deliverable B). Same pseudo-depth, same reactive safety layer, same closed-loop episode mechanics as [`tools/run_action_fusion_demo.py`](run_action_fusion_demo.md), but the heading comes from a depth-map potential field ([`foresight.planning.depth_heuristic`](../src/foresight/planning/depth_heuristic.md)) instead of a VLM call. Wraps [`foresight.sim.depth_baseline_episode.run_depth_baseline_episode`](../src/foresight/sim/depth_baseline_episode.md); same rendering layout via [`foresight.viz.render`](../src/foresight/viz/render.md).

## Usage

```bash
DISPLAY=:1 /home/thisen-ekanayake/miniforge3/envs/habitat/bin/python tools/run_depth_baseline_demo.py \
  --scene 00800-TEEsavR23oF --max-steps 30
```

Run from the repo root, `DISPLAY=:1` required. No VLM in this stack, so no model-load wait for that half — startup and per-tick cost are lower than the action-fusion demo's.

**To compare against the VLM condition:** run this with the same `--scene`/`--seed` as an `run_action_fusion_demo.py` invocation — the shared closed-loop runner ([`foresight.sim.closed_loop_episode`](../src/foresight/sim/closed_loop_episode.md)) guarantees an identical start/goal/depth-per-tick, so any difference in outcome traces back to the heading source.

| Flag | Default | Notes |
|---|---|---|
| `--dataset-config` | HM3D minival annotated config | |
| `--scene` | `00800-TEEsavR23oF` | |
| `--out` | `results/runs/depth_baseline_demo` | Output root; scene ID is appended |
| `--seed` | `0` | |
| `--width` / `--height` | `640` / `480` | |
| `--playback-fps` | `2` | 1 rendered frame = 1 control tick |
| `--mpp` | `0.05` | |
| `--panel` | `700` | |
| `--min-path-dist` | `4.0` | |
| `--goal-radius` | `0.3` | |
| `--max-steps` | `30` | |
| `--time-step` | `1.0` | |
| `--max-linear-mps` | `0.25` | |
| `--max-angular-degps` | `10.0` | |
| `--influence-range-m` | `2.0` | Potential-field repulsion range, m — obstacles beyond this don't affect the heading |
| `--repulsion-gain` | `1.0` | |
| `--keep-frames` | off | |

## Output — `<out>/<scene>/`

| File | Contents |
|---|---|
| `depth_baseline_demo.mp4` | First-person (left) + top-down (right) composite, captioned per-tick with safety level and clearance |
| `metrics.json` | `success`, `collided_steps`, `control_ticks`, `geodesic_distance_m`, `final_dist_to_goal_m`, `wall_time_s`, `achieved_hz`, and the full per-tick `steps` log (dist-to-goal, bearings, proposed heading, velocities, safety level/clearance, collision) — no `vlm_queried`/`vlm_hz_budget` fields, since there's no VLM |
| `frames/` | Per-tick PNGs, only if `--keep-frames` |

## Verified finding

Run against the same seed/scene as the milestone #6 demo (`00800-TEEsavR23oF`, seed 0): no VLM calls, so a faster loop (~3.4 Hz achieved vs. ~2.4 Hz for the VLM-driven run), but **6 collisions in 15 ticks** vs. **0** for the VLM-driven run on the identical episode. Root cause (in [`foresight.planning.depth_heuristic`](../src/foresight/planning/depth_heuristic.md)'s docs): the potential field only sees a forward horizontal depth band and misses a doorway edge the agent clips while turning — left as a genuine baseline limitation, not patched, since surfacing exactly this kind of failure is the point of the ablation.
