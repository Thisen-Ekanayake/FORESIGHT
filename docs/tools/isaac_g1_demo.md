# Through the Glass: live Unitree G1 runs in Isaac Sim (2026-11-20)

The demo video `results/isaac/foresight_through_the_glass.mp4` shows recorded **live** runs: every motion is
FORESIGHT's own decisions driving NVIDIA's pretrained G1 walking policy in Isaac Sim, re-rendered afterwards
joint for joint and path traced. This page is the record of what was run, how, and what the video does not claim.

## Pipeline

| Step | Tool | Env |
|---|---|---|
| Brain: the point-goal evaluation's decision stack (`run_pointnav_eval.build`) or FORESIGHT v1 (Laya + Qwen notes + map walls, as in the 2026-10-29 mirror-trap demo), served over a local socket | `tools/isaac/foresight_brain.py` | habitat |
| World: Isaac Lab's own G1 flat-terrain env + NVIDIA's pretrained actor (Isaac-Velocity-Flat-G1-v0, rsl_rl); FORESIGHT's (forward, turn) command each tick, held for max(0.1 s, the tick's measured compute) | `tools/isaac/g1_live.py` via `tools/isaac/isaaclab_run.sh` | Isaac Lab 3.0 |
| Pass 2: replay of the recorded root pose + 37 joints, path traced (24 spp, denoised), scripted cameras; the depth-only run as a translucent ghost | `tools/isaac/replay_render.py` | Isaac Lab 3.0 |
| Video: AR labels projected through each frame's camera; numbers from the run logs | `tools/make_glass_demo_video.py`, `foresight.viz.glass_demo_video` | habitat |

Assets: NVIDIA's public Isaac 6.0 bucket (`tools/isaac/download_isaac_assets.sh`, 2.70 GB) into `isaac_assets/`.

## Runs shown

| Scene | Condition | Start → goal (Isaac xy) | Result |
|---|---|---|---|
| NVIDIA Office | depth-only map navigator (FORESIGHT's mapper/planner on Depth Anything V2) | (-13.5, 39.5) → (-8.0, 32.5) | reached, 53.7 s |
| HM3D 00808 (scanned home) | depth-only map navigator ("ghost") | (-2.59, -0.29) → (2.4, -1.1) | crash into the hallway mirror, 28.8 s |
| HM3D 00808 | FORESIGHT v1 + impassable confirmed walls | same | reached behind the mirror, 125.9 s, no collision |

Also run (not in the video): HM3D depth-only with Isaac-rendered perception (crash 18.5 s); FORESIGHT v3 box-fill
guard (crash 28.8 s); FORESIGHT v1 with default wall cost (crash 74.6 s, see below).

## What the video does not claim

- **Office glass is not a trap here.** Depth Anything read this glass at ~2.3 m (true 2.6–2.9 m: blinds, frames and
  reflections make it visible), so the depth-only navigator walked round it. The office segment is the depth-only
  navigator; Qwen's GLASS label comes from running Qwen3-VL-2B afterwards on the rendered head-camera frames
  (`tools/isaac/office_vlm_check.py`), not from a live FORESIGHT run.
- **Home perception is Habitat's render of the same scan.** Isaac Sim (renderer) and full-precision Qwen3-VL-2B do
  not fit together on the 8 GB GPU (OOM at 7.68 GB); 8-bit Qwen took 6–15 s per look and 4-bit degraded the mirror
  check (P 0.95 → 0.76 far, 0.78 → 0.35 mid). So for the home runs Isaac simulates only the G1 (physics, colliders of
  the scan) and the brain renders the head camera in Habitat from the same HM3D scan at the robot's pose
  (`g1_live.py --perception habitat`): FORESIGHT's evaluated perception. Pass 2 re-renders everything path traced.
- **Impassable walls are a post-hoc fix.** FORESIGHT v1 flagged, double-checked and confirmed the mirror (twice) and
  added a wall, then walked through it: mapped obstacles are expensive, not impassable, and Depth Anything had mapped
  the reflected room behind the mirror as free floor, so crossing the thin wall was cheaper than the 8 m detour
  through unknown space. With confirmed walls impassable (`plan(..., virtual_cost=1e6)`, opt-in; default unchanged,
  so no evaluated result is affected) the same run reached the goal. One run, chosen after seeing the failure.
- **One encounter per scene**, not an evaluation. The G1's command is FORESIGHT's (forward, turn); the policy's own
  tracking adds small drift (it steps in place on a zero command).
- **Sign convention**: FORESIGHT's angular velocity is positive = turn right; Isaac's yaw rate positive = left. The
  first office run had it unflipped (the robot turned away from its route); fixed before every run listed above.

## Setup notes

- Isaac Lab's uv venv has no `isaacsim` package: `isaaclab_run.sh` sources Isaac Sim's `setup_conda_env.sh` and sets
  CARB_APP_PATH / ISAAC_PATH / EXP_PATH / LD_PRELOAD; any active venv (.venv-laya) must be unset.
- Isaac Lab's own kit app rendered only the sky once a textured scene was loaded; Isaac Sim's
  `isaacsim.exp.base.python.kit` renders correctly. Path tracing must be selected at launch.
- Changing instancing or physics prims after the env starts invalidates the physics views (pass 2 binds the ghost
  material on the robot root, stronger than descendants, and sets gravity to 0 in the config).
- PhysX's default GPU buffers (sized for thousands of training robots, ~2.5 GB) are shrunk for one robot.
- Office lighting: its ceiling lights don't light the glass rooms; extra rect lights are added over the shooting area.

## Natural walking with NVIDIA GEAR-SONIC (2026-11-21)

The pretrained Isaac Lab G1 policy takes fast, short steps (~5 steps/s, ~9 cm), so walking shots now use NVIDIA's
GEAR-SONIC (weights in `/ml/Models/gear_sonic`, NVIDIA Open Model License; code in `/ml/GR00T-WholeBodyControl`,
Apache-2.0). Credit NVIDIA GEAR-SONIC in any video that uses it.

- `tools/isaac/sonic.py`: a Python port of the deployed planner + v1.1 encoder/decoder tracking loop (onnxruntime-gpu).
- `tools/isaac/g1_sonic.py`: pass 1. Runs SONIC's own G1 at 200 Hz physics and 50 Hz control, follows a scripted
  mode/speed/heading profile and writes `motion.json`.
- `tools/isaac/replay_render.py --robot sonic`: pass 2. Renders Isaac Sim's textured G1, with joints matched by name.
- Gait (office walk, slow_walk 0.4 m/s): ~0.5 m/s, ~1–1.3 steps/s, 45–60 cm steps, roll sd ~2°.
- The path tracer renders flicker-free by default (`--deterministic`: caches off, synchronous geometry loading).
  `tools/isaac/deflicker.py` is only for older renders.

```
tools/isaac/isaaclab_run.sh tools/isaac/g1_sonic.py --scene office --out results/isaac/sonic/office_walk \
    --start -13.17 39.2 -44.3 --profile '[[1.5,"idle",-1,-44.3],[3,"slow_walk",0.4,-44.3],[3,"slow_walk",0.4,-44.3,-100],[3.5,"slow_walk",0.4,-100],[3,"idle",-1,-100]]'
tools/isaac/isaaclab_run.sh tools/isaac/replay_render.py --scene office --robot sonic \
    --shots results/isaac/sonic/walk10_shot.json --out results/isaac/sonic/render --fps 30 --spp 64 --iso 14
```

## Promo video "One Camera, No Map" (set up 2026-11-21, run 2026-11-23)

Plan (approved 2026-11-21): https://claude.ai/artifact/LZFmCoQ5rUzfAYmhpG3ahY (draft 2). A 2:15 square video of the
whole project. Every walk is a live FORESIGHT run walked by SONIC (`g1_live.py --walker sonic`), re-rendered path
traced. Runbook (the laptop must be free; stop Athena first):

```
athena --stop
tools/isaac/promo/live_runs.sh probe        # SONIC turn/walk check in the home, ~5 min
python tools/isaac/promo/check_runs.py      # gait, turn rate, speed gain (adjust --sonic-speed-gain if far off)
tools/isaac/promo/live_runs.sh mirror depth search   # the 7 FORESIGHT runs, ~1.5-2.5 h, resumable
python tools/isaac/promo/check_runs.py      # each run vs what its segment needs
python tools/isaac/promo/make_shots.py      # shot list timed from the runs' events + render estimate
tools/isaac/promo/render.sh stills          # one still per shot, 32 vs 64 spp side by side (look check)
tools/isaac/promo/render.sh full 32         # (or 64) every shot, resumable, ~12 h at 32 spp / ~23 h at 64
python tools/make_promo_video.py --preview 40     # spot-check composed frames
python tools/make_promo_video.py --scale 0.5      # 960x960 preview cut for sign-off
python tools/make_promo_video.py                  # 1920x1920 master: results/isaac/promo/foresight_promo.mp4
```

| Segment | Live run(s) | Shots |
|---|---|---|
| Hook | `mirror_map` (depth-only, hallway mirror) + the existing phantom-room render (`results/isaac/phantom`) | `hook_dolly` + head-camera frames from `brain.pkl` |
| One camera, two speeds | `search_v1`, first half (1–24 s; the run with live VLM looks) | `sys_front` (first 6.5 s), `sys_shoulder` |
| Same robot, different depth | `depth_{gt,da,noise,nofs}` on A-study episode a-00808-04 | `depth_crane` (the start, ×2), a 2×2 grid of the depth each robot was given, `depth_goal` (the finish from the goal, ×3), `depth_noise_stall` |
| Mirror trap | `mirror_map`, `mirror_v1` (same start/goal as 2026-11-20) | `mirror_duel`, `mirror_check`, `mirror_exit` |
| Find the sofa | `search_v1` (episode s2-00808-sofa-1), second half at real speed | `search_lapse` |
| End card | `search_v1` final pose | `end_pull` |

Results of the 2026-11-23 runs (SONIC walking): mirror_map crashed into the mirror at 19.4 s; mirror_v1 flagged the
mirror at 3.1 s, confirmed it and added a wall at 10.6 s, reached the goal at 56.5 s; on a-00808-04 GT reached at
83.6 s, DA at 77.6 s, GT + 10% noise timed out frozen 10 m away, DA without calibration timed out lost 2.3 m short
(the study's run crashed at 21 s). The first search episode, mirror-sofa-bed, crashed into the mirror at 31.7 s (the
second look dismissed it; v1 also failed it in 2 of its 3 evaluation sets), kept as
`live/search_mirror_sofa_bed_failed`; the search segment uses s2-00808-sofa-1 instead (v1 solved it in its
evaluation), found the sofa at 53.0 s.

What the promo does and doesn't claim (keep these true):

- **The depth episode was picked.** a-00808-04 is the only A-study episode on 00808's main floor where the four
  depth sources did what the aggregate says (GT and DA reached, GT + 10% noise timed out, uncalibrated DA crashed at
  21 s). The G1 re-runs must show the same pattern (`check_runs.py`); if not, try a-00808-07 / -03
  (`live_runs.sh depth_alt`) and say so. Its settings are the study's (0.35 m/s, 0.36 m success, 130 s, noise seeded
  by episode name). On-screen percentages and AbsRel values are the study's, not this episode's.
- **Walking**: FORESIGHT's (forward, turn) command each tick, turned into a walk by SONIC
  (`tools/isaac/sonic_walker.py`: slow_walk at the commanded speed × `--sonic-speed-gain`, idle + facing angle for
  turning on the spot, the commanded heading kept within 0.8 rad of the actual one). SONIC runs on the CPU in these
  runs (GPU memory); the world is paused while the brain thinks, so that changes wall time only.
- **Walking is shown at real speed** except the two four-robot shots (×2, ×3), which are labelled.
- **Two episodes were swapped after a result**: the no-calibration robot timed out instead of crashing (kept, and the
  labels say "never arrives"); the search episode was replaced after the mirror crash (both recorded above).
- **System segment**: the "fast" rate and "slow" look interval on screen are the run's medians; the point cloud is
  the depth the navigator was given (floor-calibrated), back-projected, hidden behind the render's own depth.
- The mirror segment says nothing about averages (the 2026-11-20 caveats above still apply: Habitat perception at
  home, impassable confirmed walls).
- Credit: NVIDIA GEAR-SONIC (NVIDIA Open Model License) on the end card; music "Ascending the Vale" (Kevin MacLeod,
  CC BY 4.0).
