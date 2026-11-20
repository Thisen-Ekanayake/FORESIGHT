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
