# FORESIGHT

![Last commit](https://img.shields.io/github/last-commit/Thisen-Ekanayake/FORESIGHT)
![Python](https://img.shields.io/badge/python-3.9-blue)
![Simulator](https://img.shields.io/badge/simulator-Habitat--Sim-orange)
![Status](https://img.shields.io/badge/status-research-lightgrey)

Map-Free Monocular Navigation with an On-Board VLM. RGB-only indoor navigation in Habitat-Sim: zero-shot monocular depth + a compact VLM, fused by a fast-slow controller with a reactive safety layer, benchmarked against a depth-only heuristic baseline. Full spec: `Project_FORESIGHT.pdf`.

## Try it

All commands run from the repo root with the `habitat` conda env active (see [Environment](#environment)).

```bash
# VLM scene reasoning on a captured frame
DISPLAY=:1 python tools/capture_sensors.py --scene 00800-TEEsavR23oF
python tools/describe_scene.py --capture-dir results/runs/sensor_capture/00800-TEEsavR23oF

# Closed-loop navigation demo: VLM planner + reactive safety layer
DISPLAY=:1 python tools/run_action_fusion_demo.py --scene 00800-TEEsavR23oF --max-steps 30

# Depth-only baseline on the same episode, for comparison
DISPLAY=:1 python tools/run_depth_baseline_demo.py --scene 00800-TEEsavR23oF --max-steps 30
```

Each writes a first-person + top-down `.mp4` and a `metrics.json` under `results/runs/`. Full usage/flags: [`docs/tools/`](docs/tools/).

## What's working

| Piece | Status | Docs |
|---|---|---|
| Habitat-Sim env + HM3D minival dataset | ✅ | [`data/README.md`](data/README.md) |
| RGB/depth/semantic sensor capture | ✅ | [`docs/tools/capture_sensors.md`](docs/tools/capture_sensors.md) |
| Zero-shot monocular depth (Depth Anything V2) | ✅ | [`docs/src/foresight/perception/depth/depth_anything.md`](docs/src/foresight/perception/depth/depth_anything.md) |
| Compact VLM scene reasoning (Qwen3-VL-2B) | ✅ | [`docs/src/foresight/perception/vlm/qwen_vl.md`](docs/src/foresight/perception/vlm/qwen_vl.md) |
| Action fusion controller (VLM @0.5-2Hz + reactive safety @10-30Hz) | ✅ | [`docs/src/foresight/planning/fusion.md`](docs/src/foresight/planning/fusion.md) |
| Depth-only heuristic baseline (ablation control) | ✅ | [`docs/src/foresight/planning/depth_heuristic.md`](docs/src/foresight/planning/depth_heuristic.md) |
| Trained PPO PointGoal navigator (extra baseline) | 🔵 | [`docs/rl/pointnav.md`](docs/rl/pointnav.md) |
| Difficulty-stratified benchmark (glass/mirror/low-light/blur) | ⬜ | — |

## Structure

```
src/foresight/
  sim/            Habitat-Sim env + sensors + closed-loop episode runners
  perception/
    depth/        zero-shot monocular depth (Depth Anything V2)
    vlm/          compact VLM scene reasoning (Qwen3-VL-2B)
  planning/       action fusion controller + depth-only heuristic baseline
  safety/         reactive safety layer
  viz/            demo video compositing
  rl/             PPO PointGoal navigator (comparison baseline)
data/
  raw/            downloaded datasets, untouched
  processed/      derived, Habitat-loadable scene datasets
experiments/configs/  experiment + RL configs
results/runs/         per-run outputs (videos, metrics, checkpoints)
tools/            Python CLI entry points
scripts/          shell entry points (model downloads, RL pipeline)
docs/             per-module/per-tool writeups (mirrors src/foresight/ and tools/)
logs/             dated per-session work logs
```

## Environment

```bash
conda env create -f environment.yml
conda activate habitat
pip install -e .
```

GPU rendering (`DISPLAY=:1`) is required for anything touching Habitat-Sim; VLM/depth-only tools that read an already-captured frame (e.g. `describe_scene.py`, `estimate_depth.py`) don't need it.
