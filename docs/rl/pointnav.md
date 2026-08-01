# Trained continuous-control PointGoal navigator (RL)

> New to RL? See `docs/rl/rl_primer_for_beginners.md` for a from-scratch walkthrough of the concepts and
> code below (what PPO does, why the reward is shaped this way, a worked debugging case study).

A learned PointGoal navigator — reach a target coordinate — with **stable, smooth motion**, trained with PPO
(habitat-baselines) in Habitat-Sim on the HM3D minival scenes. It is a **trained baseline** to contrast against
FORESIGHT's zero-shot RGB pipeline; it plugs into the same demo recorder as `demo_1` by swapping the action
source. (This is a deliberate addition beyond the strictly-zero-shot proposal.)

## Design

| Choice | Value | Why |
|---|---|---|
| Task | PointGoalNav + GPS/Compass | Standard, tractable; relative-goal vector given each step |
| Action | **Continuous velocity** `(linear, angular)` | Smooth arcs (turn while moving) — genuinely smooth motion, not just a smoothed video. Stop is implicit (near-zero velocities). |
| Visual input | **RGB** (256²) + pointgoal sensor | Matches the project's RGB-first thesis (`--modality depth|rgbd` for a stronger baseline) |
| Algorithm | PPO, recurrent (GRU) ResNet18 policy, Gaussian action dist | habitat-baselines default; memory needed for nav |
| Scenes | 10 HM3D minival | Only these are local; **training overfits** — fine for a working agent, expand with the HM3D train split later |

**Smoothness reward** (`src/foresight/rl/smooth_reward.py`, `PointNavSmoothReward`): the env already adds
`slack_reward` + `success_reward`, so the measure contributes
`progress − w_col·collided − w_ang·|ω| − w_jerk·‖Δaction‖`. `w_ang` suppresses spin-in-place; `w_jerk`
suppresses abrupt velocity changes → smooth, stable trajectories. Coefficients live in the config.

## Components

- `src/foresight/rl/smooth_reward.py` — the reward measure + its Hydra structured config.
- `src/foresight/rl/velocity_action.py` — `SmoothVelocityAction`: array-tolerant subclass of habitat-lab's
  `VelocityAction` (the stock one crashes on habitat-baselines' shape-`(1,)` velocity args under NumPy ≥ 1.24).
- `experiments/configs/rl/pointnav_continuous.yaml` — composes PointNav HM3D + baselines RL + our components.
- `tools/generate_pointnav_episodes.py` — episode datasets for the minival scenes.
- `tools/train_pointnav.py` — train/eval wrapper (strips discrete actions + sets modality in code).
- `tools/record_pointnav_demo.py` — roll out a checkpoint → smooth demo clip (reuses `viz/render.py`) + metrics.
- `src/foresight/rl/config.py` — loader for the per-script YAML configs (dotted-key `--set` overrides, strict).
- `scripts/run_rl_pipeline.sh` — runs the four stages in order (episodes → train → eval → demo).
- `data/processed/scene_datasets/hm3d/minival/hm3d_minival_basis_pointnav.scene_dataset_config.json` —
  registers all 10 basis meshes (the annotated config registers only the 4 semantic scenes).

## Configuration

No values are hardcoded in the RL scripts — each reads a YAML config and every setting can be overridden
from the CLI (or edited in the YAML to change the default):

| Config | Read by | Holds |
|---|---|---|
| `experiments/configs/rl/pointnav_continuous.yaml` | habitat-baselines (Hydra) | Task, sim, reward weights, PPO/DD-PPO hyperparameters, run output paths |
| `experiments/configs/rl/train_pointnav.yaml` | `tools/train_pointnav.py` | Which Hydra config to compose, modality → sensors, actions to strip, eval flag, extra Hydra overrides |
| `experiments/configs/rl/generate_episodes.yaml` | `tools/generate_pointnav_episodes.py` | Scene discovery paths, splits + episodes per scene, sampling limits, seed, output paths |
| `experiments/configs/rl/record_demo.yaml` | `tools/record_pointnav_demo.py` | Checkpoint, episode selection, rollout length, render/panel/label settings, ffmpeg encoding, output paths |
| `experiments/configs/rl/pipeline.yaml` | `scripts/run_rl_pipeline.sh` | Interpreter, `DISPLAY`, stage order, each stage's script + extra args + skip rule, log dir |

The three script configs are plain YAML (PyYAML), not Hydra. Override any key with `--set dotted.key=value`
(values parsed as YAML, so `true`, `15`, `[a, b]` work). Overrides are **strict** — the key must exist in the
YAML, so a typo fails loudly. In `train_pointnav.py`, `--set` targets the *script* config and `--habitat-set`
passes Hydra overrides through to the habitat/habitat-baselines config.

## Run it

The whole pipeline in order — episodes → train → eval → demo:

```bash
scripts/run_rl_pipeline.sh                    # all four stages, per-stage logs in results/runs/pointnav/pipeline/
scripts/run_rl_pipeline.sh --list             # what the stages are
scripts/run_rl_pipeline.sh --dry-run          # print the commands without running them
scripts/run_rl_pipeline.sh --stages eval,demo # a subset (kept in config order)
scripts/run_rl_pipeline.sh --force            # regenerate episodes too (otherwise skipped if they exist)
```

It stops at the first failing stage and prints a per-stage summary; training is resumable, so re-running
after an interruption continues from the latest checkpoint. Per-stage extra args live in
`experiments/configs/rl/pipeline.yaml`. To run a stage by hand instead — all commands from the repo root,
with the habitat python and `DISPLAY=:1`:

```bash
PY=/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python

# 1. Episodes (once): 300 train + 30 val per scene -> data/processed/pointnav_episodes/
DISPLAY=:1 $PY tools/generate_pointnav_episodes.py \
    --set splits.train.num_episodes_per_scene=300 splits.val.num_episodes_per_scene=30

# 2. Train (resumable — re-run the same command to continue from the latest checkpoint)
DISPLAY=:1 $PY tools/train_pointnav.py
#    monitor: tensorboard --logdir results/runs/pointnav/tb   (watch reward / success / spl / collisions)
#    shorter smoke run: --habitat-set habitat_baselines.total_num_steps=2000 habitat_baselines.num_environments=2

# 3. Evaluate on the val split (success / SPL / collisions); checkpoint comes from
#    habitat_baselines.eval_ckpt_path_dir — override with --habitat-set to score a different one
DISPLAY=:1 $PY tools/train_pointnav.py --eval

# 4. Record a smooth demo clip for one val episode (first-person + top-down; RL path vs geodesic-optimal)
DISPLAY=:1 $PY tools/record_pointnav_demo.py \
    --ckpt results/runs/pointnav/checkpoints/latest.pth \
    --set episode_index=0 render.duration=15 render.fps=50
```

Outputs under `results/runs/pointnav/`: `checkpoints/`, `tb/`, `train.log`, and `demo/`.

## Notes / limitations

- **Compute:** single RTX 4060 laptop GPU, ~590 fps at 4 envs → ~10M steps in ~5 h. Checkpoints are periodic
  and resumable; a usable (not SOTA) agent well before the ceiling. Overfits the 10 scenes.
- **Smoothness:** continuous velocity control already yields smooth arcs; the demo recorder can additionally
  interpolate keyframe poses (`render.duration`) exactly like `demo_1` for a polished clip.
- **Config internals:** discrete actions and the depth sensor are stripped in `train_pointnav.py::prepare_config`
  (not Hydra) because `pointnav_hm3d` re-adds them in its body — see the config's header comment.
