#!/usr/bin/env python
"""Train (or evaluate) the continuous-control PointGoal navigator with PPO via habitat-baselines.

Composes experiments/configs/rl/pointnav_continuous.yaml, post-processes it (pure-continuous action space +
visual modality), and hands off to habitat-baselines. Resumable: re-running picks up the latest checkpoint
in the config's checkpoint_folder automatically (load_resume_state_config).

Examples:
  # smoke test (few steps, verifies the pipeline runs)
  DISPLAY=:1 <py> scripts/train_pointnav.py --set habitat_baselines.total_num_steps=2000 \
      habitat_baselines.num_environments=2 habitat_baselines.log_interval=1
  # full training
  DISPLAY=:1 <py> scripts/train_pointnav.py
  # evaluate a checkpoint on the val split
  DISPLAY=:1 <py> scripts/train_pointnav.py --eval \
      --set habitat_baselines.eval_ckpt_path_dir=results/runs/pointnav/checkpoints/latest.pth
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import foresight.rl  # noqa: F401  (registers the reward measure + continuous velocity action)

from habitat.config import read_write
from habitat_baselines.config.default import get_config
from habitat_baselines.run import execute_exp

DISCRETE_ACTIONS = ["stop", "move_forward", "turn_left", "turn_right"]


def prepare_config(cfg, modality: str, data_path: str = None):
    """Force a purely continuous action space and the requested visual modality.

    Done in code (not Hydra) because pointnav_hm3d re-adds the discrete actions and the depth sensor in its
    own body, which Hydra defaults-list deletion / sensor-setup overrides don't reliably strip. `data_path`
    is also set here rather than as a Hydra override because it contains `{split}` braces the CLI parser rejects.
    """
    with read_write(cfg):
        if data_path is not None:
            cfg.habitat.dataset.data_path = data_path
        for a in DISCRETE_ACTIONS:
            cfg.habitat.task.actions.pop(a, None)
        sensors = cfg.habitat.simulator.agents.main_agent.sim_sensors
        if modality == "rgb":
            sensors.pop("depth_sensor", None)
        elif modality == "depth":
            sensors.pop("rgb_sensor", None)
        # modality == "rgbd": keep both
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-name", default="rl/pointnav_continuous.yaml")
    p.add_argument("--configs-dir", default="experiments/configs")
    p.add_argument("--modality", choices=["rgb", "depth", "rgbd"], default="rgb",
                   help="Visual input for the policy (default rgb; depth/rgbd are stronger but off-thesis)")
    p.add_argument("--data-path", default=None,
                   help="Override dataset data_path (must contain {split}); set here to dodge Hydra's brace parser")
    p.add_argument("--eval", action="store_true", help="Evaluate instead of train")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Extra Hydra overrides, e.g. habitat_baselines.total_num_steps=2000")
    args = p.parse_args()

    overrides = list(args.set)
    if args.eval:
        overrides.append("habitat_baselines.evaluate=True")

    cfg = get_config(args.config_name, overrides, configs_dir=args.configs_dir)
    cfg = prepare_config(cfg, args.modality, args.data_path)

    actions = list(cfg.habitat.task.actions.keys())
    sensors = list(cfg.habitat.simulator.agents.main_agent.sim_sensors.keys())
    print(f"[train_pointnav] actions={actions} sensors={sensors} modality={args.modality}")
    assert actions == ["velocity_control"], f"expected only velocity_control, got {actions}"

    # The trainer opens the log file before creating its own dirs — make output dirs exist first.
    hb = cfg.habitat_baselines
    os.makedirs(hb.checkpoint_folder, exist_ok=True)
    os.makedirs(hb.tensorboard_dir, exist_ok=True)
    os.makedirs(os.path.dirname(hb.log_file) or ".", exist_ok=True)

    execute_exp(cfg, "eval" if args.eval else "train")


if __name__ == "__main__":
    main()
