#!/usr/bin/env python
"""Train (or evaluate) the continuous-control PointGoal navigator with PPO via habitat-baselines.

Composes the Hydra config named in experiments/configs/rl/train_pointnav.yaml, post-processes it
(pure-continuous action space + visual modality), and hands off to habitat-baselines. Resumable:
re-running picks up the latest checkpoint in the config's checkpoint_folder automatically
(load_resume_state_config).

Every value lives in YAML, not here: script settings in experiments/configs/rl/train_pointnav.yaml
(`--set KEY=VALUE` to override), task/PPO hyperparameters in the Hydra config it names
(`--habitat-set KEY=VALUE` to override).

Examples:
  # smoke test (few steps, verifies the pipeline runs)
  DISPLAY=:1 <py> scripts/train_pointnav.py --habitat-set habitat_baselines.total_num_steps=2000 \
      habitat_baselines.num_environments=2 habitat_baselines.log_interval=1
  # full training
  DISPLAY=:1 <py> scripts/train_pointnav.py
  # evaluate a checkpoint on the val split
  DISPLAY=:1 <py> scripts/train_pointnav.py --eval \
      --habitat-set habitat_baselines.eval_ckpt_path_dir=results/runs/pointnav/checkpoints/latest.pth
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import foresight.rl  # noqa: F401  (registers the reward measure + continuous velocity action)
from foresight.rl.config import choice, get, load_config, override

from habitat.config import read_write
from habitat_baselines.config.default import get_config
from habitat_baselines.run import execute_exp

# Only literal in this script: where its config lives (relative to the repo root).
SCRIPT_CONFIG = "experiments/configs/rl/train_pointnav.yaml"


def prepare_config(cfg, script_cfg):
    """Force a purely continuous action space and the requested visual modality.

    Done in code (not Hydra) because pointnav_hm3d re-adds the discrete actions and the depth sensor in its
    own body, which Hydra defaults-list deletion / sensor-setup overrides don't reliably strip. `data_path`
    is also set here rather than as a Hydra override because it contains `{split}` braces the CLI parser rejects.
    """
    modality = choice(script_cfg, "modality", get(script_cfg, "modality"), get(script_cfg, "modalities"))
    data_path = get(script_cfg, "data_path")
    with read_write(cfg):
        if data_path is not None:
            cfg.habitat.dataset.data_path = data_path
        for action in get(script_cfg, "discrete_actions"):
            cfg.habitat.task.actions.pop(action, None)
        sensors = cfg.habitat.simulator.agents.main_agent.sim_sensors
        for sensor in get(script_cfg, "modalities")[modality]:
            sensors.pop(sensor, None)
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=SCRIPT_CONFIG, help="Script config (YAML) holding all defaults")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Override script-config keys, e.g. modality=depth")
    p.add_argument("--modality", default=None, help="Visual input for the policy (see `modalities` in the config)")
    p.add_argument("--data-path", default=None,
                   help="Override dataset data_path (must contain {split}); set here to dodge Hydra's brace parser")
    p.add_argument("--eval", action="store_true", default=None, help="Evaluate instead of train")
    p.add_argument("--habitat-set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Extra Hydra overrides for the habitat config, e.g. habitat_baselines.total_num_steps=2000")
    args = p.parse_args()

    script_cfg = load_config(args.config, args.set)
    override(script_cfg, "modality", args.modality)
    override(script_cfg, "data_path", args.data_path)
    override(script_cfg, "eval", args.eval)

    evaluate = bool(get(script_cfg, "eval"))
    overrides = list(get(script_cfg, "habitat_overrides")) + list(args.habitat_set)
    if evaluate:
        overrides.append(get(script_cfg, "eval_override"))

    cfg = get_config(
        get(script_cfg, "habitat_config.config_name"),
        overrides,
        configs_dir=get(script_cfg, "habitat_config.configs_dir"),
    )
    cfg = prepare_config(cfg, script_cfg)

    actions = list(cfg.habitat.task.actions.keys())
    sensors = list(cfg.habitat.simulator.agents.main_agent.sim_sensors.keys())
    modality = get(script_cfg, "modality")
    print(f"[train_pointnav] actions={actions} sensors={sensors} modality={modality}")
    expected = list(get(script_cfg, "expected_actions"))
    assert actions == expected, f"expected only {expected}, got {actions}"

    # The trainer opens the log file before creating its own dirs — make output dirs exist first.
    hb = cfg.habitat_baselines
    os.makedirs(hb.checkpoint_folder, exist_ok=True)
    os.makedirs(hb.tensorboard_dir, exist_ok=True)
    os.makedirs(os.path.dirname(hb.log_file) or ".", exist_ok=True)

    execute_exp(cfg, "eval" if evaluate else "train")


if __name__ == "__main__":
    main()
