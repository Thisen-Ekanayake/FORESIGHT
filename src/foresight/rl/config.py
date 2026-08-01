"""Loader for the plain-YAML script configs under `experiments/configs/rl/`.

Every tunable value used by the RL scripts lives in YAML — the scripts themselves hold no literal
defaults, only the path of the config file they read. A script loads its config, applies `--set
dotted.key=value` overrides (and any named CLI flags), then reads values out of the result.

These are *plain* YAML files read with PyYAML, not Hydra configs: they configure the scripts
(which episodes to generate, what to render, which checkpoint to load), while the habitat /
habitat-baselines task and PPO hyperparameters stay in the Hydra config
`experiments/configs/rl/pointnav_continuous.yaml`.

Overrides are strict: every segment of a dotted key must already exist in the YAML, so a typo or an
override meant for another channel (e.g. a Hydra key) fails loudly instead of being silently ignored.
"""
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import yaml

# Repo root: src/foresight/rl/config.py -> parents[3]. Config paths in the YAML files are relative to it.
REPO_ROOT = Path(__file__).resolve().parents[3]


def _split_key(dotted_key: str) -> Sequence[str]:
    parts = [p for p in dotted_key.split(".") if p]
    if not parts:
        raise ValueError(f"empty config key: {dotted_key!r}")
    return parts


def get(cfg: Dict[str, Any], dotted_key: str) -> Any:
    """Read a nested value: get(cfg, 'render.fps')."""
    node: Any = cfg
    for i, part in enumerate(_split_key(dotted_key)):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"missing config key {dotted_key!r} (at {'.'.join(_split_key(dotted_key)[:i + 1])})")
        node = node[part]
    return node


def set_(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Write a nested value; every key on the path must already exist in the config."""
    parts = _split_key(dotted_key)
    node: Any = cfg
    for i, part in enumerate(parts[:-1]):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"unknown config key {dotted_key!r} (at {'.'.join(parts[:i + 1])})")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise KeyError(f"unknown config key {dotted_key!r} — declare it in the YAML config first")
    node[parts[-1]] = value


def override(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Apply a CLI flag override, ignoring `None` (flag not given -> keep the YAML value)."""
    if value is not None:
        set_(cfg, dotted_key, value)


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Load a script config and apply `KEY=VALUE` overrides (values parsed as YAML scalars)."""
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{cfg_path}: expected a YAML mapping at the top level")
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        set_(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def choice(cfg: Dict[str, Any], dotted_key: str, value: str, options: Iterable[str]) -> str:
    """Validate a value against options declared in the config (choices come from YAML, not argparse)."""
    options = list(options)
    if value not in options:
        raise SystemExit(f"invalid {dotted_key}={value!r}; config declares {options}")
    return value
