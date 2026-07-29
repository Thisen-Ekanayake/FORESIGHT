#!/usr/bin/env python
"""Generate PointGoal navigation episodes (start pose + goal) for the local HM3D minival scenes.

We only have the 10 minival scenes locally (not the full HM3D train split), so we sample episodes on
those scenes with habitat-lab's official generator and write the `.json.gz` datasets for each split in
the config. Splits cover all scenes with disjoint episodes (train PointNav on so few scenes overfits —
acceptable for a working agent + reusable pipeline; expand with the HM3D train split later).

Every value lives in experiments/configs/rl/generate_episodes.yaml — override with `--set KEY=VALUE`.

Usage:
  DISPLAY=:1 <habitat-python> scripts/generate_pointnav_episodes.py \
      --set splits.train.num_episodes_per_scene=300 --set splits.val.num_episodes_per_scene=30
"""
import argparse
import gzip
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foresight.rl.config import get, load_config, override

import habitat
from habitat.config import read_write
from habitat.config.default import get_config
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode
from habitat.sims import make_sim

# Only literal in this script: where its config lives (relative to the repo root).
SCRIPT_CONFIG = "experiments/configs/rl/generate_episodes.yaml"


def discover_scenes(scenes_root: Path, mesh_glob: str):
    """Scene handles = subdir names like '00800-TEEsavR23oF' that contain a scene mesh."""
    scenes = []
    for d in sorted(scenes_root.iterdir()):
        if d.is_dir() and list(d.glob(mesh_glob)):
            scenes.append(d.name)
    return scenes


def make_scene_sim(scene_handle: str, scene_dataset: str, sim_config: str):
    cfg = get_config(sim_config)
    with read_write(cfg):
        cfg.habitat.simulator.scene = scene_handle
        cfg.habitat.simulator.scene_dataset = scene_dataset
    return make_sim(cfg.habitat.simulator.type, config=cfg.habitat.simulator)


def write_split(episodes, out_path: Path):
    dataset = PointNavDatasetV1()
    dataset.episodes = episodes
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write(dataset.to_json())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=SCRIPT_CONFIG, help="Script config (YAML) holding all defaults")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Override script-config keys, e.g. splits.train.num_episodes_per_scene=300")
    p.add_argument("--scenes", nargs="*", default=None, help="Scene handles (default: all discovered)")
    p.add_argument("--out", default=None, help="Output root for the split datasets")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    override(cfg, "scenes", args.scenes)
    override(cfg, "out", args.out)
    override(cfg, "seed", args.seed)

    scenes_root = Path(get(cfg, "scenes_root"))
    scenes_dir = get(cfg, "scenes_dir")
    scene_dataset = get(cfg, "scene_dataset")
    mesh_glob = get(cfg, "scene_mesh_glob")
    sampling = get(cfg, "sampling")
    splits = get(cfg, "splits")
    seed = get(cfg, "seed")

    scenes = get(cfg, "scenes") or discover_scenes(scenes_root, mesh_glob)
    if not scenes:
        sys.exit(f"No scenes matching {mesh_glob} found under {scenes_root}")
    print(f"Generating for {len(scenes)} scenes: {scenes}")

    split_eps = {name: [] for name in splits}
    skipped = []
    per_scene = sum(get(cfg, f"splits.{name}.num_episodes_per_scene") for name in splits)
    for scene in scenes:
        try:
            sim = make_scene_sim(scene, scene_dataset, get(cfg, "sim_config"))
        except Exception as e:  # scene not registered in the dataset config / unloadable — skip it
            print(f"  {scene}: SKIPPED (failed to load: {type(e).__name__})")
            skipped.append(scene)
            continue
        sim.seed(seed)
        eps = list(
            generate_pointnav_episode(
                sim,
                num_episodes=per_scene,
                closest_dist_limit=sampling["closest_dist_limit"],
                furthest_dist_limit=sampling["furthest_dist_limit"],
                geodesic_to_euclid_min_ratio=sampling["geodesic_to_euclid_min_ratio"],
            )
        )
        sim.close()
        # Store scene_id as the mesh path relative to scenes_dir, and pin the scene_dataset_config, so the
        # loader (which joins scenes_dir + scene_id and applies episode.scene_dataset_config) resolves them.
        mesh = next((scenes_root / scene).glob(mesh_glob))
        scene_rel = os.path.relpath(mesh, scenes_dir)
        for i, e in enumerate(eps):
            e.episode_id = get(cfg, "episode_id_format").format(scene=scene, index=i)
            e.scene_id = scene_rel
            e.scene_dataset_config = scene_dataset
        # Fill the splits in config order from this scene's episodes, so they stay disjoint.
        taken = 0
        counts = []
        for name, spec in splits.items():
            chunk = eps[taken: taken + spec["num_episodes_per_scene"]]
            split_eps[name].extend(chunk)
            counts.append(f"{len(chunk)} {name}")
            taken += spec["num_episodes_per_scene"]
        print(f"  {scene}: {len(eps)} episodes ({' / '.join(counts)})")

    out = Path(get(cfg, "out"))
    for name, spec in splits.items():
        write_split(split_eps[name], out / spec["out_file"])
    used = len(scenes) - len(skipped)
    totals = " + ".join(f"{len(split_eps[name])} {name}" for name in splits)
    print(f"\nWrote {totals} episodes from {used} scenes to {out}/")
    if skipped:
        print(f"Skipped {len(skipped)} unloadable scenes: {skipped}")


if __name__ == "__main__":
    main()
