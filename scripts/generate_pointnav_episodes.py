#!/usr/bin/env python
"""Generate PointGoal navigation episodes (start pose + goal) for the local HM3D minival scenes.

We only have the 10 minival scenes locally (not the full HM3D train split), so we sample episodes on
those scenes with habitat-lab's official generator and write train/val `.json.gz` datasets. Both splits
cover all scenes with disjoint episodes (train PointNav on so few scenes overfits — acceptable for a
working agent + reusable pipeline; expand with the HM3D train split later).

Usage:
  DISPLAY=:1 <habitat-python> scripts/generate_pointnav_episodes.py --num-train 400 --num-val 40
"""
import argparse
import gzip
import os
import sys
from pathlib import Path

import habitat
from habitat.config import read_write
from habitat.config.default import get_config
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode
from habitat.sims import make_sim

# scenes_dir must match habitat.dataset.scenes_dir in the training config: episode.scene_id is stored
# relative to it, and habitat joins the two (PointNavDatasetV1.from_json) to locate the scene mesh.
DEFAULT_SCENES_DIR = "data/processed/scene_datasets"
DEFAULT_SCENES_ROOT = "data/processed/scene_datasets/hm3d/minival"
# PointNav needs no semantics; this config registers all 10 minival basis meshes (the annotated config
# registers only the 4 semantic scenes). See data/README for the distinction.
DEFAULT_SCENE_DATASET = (
    "data/processed/scene_datasets/hm3d/minival/hm3d_minival_basis_pointnav.scene_dataset_config.json"
)


def discover_scenes(scenes_root: Path):
    """Scene handles = subdir names like '00800-TEEsavR23oF' that contain a .basis.glb."""
    scenes = []
    for d in sorted(scenes_root.iterdir()):
        if d.is_dir() and list(d.glob("*.basis.glb")):
            scenes.append(d.name)
    return scenes


def make_scene_sim(scene_handle: str, scene_dataset: str):
    cfg = get_config("benchmark/nav/pointnav/pointnav_hm3d.yaml")
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
    p.add_argument("--scenes-root", default=DEFAULT_SCENES_ROOT)
    p.add_argument("--scenes-dir", default=DEFAULT_SCENES_DIR,
                   help="Root that episode.scene_id is relative to (match the training config's scenes_dir)")
    p.add_argument("--scene-dataset", default=DEFAULT_SCENE_DATASET)
    p.add_argument("--out", default="data/processed/pointnav_episodes")
    p.add_argument("--num-train", type=int, default=400, help="Episodes per scene for the train split")
    p.add_argument("--num-val", type=int, default=40, help="Episodes per scene for the val split")
    p.add_argument("--scenes", nargs="*", default=None, help="Scene handles (default: all discovered)")
    p.add_argument("--closest-dist", type=float, default=1.0)
    p.add_argument("--furthest-dist", type=float, default=25.0)
    p.add_argument("--geo-to-euclid-min", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    scenes = args.scenes or discover_scenes(Path(args.scenes_root))
    if not scenes:
        sys.exit(f"No scenes with a .basis.glb found under {args.scenes_root}")
    print(f"Generating for {len(scenes)} scenes: {scenes}")

    train_eps, val_eps = [], []
    skipped = []
    per_scene = args.num_train + args.num_val
    for scene in scenes:
        try:
            sim = make_scene_sim(scene, args.scene_dataset)
        except Exception as e:  # scene not registered in the dataset config / unloadable — skip it
            print(f"  {scene}: SKIPPED (failed to load: {type(e).__name__})")
            skipped.append(scene)
            continue
        sim.seed(args.seed)
        eps = list(
            generate_pointnav_episode(
                sim,
                num_episodes=per_scene,
                closest_dist_limit=args.closest_dist,
                furthest_dist_limit=args.furthest_dist,
                geodesic_to_euclid_min_ratio=args.geo_to_euclid_min,
            )
        )
        sim.close()
        # Store scene_id as the mesh path relative to scenes_dir, and pin the scene_dataset_config, so the
        # loader (which joins scenes_dir + scene_id and applies episode.scene_dataset_config) resolves them.
        glb = next((Path(args.scenes_root) / scene).glob("*.basis.glb"))
        scene_rel = os.path.relpath(glb, args.scenes_dir)
        for i, e in enumerate(eps):
            e.episode_id = f"{scene}_{i}"
            e.scene_id = scene_rel
            e.scene_dataset_config = args.scene_dataset
        train_eps.extend(eps[: args.num_train])
        val_eps.extend(eps[args.num_train :])
        print(f"  {scene}: {len(eps)} episodes ({len(eps[:args.num_train])} train / {len(eps[args.num_train:])} val)")

    out = Path(args.out)
    write_split(train_eps, out / "train" / "train.json.gz")
    write_split(val_eps, out / "val" / "val.json.gz")
    used = len(scenes) - len(skipped)
    print(f"\nWrote {len(train_eps)} train + {len(val_eps)} val episodes from {used} scenes to {out}/")
    if skipped:
        print(f"Skipped {len(skipped)} unloadable scenes: {skipped}")


if __name__ == "__main__":
    main()
