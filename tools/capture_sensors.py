#!/usr/bin/env python
"""Capture one RGB + ground-truth depth + semantic observation from an HM3D scene and save it to disk."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foresight.sim.sensors import capture_scene_observation

SEMANTIC_SCENES = {"00800-TEEsavR23oF", "00802-wcojb4TFT35", "00803-k1cupFYWXJ6", "00808-y9hTuugGdiq"}

DEFAULT_DATASET_CONFIG = (
    "data/processed/scene_datasets/hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json"
)


def colorize_depth(depth: np.ndarray, max_depth_m: float = 10.0) -> Image.Image:
    clipped = np.nan_to_num(depth, nan=0.0, posinf=max_depth_m, neginf=0.0)
    clipped = np.clip(clipped, 0.0, max_depth_m)
    gray = (clipped / max_depth_m * 255.0).astype(np.uint8)
    return Image.fromarray(gray)


def colorize_semantic(semantic: np.ndarray) -> Image.Image:
    ids = semantic.astype(np.uint32)
    r = ((ids * 2654435761) >> 16 & 0xFF).astype(np.uint8)
    g = ((ids * 2246822519) >> 8 & 0xFF).astype(np.uint8)
    b = (ids * 3266489917 & 0xFF).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--scene", default="00800-TEEsavR23oF", help="Scene ID, e.g. 00800-TEEsavR23oF")
    parser.add_argument("--out", default="results/runs/sensor_capture", help="Output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    if args.scene not in SEMANTIC_SCENES:
        print(
            f"warning: '{args.scene}' has no semantic annotations in the minival split; "
            f"the semantic sensor output will be empty. Scenes with semantics: {sorted(SEMANTIC_SCENES)}",
            file=sys.stderr,
        )

    out_dir = Path(args.out) / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)

    obs = capture_scene_observation(
        dataset_config=args.dataset_config,
        scene_id=args.scene,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )

    Image.fromarray(obs.rgb).save(out_dir / "rgb.png")
    np.save(out_dir / "depth.npy", obs.depth)
    colorize_depth(obs.depth).save(out_dir / "depth_vis.png")
    np.save(out_dir / "semantic.npy", obs.semantic)
    colorize_semantic(obs.semantic).save(out_dir / "semantic_vis.png")

    metadata = {
        "dataset_config": args.dataset_config,
        "scene": args.scene,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "depth_units": "meters",
        "semantic_units": "per-pixel object instance ID",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Saved rgb.png, depth.npy/.png, semantic.npy/.png, metadata.json to {out_dir}")


if __name__ == "__main__":
    main()
