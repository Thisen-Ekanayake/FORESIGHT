#!/usr/bin/env bash
# Launch habitat-viewer interactively on a scene from the HM3D minival dataset.
# Usage: scripts/run_sim_interactively.sh <scene_id>   e.g. 00800-TEEsavR23oF
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <scene_id>" >&2
  exit 1
fi

SCENE_ID="$1"

DISPLAY=:1 /home/thisen-ekanayake/miniforge3/envs/habitat/bin/habitat-viewer \
  --dataset /ml/FORESIGHT/data/processed/scene_datasets/hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json \
  -- "$SCENE_ID" 2>&1 | tail -60