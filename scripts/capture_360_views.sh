#!/usr/bin/env bash
# Interactively drive an agent in a HM3D minival scene, then capture a 30%-overlap 360-degree ring of stills.
# Usage: scripts/capture_360_views.sh <scene_id>   e.g. 00800-TEEsavR23oF
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <scene_id>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

DISPLAY=:1 /home/thisen-ekanayake/miniforge3/envs/habitat/bin/python \
  "$SCRIPT_DIR/capture_360_views.py" "$1"
