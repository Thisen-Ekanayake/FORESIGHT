#!/usr/bin/env bash
# Render a top-down PNG of a Habitat-Sim .navmesh file, loaded standalone (no stage .glb needed).
# Usage: scripts/render_navmesh_smooth.sh <path/to/scene.navmesh> [--original|--smooth] [output.png]
#   --original (default)  raw per-pixel navmesh raster, no smoothing
#   --smooth               corner-rounded / anti-aliased render
# Output defaults to <repo_root>/results/<navmesh_stem>_<original|smooth>.png
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path/to/scene.navmesh> [--original|--smooth] [output.png]" >&2
  exit 1
fi

NAVMESH_PATH="$1"
shift

MODE="original"
OUTPUT_PATH=""
for arg in "$@"; do
  case "$arg" in
    --original) MODE="original" ;;
    --smooth) MODE="smooth" ;;
    *) OUTPUT_PATH="$arg" ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../results"
PYTHON=/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python

"$PYTHON" - "$NAVMESH_PATH" "$MODE" "$OUTPUT_PATH" "$RESULTS_DIR" <<'PY'
"""Render a top-down PNG of a Habitat-Sim navmesh, loaded standalone (no stage .glb needed).

Modes:
  original - raw get_topdown_view raster, pixel-exact, jagged edges.
  smooth   - supersample 2x, round corners with an elliptical open+close pass sized well under the
             scene's narrowest real corridor (so doorways don't get eaten), finish with a Gaussian
             blur + rethreshold, then downscale with area averaging for anti-aliased edges.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import habitat_sim
import numpy as np
from PIL import Image

FREE = (255, 255, 255)
BLOCKED = (20, 20, 20)

MPP = 0.02  # meters per pixel
SUPERSAMPLE = 2
ROUND_RADIUS_PX = 21  # corner-rounding kernel radius in supersampled px
BLUR_KSIZE = 31
MIN_CONTOUR_AREA = 80


def _get_topdown_mask(navmesh_path: str, height: float | None = None) -> np.ndarray:
    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(navmesh_path):
        raise RuntimeError(f"Failed to load navmesh: {navmesh_path}")

    lo, hi = pf.get_bounds()
    if height is None:
        height = lo[1] + 0.1  # just above the lowest floor

    topdown = pf.get_topdown_view(MPP, height)
    return topdown.astype(np.uint8) * 255


def render_original_topdown(navmesh_path: str) -> Image.Image:
    mask = _get_topdown_mask(navmesh_path)
    rgb = np.where(mask[..., None] > 0, np.array(FREE, np.uint8), np.array(BLOCKED, np.uint8))
    return Image.fromarray(rgb.astype(np.uint8))


def render_smoothed_topdown(navmesh_path: str) -> Image.Image:
    mask = _get_topdown_mask(navmesh_path)

    big = cv2.resize(
        mask,
        (mask.shape[1] * SUPERSAMPLE, mask.shape[0] * SUPERSAMPLE),
        interpolation=cv2.INTER_NEAREST,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ROUND_RADIUS_PX, ROUND_RADIUS_PX))
    smooth = cv2.morphologyEx(big, cv2.MORPH_OPEN, kernel)
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_CLOSE, kernel)
    smooth = cv2.GaussianBlur(smooth, (BLUR_KSIZE, BLUR_KSIZE), 0)
    _, smooth = cv2.threshold(smooth, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(smooth, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    canvas = np.full((*smooth.shape, 3), BLOCKED, dtype=np.uint8)
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        is_hole = hierarchy[0][i][3] != -1
        color = BLOCKED if is_hole else FREE
        cv2.drawContours(canvas, [cnt], -1, color, thickness=cv2.FILLED, lineType=cv2.LINE_AA)

    canvas = cv2.resize(canvas, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    return Image.fromarray(canvas)


def main():
    navmesh_path = Path(sys.argv[1])
    mode = sys.argv[2]
    out_arg = sys.argv[3] if len(sys.argv) > 3 else ""
    results_dir = Path(sys.argv[4])
    out_path = Path(out_arg) if out_arg else results_dir / f"{navmesh_path.stem}_{mode}.png"

    img = render_original_topdown(str(navmesh_path)) if mode == "original" else render_smoothed_topdown(str(navmesh_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
PY
