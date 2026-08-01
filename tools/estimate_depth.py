#!/usr/bin/env python
"""Run zero-shot monocular depth estimation on a captured RGB frame and score it against ground truth."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foresight.perception.depth.depth_anything import DEFAULT_MODEL, ZeroShotDepthEstimator


def colorize_depth(depth: np.ndarray, max_depth_m: float) -> Image.Image:
    clipped = np.nan_to_num(depth, nan=0.0, posinf=max_depth_m, neginf=0.0)
    clipped = np.clip(clipped, 0.0, max_depth_m)
    gray = (clipped / max_depth_m * 255.0).astype(np.uint8)
    return Image.fromarray(gray)


def _metrics(pred_v: np.ndarray, gt_v: np.ndarray) -> dict:
    abs_rel = float(np.mean(np.abs(pred_v - gt_v) / gt_v))
    rmse = float(np.sqrt(np.mean((pred_v - gt_v) ** 2)))
    ratio = np.maximum(pred_v / gt_v, gt_v / pred_v)
    delta1 = float(np.mean(ratio < 1.25))
    return {"abs_rel": abs_rel, "rmse_m": rmse, "delta1": delta1}


def depth_errors(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Standard monocular depth metrics (AbsRel, RMSE, delta1), computed only where GT is valid (> 0).

    Reports both the raw metric error (checkpoint's absolute scale, as-is) and the
    scale-aligned error (pred rescaled by the median pred/gt ratio) — the gap between the
    two separates a systematic calibration offset from genuine structural error, which
    matters for deciding whether the pipeline needs a scale-correction step.
    """
    valid = gt > 0
    pred_v, gt_v = pred[valid], gt[valid]
    scale = float(np.median(pred_v / gt_v))
    return {
        "raw": _metrics(pred_v, gt_v),
        "scale_aligned": {"scale_factor": scale, **_metrics(pred_v / scale, gt_v)},
        "n_valid_px": int(valid.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture-dir",
        required=True,
        help="A results/runs/sensor_capture/<scene>/ dir containing rgb.png (and optionally depth.npy as ground truth)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=None, help="Output dir, default: <capture-dir>/depth_estimate")
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir)
    rgb = np.array(Image.open(capture_dir / "rgb.png").convert("RGB"))

    out_dir = Path(args.out) if args.out else capture_dir / "depth_estimate"
    out_dir.mkdir(parents=True, exist_ok=True)

    estimator = ZeroShotDepthEstimator(model_name=args.model)
    pred_depth = estimator.estimate(rgb)

    np.save(out_dir / "depth_pred.npy", pred_depth)
    colorize_depth(pred_depth, max_depth_m=float(pred_depth.max())).save(out_dir / "depth_pred_vis.png")

    result = {"model": args.model, "capture_dir": str(capture_dir)}

    gt_path = capture_dir / "depth.npy"
    if gt_path.exists():
        gt_depth = np.load(gt_path)
        result["errors_vs_gt"] = depth_errors(pred_depth, gt_depth)
    else:
        print(f"warning: no ground-truth depth.npy found in {capture_dir}; skipping error metrics", file=sys.stderr)

    (out_dir / "metadata.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
