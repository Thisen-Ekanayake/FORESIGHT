# `tools/estimate_depth.py`

CLI wrapper around [`foresight.perception.depth.depth_anything.ZeroShotDepthEstimator`](../src/foresight/perception/depth/depth_anything.md). Runs zero-shot monocular depth on a captured RGB frame and, if ground truth is available, scores it.

## Usage

```bash
/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python tools/estimate_depth.py \
  --capture-dir results/runs/sensor_capture/<scene> \
  [--model HF_MODEL_ID] [--out DIR]
```

Run from the repo root. `--capture-dir` must contain an `rgb.png` — normally the output of [`tools/capture_sensors.py`](capture_sensors.md); if that same directory also has a `depth.npy` (ground truth), error metrics are computed and printed, otherwise a warning is printed to stderr and only the prediction is saved.

| Flag | Default | Notes |
|---|---|---|
| `--capture-dir` | *(required)* | e.g. `results/runs/sensor_capture/00800-TEEsavR23oF` |
| `--model` | `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | Any HF depth-estimation checkpoint with the same processor/model API |
| `--out` | `<capture-dir>/depth_estimate` | Output directory |

## Output — `<out>/`

| File | Contents |
|---|---|
| `depth_pred.npy` | Predicted depth, float32, meters, `(H, W)` — same shape as the input RGB, at the model's absolute (uncorrected) scale |
| `depth_pred_vis.png` | Grayscale visualization, normalized to the frame's own max depth (not the same scale as `capture_sensors.py`'s `depth_vis.png` — compare shape/structure, not brightness, between the two) |
| `metadata.json` | Model ID, capture dir, and `errors_vs_gt` if ground truth was present |

## Error metrics (`depth_errors`)

Only computed where GT `> 0` (Habitat's depth sensor returns `0` for no-hit pixels). Two variants, both standard monocular-depth metrics (AbsRel, RMSE, δ1 = fraction of pixels with `max(pred/gt, gt/pred) < 1.25`):

- **`raw`** — the checkpoint's output as-is, no correction. This is the number that matters for the actual pipeline and for deliverable A (depth-error-propagation study) — at deployment there's no ground truth to calibrate against, so this is the real error the navigation stack has to tolerate.
- **`scale_aligned`** — `pred` divided by the median `pred/gt` ratio first, then the same three metrics, plus the `scale_factor` used. A diagnostic only, to tell whether a bad `raw` score comes from a systematic offset or genuinely wrong structure — see the finding below.

## Verified finding: scale bias is scene-dependent, don't hardcode a fix

Run against two scenes captured via `capture_sensors.py`:

```bash
python tools/estimate_depth.py --capture-dir results/runs/sensor_capture/00800-TEEsavR23oF
python tools/estimate_depth.py --capture-dir results/runs/sensor_capture/00802-wcojb4TFT35
```

`00800` (living room): raw AbsRel 0.66, raw δ1 0.6% — looks broken until you see the scale-aligned numbers: AbsRel 0.058, δ1 97.2% at a 1.66× scale factor. `00802` (garage, glossy car): scale factor 2.26×, and even after alignment AbsRel 0.27 / δ1 33% — meaningfully worse structure, not just a worse offset. Full writeup and the reflectivity connection: see the depth_anything.py docs linked above.
