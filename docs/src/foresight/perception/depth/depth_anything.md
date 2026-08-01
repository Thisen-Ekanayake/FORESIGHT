# `src/foresight/perception/depth/depth_anything.py`

Zero-shot monocular metric depth via [Depth Anything V2](https://huggingface.co/depth-anything), metric-indoor checkpoint. Library code — no file I/O, no CLI; consumed by [`tools/estimate_depth.py`](../../../../tools/estimate_depth.md).

## `DEFAULT_MODEL`

`"depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"` — the small (~25M param) variant, fine-tuned for metric (real-scale) indoor depth. Chosen over the base/large checkpoints for the proposal's "modest on-board compute" constraint, and over the relative-depth (non-metric) checkpoints because the depth-error-propagation study (`PROGRESS.md` deliverable A) needs output comparable in units (meters) to Habitat's ground-truth depth sensor.

## `ZeroShotDepthEstimator`

```python
estimator = ZeroShotDepthEstimator()          # loads DEFAULT_MODEL, picks cuda if available
depth = estimator.estimate(rgb)               # rgb: (H, W, 3) uint8 -> depth: (H, W) float32, meters
```

- `__init__(model_name=DEFAULT_MODEL, device=None)` — loads the HF `AutoImageProcessor` + `AutoModelForDepthEstimation` for `model_name`, moves the model to `device` (`cuda` if available, else `cpu`) and sets eval mode. Any model on the HF depth-estimation hub with this architecture works if swapped in.
- `estimate(rgb)` — runs the model under `torch.inference_mode()` and calls the processor's `post_process_depth_estimation(outputs, target_sizes=[(h, w)])` to resize the prediction back to the input's original resolution. Returns a plain numpy array, not a torch tensor — no GPU/framework leakage into caller code.

This is used **as-is, zero-shot** — no fine-tuning, no scale/shift fitting against HM3D. That's deliberate: the whole point of deliverable A is to characterize how this exact off-the-shelf model's error propagates into navigation failures.

## Known finding: scale bias is scene-dependent

Verified against two captured HM3D scenes (ground truth from `tools/capture_sensors.py`):

| Scene | Raw AbsRel | Raw δ1 | Scale factor (median pred/gt) | Scale-aligned AbsRel | Scale-aligned δ1 |
|---|---|---|---|---|---|
| `00800-TEEsavR23oF` (living room) | 0.66 | 0.6% | 1.66× | 0.058 | 97.2% |
| `00802-wcojb4TFT35` (garage, reflective car) | 1.19 | 1.0% | 2.26× | 0.266 | 32.8% |

Two takeaways:
1. **Raw metric output is not trustworthy as absolute distance** on HM3D — the checkpoint's calibration doesn't transfer cleanly to this render/camera domain, and the scale offset is not a fixed constant (1.66× vs 2.26×), so a single global correction factor won't fix it.
2. **Once the scene-specific scale is corrected, relative structure quality itself varies** — near-perfect on the living room (δ1 97%), much worse on the garage scene, which has a glossy/reflective car — an early, unprompted confirmation of exactly the reflectivity failure mode the proposal's difficulty-stratified benchmark (deliverable C) is designed to stress-test.

Do not "fix" this with a global scale constant in the perception module — the raw, uncorrected error is the realistic zero-shot signal deliverable A is supposed to measure. `tools/estimate_depth.py`'s scale-aligned metric exists as a diagnostic to separate "badly calibrated" from "structurally wrong," not as a correction to ship.
