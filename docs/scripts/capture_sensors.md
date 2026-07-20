# `scripts/capture_sensors.py`

CLI wrapper around [`foresight.sim.sensors.capture_scene_observation`](../src/foresight/sim/sensors.md). Captures one RGB + ground-truth depth + semantic frame from an HM3D scene and writes it to disk.

## Usage

```bash
/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python scripts/capture_sensors.py \
  --scene 00800-TEEsavR23oF \
  [--dataset-config PATH] [--out DIR] [--seed N] [--width W] [--height H]
```

Run from the repo root — `--out` and the default `--dataset-config` are relative paths.

| Flag | Default | Notes |
|---|---|---|
| `--dataset-config` | `data/processed/scene_datasets/hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json` | |
| `--scene` | `00800-TEEsavR23oF` | Scene ID from the dataset config |
| `--out` | `results/runs/sensor_capture` | Output root; actual files land in `<out>/<scene>/` |
| `--seed` | `0` | Seeds the random navigable-point spawn |
| `--width`, `--height` | `640`, `480` | Sensor resolution in pixels |

Prints a warning to stderr (doesn't abort) if `--scene` isn't one of the 4 minival scenes with semantic annotations — `00800-TEEsavR23oF`, `00802-wcojb4TFT35`, `00803-k1cupFYWXJ6`, `00808-y9hTuugGdiq` (`SEMANTIC_SCENES` in the script). For any other scene the `semantic.npy`/`semantic_vis.png` output will be all zeros.

## Output — `<out>/<scene>/`

| File | Contents |
|---|---|
| `rgb.png` | RGB frame, uint8 |
| `depth.npy` | Raw ground-truth depth, float32, meters, shape `(H, W)` |
| `depth_vis.png` | Depth for eyeballing: clipped to `[0, 10]` m, normalized to grayscale (`colorize_depth`) |
| `semantic.npy` | Raw per-pixel object instance IDs, `(H, W)` |
| `semantic_vis.png` | Semantic map for eyeballing: each instance ID hashed to an RGB color (`colorize_semantic`) — colors are stable per ID within a run but arbitrary, not a category legend |
| `metadata.json` | The capture params (`dataset_config`, `scene`, `width`, `height`, `seed`) plus units (`depth_units: "meters"`, `semantic_units: "per-pixel object instance ID"`) |

To recover category names for a semantic ID, cross-reference `semantic.npy` against the scene's `*.semantic.txt` in `data/raw/hm3d-minival-semantic-annots-v0.2/<scene>/` (not copied into the output — see `data/README.md`).

## Example

```bash
cd /ml/FORESIGHT
/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python scripts/capture_sensors.py --scene 00802-wcojb4TFT35
# -> results/runs/sensor_capture/00802-wcojb4TFT35/{rgb.png,depth.npy,depth_vis.png,semantic.npy,semantic_vis.png,metadata.json}
```
