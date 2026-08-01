# `tools/describe_scene.py`

CLI wrapper around [`foresight.perception.vlm.qwen_vl.Qwen3VLSceneReasoner`](../src/foresight/perception/vlm/qwen_vl.md). Runs zero-shot VLM scene reasoning (free-text `describe`, not the structured `propose_heading` used by the action fusion controller) on a captured RGB frame.

## Usage

```bash
/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python tools/describe_scene.py \
  --capture-dir results/runs/sensor_capture/<scene> \
  [--model MODEL_PATH] [--prompt TEXT] [--system-prompt TEXT] \
  [--max-new-tokens N] [--repetition-penalty F] [--out DIR]
```

Run from the repo root. No `DISPLAY` needed — this doesn't touch Habitat-Sim, only loads the VLM and reads an existing `rgb.png`.

| Flag | Default | Notes |
|---|---|---|
| `--capture-dir` | *(required)* | Must contain `rgb.png` — normally the output of [`tools/capture_sensors.py`](capture_sensors.md) |
| `--model` | `models/Qwen3-VL-2B-Instruct` | Local checkpoint path; download via `scripts/download_qwen3_vl_2b.sh` |
| `--prompt` | `"Describe this scene for a robot deciding where it is safe to move next."` | User-turn text |
| `--system-prompt` | the module's `DEFAULT_SYSTEM_PROMPT` | System-turn text — navigation-framed by default (free space, obstacles, reflective/textureless hazards) |
| `--max-new-tokens` | `256` | |
| `--repetition-penalty` | `1.15` | Non-default — the checkpoint's own default (1.0) degenerated into a repeated-phrase loop toward the token cap on the first run; see the module docs |
| `--out` | `<capture-dir>/vlm_description` | Output directory |

## Output — `<out>/`

| File | Contents |
|---|---|
| `description.json` | `model`, `capture_dir`, `prompt`, `description` (the model's free-text reply) |

## Verified

```bash
python tools/describe_scene.py --capture-dir results/runs/sensor_capture/00800-TEEsavR23oF
python tools/describe_scene.py --capture-dir results/runs/sensor_capture/00802-wcojb4TFT35
```

Both produced coherent, grounded descriptions — furniture/layout/obstacle callouts for the living-room scene, obstacle + hazard callouts (car, clutter, hanging hose) for the garage scene — at the default `repetition_penalty=1.15`.
