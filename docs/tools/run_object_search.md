# `tools/run_object_search.py`

Object search, baseline vs FORESIGHT, on HM3D homes in Habitat-Sim: runs episodes, scores them against the scene
annotations, renders the narrated 1920×1920 video. Agent design: [`foresight.planning.object_search`](../src/foresight/planning/object_search.md).

| Condition | Explorer + object memory + safety | Qwen3-VL per look | Laya |
|---|---|---|---|
| `baseline` | yes | presence check, grounding, search hint (~0.5 s) | — |
| `foresight` | same | + scene note (reflection-aware prompt) + mirror grounding (~3.8 s) | reads every note each tick; double-check → wall |

Sim time advances by measured compute ([`timed_episode`](../../src/foresight/sim/timed_episode.py)); background
looks land after their measured latency. A crash ends the run (driving forward with < 0.3 m true clearance).

## Usage

```bash
PY=/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python
# fixed evaluation set (written once by tools/make_object_search_episodes.py, committed before any run)
$PY tools/make_object_search_episodes.py
# demo episode + evaluation set, both conditions (~3 h on an RTX 4060)
$PY tools/run_object_search.py --episodes mirror-sofa-bed <names from episodes.json> --out results/runs/object_search/eval
$PY tools/run_object_search.py --phase eval --episodes eval --out results/runs/object_search/eval      # -> eval.json
$PY tools/run_object_search.py --phase assets --episodes mirror-sofa-bed --out results/runs/object_search/eval
$PY tools/run_object_search.py --phase render --episodes mirror-sofa-bed --out results/runs/object_search/eval
```

`--episodes eval` expands to every episode in `experiments/configs/object_search/episodes.json` plus the demo
episode (zsh: split a `$EPS` list with `${=EPS}`). `--oracle` swaps grounding for the simulator's labels (and a
scripted mirror note) — a harness for developing the explorer without the VLM, never for reported numbers.

| Flag | Default | Notes |
|---|---|---|
| `--phase` | `run` | `run`, `eval` (score), `assets` (top-down cutaway + start-frame depth), `render` (video) |
| `--conditions` | both | |
| `--out` | `results/runs/object_search/demo` | `<out>/<episode>/<condition>.pkl` + `_meta.json` |
| `--max-sim-s` | episode's | override the time budget (quick tests) |
| `--no-prior` | off | explore without Qwen's search hint |
| `--threshold` | `0.6` | Laya P(mirror) that raises a flag |
| `--contact-m` | `0.3` | crash rule (sim GT depth ahead while driving forward) |
| `--eval` | `results/runs/object_search/eval/eval.json` | scoreboard source for `render` |

GPU memory is tight on 8 GB: Qwen3-VL-2B (4.1 GB) + Depth Anything + renderer (1.2 GB without the semantic mesh)
+ the Laya worker (1 GB) + the desktop. Models are loaded once per process; don't run two jobs at once.
