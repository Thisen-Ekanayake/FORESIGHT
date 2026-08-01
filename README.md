# FORESIGHT

Map-Free Monocular Navigation with an On-Board VLM. Full spec: `Project_FORESIGHT.pdf`. Live status: `PROGRESS.md`.

## Structure

```
data/
  raw/            downloaded datasets, untouched (see data/README.md)
  processed/      derived, Habitat-loadable scene datasets
src/foresight/
  sim/            Habitat-Sim environment + sensor (RGB/depth/semantic) wrappers
  perception/
    depth/        zero-shot monocular depth model
    vlm/          compact VLM scene reasoning
  planning/       action fusion controller + depth-only heuristic baseline
  safety/         reactive safety layer
  benchmark/      difficulty stratification + metrics (collision rate, success rate, SPL)
  utils/
experiments/
  configs/        experiment configs (depth-error sweeps, ablations, benchmark runs)
results/
  figures/        generated plots
  metrics/        generated metric tables
  runs/           raw per-run outputs
scripts/          shell entry points
tools/            Python CLI entry points
notebooks/        exploratory analysis
tests/            unit tests
docs/             supplementary write-ups
logs/             dated per-session work logs (see CLAUDE.md)
```

## Environment

Conda env `habitat` (see `environment.yml`) at `/home/thisen-ekanayake/miniforge3/envs/habitat`. The `conda` shell wrapper is broken on this machine — call env binaries directly, e.g. `/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python`.

Install the local package in editable mode once source code lands under `src/foresight/`:

```bash
/home/thisen-ekanayake/miniforge3/envs/habitat/bin/pip install -e .
```
