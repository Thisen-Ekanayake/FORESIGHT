#!/usr/bin/env bash
# Mirror-guard precision/recall on the dev sets (VLM, alone on the GPU). $1 = output dir, then extra runner args.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python}
export DISPLAY=${DISPLAY:-:1}
OUT=$1; shift
LOG=results/runs/pointnav/logs; mkdir -p "$LOG"
echo "== start $(date +%T)"
$PY tools/run_pointnav_eval.py --set S_dev --limit 4 --conditions foresight --depth da --out "$OUT" "$@" > "$LOG/$(basename $OUT)_S.log" 2>&1
$PY tools/run_pointnav_eval.py --set E_dev --conditions foresight --depth da --out "$OUT" "$@" > "$LOG/$(basename $OUT)_E.log" 2>&1
$PY tools/run_pointnav_eval.py --set E_dev --conditions map --depth da --out results/runs/pointnav/dev_guard_map > "$LOG/dev_guard_map_E.log" 2>&1
echo "== done $(date +%T)"
