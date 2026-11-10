#!/usr/bin/env bash
# FORESIGHT v2.2 mirror guard on the dev sets (VLM: one worker, alone on the GPU): the 30-episode point-goal pilot set
# (vs map:da and the v2.1 pilot) and 16 annotated-home episodes for wall precision. Waits for the fix sweep first.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python}
export DISPLAY=${DISPLAY:-:1}
LOG=results/runs/pointnav/logs; mkdir -p "$LOG"
while ps -eo cmd | grep -q '[r]un_fix_sweep.sh'; do sleep 60; done
echo "== b_dev $(date +%T)"
$PY tools/run_pointnav_eval.py --set A_dev --limit 3 --conditions foresight --depth da --out results/runs/pointnav/dev_b > "$LOG/b_dev_A.log" 2>&1
$PY tools/run_pointnav_eval.py --set S_dev --limit 4 --conditions map foresight --depth da --out results/runs/pointnav/dev_b > "$LOG/b_dev_S.log" 2>&1
echo "== done $(date +%T)"
