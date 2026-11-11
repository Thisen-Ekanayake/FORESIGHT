#!/usr/bin/env bash
# v2.3 mirror guard on the same dev episodes as v2.2 (annotated homes, dev encounters) plus the 30-episode A_dev pilot.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python}
export DISPLAY=${DISPLAY:-:1}
OUT=results/runs/pointnav/dev_guard_v23; LOG=results/runs/pointnav/logs
echo "== start $(date +%T)"
$PY tools/run_pointnav_eval.py --set S_dev --limit 4 --conditions foresight --guard v23 --depth da --out $OUT > $LOG/dev_guard_v23_S.log 2>&1
$PY tools/run_pointnav_eval.py --set E_dev --conditions foresight --guard v23 --depth da --out $OUT > $LOG/dev_guard_v23_E.log 2>&1
$PY tools/run_pointnav_eval.py --set A_dev --limit 3 --conditions foresight --guard v23 --depth da --out $OUT > $LOG/dev_guard_v23_A.log 2>&1
echo "== done $(date +%T)"
