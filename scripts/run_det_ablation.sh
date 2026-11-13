#!/usr/bin/env bash
# Detector-quality ablation of the mirror guard (tools/run_detector_ablation.py), queued after the B evaluation.
# Phase 1 (development, E_dev): oracle detector with the v2.3 vs v2.4 guard.
# Phase 2 (reporting, E): v2.3 guard with oracle / lower-recall / false-positive detectors.
# Each level is one process (no VLM: renderer + Depth Anything Small, ~1 GB); three run at a time. Resumable.
set -u
cd "$(dirname "$0")/.."
PY=/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python
export DISPLAY=:1 PYTHONPATH=src
LOG=results/runs/pointnav/logs; mkdir -p "$LOG"
while pgrep -f run_b_eval.sh >/dev/null; do sleep 60; done
echo "== start $(date +%T)"

run() {  # set recall fp guard out
  $PY tools/run_detector_ablation.py --set "$1" --recall "$2" --fp "$3" --guard "$4" --out "$5" \
    > "$LOG/det_$1_r$2_fp$3_$4.log" 2>&1
}
run E_dev 1 0 v23 results/runs/pointnav/det/E_dev &
run E_dev 1 0 v24 results/runs/pointnav/det/E_dev &
run E 1 0 v23 results/runs/pointnav/det/E &
wait; echo "== phase 1 + oracle E $(date +%T)"
run E 1 0.3 v23 results/runs/pointnav/det/E &
run E 0.5 0 v23 results/runs/pointnav/det/E &
run E 1 0.1 v23 results/runs/pointnav/det/E &
wait
echo "== done $(date +%T)"
