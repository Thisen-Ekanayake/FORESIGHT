#!/usr/bin/env bash
# Deliverable B evaluation: FORESIGHT v2.3 (wedge-only mirror guard) with Depth Anything V2 on the evaluation sets,
# paired with the map / heuristic cells already in results/runs/pointnav/{E,S,A}. VLM: one worker, alone on the GPU
# (stop Athena's llama-server first). Order E -> S -> A: the mirror encounters matter most if time runs out.
# Resumable: finished episodes are skipped.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python}
export DISPLAY=${DISPLAY:-:1}
LOG=results/runs/pointnav/logs; mkdir -p "$LOG"
echo "== start $(date +%T)"
for S in E S A; do
  echo "== set $S $(date +%T)"
  $PY tools/run_pointnav_eval.py --set $S --conditions foresight --guard v23 --depth da --out results/runs/pointnav/$S > "$LOG/b_eval_$S.log" 2>&1
done
echo "== done $(date +%T)"
