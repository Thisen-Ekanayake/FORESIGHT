#!/usr/bin/env bash
# Deliverable A: the depth-error-propagation sweep (PLAN.md Phase 3). Resumable: finished runs are skipped.
# Non-VLM runs, three workers in parallel on one 8 GB GPU (~1.5 GB each). Output: results/runs/pointnav/A, .../S.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python}
export DISPLAY=${DISPLAY:-:1}
N=${N:-3}
LOG=results/runs/pointnav/logs; mkdir -p "$LOG"

UNIFORM=(gt da
  "gt:scale=0.5" "gt:scale=0.7" "gt:scale=1.4" "gt:scale=2.0" "gt:scale=2.5"
  "gt:bias_m=-0.3" "gt:bias_m=0.3" "gt:bias_m=0.6"
  "gt:noise=0.1" "gt:noise=0.2" "gt:noise=0.4" "gt:noise=0.1,noise_px=16" "gt:noise=0.2,noise_px=16"
  "gt:holes=0.1" "gt:holes=0.3" "gt:holes=0.5"
  "gt:quant_m=0.1" "gt:quant_m=0.25" "gt:quant_m=0.5")
SCALE=(gt da "gt:scale=0.5" "gt:scale=0.7" "gt:scale=1.4" "gt:scale=2.0" "gt:scale=2.5")
KEY=(gt da "gt:scale=0.7" "gt:scale=1.4" "gt:scale=2.0" "gt:noise=0.2" "gt:holes=0.3" "gt:quant_m=0.25")
STRUCT=(gt da "gt:reflect=1.5" "gt:reflect=2.0" "gt:reflect=3.0" "gt:flat_tex=1"
  "mix:gt@reflective=da" "mix:da@reflective=gt" "mix:gt@textureless=da" "mix:da@textureless=gt")

stage() {  # name, then run_pointnav_eval args
  local name=$1; shift
  echo "== $name $(date +%T)"
  for i in $(seq 0 $((N - 1))); do
    $PY tools/run_pointnav_eval.py "$@" --shard "$i" --nshards "$N" > "$LOG/$name.$i.log" 2>&1 &
  done
  wait
  grep -h "FAILED" "$LOG/$name".*.log | head
}

stage map_uniform   --set A --conditions map --depth "${UNIFORM[@]}" --out results/runs/pointnav/A
stage map_nofs      --set A --conditions map --no-floor-scale --depth "${SCALE[@]}" --out results/runs/pointnav/A
stage heur_key      --set A --conditions heuristic --depth "${KEY[@]}" --out results/runs/pointnav/A
stage struct        --set S --conditions map heuristic --depth "${STRUCT[@]}" --out results/runs/pointnav/S
echo "== done $(date +%T)"
