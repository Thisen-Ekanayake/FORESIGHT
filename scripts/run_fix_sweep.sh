#!/usr/bin/env bash
# 2026-11-10 follow-ups to deliverable A (non-VLM, three workers): the robust-scan ablation on set A, the set-S
# reflective cells re-run with the fixed reflective categories, and the non-VLM conditions on the encounter set E.
# Resumable: finished runs are skipped.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python}
export DISPLAY=${DISPLAY:-:1}
N=${N:-3}
LOG=results/runs/pointnav/logs; mkdir -p "$LOG"
ROBUST=(gt da "gt:noise=0.1" "gt:noise=0.2" "gt:noise=0.4" "gt:noise=0.1,noise_px=16" "gt:noise=0.2,noise_px=16"
  "gt:bias_m=-0.3" "gt:bias_m=0.3" "gt:scale=2.5")
REFLECT=("gt:reflect=1.5" "gt:reflect=2.0" "gt:reflect=3.0" "mix:gt@reflective=da" "mix:da@reflective=gt")
ENC=(gt da "gt:reflect=2.0" "gt:reflect=3.0" "mix:gt@reflective=da" "mix:da@reflective=gt")

stage() {
  local name=$1; shift
  echo "== $name $(date +%T)"
  for i in $(seq 0 $((N - 1))); do
    $PY tools/run_pointnav_eval.py "$@" --shard "$i" --nshards "$N" > "$LOG/$name.$i.log" 2>&1 &
  done
  wait
  grep -h "FAILED" "$LOG/$name".*.log | head
}

stage robust_A    --set A --conditions map --robust-scan --depth "${ROBUST[@]}" --out results/runs/pointnav/A
stage reflect_S   --set S --conditions map heuristic --depth "${REFLECT[@]}" --out results/runs/pointnav/S
stage enc_map     --set E --conditions map --depth "${ENC[@]}" --out results/runs/pointnav/E
stage enc_robust  --set E --conditions map --robust-scan --depth gt da --out results/runs/pointnav/E
stage enc_heur    --set E --conditions heuristic --depth gt da "gt:reflect=3.0" --out results/runs/pointnav/E
echo "== done $(date +%T)"
