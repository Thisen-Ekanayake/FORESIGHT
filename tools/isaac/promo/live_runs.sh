#!/usr/bin/env bash
# FORESIGHT promo ("One Camera, No Map"), pass 1: every live G1 run the video needs, walked by NVIDIA GEAR-SONIC.
# Plan: https://claude.ai/artifact/LZFmCoQ5rUzfAYmhpG3ahY (draft 2). Resumable: a run whose motion.json exists is
# skipped (delete its folder to redo it). Output: results/isaac/promo/live/<run>/{motion.json,brain.pkl,world.log}.
#
#   athena --stop                                  # free the GPU first (Athena's llama-server holds 5.5 GB)
#   tools/isaac/promo/live_runs.sh                 # all stages in order
#   tools/isaac/promo/live_runs.sh probe           # one stage: probe | mirror | depth | depth_alt | search
#   then: python tools/isaac/promo/check_runs.py   # did each run do what the plan needs?
#
# Stages:
#   probe      no brain: SONIC walks a scripted turn-on-the-spot + walk profile in the scanned home (gait, turn rate,
#              speed gain check; ~2 min). Look at check_runs.py's numbers before the long runs.
#   mirror     the Through the Glass pair re-run with SONIC: depth-only map navigator, FORESIGHT v1 (hook + mirror)
#   depth      the A study's episode a-00808-04 four times: ground truth, Depth Anything V2, GT + 10% noise,
#              Depth Anything without floor calibration (the "same robot, different depth" centrepiece; the DA run
#              also serves the "one camera, two speeds" segment)
#   depth_alt  fallback episodes (a-00808-07, a-00808-03), only if a-00808-04 doesn't show the study's pattern
#   search     "find the sofa, then the bed" (FORESIGHT v1 object search, the mirror-sofa-bed episode)
set -euo pipefail
cd "$(dirname "$0")/../../.."
ROOT=$PWD
OUT=$ROOT/results/isaac/promo/live
HAB=/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python
WORLD="tools/isaac/isaaclab_run.sh tools/isaac/g1_live.py"
# SONIC's ONNX planner on the CPU: the brain's Qwen3-VL + Depth Anything need the GPU, and the world is paused while
# the brain thinks, so a slower planner only costs wall time, never behaviour. --sonic-cpu off = faster if it fits.
COMMON=(--scene hm3d --perception habitat --walker sonic --sonic-cpu --cam-height 1.22)
mkdir -p "$OUT"

check_gpu() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -gt 1500 ]; then
    echo "GPU already has ${used} MiB in use. Stop Athena (athena --stop) and other GPU jobs first." >&2
    exit 1
  fi
}

BRAIN_PID=""
start_brain() {
  [ -n "$BRAIN_PID" ] && return
  echo "starting FORESIGHT brain (habitat env)"
  "$HAB" tools/isaac/foresight_brain.py > "$OUT/brain.log" 2>&1 &
  BRAIN_PID=$!
  for _ in $(seq 1 180); do
    grep -q "brain ready" "$OUT/brain.log" 2>/dev/null && { echo "brain ready"; return; }
    kill -0 "$BRAIN_PID" 2>/dev/null || { echo "brain died, see $OUT/brain.log" >&2; exit 1; }
    sleep 2
  done
  echo "brain did not start in 6 min, see $OUT/brain.log" >&2
  exit 1
}
stop_brain() { [ -n "$BRAIN_PID" ] && kill "$BRAIN_PID" 2>/dev/null || true; }
trap stop_brain EXIT

run() {  # run <name> <args...>: one live run, skipped if done
  local name=$1; shift
  local d=$OUT/$name
  if [ -f "$d/motion.json" ]; then echo "skip $name (done)"; return; fi
  mkdir -p "$d"
  echo "=== $name  $(date +%H:%M)"
  $WORLD "${COMMON[@]}" --out "$d" "$@" > "$d/world.log" 2>&1 || echo "  $name exited with an error, see $d/world.log"
  grep -E "^hm3d|FELL|fell|profile done" "$d/world.log" | tail -2 || true
}

# the hallway-mirror episode (Through the Glass, 2026-11-20): Isaac xy, yaw 0 = +x
MIRROR=(--start -2.59 -0.29 0 --goal 2.4 -1.1)
# A study episode a-00808-04 (experiments/configs/pointnav/episodes_A_3003.json), Habitat -> Isaac (x, -z),
# yaw = Habitat yaw + 90; the A study's settings: 0.35 m/s, success 0.36 m, budget 130 s, noise seeded by episode
EP04=(--start -4.925 -2.226 -15.6 --goal 5.231 -4.195 --max-linear 0.35 --success-m 0.36 --max-sim-s 130 --depth-seed 895999734)
EP07=(--start -0.245 -3.765 -34.6 --goal 6.118 -4.240 --max-linear 0.35 --success-m 0.36 --max-sim-s 94.4 --depth-seed 639133851)
EP03=(--start -1.678 0.586 -65.0 --goal -10.832 -3.197 --max-linear 0.35 --success-m 0.36 --max-sim-s 126.5 --depth-seed 1099685985)

stage_probe() {
  # stand 2 s, turn on the spot left then right at 0.6 rad/s (FORESIGHT turns at up to 0.70), walk 0.35 m/s, curve, stop
  run probe_sonic --profile '[[2,0,0],[3,0,0.6],[3,0,-0.6],[4,0.35,0],[3,0.35,0.3],[2,0,0]]' --start -2.59 -0.29 0
}
stage_mirror() {
  start_brain
  run mirror_map --cond map --max-linear 0.45 --max-sim-s 90 "${MIRROR[@]}"
  run mirror_v1 --cond foresight_v1 --max-linear 0.35 --max-sim-s 150 "${MIRROR[@]}"
}
stage_depth() {
  start_brain
  run depth_gt --cond map --depth gt "${EP04[@]}"
  run depth_da --cond map --depth da "${EP04[@]}"
  run depth_noise --cond map --depth gt:noise=0.1 "${EP04[@]}"
  run depth_nofs --cond map --depth da --no-floor-scale "${EP04[@]}"
}
stage_depth_alt() {
  start_brain
  for ep in 07 03; do
    local -n E="EP$ep"
    run "depth${ep}_gt" --cond map --depth gt "${E[@]}"
    run "depth${ep}_da" --cond map --depth da "${E[@]}"
    run "depth${ep}_noise" --cond map --depth gt:noise=0.1 "${E[@]}"
    run "depth${ep}_nofs" --cond map --depth da --no-floor-scale "${E[@]}"
  done
}
stage_search() {
  start_brain
  # the object search episode "mirror-sofa-bed" (tools/run_object_search.py): Habitat yaw -89.1 -> Isaac 0.9
  run search_v1 --task search --instructions sofa bed --cond foresight_v1 --max-linear 0.35 --max-sim-s 300 \
      --start -2.592 -0.290 0.9
}

check_gpu
stages=("${@:-probe mirror depth search}")
for s in ${stages[@]}; do
  "stage_$s"
done
echo "done. Next: $HAB tools/isaac/promo/check_runs.py"
