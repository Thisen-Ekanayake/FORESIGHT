#!/usr/bin/env bash
# FORESIGHT promo, pass 2: path-traced re-render of the live runs (tools/isaac/replay_render.py, SONIC's G1 look).
#
#   python tools/isaac/promo/make_shots.py          # shots.json + shots_stills.json from the live runs
#   tools/isaac/promo/render.sh stills              # one still per shot at 32 and 64 spp -> results/isaac/promo/stills{32,64}
#   tools/isaac/promo/render.sh full [SPP]          # every shot (default 64 spp) -> results/isaac/promo/film/<shot>/
#   tools/isaac/promo/render.sh full 32 depth_crane # just these shots
# Full renders launch Isaac once per shot and skip shots already rendered (frames.json present): stop it any time and
# start it again. --deterministic is on by default in replay_render.py (no warm-up flicker); --iso 14 = the fixed
# exposure of the office test clip, to re-check on the home stills.
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=results/isaac/promo
RR="tools/isaac/isaaclab_run.sh tools/isaac/replay_render.py --scene hm3d --robot sonic --res 1440 --fps 30 --iso ${ISO:-20}"
mode=${1:-stills}

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$used" -gt 1500 ]; then
  echo "GPU already has ${used} MiB in use (Athena? the brain?). Free it first." >&2
  exit 1
fi

if [ "$mode" = stills ]; then
  for spp in 32 64; do
    rm -rf "$P/stills$spp"
    $RR --spp "$spp" --still --shots "$P/shots_stills.json" --out "$P/stills$spp" > "$P/stills$spp.log" 2>&1
  done
  # side by side, 32 left / 64 right, for the look check
  mkdir -p "$P/stills_compare"
  for d in "$P"/stills64/*/; do
    n=$(basename "$d")
    ffmpeg -y -v error -i "$P/stills32/$n/00000.jpg" -i "$P/stills64/$n/00000.jpg" -filter_complex hstack \
      "$P/stills_compare/$n.jpg"
  done
  echo "stills: $P/stills_compare/ (left 32 spp, right 64 spp)"
  exit 0
fi

spp=${2:-64}
shift $(( $# >= 2 ? 2 : $# ))
names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  mapfile -t names < <(python3 -c "import json; [print(s['name']) for s in json.load(open('$P/shots.json'))]")
fi
mkdir -p "$P/film"
for n in "${names[@]}"; do
  if [ -f "$P/film/$n/frames.json" ]; then echo "skip $n (rendered)"; continue; fi
  echo "=== $n  $(date +%H:%M)"
  $RR --spp "$spp" --shots "$P/shots.json" --only "$n" --out "$P/film" > "$P/film/$n.log" 2>&1 \
    || { echo "  $n failed, see $P/film/$n.log"; continue; }
  tail -1 "$P/film/$n.log"
done
echo "film: $P/film/  next: python tools/make_promo_video.py --preview 10"
