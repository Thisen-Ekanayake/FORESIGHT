#!/usr/bin/env bash
# Run the PointGoal RL pipeline in order: episodes -> train -> eval -> demo.
#
# Every value (interpreter, display, stage order, per-stage script + args, skip rules, log dir) comes from
# experiments/configs/rl/pipeline.yaml — the only literal here is that config's path. Each stage's own
# settings live in its script config; see docs/rl/pointnav.md.
#
#   scripts/run_rl_pipeline.sh                  # whole pipeline
#   scripts/run_rl_pipeline.sh --stages train,demo
#   scripts/run_rl_pipeline.sh --dry-run        # print the commands without running them
#   scripts/run_rl_pipeline.sh --force          # run skippable stages (episode generation) anyway
#
# Training is resumable: re-running continues from the latest checkpoint, so re-running the pipeline
# after an interruption is safe.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${PROJECT_ROOT}/experiments/configs/rl/pipeline.yaml"

DRY_RUN=0
FORCE=0
STAGES_ARG=""

usage() {
  sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
  exit "${1:-0}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --stages) STAGES_ARG="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    --list) STAGES_ARG=""; LIST=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done

[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }

# Bootstrap: read the interpreter path out of the config with awk, because reading the rest of the
# config needs a python with PyYAML — which is the very interpreter named there. $FORESIGHT_PYTHON wins.
PY="${FORESIGHT_PYTHON:-$(awk '$1=="python:"{print $2; exit}' "$CONFIG")}"
[ -x "$PY" ] || { echo "python interpreter not executable: '$PY' (set FORESIGHT_PYTHON to override)" >&2; exit 1; }

# cfg <dotted.key> -> scalar on one line, list one item per line, null/missing-optional as nothing.
cfg() {
  "$PY" - "$CONFIG" "$1" <<'PYEOF'
import sys
import yaml

path, key = sys.argv[1], sys.argv[2]
with open(path) as f:
    node = yaml.safe_load(f)
for part in key.split("."):
    if not isinstance(node, dict) or part not in node:
        sys.exit(f"run_rl_pipeline: missing key '{key}' in {path}")
    node = node[part]
if isinstance(node, list):
    for item in node:  # one per line, nothing at all for an empty list (-> an empty bash array)
        print(item)
elif isinstance(node, bool):
    print(str(node).lower())
elif node is not None:
    print(node)
PYEOF
}

cd "$PROJECT_ROOT"

DISPLAY_CFG="$(cfg display)"
LOG_DIR="$(cfg log_dir)"
mapfile -t ALL_STAGES < <(cfg stages)
[ "${#ALL_STAGES[@]}" -gt 0 ] || { echo "no stages listed in $CONFIG" >&2; exit 1; }

if [ -n "${LIST:-}" ]; then
  echo "stages in $CONFIG:"
  for stage in "${ALL_STAGES[@]}"; do
    printf '  %-10s %s\n' "$stage" "$(cfg "$stage.description")"
  done
  exit 0
fi

# Stage selection: --stages keeps the config's order, and unknown names fail before anything runs.
if [ -n "$STAGES_ARG" ]; then
  IFS=',' read -r -a REQUESTED <<< "$STAGES_ARG"
  for want in "${REQUESTED[@]}"; do
    found=0
    for stage in "${ALL_STAGES[@]}"; do [ "$stage" = "$want" ] && found=1; done
    [ "$found" -eq 1 ] || { echo "unknown stage: '$want' (have: ${ALL_STAGES[*]})" >&2; exit 1; }
  done
  STAGES=()
  for stage in "${ALL_STAGES[@]}"; do
    for want in "${REQUESTED[@]}"; do [ "$stage" = "$want" ] && STAGES+=("$stage"); done
  done
else
  STAGES=("${ALL_STAGES[@]}")
fi

mkdir -p "$LOG_DIR"
echo "RL pipeline: ${STAGES[*]}   (config: ${CONFIG#$PROJECT_ROOT/}, logs: $LOG_DIR)"
SUMMARY=()

for stage in "${STAGES[@]}"; do
  script="$(cfg "$stage.script")"
  description="$(cfg "$stage.description")"
  skip_if_exists="$(cfg "$stage.skip_if_exists")"
  mapfile -t args < <(cfg "$stage.args")
  log_file="$LOG_DIR/$stage.log"

  echo
  echo "=== [$stage] $description"

  if [ -n "$skip_if_exists" ] && [ -e "$skip_if_exists" ] && [ "$FORCE" -eq 0 ]; then
    echo "    skipped — $skip_if_exists already exists (--force to run anyway)"
    SUMMARY+=("$stage: skipped")
    continue
  fi

  printf '    DISPLAY=%s %q %s' "$DISPLAY_CFG" "$PY" "$script"
  [ "${#args[@]}" -gt 0 ] && printf ' %s' "${args[@]}"
  printf '\n'

  if [ "$DRY_RUN" -eq 1 ]; then
    SUMMARY+=("$stage: dry-run")
    continue
  fi

  echo "    log: $log_file"
  started=$SECONDS
  set +e
  DISPLAY="$DISPLAY_CFG" "$PY" "$script" "${args[@]}" 2>&1 | tee "$log_file"
  rc=${PIPESTATUS[0]}
  set -e
  elapsed=$((SECONDS - started))

  if [ "$rc" -ne 0 ]; then
    SUMMARY+=("$stage: FAILED (exit $rc, ${elapsed}s)")
    echo
    echo "[$stage] failed with exit $rc after ${elapsed}s — see $log_file" >&2
    printf '%s\n' "${SUMMARY[@]}" >&2
    exit "$rc"
  fi
  SUMMARY+=("$stage: ok (${elapsed}s)")
done

echo
echo "=== summary"
printf '  %s\n' "${SUMMARY[@]}"
