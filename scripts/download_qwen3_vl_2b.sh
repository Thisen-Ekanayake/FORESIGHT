#!/usr/bin/env bash
# Download Qwen3-VL-2B-Instruct (~4.3GB) into models/. Resumable: safe to re-run
# after an interruption, curl continues each file from where it left off.
set -euo pipefail

REPO="Qwen/Qwen3-VL-2B-Instruct"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${PROJECT_ROOT}/models/Qwen3-VL-2B-Instruct"

FILES=(
  config.json
  generation_config.json
  chat_template.json
  preprocessor_config.json
  video_preprocessor_config.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
  model.safetensors
)

# Expected sizes in bytes, from the HF API, used to confirm each file landed complete.
declare -A EXPECTED_SIZES=(
  [config.json]=1505
  [generation_config.json]=269
  [chat_template.json]=5502
  [preprocessor_config.json]=390
  [video_preprocessor_config.json]=385
  [tokenizer.json]=7032403
  [tokenizer_config.json]=10868
  [vocab.json]=2776833
  [merges.txt]=1671839
  [model.safetensors]=4255140312
)

mkdir -p "$DEST"

AUTH_HEADER=()
if [ -n "${HF_TOKEN:-}" ]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${HF_TOKEN}")
else
  echo "note: no HF_TOKEN set, downloading unauthenticated (fine, this repo is public, but rate limits are lower)"
fi

echo "Downloading ${REPO} to ${DEST} (about 4.3GB total)"

for f in "${FILES[@]}"; do
  url="https://huggingface.co/${REPO}/resolve/main/${f}"
  echo "==> ${f}"
  curl -L -C - --progress-bar "${AUTH_HEADER[@]}" -o "${DEST}/${f}" "$url"

  actual_size=$(stat -c%s "${DEST}/${f}")
  expected_size="${EXPECTED_SIZES[$f]}"
  if [ "$actual_size" -ne "$expected_size" ]; then
    echo "warning: ${f} is ${actual_size} bytes, expected ${expected_size}. Re-run this script to resume/retry." >&2
  fi
done

echo "Done. Model files in: ${DEST}"
