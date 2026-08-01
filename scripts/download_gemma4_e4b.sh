#!/usr/bin/env bash
# Download the Gemma 4 E4B checkpoint (QAT, mobile/transformers-compatible, ~3.5GB) into models/.
# Resumable: safe to re-run after an interruption, curl continues each file from where it left off.
set -euo pipefail

REPO="google/gemma-4-E4B-it-qat-mobile-transformers"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${PROJECT_ROOT}/models/gemma-4-E4B-it-qat-mobile-transformers"

FILES=(
  config.json
  generation_config.json
  chat_template.jinja
  preprocessor_config.json
  processor_config.json
  tokenizer.json
  tokenizer_config.json
  model.safetensors
)

# Expected sizes in bytes, from the HF API, used to confirm each file landed complete.
declare -A EXPECTED_SIZES=(
  [config.json]=6305
  [generation_config.json]=209
  [chat_template.jinja]=18569
  [preprocessor_config.json]=511
  [processor_config.json]=1689
  [tokenizer.json]=32169626
  [tokenizer_config.json]=3082
  [model.safetensors]=3525094516
)

mkdir -p "$DEST"

AUTH_HEADER=()
if [ -n "${HF_TOKEN:-}" ]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${HF_TOKEN}")
else
  echo "note: no HF_TOKEN set, downloading unauthenticated (fine, this repo is public, but rate limits are lower)"
fi

echo "Downloading ${REPO} to ${DEST} (about 3.5GB total)"

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
