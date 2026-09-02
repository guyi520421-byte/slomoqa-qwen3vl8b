#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/pipeline_env.sh"
configure_split
cd "$ROOT"
exec "$QWEN_PYTHON" "$ROOT/experiments/qwen3_ocr_experiment.py" \
  --split "$SPLIT" --gpu-ids "$GPU_ID" --qwen-path "$QWEN3_MODEL" \
  --ocr-decode-max-side 1024 --ocr-min-visual-tokens 256 \
  --ocr-max-visual-tokens 512 --ocr-batch-size 2 \
  --answer-decode-max-side 1024 --answer-min-visual-tokens 512 \
  --answer-max-visual-tokens 512 --max-text-evidence-items 64 \
  --max-evidence-chars 20000 --evidence-temporal-window-seconds 20 \
  --quantization 4bit --dtype bf16 --attention sdpa --resume "$@"
