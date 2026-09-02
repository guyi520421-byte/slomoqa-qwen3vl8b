#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/pipeline_env.sh"
configure_split
OUTPUT=${OUTPUT:-$ROOT/outputs/qwen3vl8b_lora_${SPLIT}.json}
CUSTOM=()
if [[ "$SPLIT" == public ]]; then CUSTOM+=(--allow-custom-public-inputs); fi
cd "$ROOT"
exec "$QWEN_PYTHON" "$ROOT/infer_public.py" \
  --split "$SPLIT" --gpu-ids "$GPU_ID" --qwen-path "$QWEN3_MODEL" \
  --lora-path "$ADAPTER_PATH" \
  --questions "$QUESTIONS" --video-dir "$VIDEO_DIR" \
  --selections "$SELECTIONS" --evidence "$CLEAN_EVIDENCE" \
  --output "$OUTPUT" --topk 32 --min-answer-frames 12 \
  --decode-max-side 1024 --min-visual-tokens 512 --max-visual-tokens 512 \
  --max-text-evidence-items 64 --max-evidence-chars 20000 \
  --evidence-temporal-window-seconds 20 --quantization 4bit \
  --dtype bf16 --attention sdpa --max-new-tokens 128 --resume \
  "${CUSTOM[@]}" "$@"
