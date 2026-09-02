#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/pipeline_env.sh"
configure_split
cd "$ROOT"
exec "$QWEN_PYTHON" "$ROOT/clean_ocr.py" \
  --stage clean --split "$SPLIT" \
  --questions "$QUESTIONS" --video-dir "$VIDEO_DIR" \
  --source-evidence "$RAW_EVIDENCE" --selections "$SELECTIONS" \
  --cleaned-evidence "$CLEAN_EVIDENCE" --resume "$@"
