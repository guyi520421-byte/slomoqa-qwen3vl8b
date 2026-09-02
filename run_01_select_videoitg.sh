#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/pipeline_env.sh"
configure_split
cd "$ROOT"
exec "$VIDEOITG_PYTHON" "$ROOT/feature_pipeline.py" \
  --stage select --split "$SPLIT" --gpu-ids "$GPU_ID" \
  --questions "$QUESTIONS" --video-dir "$VIDEO_DIR" \
  --selector-path "$VIDEOITG_MODEL" --selections "$SELECTIONS" \
  --candidate-frames 512 --min-candidate-frames 128 --target-fps 2.0 \
  --dino-pool-size 64 --topk 32 --min-answer-frames 12 --resume "$@"
