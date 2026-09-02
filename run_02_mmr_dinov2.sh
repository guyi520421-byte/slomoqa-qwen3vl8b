#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/pipeline_env.sh"
configure_split
cd "$ROOT"
exec "$VIDEOITG_PYTHON" "$ROOT/feature_pipeline.py" \
  --stage mmr --split "$SPLIT" --gpu-ids "$GPU_ID" \
  --questions "$QUESTIONS" --video-dir "$VIDEO_DIR" \
  --dino-path "$DINOV2_MODEL" --selections "$SELECTIONS" \
  --candidate-frames 512 --min-candidate-frames 128 --target-fps 2.0 \
  --dino-pool-size 64 --topk 32 --min-answer-frames 12 \
  --dino-batch-size 8 --dino-dtype fp16 \
  --mmr-relevance-weight 1.0 --mmr-visual-weight 0.30 \
  --mmr-temporal-weight 0.15 --mmr-global-weight 0.60 \
  --mmr-patch-weight 0.40 --mmr-temporal-tau-seconds 3.0 \
  --mmr-patch-grid-size 7 --resume "$@"
