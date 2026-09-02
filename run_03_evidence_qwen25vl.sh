#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/pipeline_env.sh"
configure_split
OCR_DECODE_MAX_SIDE=${OCR_DECODE_MAX_SIDE:-1024}
OCR_MIN_VISUAL_TOKENS=${OCR_MIN_VISUAL_TOKENS:-256}
OCR_MAX_VISUAL_TOKENS=${OCR_MAX_VISUAL_TOKENS:-512}
cd "$ROOT"
exec "$QWEN_PYTHON" "$ROOT/feature_pipeline.py" \
  --stage evidence --split "$SPLIT" --gpu-ids "$GPU_ID" \
  --questions "$QUESTIONS" --video-dir "$VIDEO_DIR" \
  --selections "$SELECTIONS" --evidence "$RAW_EVIDENCE" \
  --dino-path "$DINOV2_MODEL" --dino-pool-size 64 --topk 32 \
  --asr-model-path "$ASR_MODEL" --asr-device cuda:0 --asr-dtype fp16 \
  --asr-language en --asr-task transcribe --asr-chunk-seconds 30 \
  --ocr --ocr-model-path "$QWEN25_OCR_MODEL" --ocr-batch-size 3 \
  --ocr-max-frames-per-video 96 --ocr-max-new-tokens 512 \
  --decode-max-side "$OCR_DECODE_MAX_SIDE" \
  --min-visual-tokens "$OCR_MIN_VISUAL_TOKENS" \
  --max-visual-tokens "$OCR_MAX_VISUAL_TOKENS" \
  --quantization 4bit --dtype bf16 --attention sdpa --resume "$@"
