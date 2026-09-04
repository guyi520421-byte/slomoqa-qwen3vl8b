#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/pipeline_env.sh"
configure_split
# These defaults reproduce the OCR-generation profile behind the packaged
# cleaned-evidence cache used by qwen3vl8b_base512_public3.json.  They are
# intentionally separate from the final answer model's 1024/512-512 settings.
OCR_DECODE_MAX_SIDE=${OCR_DECODE_MAX_SIDE:-740}
OCR_MIN_VISUAL_TOKENS=${OCR_MIN_VISUAL_TOKENS:-128}
OCR_MAX_VISUAL_TOKENS=${OCR_MAX_VISUAL_TOKENS:-256}
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
