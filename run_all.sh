#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/pipeline_env.sh"

SPLIT=${SPLIT:-public}
GPU_ID=${GPU_ID:-0}
RUN_DIR=${RUN_DIR:-$ROOT/runs/${SPLIT}_$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$RUN_DIR"
RUN_DIR=$(cd -- "$RUN_DIR" && pwd)

QUESTIONS=${QUESTIONS:-$DATA_ROOT/sf20k_${SPLIT}_test_questions.csv}
VIDEO_DIR=${VIDEO_DIR:-$DATA_ROOT/sf20k_${SPLIT}_test_videos}
SELECTIONS=${SELECTIONS:-$RUN_DIR/${SPLIT}_selections.jsonl}
RAW_EVIDENCE=${RAW_EVIDENCE:-$RUN_DIR/${SPLIT}_evidence_raw.jsonl}
CLEAN_EVIDENCE=${CLEAN_EVIDENCE:-$RUN_DIR/${SPLIT}_evidence_ocr_clean.jsonl}
OUTPUT=${OUTPUT:-$RUN_DIR/qwen3vl8b_base_${SPLIT}.json}
ALLOW_CUSTOM_PUBLIC_INPUTS=1
export SPLIT GPU_ID QUESTIONS VIDEO_DIR SELECTIONS RAW_EVIDENCE CLEAN_EVIDENCE OUTPUT
export ALLOW_CUSTOM_PUBLIC_INPUTS

configure_split
echo "Full pipeline run"
echo "  split=$SPLIT gpu=$GPU_ID"
echo "  run_dir=$RUN_DIR"
echo "  OCR=${OCR_DECODE_MAX_SIDE:-740}px/${OCR_MIN_VISUAL_TOKENS:-128}-${OCR_MAX_VISUAL_TOKENS:-256} visual tokens"
echo "  answer=1024px/512 visual tokens"

bash "$ROOT/run_01_select_videoitg.sh"
bash "$ROOT/run_02_mmr_dinov2.sh"
"$QWEN_PYTHON" "$ROOT/validate_artifacts.py" \
  --split "$SPLIT" --questions "$QUESTIONS" --selections "$SELECTIONS"

bash "$ROOT/run_03_evidence_qwen25vl.sh"
"$QWEN_PYTHON" "$ROOT/validate_artifacts.py" \
  --split "$SPLIT" --questions "$QUESTIONS" --evidence "$RAW_EVIDENCE"

bash "$ROOT/run_04_clean_ocr.sh"
"$QWEN_PYTHON" "$ROOT/validate_artifacts.py" \
  --split "$SPLIT" --questions "$QUESTIONS" --evidence "$CLEAN_EVIDENCE"

bash "$ROOT/run_05_infer_base.sh"
"$QWEN_PYTHON" "$ROOT/validate_artifacts.py" \
  --split "$SPLIT" --questions "$QUESTIONS" \
  --selections "$SELECTIONS" --evidence "$CLEAN_EVIDENCE" \
  --predictions "$OUTPUT"
echo "Full run complete: $RUN_DIR"
