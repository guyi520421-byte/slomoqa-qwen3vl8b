#!/usr/bin/env bash
# Shared paths for the independent SloMoQA pipeline. Override any variable in
# the shell before launching a stage; model weights and raw videos are external.
PIPELINE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VIDEOITG_PYTHON=${VIDEOITG_PYTHON:-/data/miniconda3/envs/videoitg/bin/python}
QWEN_PYTHON=${QWEN_PYTHON:-/data/miniconda3/envs/egoVqa/bin/python}
DATA_ROOT=${DATA_ROOT:-/data/wen1/datasets/slomo_qa}
VIDEOITG_MODEL=${VIDEOITG_MODEL:-/data/wen1/MLLMs/VideoITG-8B}
DINOV2_MODEL=${DINOV2_MODEL:-/data/wen1/MLLMs/dinov2_vitb14_reg4_pretrain.pth}
ASR_MODEL=${ASR_MODEL:-/data/wen1/MLLMs/whisper-large-v3-turbo}
QWEN25_OCR_MODEL=${QWEN25_OCR_MODEL:-/data/wen1/MLLMs/Qwen_25vl_7b}
QWEN3_MODEL=${QWEN3_MODEL:-/data/wen1/MLLMs/qwen3vl8b}
HISTORICAL_ADAPTER=${HISTORICAL_ADAPTER:-/data/wen1/slomoQa/lora_train_frames_legacy_exact_qwen3vl8b/outputs/qwen3vl8b_frame_legacy_exact_lora}
LOCAL_ADAPTER=${LOCAL_ADAPTER:-$PIPELINE_ROOT/outputs/qwen3vl8b_frame_legacy_exact_lora}
if [[ -f "$LOCAL_ADAPTER/adapter_model.safetensors" ]]; then
  ADAPTER_PATH=${ADAPTER_PATH:-$LOCAL_ADAPTER}
else
  ADAPTER_PATH=${ADAPTER_PATH:-$HISTORICAL_ADAPTER}
fi

configure_split() {
  SPLIT=${SPLIT:-public}
  case "$SPLIT" in
    public|private) ;;
    *) echo "SPLIT must be public or private" >&2; return 2 ;;
  esac
  QUESTIONS=${QUESTIONS:-$DATA_ROOT/sf20k_${SPLIT}_test_questions.csv}
  VIDEO_DIR=${VIDEO_DIR:-$DATA_ROOT/sf20k_${SPLIT}_test_videos}
  SELECTIONS=${SELECTIONS:-$PIPELINE_ROOT/inputs/${SPLIT}_selections.jsonl}
  RAW_EVIDENCE=${RAW_EVIDENCE:-$PIPELINE_ROOT/inputs/${SPLIT}_evidence_raw.jsonl}
  CLEAN_EVIDENCE=${CLEAN_EVIDENCE:-$PIPELINE_ROOT/inputs/${SPLIT}_evidence_ocr_clean.jsonl}
  GPU_ID=${GPU_ID:-0}
}
