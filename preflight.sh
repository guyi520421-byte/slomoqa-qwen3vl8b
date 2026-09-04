#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/pipeline_env.sh"
configure_split

require_file() {
  [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 1; }
}

require_file "$VIDEOITG_PYTHON"
require_file "$QWEN_PYTHON"
require_file "$QUESTIONS"
require_dir "$VIDEO_DIR"
require_dir "$VIDEOITG_MODEL"
require_file "$VIDEOITG_MODEL/config.json"
require_file "$DINOV2_MODEL"
require_dir "$ASR_MODEL"
require_file "$ASR_MODEL/config.json"
require_dir "$QWEN25_OCR_MODEL"
require_file "$QWEN25_OCR_MODEL/config.json"
require_dir "$QWEN3_MODEL"
require_file "$QWEN3_MODEL/config.json"

"$VIDEOITG_PYTHON" -c \
  'import av, decord, numpy, PIL, timm, torch, transformers'
"$QWEN_PYTHON" -c \
  'import av, bitsandbytes, cv2, numpy, PIL, qwen_vl_utils, torch, transformers'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$VIDEOITG_PYTHON" -c \
  'from transformers import AutoConfig; AutoConfig.from_pretrained("google/siglip-so400m-patch14-384", local_files_only=True)'

"$QWEN_PYTHON" "$ROOT/verify_package.py"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -i "$GPU_ID" \
    --query-gpu=index,name,memory.total,memory.free \
    --format=csv,noheader
fi
echo "Runtime preflight: PASS (split=$SPLIT, gpu=$GPU_ID)"
