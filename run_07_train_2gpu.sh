#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/pipeline_env.sh"
TRAIN_GPU_IDS=${TRAIN_GPU_IDS:-0,1}
IFS=',' read -r -a GPU_ARRAY <<< "$TRAIN_GPU_IDS"
if [[ ${#GPU_ARRAY[@]} -ne 2 || ${GPU_ARRAY[0]} == "${GPU_ARRAY[1]}" ]]; then
  echo "TRAIN_GPU_IDS must contain two different GPU IDs, for example 0,1" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
cd "$ROOT"
echo "Standard 2-rank DDP: each GPU loads one complete 4-bit Qwen3-VL model." >&2
exec "$QWEN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "$ROOT/train_lora.py" \
  --model "$QWEN3_MODEL" \
  --manifest "$ROOT/artifacts/train_manifest_qwen3vl8b.jsonl" \
  --output-dir "$LOCAL_ADAPTER" \
  --min-free-gpu-memory-gib 22 --dtype bf16 --quantization 4bit \
  --attention sdpa --epochs 3 --learning-rate 1e-5 --weight-decay 0.01 \
  --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --label-smoothing-factor 0 --gradient-accumulation-steps 4 \
  --warmup-ratio 0.05 --validation-ratio 0.10 \
  --save-steps 25 --eval-steps 25 --logging-steps 5 \
  --early-stopping-patience 3 --max-sequence-length 32768 \
  --resume-from-checkpoint "$@"
