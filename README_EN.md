# SloMoQA Qwen3-VL-8B Reproduction


The final system uses Base Qwen3-VL-8B-Instruct without LoRA or task-specific
training. Two entry points are provided:

- `run_best_cached.sh`: reproduce the current best configuration with the
  packaged selections and cleaned evidence.
- `run_all.sh`: rerun the complete pipeline from question CSV files and raw
  videos.

## Environment

The reference system uses Ubuntu 22.04, NVIDIA driver 555.42.06, PyTorch
2.6.0+cu118, and RTX 3090 24 GiB GPUs. Each command uses one complete GPU; no
cross-GPU model splitting is used. On a single-GPU machine, set `GPU_ID=0` for
both splits.

Two Conda environments are required because VideoITG and Qwen3-VL use
different Transformers versions.

### VideoITG and DINOv2

```bash
conda create -y -n slomoqa-videoitg python=3.12
conda run -n slomoqa-videoitg python -m pip install \
  torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
conda run -n slomoqa-videoitg python -m pip install \
  transformers==4.47.1 accelerate==1.3.0 decord==0.6.0 av==14.1.0 \
  timm==0.9.11 numpy==1.26.4 Pillow==10.4.0 einops==0.6.1 \
  scipy==1.14.1 fvcore==0.1.5.post20221221 iopath==0.1.10 \
  peft==0.14.0 safetensors==0.5.2 tokenizers==0.21.4 \
  huggingface-hub==0.28.1
```

### Qwen3-VL, Qwen2.5-VL OCR, and Whisper

```bash
conda create -y -n slomoqa-qwen python=3.10
conda run -n slomoqa-qwen python -m pip install \
  torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
conda run -n slomoqa-qwen python -m pip install \
  transformers==4.57.0 accelerate==1.14.0 bitsandbytes==0.49.2 \
  qwen-vl-utils==0.0.14 decord==0.6.0 av==17.1.0 \
  opencv-python-headless==5.0.0.93 numpy==2.2.6 Pillow==12.2.0 \
  timm==1.0.28 einops==0.8.2 scipy==1.15.3 \
  fvcore==0.1.5.post20221221 iopath==0.1.10 peft==0.14.0 \
  safetensors==0.8.0 tokenizers==0.22.2 huggingface-hub==0.36.2
```

FlashAttention and a system `ffmpeg` executable are not required.

## Data, models, and paths

`DATA_ROOT` must contain:

```text
sf20k_public_test_questions.csv
sf20k_public_test_videos/<video_id>.mp4
sf20k_private_test_questions.csv
sf20k_private_test_videos/<video_id>.mp4
```

Download complete local copies of VideoITG-8B,
`dinov2_vitb14_reg4_pretrain.pth`, Whisper Large-v3-Turbo,
Qwen2.5-VL-7B-Instruct, and Qwen3-VL-8B-Instruct. VideoITG also requires
`google/siglip-so400m-patch14-384` in the Hugging Face cache because the code
runs offline.

Set the paths before running:

```bash
cd /absolute/path/to/slomoqa_qwen3vl8b_final_pipeline

unset QUESTIONS VIDEO_DIR SELECTIONS RAW_EVIDENCE CLEAN_EVIDENCE OUTPUT
export VIDEOITG_PYTHON=/absolute/path/to/envs/slomoqa-videoitg/bin/python
export QWEN_PYTHON=/absolute/path/to/envs/slomoqa-qwen/bin/python
export DATA_ROOT=/absolute/path/to/slomo_qa
export VIDEOITG_MODEL=/absolute/path/to/VideoITG-8B
export DINOV2_MODEL=/absolute/path/to/dinov2_vitb14_reg4_pretrain.pth
export ASR_MODEL=/absolute/path/to/whisper-large-v3-turbo
export QWEN25_OCR_MODEL=/absolute/path/to/Qwen2.5-VL-7B-Instruct
export QWEN3_MODEL=/absolute/path/to/Qwen3-VL-8B-Instruct
export HF_HOME=/absolute/path/to/huggingface-cache
```

Check the setup:

```bash
"$QWEN_PYTHON" verify_package.py
SPLIT=public GPU_ID=0 bash preflight.sh
```

## Reproduce the current best configuration

This command only reruns Base Qwen3-VL answer generation with the packaged
golden selections and cleaned evidence.

Public:

```bash
cd /data/wen1/slomoQa/slomoqa_qwen3vl8b_final_pipeline

SPLIT=public GPU_ID=0 bash preflight.sh

SPLIT=public GPU_ID=0 \
OUTPUT="$PWD/outputs/qwen3vl8b_base512_public3.json" \
  bash run_best_cached.sh
```

Private:

```bash
SPLIT=private GPU_ID=1 \
OUTPUT="$PWD/outputs/qwen3vl8b_base_private_notfullregen_evidence2.json" \
  bash run_best_cached.sh
```

These commands reuse the verified selections and cleaned OCR evidence, rerun
only Base Qwen3-VL-8B answer generation, and automatically validate question
coverage and the historical result hash. Reusing an existing `OUTPUT` resumes
from its matching metadata sidecar.

## Complete rerun

Public:

```bash
SPLIT=public GPU_ID=0 \
RUN_DIR="$PWD/runs/public_full_01" \
  bash run_all.sh
```

Private:

```bash
SPLIT=private GPU_ID=1 \
RUN_DIR="$PWD/runs/private_full_01" \
  bash run_all.sh
```

All newly generated intermediate files and predictions are written under
`RUN_DIR`; the packaged golden caches are not overwritten. Reuse the same
`RUN_DIR` to resume an interrupted run, or choose a new directory to start a
fresh complete rerun.
