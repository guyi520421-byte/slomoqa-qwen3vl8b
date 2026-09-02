# SloMoQA Qwen3-VL-8B 最终整理版

这是从现有实验目录中提取的独立代码包。原目录没有被修改；选帧、MMR、evidence、OCR清洗、训练和Qwen3-VL推理所需Python源码都在本文件夹内，不再从其他SloMoQA或EgoLongQA目录动态导入代码。

技术细节见 [TECHNICAL_SOLUTION.md](TECHNICAL_SOLUTION.md)，环境与模型路径见 [ENVIRONMENTS.md](ENVIRONMENTS.md) 和 `pipeline_env.sh`。

## 每个阶段使用哪个代码

| 阶段 | 直接运行入口 | 核心代码 | 主要输出 |
|---|---|---|---|
| 1. VideoITG选帧 | `run_01_select_videoitg.sh` | `feature_pipeline.py`、`videoitg_base.py`、`eagle/` | `inputs/{split}_selections.jsonl`中的VideoITG Top64 |
| 2. DINOv2 MMR | `run_02_mmr_dinov2.sh` | `feature_pipeline.py`、`dinov2_frame_filter.py` | 同一selections文件中的MMR Top32 |
| 3. ASR/字幕/OCR | `run_03_evidence_qwen25vl.sh` | `feature_pipeline.py` | `inputs/{split}_evidence_raw.jsonl` |
| 4. OCR清洗 | `run_04_clean_ocr.sh` | `clean_ocr.py`、`ocr_clean_core.py` | `inputs/{split}_evidence_ocr_clean.jsonl` |
| 5. 训练清单 | `run_05_prepare_train.sh` | `prepare_data.py` | `artifacts/train_manifest_qwen3vl8b.jsonl` |
| 6. 训练预检 | `run_06_preflight_train.sh` | `preflight.py`、`train_lora.py` | token/显存前置检查日志 |
| 7. 两卡QLoRA | `run_07_train_2gpu.sh` | `train_lora.py` | `outputs/qwen3vl8b_frame_legacy_exact_lora/` |
| 8. Base推理 | `run_08_infer_base.sh` | `infer_public.py` | `outputs/qwen3vl8b_base_{split}.json` |
| 9. LoRA推理 | `run_09_infer_lora.sh` | `infer_public.py` | `outputs/qwen3vl8b_lora_{split}.json` |
| 可选Qwen3 OCR | `run_experiment_qwen3_ocr.sh` | `experiments/qwen3_ocr_experiment.py` | 独立的Qwen3 OCR evidence和预测 |

`videoitg_base.py`、`eagle/`属于本地内置运行库，不需要单独启动。历史的双卡按视频分片脚本没有放进主流程；主流程的evidence和推理均按单卡完整运行，训练才使用标准两卡DDP。

## public完整流程

```bash
cd /data/wen1/slomoQa/slomoqa_qwen3vl8b_final_pipeline
SPLIT=public GPU_ID=0 bash run_01_select_videoitg.sh
SPLIT=public GPU_ID=0 bash run_02_mmr_dinov2.sh
SPLIT=public GPU_ID=0 bash run_03_evidence_qwen25vl.sh
SPLIT=public GPU_ID=0 bash run_04_clean_ocr.sh
SPLIT=public GPU_ID=0 bash run_08_infer_base.sh
```

## private完整流程

```bash
cd /data/wen1/slomoQa/slomoqa_qwen3vl8b_final_pipeline
SPLIT=private GPU_ID=1 bash run_01_select_videoitg.sh
SPLIT=private GPU_ID=1 bash run_02_mmr_dinov2.sh
SPLIT=private GPU_ID=1 bash run_03_evidence_qwen25vl.sh
SPLIT=private GPU_ID=1 bash run_04_clean_ocr.sh
SPLIT=private GPU_ID=1 bash run_08_infer_base.sh
```

public/private入口使用同一套实现和参数，只有问题CSV、视频目录及输出文件不同。各脚本末尾都接受额外参数覆盖；每一行续写时必须保留反斜杠。

## 使用已有 evidence 直接推理

代码包已复制目前确认有效的public/private selections与clean evidence，因此可以跳过1–4阶段：

```bash
SPLIT=public GPU_ID=0 bash run_08_infer_base.sh
SPLIT=private GPU_ID=1 bash run_08_infer_base.sh
```

private若要切换到重新生成的OCR evidence：

```bash
SPLIT=private GPU_ID=1 \
CLEAN_EVIDENCE=$PWD/inputs/private_evidence_ocr_clean_fullregen_gpu1.jsonl \
OUTPUT=$PWD/outputs/qwen3vl8b_base_private_fullregen.json \
  bash run_08_infer_base.sh
```

默认答题参数为Top32、1024、每图固定512 visual tokens、最多64条/20000字符text evidence。

## 两卡训练

```bash
bash run_05_prepare_train.sh
bash run_06_preflight_train.sh
TRAIN_GPU_IDS=0,1 bash run_07_train_2gpu.sh
```

这是标准2-rank DDP；每张GPU都有完整4-bit模型，不做“GPU0视觉塔+前层、GPU1后层”的手工拆分。真实复现配置为840、每图128–256 visual tokens、LoRA r32/alpha64。

## LoRA推理

若本目录训练出了adapter，入口自动优先使用本目录adapter；否则默认引用历史已完成adapter：

```bash
SPLIT=public GPU_ID=0 bash run_09_infer_lora.sh
SPLIT=private GPU_ID=1 bash run_09_infer_lora.sh
```

也可以显式指定：

```bash
ADAPTER_PATH=/path/to/adapter SPLIT=private GPU_ID=1 bash run_09_infer_lora.sh
```

## 验证整理包

```bash
/data/miniconda3/envs/egoVqa/bin/python verify_package.py
/data/miniconda3/envs/egoVqa/bin/python -m unittest -v test_pipeline.py
```

模型权重和原始视频没有复制；它们在 `pipeline_env.sh` 里作为可覆盖的外部资源。代码、Eagle运行库、DINOv2过滤器、已验证的小型缓存和真实训练配置副本均已收拢到当前文件夹。
