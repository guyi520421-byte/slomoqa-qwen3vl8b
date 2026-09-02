# SloMoQA Base Qwen3-VL-8B 最终推理流程

这是从现有实验目录中提取的独立推理代码包。最终方案使用 **Base Qwen3-VL-8B**，不使用训练脚本、LoRA adapter 或手工跨卡分层。原实验目录没有被修改；选帧、MMR、evidence、OCR 清洗和 Base 推理所需源码都在本文件夹内，不再从其他 SloMoQA 或 EgoLongQA 目录动态导入代码。

技术细节见 [TECHNICAL_SOLUTION.md](TECHNICAL_SOLUTION.md)，环境与模型路径见 [ENVIRONMENTS.md](ENVIRONMENTS.md) 和 `pipeline_env.sh`。

## 每个阶段使用哪个代码

| 阶段 | 直接运行入口 | 核心代码 | 主要输出 |
|---|---|---|---|
| 1. VideoITG 选帧 | `run_01_select_videoitg.sh` | `feature_pipeline.py`、`videoitg_base.py`、`eagle/` | `inputs/{split}_selections.jsonl` 中的 VideoITG Top64 |
| 2. DINOv2 MMR | `run_02_mmr_dinov2.sh` | `feature_pipeline.py`、`dinov2_frame_filter.py` | 同一 selections 文件中的 MMR Top32 |
| 3. ASR/字幕/OCR | `run_03_evidence_qwen25vl.sh` | `feature_pipeline.py` | `inputs/{split}_evidence_raw.jsonl` |
| 4. OCR 清洗 | `run_04_clean_ocr.sh` | `clean_ocr.py`、`ocr_clean_core.py` | `inputs/{split}_evidence_ocr_clean.jsonl` |
| 5. Base 推理 | `run_05_infer_base.sh` | `infer_public.py` | `outputs/qwen3vl8b_base_{split}.json` |
| 可选 Qwen3 OCR 实验 | `run_experiment_qwen3_ocr.sh` | `experiments/qwen3_ocr_experiment.py` | 独立的 Qwen3 OCR evidence 和预测 |

`videoitg_base.py` 和 `eagle/` 是本地内置运行库，不需要单独启动。public/private 使用相同实现和默认参数，只有问题 CSV、视频目录及输出文件不同。主流程每个阶段均在指定的一张 GPU 上完整执行；这和训练时的数据并行或模型分层无关。

## public 完整流程

```bash
cd /data/wen1/slomoQa/slomoqa_qwen3vl8b_final_pipeline
SPLIT=public GPU_ID=0 bash run_01_select_videoitg.sh
SPLIT=public GPU_ID=0 bash run_02_mmr_dinov2.sh
SPLIT=public GPU_ID=0 bash run_03_evidence_qwen25vl.sh
SPLIT=public GPU_ID=0 bash run_04_clean_ocr.sh
SPLIT=public GPU_ID=0 bash run_05_infer_base.sh
```

## private 完整流程

```bash
cd /data/wen1/slomoQa/slomoqa_qwen3vl8b_final_pipeline
SPLIT=private GPU_ID=1 bash run_01_select_videoitg.sh
SPLIT=private GPU_ID=1 bash run_02_mmr_dinov2.sh
SPLIT=private GPU_ID=1 bash run_03_evidence_qwen25vl.sh
SPLIT=private GPU_ID=1 bash run_04_clean_ocr.sh
SPLIT=private GPU_ID=1 bash run_05_infer_base.sh
```

各脚本末尾都接受额外参数覆盖；多行命令的每一条续写行末必须保留反斜杠。

## 使用已有 evidence 直接推理

代码包已包含目前确认有效的 public/private selections 与 clean evidence，因此可以跳过阶段 1–4：

```bash
SPLIT=public GPU_ID=0 bash run_05_infer_base.sh
SPLIT=private GPU_ID=1 bash run_05_infer_base.sh
```

private 若要切换到重新生成的 OCR evidence：

```bash
SPLIT=private GPU_ID=1 \
CLEAN_EVIDENCE=$PWD/inputs/private_evidence_ocr_clean_fullregen_gpu1.jsonl \
OUTPUT=$PWD/outputs/qwen3vl8b_base_private_fullregen.json \
  bash run_05_infer_base.sh
```

Base 答题默认参数为 MMR Top32、图片最长边 1024、每图固定 512 visual tokens、最多 64 条/20000 字符 text evidence、20 秒时间窗口、4-bit NF4、BF16、SDPA 和最多生成 128 tokens。

evidence 阶段的 OCR 图片参数在 `run_03_evidence_qwen25vl.sh` 中设置：

```bash
OCR_DECODE_MAX_SIDE=1024 \
OCR_MIN_VISUAL_TOKENS=256 \
OCR_MAX_VISUAL_TOKENS=512 \
SPLIT=private GPU_ID=1 bash run_03_evidence_qwen25vl.sh
```

答题阶段的图片参数位于 `run_05_infer_base.sh` 的 `--decode-max-side`、`--min-visual-tokens` 和 `--max-visual-tokens`。

## 可选：用 Qwen3-VL 重新提取 OCR

该实验只替换 OCR 模型，MMR selections、ASR 和字幕仍按主流程处理：

```bash
SPLIT=public GPU_ID=0 bash run_experiment_qwen3_ocr.sh
SPLIT=private GPU_ID=1 bash run_experiment_qwen3_ocr.sh
```

## 验证整理包

```bash
/data/miniconda3/envs/egoVqa/bin/python verify_package.py
```

模型权重、原始视频和问题 CSV 没有复制进仓库，它们在 `pipeline_env.sh` 中作为可覆盖的外部资源。历史训练脚本和 adapter 仍留在原实验目录中，但不属于这个最终 Base 推理仓库。
