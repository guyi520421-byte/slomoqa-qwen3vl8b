# SloMoQA Base Qwen3-VL-8B 最终推理流程

这是从现有实验目录中提取的独立推理代码包。最终方案使用 **Base Qwen3-VL-8B**，不使用训练脚本、LoRA adapter 或手工跨卡分层。原实验目录没有被修改；选帧、MMR、evidence、OCR 清洗和 Base 推理所需源码都在本文件夹内，不再从其他 SloMoQA 或 EgoLongQA 目录动态导入代码。

技术细节见 [TECHNICAL_SOLUTION.md](TECHNICAL_SOLUTION.md)，环境与模型路径见 [ENVIRONMENTS.md](ENVIRONMENTS.md) 和 `pipeline_env.sh`。重新运行时优先阅读 [RERUN_GUIDE.md](RERUN_GUIDE.md)。

## 最常用的两种重跑

### 复现当前最佳配置

Public：

```bash
cd /data/wen1/slomoQa/slomoqa_qwen3vl8b_final_pipeline

SPLIT=public GPU_ID=0 bash preflight.sh

SPLIT=public GPU_ID=0 \
OUTPUT=$PWD/outputs/qwen3vl8b_base512_public3.json \
  bash run_best_cached.sh
```

Private：

```bash
SPLIT=private GPU_ID=1 \
OUTPUT=$PWD/outputs/qwen3vl8b_base_private_notfullregen_evidence2.json \
  bash run_best_cached.sh
```

这会复用已验证的 selections 和 cleaned OCR evidence，只重新执行 Base Qwen3-VL-8B 答题，并自动检查题目覆盖及历史结果哈希。

### 从视频开始完整重跑

```bash
SPLIT=public GPU_ID=0 \
RUN_DIR=$PWD/runs/public_full_01 \
  bash run_all.sh

SPLIT=private GPU_ID=1 \
RUN_DIR=$PWD/runs/private_full_01 \
  bash run_all.sh
```

所有新产物会进入独立的 `RUN_DIR`，不会覆盖最佳缓存。中断后使用同一个 `RUN_DIR` 重跑即可续跑。

## 每个阶段使用哪个代码

| 阶段 | 直接运行入口 | 核心代码 | 主要输出 |
|---|---|---|---|
| 1. VideoITG 选帧 | `run_01_select_videoitg.sh` | `feature_pipeline.py`、`videoitg_base.py`、`eagle/` | `inputs/{split}_selections.jsonl` 中的 VideoITG Top64 |
| 2. DINOv2 MMR | `run_02_mmr_dinov2.sh` | `feature_pipeline.py`、`dinov2_frame_filter.py` | 同一 selections 文件中的 MMR Top32 |
| 3. ASR/字幕/OCR | `run_03_evidence_qwen25vl.sh` | `feature_pipeline.py` | `inputs/{split}_evidence_raw.jsonl` |
| 4. OCR 清洗 | `run_04_clean_ocr.sh` | `clean_ocr.py`、`ocr_clean_core.py` | `inputs/{split}_evidence_ocr_clean.jsonl` |
| 5. Base 推理 | `run_05_infer_base.sh` | `infer_public.py` | `outputs/qwen3vl8b_base_{split}.json` |
| 最佳缓存复现 | `run_best_cached.sh` | 阶段 5、`validate_artifacts.py` | 历史命名对应的完整预测 |
| 全流程独立重跑 | `run_all.sh` | 顺序执行阶段 1–5 | `runs/<name>/` 下的全套产物 |
| 可选 Qwen3 OCR 实验 | `run_experiment_qwen3_ocr.sh` | `experiments/qwen3_ocr_experiment.py` | 独立的 Qwen3 OCR evidence 和预测 |

`videoitg_base.py` 和 `eagle/` 是本地内置运行库，不需要单独启动。public/private 使用相同实现和默认参数，只有问题 CSV、视频目录及输出文件不同。主流程每个阶段均在指定的一张 GPU 上完整执行；这和训练时的数据并行或模型分层无关。

## 手动运行各阶段

以下阶段入口默认带 `--resume`。包内已经存在 selections，因此直接使用默认路径时会命中缓存，并不表示从零生成。真正的完整重跑请使用 `run_all.sh`，避免覆盖 `inputs/` 中已验证的 golden caches。

先指定 split、GPU 和独立产物路径，再逐个运行：

```bash
cd /data/wen1/slomoQa/slomoqa_qwen3vl8b_final_pipeline
export SPLIT=public GPU_ID=0
export RUN_DIR=$PWD/runs/public_manual_01
mkdir -p "$RUN_DIR"
export SELECTIONS=$RUN_DIR/public_selections.jsonl
export RAW_EVIDENCE=$RUN_DIR/public_evidence_raw.jsonl
export CLEAN_EVIDENCE=$RUN_DIR/public_evidence_ocr_clean.jsonl
export OUTPUT=$RUN_DIR/qwen3vl8b_base_public.json

bash run_01_select_videoitg.sh
bash run_02_mmr_dinov2.sh
bash run_03_evidence_qwen25vl.sh
bash run_04_clean_ocr.sh
ALLOW_CUSTOM_PUBLIC_INPUTS=1 bash run_05_infer_base.sh
```

Private 将 `SPLIT=private`、`GPU_ID=1` 和文件名前缀改为 `private` 即可。完整命令及中断续跑方法见 `RERUN_GUIDE.md`。

各脚本末尾都接受额外参数覆盖；多行命令的每一条续写行末必须保留反斜杠。

## 使用已有 evidence 直接推理

代码包已包含目前确认有效的 public/private selections 与 clean evidence，因此可以跳过阶段 1–4。推荐使用带覆盖和历史哈希检查的入口：

```bash
SPLIT=public GPU_ID=0 bash run_best_cached.sh
SPLIT=private GPU_ID=1 bash run_best_cached.sh
```

private 若要切换到重新生成的 OCR evidence：

```bash
SPLIT=private GPU_ID=1 \
CLEAN_EVIDENCE=$PWD/inputs/private_evidence_ocr_clean_fullregen_gpu1.jsonl \
OUTPUT=$PWD/outputs/qwen3vl8b_base_private_fullregen.json \
  bash run_05_infer_base.sh
```

Base 答题默认参数为 MMR Top32、图片最长边 1024、每图固定 512 visual tokens、最多 64 条/20000 字符 text evidence、20 秒时间窗口、4-bit NF4、BF16、SDPA 和最多生成 128 tokens。

evidence 阶段的 OCR 图片参数独立于答题图片。主线默认值为 740 / 128–256，对应当前 golden cleaned-evidence 的来源配置。后来 Private full-regeneration 使用的 1024 / 256–512 可这样启用：

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
