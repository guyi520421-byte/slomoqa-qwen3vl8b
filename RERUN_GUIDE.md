# SloMoQA 重跑手册

本目录把“复现当前最佳答案”和“从视频开始重新生成全部中间结果”分成两个入口。两者不要混用：前者锁定已验证缓存，后者会重新运行带生成性的 VideoITG、ASR 和 OCR 阶段，输出不保证与历史 JSON 逐字节一致。

## 1. 当前采用的配置

主结果是 Base Qwen3-VL-8B，不加载 LoRA：

```text
Public 历史文件:  qwen3vl8b_base512_public3.json
Private 对应文件: qwen3vl8b_base_private_notfullregen_evidence2.json

MMR Top32
答题图片最长边 1024
每张答题图片固定 512 visual tokens
文本 evidence 最多 64 条 / 20000 字符 / 20 秒窗口
4-bit NF4 / BF16 / SDPA / greedy / max_new_tokens=128
```

主结果复用目录内的 golden caches：

```text
inputs/public_selections.jsonl
inputs/public_evidence_ocr_clean.jsonl
inputs/private_selections.jsonl
inputs/private_evidence_ocr_clean.jsonl
```

这里需要区分两组图片参数：最终 Qwen3-VL **答题**使用 1024 / 512；golden evidence 中的 Qwen2.5-VL **OCR 提取**来自旧配置 740 / 128–256。后来的 Private full-regeneration OCR 实验才使用 1024 / 256–512。

## 2. 运行前检查

```bash
cd /data/wen1/slomoQa/slomoqa_qwen3vl8b_final_pipeline
/data/miniconda3/envs/egoVqa/bin/python verify_package.py
SPLIT=public GPU_ID=0 bash preflight.sh
```

第一条检查独立源码、缓存数量和 golden cache 哈希；第二条还检查运行环境、外部模型、数据、SigLIP 本地缓存和指定 GPU。默认模型、数据和 Python 路径集中在 `pipeline_env.sh`，换机器时只需要覆盖其中的环境变量。

## 3. 推荐：锁定缓存重新生成答案

这是复现 `public3` 最直接、变量最少的方式，不重新运行 VideoITG、DINOv2、ASR 或 OCR。

Public 使用 GPU 0：

```bash
cd /data/wen1/slomoQa/slomoqa_qwen3vl8b_final_pipeline
SPLIT=public GPU_ID=0 \
OUTPUT=$PWD/outputs/qwen3vl8b_base512_public3.json \
  bash run_best_cached.sh
```

Private 使用 GPU 1：

```bash
cd /data/wen1/slomoQa/slomoqa_qwen3vl8b_final_pipeline
SPLIT=private GPU_ID=1 \
OUTPUT=$PWD/outputs/qwen3vl8b_base_private_notfullregen_evidence2.json \
  bash run_best_cached.sh
```

脚本结束时会验证 538/441 道题是否全部存在且答案非空，并比较历史预测 SHA256。历史参考值为：

```text
Public:  6fbaa7aeb24ad21483800bb26b6c8c5aaf9dc3506eb067ac8d732f018ca0712a
Private: 66fe3508dcf73462676ee394f3b2d0f413cb60516ebae165fc55c085ff3b95ff
```

如果覆盖检查通过但哈希不同，应检查模型文件、Transformers/bitsandbytes 版本、GPU OOM 降帧日志和输入是否被改动。

## 4. 从视频开始完整重跑

`run_all.sh` 顺序执行五个阶段，并把所有新文件写入独立的 `RUN_DIR`，不会覆盖仓库内的 golden caches。

```bash
cd /data/wen1/slomoQa/slomoqa_qwen3vl8b_final_pipeline

SPLIT=public GPU_ID=0 \
RUN_DIR=$PWD/runs/public_full_01 \
  bash run_all.sh

SPLIT=private GPU_ID=1 \
RUN_DIR=$PWD/runs/private_full_01 \
  bash run_all.sh
```

每个运行目录包含：

```text
{split}_selections.jsonl
{split}_evidence_raw.jsonl
{split}_evidence_ocr_clean.jsonl
qwen3vl8b_base_{split}.json
```

中断后使用相同的 `RUN_DIR` 再执行同一命令即可按缓存继续。若要真正开始一轮全新实验，请换一个新的 `RUN_DIR`，不要删除或覆盖 `inputs/` 中的 golden caches。

完整重跑默认使用与 golden public OCR 来源一致的 740 / 128–256。如果要运行后来使用的高分辨率 OCR 实验：

```bash
OCR_DECODE_MAX_SIDE=1024 \
OCR_MIN_VISUAL_TOKENS=256 \
OCR_MAX_VISUAL_TOKENS=512 \
SPLIT=private GPU_ID=1 \
RUN_DIR=$PWD/runs/private_ocr1024_01 \
  bash run_all.sh
```

这会产生新的 OCR evidence，属于对比实验，不再是严格的 `public3` 缓存复现。

## 5. 手动分阶段运行

五个入口及输出关系如下：

```text
run_01_select_videoitg.sh       VideoITG Top64
run_02_mmr_dinov2.sh            DINOv2 MMR Top32
run_03_evidence_qwen25vl.sh     Whisper ASR + subtitle + Qwen2.5-VL OCR
run_04_clean_ocr.sh             只清洗 OCR 通道
run_05_infer_base.sh            Base Qwen3-VL-8B 答题
```

若手动从头运行，建议先把路径指向独立目录：

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

直接运行 `run_05_infer_base.sh` 时，Public golden inputs 默认会执行 SHA256 校验；只有明确使用自定义 Public selections/evidence 时才设置 `ALLOW_CUSTOM_PUBLIC_INPUTS=1`。

## 6. Private full-regeneration evidence 对比

目录里还保留了一份 1024 / 256–512 OCR 生成的 Private cleaned evidence。只切换这份 evidence 并保持答题配置不变：

```bash
cd /data/wen1/slomoQa/slomoqa_qwen3vl8b_final_pipeline
SPLIT=private GPU_ID=1 \
CLEAN_EVIDENCE=$PWD/inputs/private_evidence_ocr_clean_fullregen_gpu1.jsonl \
OUTPUT=$PWD/outputs/qwen3vl8b_base_private_fullregen_1024_v256_512.json \
  bash run_05_infer_base.sh

/data/miniconda3/envs/egoVqa/bin/python validate_artifacts.py \
  --split private \
  --selections inputs/private_selections.jsonl \
  --evidence inputs/private_evidence_ocr_clean_fullregen_gpu1.jsonl \
  --predictions outputs/qwen3vl8b_base_private_fullregen_1024_v256_512.json
```

`answer_prompt.txt` 目前仅作为可读参考；严格复现所用 prompt 内联在 `infer_public.py` 的 `render_legacy_exact_prompt()` 中，修改文本文件不会改变主结果。
