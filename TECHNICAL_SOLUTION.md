# SloMoQA 最终技术方案

## 主流程

```text
问题 + 完整视频
  → VideoITG 问题相关性排序（最多 512 个候选）
  → DINOv2 全局/patch/时间 MMR（Top64 → Top32）
  → Whisper ASR + 字幕 + Qwen2.5-VL OCR
  → strict OCR 清洗
  → 按问题词汇相关性与选中帧时间邻近度检索 text evidence
  → Base Qwen3-VL-8B 生成短答案
```

最终提交方法不包含训练或 LoRA adapter。整个 public/private 推理流程使用同一套代码和参数，仅数据路径与输出路径不同。

## 1. VideoITG 候选帧

`feature_pipeline.py --stage select` 调用本地 `videoitg_base.py` 和 `eagle/`。视频以目标 2 FPS 采样，最多均匀保留 512 个候选；VideoITG 根据问题对帧排序，保留前 64 个供 MMR 处理。答题模型的图片分辨率与 visual-token 参数不会改变这一阶段的 selection。

## 2. DINOv2 时间 MMR

`feature_pipeline.py --stage mmr` 调用 `dinov2_frame_filter.py`。DINOv2 提取全局 token 和 patch token，patch 自适应池化为 7×7；视觉相似度由 0.60 全局相似度与 0.40 patch 相似度组成。时间冗余为 `exp(-|Δt|/3s)`。

```text
utility = 1.00 × relevance - 0.30 × visual_redundancy - 0.15 × temporal_redundancy
relevance = 0.85 × normalized_VideoITG_score + 0.15 × rank_prior
```

从 VideoITG Top64 中按效用选出 32 帧。public 和 private 使用完全相同的代码及参数。

## 3. 多源文本 evidence

`feature_pipeline.py --stage evidence` 生成三类文本：

- ASR：Whisper Large-v3-Turbo，16 kHz 单声道、英文转写、30 秒分块。
- Subtitle：读取视频内嵌字幕或可选 sidecar 字幕。
- OCR：Qwen2.5-VL-7B，对各视频跨问题合并后的 MMR 帧按相关分数取最多 96 帧；batch=3，`max_new_tokens=512`。

整理版默认 OCR 输入为最长边 1024、每图 256–512 visual tokens。图片参数位于 `run_03_evidence_qwen25vl.sh`，不会回头改变 VideoITG/DINOv2 已生成的 selections。

## 4. OCR 清洗

`clean_ocr.py --stage clean` 调用 `ocr_clean_core.py`，只改变 OCR 通道：删除帧 ID 模板、固定平台水印、跨长时间重复水印和邻近重复项。ASR 和 subtitle 保持原样，credits 会保留。

## 5. Text evidence 检索

`infer_public.py` 对 subtitle、ASR、OCR 统一打分：

```text
score = 2.5 × question_word_overlap + temporal_proximity + source_bonus
source_bonus: subtitle=0.15, ASR=0.10, OCR=0.05
proximity = 1 / (1 + distance/5)，仅保留 20 秒窗口内的邻近加分
```

随后去除完全重复文本，再受 `max_text_evidence_items=64` 和 `max_evidence_chars=20000` 双重上限约束，最后按时间排序输入模型。64 是上限；若过滤后只有 25 条有效 evidence，日志就会显示 `text_evidence=25`。

## 6. Base Qwen3-VL-8B 回答

`run_05_infer_base.sh` 调用 `infer_public.py`，不传入任何 LoRA adapter。最终设置为 MMR Top32、至少 12 帧、最长边 1024、每图固定 512 visual tokens、4-bit NF4、BF16、SDPA、最多生成 128 tokens。

推理在指定的单张 GPU 上按视频/问题顺序运行。双卡按视频分片只属于任务调度；只要输入缓存和参数一致，模型输入及生成逻辑不变。本最终入口采用单卡完整模型，未做跨 GPU 层切分。

## 7. 可选 Qwen3 OCR 替换实验

`experiments/qwen3_ocr_experiment.py` 只将 OCR 模型替换为 Base Qwen3-VL-8B；MMR、ASR 和 subtitle 继续使用相同流程。它是独立对比实验，不是最终主流程，入口为 `run_experiment_qwen3_ocr.sh`。
