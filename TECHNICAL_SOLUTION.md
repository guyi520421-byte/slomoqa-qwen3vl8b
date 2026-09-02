# SloMoQA 技术方案

## 主流程

```text
问题 + 完整视频
  → VideoITG 问题相关性排序（512 个候选）
  → DINOv2 全局/patch/时间 MMR（Top64 → Top32）
  → Whisper ASR + 字幕 + Qwen2.5-VL OCR
  → strict OCR 清洗
  → 按问题词汇相关性与选中帧时间邻近度检索 text evidence
  → Base Qwen3-VL-8B 或 Qwen3-VL-8B QLoRA 生成短答案
```

## 1. VideoITG 候选帧

`feature_pipeline.py --stage select` 调用本地 `videoitg_base.py` 和 `eagle/`。视频以目标 2 FPS 采样，最多均匀保留512个候选；VideoITG根据问题对帧排序，保留前64个供MMR处理。图片推理分辨率参数不会改变这一阶段的选帧结果。

## 2. DINOv2 时间 MMR

`feature_pipeline.py --stage mmr` 调用 `dinov2_frame_filter.py`。DINOv2提取全局token和patch token，patch自适应池化为7×7；视觉相似度由0.60全局相似度与0.40 patch相似度组成。时间冗余为 `exp(-|Δt|/3s)`。

MMR效用为：

```text
utility = 1.00 × relevance - 0.30 × visual_redundancy - 0.15 × temporal_redundancy
relevance = 0.85 × normalized_VideoITG_score + 0.15 × rank_prior
```

从VideoITG Top64中按效用选出32帧。public和private使用完全相同的代码及参数，只有CSV和视频不同。

## 3. 多源文本 evidence

`feature_pipeline.py --stage evidence` 生成：

- ASR：Whisper Large-v3-Turbo，16 kHz单声道，英文转写，30秒分块。
- Subtitle：读取视频内嵌字幕或可选sidecar字幕。
- OCR：Qwen2.5-VL-7B，对各视频跨问题合并后的MMR帧按相关分数取最多96帧；batch=3，`max_new_tokens=512`。

整理版推荐OCR输入为最长边1024、每图256–512 visual tokens。历史已保存的标准 evidence 来自旧流程（实际为740、128–256）；旧 orchestrator 没有转发图片覆盖参数，因此历史 `fullregen_gpu1` 的差异主要来自OCR重新生成，而不是selection变化。

## 4. OCR 清洗

`clean_ocr.py --stage clean` 调用 `ocr_clean_core.py`，只改变OCR通道：删除帧ID模板、固定平台水印、跨长时间重复水印和邻近重复项。ASR和subtitle保持原样。credits会保留，不按普通噪声规则删除。

## 5. Text evidence 检索

`infer_public.py` 对subtitle、ASR、OCR统一打分：

```text
score = 2.5 × question_word_overlap + temporal_proximity + source_bonus
source_bonus: subtitle=0.15, ASR=0.10, OCR=0.05
proximity = 1 / (1 + distance/5)，仅保留20秒窗口内的邻近加分
```

随后去除完全重复文本，受 `max_text_evidence_items` 和 `max_evidence_chars` 双重上限约束，再按时间排序输入模型。当前Base/LoRA对比入口使用最多64条、20000字符。

## 6. Qwen3-VL-8B 回答

`infer_public.py` 同时支持public/private，二者 evidence 生成与检索代码一致。主对比设置为MMR Top32、最长边1024、每图固定512 visual tokens、4-bit NF4、BF16、SDPA、最多生成128 tokens。未传 `--lora-path` 时为Base；传入adapter时为LoRA。

按视频单卡顺序推理只改变调度，不改变模型或输入；它不同于训练时错误地把语言层拆到两张卡。

## 7. QLoRA训练

训练数据复用best43对齐快照：634个QA、415个视频，答案8–11词，MMR帧与最多25条/15000字符 evidence 已固定。已完成adapter的真实图片配置是最长边840、每图128–256 visual tokens。

训练使用标准双进程DDP：两张3090各自加载完整4-bit Qwen3-VL-8B，并同步梯度，不进行手工层切分。参数如下：

- 3 epochs，per-device batch=1，gradient accumulation=4，全局有效batch=8。
- learning rate `1e-5`，weight decay `0.01`，warmup `0.05`，cosine scheduler。
- LoRA `r=32`、`alpha=64`、dropout `0.05`。
- 目标为36层语言模型中的 `q/k/v/o_proj` 与 `gate/up/down_proj`，共7类projection；视觉塔冻结。
- gradient checkpointing、paged AdamW 8-bit、BF16、SDPA。
- 按视频切分90%训练/10%验证，避免视频泄漏；seed=42。

已完成训练从初始验证loss 3.6814下降到最终1.3897。真实配置副本位于 `docs/reference/`。

## 8. Qwen3 OCR替换实验

`experiments/qwen3_ocr_experiment.py` 仅替换OCR模型为Base Qwen3-VL-8B；MMR、ASR和subtitle继续复用。它属于独立实验，不是已验证主流程。运行入口为 `run_experiment_qwen3_ocr.sh`。
