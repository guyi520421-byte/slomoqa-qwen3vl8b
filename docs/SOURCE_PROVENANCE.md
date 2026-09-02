# 核心代码来源

所有源文件都已复制进当前目录，运行时不再从以下原工程导入Python代码。原工程只作为审计来源，未被修改。

| 整理后文件 | 审计来源 | 整理时修改 |
|---|---|---|
| `feature_pipeline.py` | `ideoITG_dinov2MMR_OCR/run_slomoqa.py` | 将底层实现改为本地 `videoitg_base.py`；默认输入输出改到当前目录；把OCR分辨率/token加入缓存身份 |
| `videoitg_base.py` | `videoITG_dinov2GloabalPatch/run_slomoqa.py` | 将Eagle与DINO helper根目录改为当前目录 |
| `dinov2_frame_filter.py` | `VideoITG_timeStamps_DINOv2_patch_edit/dinov2_frame_filter.py` | 未改算法，SHA256仍为 `47770148...d51` |
| `eagle/` | `VideoITG_timeStamps_DINOv2_patch_edit/eagle/` | 仅保留模型推理所需源码，去除eval/train工具和缓存 |
| `clean_ocr.py` | `ideoITG_dinov2MMR_OCRClean/run_ocr_clean.py` | 默认路径改到当前目录；主流程只使用其clean阶段，回答统一由 `infer_public.py` 完成 |
| `ocr_clean_core.py` | 原Qwen3VL8B目录的本地严格cleaner | 未改变清洗规则 |
| `infer_public.py` | 原Qwen3VL8B目录 | 未改，SHA256仍为 `25da6947...b8` |
| `prepare_data.py` | 原Qwen3VL8B目录 | 仅修正过期profile名称，使其与真实840配置一致 |
| `train_lora.py` | 原Qwen3VL8B目录 | 默认LoRA改为已完成adapter的r32/alpha64 |
| `experiments/qwen3_ocr_experiment.py` | 原Qwen3VL8B目录 | 改为从项目根目录导入公共模块 |

模型权重、视频和问题CSV仍是外部资源，统一由 `pipeline_env.sh` 配置。这不属于跨目录代码混用。
