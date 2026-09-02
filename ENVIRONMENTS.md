# 运行环境

本代码包使用两个现有 Conda 环境，阶段脚本会自动选择。

- `/data/miniconda3/envs/videoitg/bin/python`：VideoITG 与 DINOv2 MMR。核心依赖包括 PyTorch、Transformers、Decord、timm、Pillow、NumPy、PyAV。
- `/data/miniconda3/envs/egoVqa/bin/python`：Whisper ASR、Qwen2.5-VL OCR、Qwen3-VL 推理与 QLoRA。核心依赖包括 PyTorch、Transformers、PEFT、bitsandbytes、Accelerate、qwen-vl-utils、OpenCV、Pillow、PyAV。

默认模型路径集中在 `pipeline_env.sh`。模型权重、原始 CSV 和视频不复制进代码包，因为它们属于外部数据/模型资源；所有 Python 源码依赖已放在本文件夹内。
