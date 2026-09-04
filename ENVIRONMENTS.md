# 运行环境

本代码包使用两个现有 Conda 环境，阶段脚本会自动选择。

- `/data/miniconda3/envs/videoitg/bin/python`：VideoITG 与 DINOv2 MMR。当前环境为 PyTorch 2.6.0+cu118、Transformers 4.47.1、Decord 0.6.0、timm 0.9.11、PyAV 14.1.0。
- `/data/miniconda3/envs/egoVqa/bin/python`：Whisper ASR、Qwen2.5-VL OCR 与 Base Qwen3-VL 推理。当前环境为 PyTorch 2.6.0+cu118、Transformers 4.57.0、bitsandbytes 0.49.2、qwen-vl-utils 0.0.14、PyAV 17.1.0 和 OpenCV 5.0.0。

默认模型路径集中在 `pipeline_env.sh`。VideoITG 的配置还引用 `google/siglip-so400m-patch14-384`，离线运行前必须确保它已在 Hugging Face cache 中。模型权重、原始 CSV 和视频不复制进代码包，因为它们属于外部数据/模型资源；所有项目 Python 源码已放在本文件夹内。

运行前检查：

```bash
SPLIT=public GPU_ID=0 bash preflight.sh
```
