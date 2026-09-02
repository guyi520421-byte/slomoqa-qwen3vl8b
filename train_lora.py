#!/usr/bin/env python3
"""Stable QLoRA training for folder-local Qwen3-VL legacy-exact frames."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

import infer_public as inference
import prepare_data as profile


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = Path("/data/wen1/MLLMs/qwen3vl8b")
DEFAULT_MANIFEST = HERE / "artifacts/train_manifest_qwen3vl8b.jsonl"
DEFAULT_OUTPUT = HERE / "outputs/qwen3vl8b_frame_legacy_exact_lora"

TOPK = profile.TOPK
MIN_FRAMES = profile.MIN_FRAMES
DECODE_MAX_SIDE = profile.DECODE_MAX_SIDE
MIN_VISUAL_TOKENS = profile.MIN_VISUAL_TOKENS
MAX_VISUAL_TOKENS = profile.MAX_VISUAL_TOKENS
EXPECTED_PIPELINE = profile.PIPELINE_METHOD
EXPECTED_PROFILE = profile.PROFILE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--label-smoothing-factor", type=float, default=0.0)
    parser.add_argument("--min-answer-words", type=int, default=8)
    parser.add_argument("--max-answer-words", type=int, default=11)
    parser.add_argument("--max-sequence-length", type=int, default=32768)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-index", type=int)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--attention", choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--quantization", choices=("4bit", "none"), default="4bit")
    parser.add_argument(
        "--min-free-gpu-memory-gib",
        type=float,
        default=22.0,
        help="Fail if the GPU assigned to this DDP rank has less free memory; 0 disables.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const="auto",
        help="Resume a checkpoint path, or the latest checkpoint when omitted.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--eval-before-training",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--preflight-samples", type=int, default=12,
        help="Records to decode; 0 validates all records.",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", -1)),
    )
    return parser.parse_args()


def raise_value_error(message: str) -> None:
    raise ValueError(message)


def validate_args(args: argparse.Namespace) -> None:
    inference.require_qwen3_checkpoint(args.model, raise_value_error)
    if not args.manifest.is_file():
        raise FileNotFoundError(
            f"Training manifest is missing: {args.manifest}; run prepare.sh first"
        )
    if args.init_adapter is not None:
        inference.require_adapter(args.init_adapter, raise_value_error)
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max-steps must be -1 or positive")
    if args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("--epochs and --learning-rate must be positive")
    if args.weight_decay < 0 or not 0 <= args.label_smoothing_factor < 1:
        raise ValueError("Invalid weight decay or label smoothing")
    if not 0 <= args.warmup_ratio <= 1:
        raise ValueError("--warmup-ratio must be in [0, 1]")
    if args.min_answer_words <= 0 or args.max_answer_words < args.min_answer_words:
        raise ValueError("Invalid answer word-count range")
    if not 0 <= args.validation_ratio < 1:
        raise ValueError("--validation-ratio must be in [0, 1)")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.max_sequence_length <= 0 or args.preflight_samples < 0:
        raise ValueError("Invalid sequence/preflight limit")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.save_steps <= 0 or args.eval_steps <= 0 or args.logging_steps <= 0:
        raise ValueError("Save/eval/logging steps must be positive")
    if args.save_steps % args.eval_steps:
        raise ValueError("--save-steps must be a multiple of --eval-steps")
    if args.lora_r <= 0 or args.lora_alpha <= 0 or not 0 <= args.lora_dropout < 1:
        raise ValueError("Invalid LoRA parameters")
    if args.min_free_gpu_memory_gib < 0:
        raise ValueError("--min-free-gpu-memory-gib must be non-negative")


def validate_gpu_runtime(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-VL training")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if world_size not in (1, 2):
        raise RuntimeError(f"Expected WORLD_SIZE 1 or 2, got {world_size}")
    if world_size == 2:
        if torch.cuda.device_count() != 2 or local_rank not in (0, 1):
            raise RuntimeError(
                "Two-rank DDP requires exactly two visible GPUs and LOCAL_RANK 0/1; "
                f"devices={torch.cuda.device_count()} local_rank={local_rank}"
            )
        device_index = local_rank
    else:
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "A non-DDP run requires exactly one visible GPU. Use train_2gpu.sh "
                "for normal two-card training."
            )
        device_index = 0
    torch.cuda.set_device(device_index)
    with torch.cuda.device(device_index):
        free_bytes, _ = torch.cuda.mem_get_info()
    free_gib = free_bytes / 1024**3
    rank = int(os.environ.get("RANK", "0"))
    print(
        f"Rank{rank}/local_rank{local_rank}: "
        f"{torch.cuda.get_device_name(device_index)}, free={free_gib:.2f} GiB",
        flush=True,
    )
    if args.min_free_gpu_memory_gib > 0 and free_gib < args.min_free_gpu_memory_gib:
        raise RuntimeError(
            f"At least {args.min_free_gpu_memory_gib:.1f} GiB free is required on "
            f"this rank's GPU for a stable start; observed={free_gib:.2f} GiB. "
            "Stop GPU jobs or reduce the image resolution/visual-token budget."
        )
    return device_index


def read_manifest(
    path: Path, min_answer_words: int = 8, max_answer_words: int = 11
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            required = (
                "question_id", "video_id", "video_path", "question", "answer", "fps",
                "frame_indices", "text_evidence", "pipeline",
            )
            missing = [key for key in required if key not in record]
            if missing:
                raise ValueError(f"Incomplete record at {path}:{line_number}: {missing}")
            pipeline = record["pipeline"]
            fixed = {
                "method": EXPECTED_PIPELINE,
                "profile": EXPECTED_PROFILE,
                "topk": TOPK,
                "decode_max_side": DECODE_MAX_SIDE,
                "min_visual_tokens": MIN_VISUAL_TOKENS,
                "max_visual_tokens": MAX_VISUAL_TOKENS,
                "max_text_evidence_items": profile.MAX_TEXT_EVIDENCE_ITEMS,
                "max_evidence_chars": profile.MAX_EVIDENCE_CHARS,
                "evidence_temporal_window_seconds": profile.EVIDENCE_TEMPORAL_WINDOW_SECONDS,
                "prompt": "legacy-exact-inline-v1",
            }
            mismatches = {
                key: (expected, pipeline.get(key))
                for key, expected in fixed.items() if pipeline.get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    f"Pipeline mismatch for {record['question_id']}: {mismatches}"
                )
            frames = [int(value) for value in record["frame_indices"]]
            if frames != sorted(set(frames)) or not MIN_FRAMES <= len(frames) <= TOPK:
                raise ValueError(f"Invalid frames for {record['question_id']}")
            record["frame_indices"] = frames
            if float(record["fps"]) <= 0:
                raise ValueError(f"Invalid FPS for {record['question_id']}")
            if not Path(record["video_path"]).is_file():
                raise FileNotFoundError(record["video_path"])
            answer_words = len(str(record["answer"]).split())
            if not min_answer_words <= answer_words <= max_answer_words:
                raise ValueError(
                    f"Answer is not {min_answer_words}-{max_answer_words} words for "
                    f"{record['question_id']}"
                )
            records.append(record)
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    ids = [str(record["question_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate question_id values")
    return records


def split_by_video(
    records: list[dict[str, Any]], validation_ratio: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    videos = sorted({str(record["video_id"]) for record in records})
    if validation_ratio <= 0 or len(videos) < 2:
        return records, []
    generator = random.Random(seed)
    generator.shuffle(videos)
    validation_count = min(
        len(videos) - 1, max(1, round(validation_ratio * len(videos)))
    )
    validation_videos = set(videos[:validation_count])
    return (
        [record for record in records if record["video_id"] not in validation_videos],
        [record for record in records if record["video_id"] in validation_videos],
    )


class ManifestDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def decode_record(record: dict[str, Any]) -> tuple[dict[int, Any], float]:
    import cv2
    from PIL import Image

    frame_map, actual_fps = inference.decode_frame_map(
        Path(record["video_path"]),
        [int(value) for value in record["frame_indices"]],
        DECODE_MAX_SIDE,
        cv2,
        Image,
    )
    expected_fps = float(record["fps"])
    if abs(actual_fps - expected_fps) > max(1e-3, expected_fps * 1e-4):
        raise RuntimeError(
            f"{record['question_id']}: FPS changed from {expected_fps} to {actual_fps}"
        )
    if len(frame_map) != len(record["frame_indices"]):
        raise RuntimeError(
            f"{record['question_id']}: decoded {len(frame_map)}/"
            f"{len(record['frame_indices'])} frames"
        )
    return frame_map, actual_fps


def build_user_content(
    record: dict[str, Any], frame_map: dict[int, Any], fps: float
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for frame_index in record["frame_indices"]:
        if frame_index not in frame_map:
            raise RuntimeError(
                f"{record['question_id']}: frame {frame_index} could not be decoded"
            )
        content.append(
            {"type": "text", "text": f"Visual evidence at {frame_index / fps:.1f}s:"}
        )
        content.append({"type": "image", "image": frame_map[frame_index]})
    content.append(
        {
            "type": "text",
            "text": inference.render_legacy_exact_prompt(
                record, list(record.get("text_evidence", []))
            ),
        }
    )
    return content


class LegacyExactFrameCollator:
    def __init__(
        self, processor: Any, process_vision_info: Any, max_sequence_length: int
    ) -> None:
        self.processor = processor
        self.process_vision_info = process_vision_info
        self.max_sequence_length = max_sequence_length

    def process(
        self,
        messages: list[dict[str, Any]],
        image_inputs: Any,
        video_inputs: Any,
        *,
        add_generation_prompt: bool,
    ) -> Any:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if len(features) != 1:
            raise ValueError("Frame QLoRA requires per-device batch size 1")
        record = features[0]
        frame_map, fps = decode_record(record)
        user_messages = [
            {"role": "user", "content": build_user_content(record, frame_map, fps)}
        ]
        full_messages = [
            *user_messages,
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(record["answer"])}],
            },
        ]
        image_inputs, video_inputs = self.process_vision_info(user_messages)
        if video_inputs:
            raise RuntimeError(f"Unexpected video input for {record['question_id']}")
        if not image_inputs or len(image_inputs) != len(record["frame_indices"]):
            raise RuntimeError(
                f"{record['question_id']}: processor received "
                f"{len(image_inputs or [])}/{len(record['frame_indices'])} images"
            )
        prompt_inputs = self.process(
            user_messages, image_inputs, video_inputs, add_generation_prompt=True
        )
        full_inputs = self.process(
            full_messages, image_inputs, video_inputs, add_generation_prompt=False
        )
        input_ids = full_inputs["input_ids"]
        prompt_ids = prompt_inputs["input_ids"]
        if input_ids.shape[1] > self.max_sequence_length:
            raise ValueError(
                f"{record['question_id']}: {input_ids.shape[1]} tokens exceed "
                f"--max-sequence-length={self.max_sequence_length}"
            )
        common = min(input_ids.shape[1], prompt_ids.shape[1])
        difference = (input_ids[0, :common] != prompt_ids[0, :common]).nonzero()
        if difference.numel():
            common = int(difference[0].item())
        if common < prompt_ids.shape[1] - 2:
            raise RuntimeError(
                f"Could not locate assistant boundary for {record['question_id']}"
            )
        labels = input_ids.clone()
        labels[:, :common] = -100
        if "attention_mask" in full_inputs:
            labels[full_inputs["attention_mask"] == 0] = -100
        batch = {
            key: value for key, value in full_inputs.items() if torch.is_tensor(value)
        }
        batch["labels"] = labels
        frame_map.clear()
        return batch


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import (
        AutoConfig,
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
    )

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if config.model_type != "qwen3_vl":
        raise ValueError(f"Expected qwen3_vl, got {config.model_type!r}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    device_index = local_rank if local_rank >= 0 else 0
    kwargs: dict[str, Any] = {
        "config": config,
        "dtype": dtype,
        "attn_implementation": args.attention,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        "trust_remote_code": False,
        "device_map": {"": device_index},
    }
    if args.quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **kwargs)
    model.config.use_cache = False

    vision = config.vision_config
    token_side = int(vision.patch_size) * int(vision.spatial_merge_size)
    pixel_area = token_side**2
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=MIN_VISUAL_TOKENS * pixel_area,
        max_pixels=MAX_VISUAL_TOKENS * pixel_area,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    return model, processor


def add_lora(model: Any, args: argparse.Namespace) -> Any:
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    if args.quantization == "4bit":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    model.config.use_cache = False
    model.enable_input_require_grads()
    if args.init_adapter is not None:
        model = PeftModel.from_pretrained(
            model, str(args.init_adapter), is_trainable=True
        )
    else:
        target_modules = (
            r"model\.language_model\.layers\.\d+\.(?:self_attn\."
            r"(?:q_proj|k_proj|v_proj|o_proj)|mlp\."
            r"(?:gate_proj|up_proj|down_proj))"
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            ),
        )
    model.print_trainable_parameters()
    return model


def create_answer_only_trainer(trainer_base: Any) -> Any:
    class AnswerOnlyTrainer(trainer_base):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, torch.Tensor],
            return_outputs: bool = False,
            num_items_in_batch: torch.Tensor | None = None,
        ) -> Any:
            inputs = dict(inputs)
            labels = inputs.pop("labels")
            if labels.ndim != 2 or labels.shape[0] != 1:
                raise ValueError("Answer-only logits require batch size 1")
            positions = (labels[0] != -100).nonzero(as_tuple=False).flatten()
            if positions.numel() == 0 or int(positions.min()) <= 0:
                raise ValueError("Batch has no causal answer targets")
            logits_positions = [int(value) - 1 for value in positions.tolist()]
            outputs = model(**inputs, logits_to_keep=logits_positions)
            targets = labels.index_select(1, positions).to(outputs.logits.device)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits.reshape(-1, outputs.logits.shape[-1]).float(),
                targets.reshape(-1),
                reduction="sum",
                label_smoothing=float(self.args.label_smoothing_factor),
            )
            denominator: Any = max(1, targets.numel())
            if num_items_in_batch is not None:
                denominator = (
                    num_items_in_batch.to(loss.device, loss.dtype)
                    if torch.is_tensor(num_items_in_batch)
                    else max(1, int(num_items_in_batch))
                )
            loss = loss / denominator
            return (loss, outputs) if return_outputs else loss

    return AnswerOnlyTrainer


def run_decode_preflight(records: list[dict[str, Any]], count: int) -> None:
    selected = records if count == 0 else records[:count]
    for position, record in enumerate(selected, start=1):
        frame_map, fps = decode_record(record)
        content = build_user_content(record, frame_map, fps)
        images = [item for item in content if item.get("type") == "image"]
        if len(images) != len(record["frame_indices"]):
            raise RuntimeError(f"Image count mismatch for {record['question_id']}")
        frame_map.clear()
        print(
            f"Decode preflight [{position}/{len(selected)}] {record['question_id']}: "
            f"frames={len(images)} text_evidence={len(record['text_evidence'])}",
            flush=True,
        )
    print(f"Decode preflight passed for {len(selected)} samples")


def save_dataset_summary(
    args: argparse.Namespace,
    all_records: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "dataset_split.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "pipeline": EXPECTED_PIPELINE,
        "profile": EXPECTED_PROFILE,
        "model": str(args.model.resolve()),
        "distributed": {
            "strategy": "ddp-full-model-per-rank",
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        },
        "image_profile": {
            "topk": TOPK,
            "decode_max_side": DECODE_MAX_SIDE,
            "min_visual_tokens": MIN_VISUAL_TOKENS,
            "max_visual_tokens": MAX_VISUAL_TOKENS,
        },
        "training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "validation_ratio": args.validation_ratio,
            "seed": args.seed,
        },
        "all_samples": len(all_records),
        "all_videos": len({record["video_id"] for record in all_records}),
        "train_question_ids": [record["question_id"] for record in train_records],
        "validation_question_ids": [
            record["question_id"] for record in validation_records
        ],
        "train_videos": sorted({record["video_id"] for record in train_records}),
        "validation_videos": sorted(
            {record["video_id"] for record in validation_records}
        ),
    }
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    from transformers import Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    set_seed(args.seed)
    records = read_manifest(
        args.manifest, args.min_answer_words, args.max_answer_words
    )
    if args.sample_index is not None:
        if not 0 <= args.sample_index < len(records):
            raise IndexError(f"--sample-index must be in [0, {len(records) - 1}]")
        records = [records[args.sample_index]]
    if args.max_samples is not None:
        records = records[:args.max_samples]
    if args.preflight_only:
        run_decode_preflight(records, args.preflight_samples)
        return

    device_index = validate_gpu_runtime(args)
    rank = int(os.environ.get("RANK", "0"))
    train_records, validation_records = split_by_video(
        records, args.validation_ratio, args.seed
    )
    if {r["video_id"] for r in train_records} & {
        r["video_id"] for r in validation_records
    }:
        raise RuntimeError("Video leakage between training and validation splits")
    if rank == 0:
        print(
            f"Dataset: all={len(records)}/{len({r['video_id'] for r in records})} videos; "
            f"train={len(train_records)}/{len({r['video_id'] for r in train_records})}; "
            f"validation={len(validation_records)}/"
            f"{len({r['video_id'] for r in validation_records})}",
            flush=True,
        )
    save_dataset_summary(args, records, train_records, validation_records)
    from qwen_vl_utils import process_vision_info

    torch.cuda.reset_peak_memory_stats(device_index)
    model, processor = load_model_and_processor(args)
    model = add_lora(model, args)
    collator = LegacyExactFrameCollator(
        processor, process_vision_info, args.max_sequence_length
    )
    use_bf16 = args.dtype == "bf16"
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing_factor=args.label_smoothing_factor,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if args.quantization == "4bit" else "adamw_torch",
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if validation_records else "no",
        eval_steps=args.eval_steps if validation_records else None,
        load_best_model_at_end=bool(validation_records),
        metric_for_best_model="eval_loss" if validation_records else None,
        greater_is_better=False if validation_records else None,
        prediction_loss_only=True,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
        local_rank=int(os.environ.get("LOCAL_RANK", args.local_rank)),
    )
    callbacks = []
    if validation_records and args.early_stopping_patience > 0:
        from transformers import EarlyStoppingCallback

        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )
    trainer_class = create_answer_only_trainer(Trainer)
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=ManifestDataset(train_records),
        eval_dataset=(
            ManifestDataset(validation_records) if validation_records else None
        ),
        data_collator=collator,
        processing_class=processor,
        callbacks=callbacks,
    )
    if args.eval_only:
        if not validation_records:
            raise ValueError("--eval-only requires validation records")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        return
    if args.eval_before_training and validation_records:
        initial_metrics = trainer.evaluate(metric_key_prefix="initial_eval")
        trainer.log_metrics("initial_eval", initial_metrics)
        trainer.save_metrics("initial_eval", initial_metrics)
    checkpoint: str | bool | None = None
    if args.resume_from_checkpoint:
        checkpoint = (
            get_last_checkpoint(str(args.output_dir))
            if args.resume_from_checkpoint == "auto"
            else args.resume_from_checkpoint
        )
        if args.resume_from_checkpoint == "auto" and checkpoint is None:
            print("No checkpoint found; starting a new run.")
    trainer.train(resume_from_checkpoint=checkpoint)
    if validation_records:
        metrics = trainer.evaluate(metric_key_prefix="final_eval")
        trainer.log_metrics("final_eval", metrics)
        trainer.save_metrics("final_eval", metrics)
    trainer.save_model(str(args.output_dir))
    if rank == 0:
        processor.save_pretrained(args.output_dir)
        print(f"LoRA adapter and processor saved to {args.output_dir}")
    print(
        f"Rank{rank}/GPU{device_index} peak allocated="
        f"{torch.cuda.max_memory_allocated(device_index) / 1024**3:.2f} GiB, "
        f"peak reserved="
        f"{torch.cuda.max_memory_reserved(device_index) / 1024**3:.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
