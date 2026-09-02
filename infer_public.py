#!/usr/bin/env python3
"""Self-contained legacy-exact public/private inference with Qwen3-VL-8B."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DATA_ROOT = Path("/data/wen1/datasets/slomo_qa")
DEFAULT_QUESTIONS = DATA_ROOT / "sf20k_public_test_questions.csv"
DEFAULT_VIDEO_DIR = DATA_ROOT / "sf20k_public_test_videos"
DEFAULT_SELECTIONS = HERE / "inputs/public_selections.jsonl"
DEFAULT_EVIDENCE = HERE / "inputs/public_evidence_ocr_clean.jsonl"
DEFAULT_OUTPUT = HERE / "outputs/qwen3vl8b_frame_legacy_exact_2.json"
PRIVATE_QUESTIONS = DATA_ROOT / "sf20k_private_test_questions.csv"
PRIVATE_VIDEO_DIR = DATA_ROOT / "sf20k_private_test_videos"
PRIVATE_SELECTIONS = HERE / "inputs/private_selections.jsonl"
PRIVATE_EVIDENCE = HERE / "inputs/private_evidence_ocr_clean.jsonl"
PRIVATE_OUTPUT = HERE / "outputs/qwen3vl8b_frame_legacy_exact_private.json"

EXPECTED_EVIDENCE_SHA256 = (
    "20a20f417927402a108638605914a8e5b6731bfad4773d70e1d0e5a2a8fc7798"
)
EXPECTED_SELECTIONS_SHA256 = (
    "8bf79b8f11287c14f6fd93bf259ee29637a98eb22a0b680219249e5daf35d5e5"
)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "how", "in",
    "is", "it", "of", "on", "or", "she", "that", "the", "they", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "with",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, on_error) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        on_error(f"Could not read JSON file {path}: {error}")
    if not isinstance(payload, dict):
        on_error(f"Expected a JSON object in {path}")
    return payload


def checkpoint_has_weights(path: Path) -> bool:
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shards = set(index["weight_map"].values())
        except (KeyError, OSError, TypeError, json.JSONDecodeError):
            return False
        return bool(shards) and all((path / shard).is_file() for shard in shards)
    return (path / "model.safetensors").is_file()


def choose_default_qwen() -> Path:
    override = os.environ.get("QWEN3_VL_PATH")
    if override:
        return Path(override).expanduser()
    candidates = (
        Path("/data/wen1/MLLMs/qwen3vl8b"),
        Path("/data/dudu/cache/modelscope/Qwen3-VL-8B-Instruct"),
    )
    return next(
        (path for path in candidates if checkpoint_has_weights(path)), candidates[0]
    )


def require_qwen3_checkpoint(qwen_path: Path, on_error) -> None:
    config_path = qwen_path / "config.json"
    if not config_path.is_file():
        on_error(f"Qwen checkpoint config not found: {config_path}")
    config = load_json(config_path, on_error)
    if config.get("model_type") != "qwen3_vl":
        on_error(
            f"--qwen-path must be a Qwen3-VL checkpoint; "
            f"found model_type={config.get('model_type')!r} in {config_path}"
        )

    index_path = qwen_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = load_json(index_path, on_error)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            on_error(f"Invalid or empty weight_map in {index_path}")
        missing = sorted(
            qwen_path / name
            for name in set(weight_map.values())
            if not (qwen_path / name).is_file()
        )
        if missing:
            on_error("Missing Qwen weight shards: " + ", ".join(map(str, missing)))
    elif not (qwen_path / "model.safetensors").is_file():
        on_error(
            f"No complete safetensors weights found under {qwen_path}. "
            "Files ending in .incomplete cannot be used."
        )

    processor_files = (
        "chat_template.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
    )
    missing_processor = [
        qwen_path / name for name in processor_files if not (qwen_path / name).is_file()
    ]
    if missing_processor:
        on_error(
            "Incomplete Qwen processor/tokenizer files; missing: "
            + ", ".join(map(str, missing_processor))
        )


def require_adapter(lora_path: Path, on_error) -> None:
    config_path = lora_path / "adapter_config.json"
    weights_path = lora_path / "adapter_model.safetensors"
    missing = [path for path in (config_path, weights_path) if not path.is_file()]
    if missing:
        on_error("Incomplete --lora-path; missing: " + ", ".join(map(str, missing)))
    config = load_json(config_path, on_error)
    base_name = str(config.get("base_model_name_or_path", "")).lower()
    if "qwen2" in base_name or "qwen_25" in base_name or "qwen-25" in base_name:
        on_error(
            "This adapter targets Qwen2/2.5-VL and is incompatible with Qwen3-VL: "
            f"{lora_path}"
        )


def require_exact_input(path: Path, expected: str, label: str, on_error) -> None:
    if not path.is_file():
        on_error(f"Missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        on_error(
            f"{label} is not the fixed legacy-exact input: expected sha256={expected}, "
            f"actual={actual}, path={path}"
        )


def configure_visible_gpus(gpu_ids: str, on_error) -> None:
    values = [value.strip() for value in gpu_ids.split(",")]
    if not values or any(not value.isdigit() for value in values):
        on_error("--gpu-ids must look like 0 or 0,1")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer a SloMo-QA split with Qwen3-VL-8B and folder-local "
            "legacy-exact Top32 selections and OCR-clean evidence. Public "
            "inputs remain hash checked; private inputs are coverage checked."
        )
    )
    parser.add_argument("--split", choices=("public", "private"), default="public")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument(
        "--allow-custom-public-inputs",
        action="store_true",
        help=(
            "Allow explicitly supplied public selections/evidence without the "
            "legacy-exact SHA256 check. Coverage checks still run."
        ),
    )
    parser.add_argument(
        "--qwen-path", type=Path, default=choose_default_qwen(),
        help="Local Qwen3-VL Instruct checkpoint (or set QWEN3_VL_PATH).",
    )
    parser.add_argument(
        "--lora-path", type=Path,
        help="Optional LoRA trained specifically from the selected Qwen3-VL base.",
    )
    parser.add_argument("--output", type=Path)

    image = parser.add_argument_group("image/frame parameters")
    image.add_argument("--topk", type=int, default=32,
                       help="Maximum selected frames supplied per question.")
    image.add_argument("--min-answer-frames", type=int, default=15,
                       help="Minimum frame count after CUDA OOM fallback.")
    image.add_argument("--decode-max-side", type=int, default=1024,
                       help="Downscale decoded frames to this maximum side.")
    image.add_argument("--min-visual-tokens", type=int, default=512,
                       help="Minimum Qwen3-VL visual tokens per image.")
    image.add_argument("--max-visual-tokens", type=int, default=512,
                       help="Maximum Qwen3-VL visual tokens per image.")

    parser.add_argument("--max-text-evidence-items", type=int, default=25)
    parser.add_argument("--max-evidence-chars", type=int, default=15000)
    parser.add_argument("--evidence-temporal-window-seconds", type=float, default=20.0)
    parser.add_argument(
        "--quantization", choices=("4bit", "8bit", "none"), default="4bit"
    )
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--attention", choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    if args.split == "public":
        args.questions = args.questions or DEFAULT_QUESTIONS
        args.video_dir = args.video_dir or DEFAULT_VIDEO_DIR
        args.selections = args.selections or DEFAULT_SELECTIONS
        args.evidence = args.evidence or DEFAULT_EVIDENCE
        args.output = args.output or DEFAULT_OUTPUT
    else:
        args.questions = args.questions or PRIVATE_QUESTIONS
        args.video_dir = args.video_dir or PRIVATE_VIDEO_DIR
        args.selections = args.selections or PRIVATE_SELECTIONS
        args.evidence = args.evidence or PRIVATE_EVIDENCE
        args.output = args.output or PRIVATE_OUTPUT

    positive = {
        "--topk": args.topk,
        "--min-answer-frames": args.min_answer_frames,
        "--decode-max-side": args.decode_max_side,
        "--min-visual-tokens": args.min_visual_tokens,
        "--max-visual-tokens": args.max_visual_tokens,
        "--max-text-evidence-items": args.max_text_evidence_items,
        "--max-evidence-chars": args.max_evidence_chars,
        "--evidence-temporal-window-seconds": args.evidence_temporal_window_seconds,
        "--max-new-tokens": args.max_new_tokens,
    }
    if args.max_samples is not None:
        positive["--max-samples"] = args.max_samples
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.min_answer_frames > args.topk:
        parser.error("--min-answer-frames cannot exceed --topk")
    if args.min_visual_tokens > args.max_visual_tokens:
        parser.error("--min-visual-tokens cannot exceed --max-visual-tokens")
    if not args.questions.is_file():
        parser.error(f"--questions not found: {args.questions}")
    if not args.video_dir.is_dir():
        parser.error(f"--video-dir not found: {args.video_dir}")
    if args.split == "public" and not args.allow_custom_public_inputs:
        require_exact_input(
            args.selections, EXPECTED_SELECTIONS_SHA256, "MMR selections", parser.error
        )
        require_exact_input(
            args.evidence, EXPECTED_EVIDENCE_SHA256, "cleaned evidence", parser.error
        )
    else:
        if not args.selections.is_file():
            parser.error(f"Missing {args.split} MMR selections: {args.selections}")
        if not args.evidence.is_file():
            parser.error(f"Missing {args.split} cleaned evidence: {args.evidence}")
    require_qwen3_checkpoint(args.qwen_path, parser.error)
    if args.lora_path is not None:
        require_adapter(args.lora_path, parser.error)
    configure_visible_gpus(args.gpu_ids, parser.error)
    return args


def load_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    with args.questions.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"question_id", "video_id", "question"}
    if not rows or not required <= rows[0].keys():
        raise ValueError(f"CSV must contain {sorted(required)}")
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    missing = [
        row["video_id"] for row in rows
        if not (args.video_dir / f'{row["video_id"]}.mp4').is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(set(missing))} videos under {args.video_dir}; "
            f"first={missing[0]}.mp4"
        )
    return rows


def load_jsonl_by(path: Path, key: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            value = record.get(key)
            if value is not None:
                records[str(value)] = record
    return records


def group_by_video(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)
    return grouped


def lexical_terms(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9']+", text.lower())
        if token not in STOP_WORDS and len(token) > 1
    }


def select_text_evidence(
    question: str,
    selected_timestamps: list[float],
    video_evidence: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    query_terms = lexical_terms(question)
    candidates: list[tuple[float, dict[str, Any]]] = []
    source_bonus = {"subtitles": 0.15, "asr": 0.10, "ocr": 0.05}
    for group in ("subtitles", "asr", "ocr"):
        for item in video_evidence.get(group, []):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            overlap = len(query_terms & lexical_terms(text)) / max(1, len(query_terms))
            midpoint = (float(item.get("start", 0)) + float(item.get("end", 0))) / 2
            distance = min(
                (abs(midpoint - timestamp) for timestamp in selected_timestamps),
                default=float("inf"),
            )
            proximity = (
                1.0 / (1.0 + distance / 5.0)
                if distance <= args.evidence_temporal_window_seconds else 0.0
            )
            enriched = dict(item)
            enriched["evidence_type"] = group[:-1] if group.endswith("s") else group
            candidates.append((2.5 * overlap + proximity + source_bonus[group], enriched))
    candidates.sort(key=lambda pair: (-pair[0], float(pair[1].get("start", 0))))

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    characters = 0
    for _, item in candidates:
        normalized = " ".join(str(item["text"]).lower().split())
        if normalized in seen:
            continue
        rendered_length = len(str(item["text"])) + 40
        if selected and characters + rendered_length > args.max_evidence_chars:
            continue
        seen.add(normalized)
        selected.append(item)
        characters += rendered_length
        if len(selected) >= args.max_text_evidence_items:
            break
    return sorted(selected, key=lambda item: float(item.get("start", 0)))


def format_text_evidence(records: list[dict[str, Any]]) -> str:
    if not records:
        return "No ASR, subtitle, or OCR evidence was retrieved."
    lines: list[str] = []
    for item in records:
        source = str(item.get("evidence_type", item.get("source", "text"))).upper()
        start, end = float(item.get("start", 0)), float(item.get("end", 0))
        stamp = f"{start:.1f}-{end:.1f}s" if end > start else f"{start:.1f}s"
        lines.append(f"[{source} {stamp}] {item['text']}")
    return "\n".join(lines)


def question_answer_instruction(question: str) -> str:
    lowered = question.lower().strip()
    if re.match(r"^(who|where|when|which)\b", lowered):
        return "Return only the requested entity, location, time, or choice, normally 1-8 words."
    if re.match(
        r"^(is|are|was|were|do|does|did|can|could|will|would|has|have|had)\b",
        lowered,
    ):
        return "Begin with Yes or No when applicable, followed by at most one short clause."
    if re.match(r"^(why|how)\b", lowered):
        return "Return one concise causal or procedural clause, normally no more than 15 words."
    return "Return one exact short answer, normally no more than 12 words."


def render_legacy_exact_prompt(
    row: dict[str, str], text_records: list[dict[str, Any]]
) -> str:
    title = str(row.get("movie_title", "")).strip()
    return (
        f"Movie title metadata: {title or 'unknown'}\n"
        "Timestamped ASR/subtitle/OCR evidence:\n"
        f"{format_text_evidence(text_records)}\n\n"
        f"Question: {row['question']}\n"
        f"{question_answer_instruction(row['question'])}\n"
        "Return only the answer with no prefix or explanation."
    )


def load_qwen3vl(args: argparse.Namespace, torch: Any) -> tuple[Any, Any]:
    from transformers import (
        AutoConfig,
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
    )
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    config = AutoConfig.from_pretrained(
        args.qwen_path, local_files_only=True, trust_remote_code=False
    )
    if config.model_type != "qwen3_vl":
        raise ValueError(f"Expected qwen3_vl, got {config.model_type!r}")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    kwargs: dict[str, Any] = {
        "config": config,
        "dtype": dtype,
        "device_map": "auto",
        "attn_implementation": args.attention,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if args.quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif args.quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.qwen_path, **kwargs
    ).eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    vision_config = config.vision_config
    token_side = int(vision_config.patch_size) * int(vision_config.spatial_merge_size)
    pixel_area = token_side**2
    processor = AutoProcessor.from_pretrained(
        args.qwen_path,
        min_pixels=args.min_visual_tokens * pixel_area,
        max_pixels=args.max_visual_tokens * pixel_area,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    return model, processor


def model_input_device(model: Any, torch: Any) -> Any:
    try:
        device = model.get_input_embeddings().weight.device
        if device.type != "meta":
            return device
    except (AttributeError, RuntimeError):
        pass
    return torch.device("cuda:0")


def decode_frame_map(
    video_path: Path,
    indices: list[int],
    max_side: int,
    cv2: Any,
    image_cls: Any,
) -> tuple[dict[int, Any], float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: dict[int, Any] = {}
    try:
        for index in sorted(set(indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, max_side / max(height, width))
            if scale < 1:
                frame = cv2.resize(
                    frame,
                    (round(width * scale), round(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            frames[index] = image_cls.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames, fps


def generate_answer(
    row: dict[str, str],
    frames: list[Any],
    timestamps: list[float],
    text_records: list[dict[str, Any]],
    model: Any,
    processor: Any,
    process_vision_info: Any,
    torch: Any,
    args: argparse.Namespace,
) -> str:
    content: list[dict[str, Any]] = []
    for frame, timestamp in zip(frames, timestamps):
        content.append({"type": "text", "text": f"Visual evidence at {timestamp:.1f}s:"})
        content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": render_legacy_exact_prompt(row, text_records)})
    messages = [{"role": "user", "content": content}]
    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model_input_device(model, torch))
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    new_tokens = generated[0, inputs.input_ids.shape[1]:]
    answer = processor.decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    answer = re.sub(r"^\s*(?:answer|prediction)\s*:\s*", "", answer, flags=re.I)
    answer = " ".join(answer.replace("```", "").split())
    if not answer:
        raise ValueError("Qwen returned an empty answer")
    return answer


def file_identity(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        stat = resolved.stat()
        return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if resolved.is_dir():
        config = resolved / "config.json"
        stat = config.stat() if config.is_file() else resolved.stat()
        return {"path": str(resolved), "config_mtime_ns": stat.st_mtime_ns}
    return {"path": str(resolved), "missing": True}


def inference_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": "qwen3vl-legacy-exact-standalone-v1",
        "split": args.split,
        "qwen": file_identity(args.qwen_path),
        "lora": file_identity(args.lora_path),
        "questions": file_identity(args.questions),
        "selections": file_identity(args.selections),
        "evidence": file_identity(args.evidence),
        "topk": args.topk,
        "min_answer_frames": args.min_answer_frames,
        "decode_max_side": args.decode_max_side,
        "min_visual_tokens": args.min_visual_tokens,
        "max_visual_tokens": args.max_visual_tokens,
        "max_text_evidence_items": args.max_text_evidence_items,
        "max_evidence_chars": args.max_evidence_chars,
        "evidence_temporal_window_seconds": args.evidence_temporal_window_seconds,
        "quantization": args.quantization,
        "dtype": args.dtype,
        "attention": args.attention,
        "max_new_tokens": args.max_new_tokens,
    }


def metadata_path(output: Path) -> Path:
    return output.with_suffix(".meta.json")


def load_predictions(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Predictions must be a JSON list: {path}")
    return {
        str(item["question_id"]): str(item["prediction"]).strip()
        for item in data
        if item.get("question_id") and str(item.get("prediction", "")).strip()
    }


def load_cached_predictions(args: argparse.Namespace) -> dict[str, str]:
    if not args.resume:
        return {}
    meta_path = metadata_path(args.output)
    if not meta_path.is_file():
        if args.output.is_file():
            print("Answer cache ignored because its metadata sidecar is missing.")
        return {}
    try:
        cached_config = json.loads(meta_path.read_text(encoding="utf-8"))
        if cached_config != inference_config(args):
            print("Answer cache ignored because the inference configuration changed.")
            return {}
        return load_predictions(args.output)
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
        print(f"Answer cache ignored because predictions are invalid: {error}")
        return {}


def write_answer_cache(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    predictions: dict[str, str],
) -> None:
    result = [
        {"question_id": row["question_id"], "prediction": predictions[row["question_id"]]}
        for row in rows if row["question_id"] in predictions
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, args.output)

    meta_path = metadata_path(args.output)
    temporary_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    temporary_meta.write_text(
        json.dumps(inference_config(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_meta, meta_path)


def run_inference(args: argparse.Namespace) -> None:
    import cv2
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    rows = load_rows(args)
    selections = load_jsonl_by(args.selections, "question_id")
    evidence = load_jsonl_by(args.evidence, "video_id")
    missing_selections = [
        row["question_id"] for row in rows
        if not selections.get(row["question_id"], {}).get("mmr_ranked_frame_indices")
    ]
    missing_evidence = sorted({
        row["video_id"] for row in rows if row["video_id"] not in evidence
    })
    if missing_selections:
        raise RuntimeError(
            f"MMR cache lacks {len(missing_selections)} requested questions; "
            f"first={missing_selections[0]}"
        )
    if missing_evidence:
        raise RuntimeError(
            f"Evidence cache lacks {len(missing_evidence)} requested videos; "
            f"first={missing_evidence[0]}"
        )
    predictions = load_cached_predictions(args)
    valid_ids = {row["question_id"] for row in rows}
    predictions = {key: value for key, value in predictions.items() if key in valid_ids}
    pending = [
        row for row in rows
        if row["question_id"] not in predictions
        and selections.get(row["question_id"], {}).get("mmr_ranked_frame_indices")
    ]
    print(
        f"Answer: questions={len(rows)} cached={len(predictions)} "
        f"pending={len(pending)} missing_mmr=0 evidence_videos={len(evidence)}"
    )
    if not pending:
        write_answer_cache(args, rows, predictions)
        return

    model, processor = load_qwen3vl(args, torch)
    if args.lora_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, str(args.lora_path), is_trainable=False
        ).eval()
        print(f"Loaded LoRA adapter: {args.lora_path}", flush=True)

    grouped = group_by_video(pending)
    errors = 0
    for video_position, (video_id, questions) in enumerate(grouped.items(), start=1):
        union_indices = sorted({
            int(index)
            for row in questions
            for index in selections[row["question_id"]]["mmr_ranked_frame_indices"][:args.topk]
        })
        video_path = args.video_dir / f"{video_id}.mp4"
        try:
            frame_map, fps = decode_frame_map(
                video_path, union_indices, args.decode_max_side, cv2, Image
            )
            for row in questions:
                ranked = [
                    int(index)
                    for index in selections[row["question_id"]]["mmr_ranked_frame_indices"]
                ]
                budget = min(args.topk, len(ranked))
                while True:
                    chosen = sorted(index for index in ranked[:budget] if index in frame_map)
                    timestamps = [index / fps for index in chosen]
                    text_records = select_text_evidence(
                        row["question"], timestamps, evidence.get(video_id, {}), args
                    )
                    try:
                        predictions[row["question_id"]] = generate_answer(
                            row,
                            [frame_map[index] for index in chosen],
                            timestamps,
                            text_records,
                            model,
                            processor,
                            process_vision_info,
                            torch,
                            args,
                        )
                        write_answer_cache(args, rows, predictions)
                        break
                    except RuntimeError as error:
                        next_budget = max(args.min_answer_frames, budget // 2)
                        if "out of memory" not in str(error).lower() or next_budget >= budget:
                            raise
                        budget = next_budget
                        gc.collect()
                        torch.cuda.empty_cache()
                print(
                    f"[{len(predictions)}/{len(rows)}] {row['question_id']}: "
                    f"frames={len(chosen)} text_evidence={len(text_records)}",
                    flush=True,
                )
            print(
                f"video [{video_position}/{len(grouped)}] {video_path.name}: "
                f"questions={len(questions)} decoded_union={len(frame_map)}",
                flush=True,
            )
            frame_map.clear()
        except Exception as error:
            errors += len(questions)
            print(
                f"ERROR answer {video_path.name}: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    write_answer_cache(args, rows, predictions)
    print(f"Predictions written to {args.output}; errors={errors}")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(f"Qwen3-VL checkpoint: {args.qwen_path.resolve()}", flush=True)
    print(f"LoRA adapter: {args.lora_path.resolve() if args.lora_path else 'none'}", flush=True)
    print(
        f"Images: topk={args.topk} decode_max_side={args.decode_max_side} "
        f"visual_tokens={args.min_visual_tokens}-{args.max_visual_tokens}",
        flush=True,
    )
    run_inference(args)


if __name__ == "__main__":
    main()
