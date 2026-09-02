#!/usr/bin/env python3
"""Replace only OCR with Base Qwen3-VL, reuse ASR/subtitles, clean, and answer."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# This experiment lives in experiments/, while its reusable inference and OCR
# cleaning modules live at the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import infer_public as inference
from ocr_clean_core import clean_ocr_only, compact_text


HERE = PROJECT_ROOT
DATA_ROOT = Path("/data/wen1/datasets/slomo_qa")
PYTHON = Path("/data/miniconda3/envs/egoVqa/bin/python")
DEFAULT_QWEN3 = Path("/data/wen1/MLLMs/qwen3vl8b")
CLEANING_METHOD = "strict-obvious-ocr-noise-only-v1"


def split_defaults(split: str) -> dict[str, Path]:
    return {
        "questions": DATA_ROOT / f"sf20k_{split}_test_questions.csv",
        "video_dir": DATA_ROOT / f"sf20k_{split}_test_videos",
        "selections": HERE / f"inputs/{split}_selections.jsonl",
        "source_evidence": HERE / f"inputs/{split}_evidence_ocr_clean.jsonl",
        "raw_evidence": HERE / f"inputs/qwen3_ocr/{split}_evidence_raw.jsonl",
        "cleaned_evidence": (
            HERE / f"inputs/qwen3_ocr/{split}_evidence_ocr_clean.jsonl"
        ),
        "output": HERE / f"outputs/compare/qwen3vl8b_base_qwen3ocr_{split}.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Base Qwen3-VL OCR over existing MMR frames while reusing prior "
            "ASR/subtitles, apply folder-local strict OCR cleaning, then optionally "
            "run folder-local Qwen3-VL answer inference."
        )
    )
    parser.add_argument("--stage", choices=("all", "ocr", "clean", "answer"), default="all")
    parser.add_argument("--split", choices=("public", "private"), default="public")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--source-evidence", type=Path)
    parser.add_argument("--raw-evidence", type=Path)
    parser.add_argument("--cleaned-evidence", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qwen-path", type=Path, default=DEFAULT_QWEN3)
    parser.add_argument("--lora-path", type=Path)

    ocr = parser.add_argument_group("Base Qwen3-VL OCR settings")
    ocr.add_argument("--ocr-batch-size", type=int, default=2)
    ocr.add_argument("--ocr-max-frames-per-video", type=int, default=96)
    ocr.add_argument("--ocr-max-new-tokens", type=int, default=512)
    ocr.add_argument("--ocr-decode-max-side", type=int, default=1024)
    ocr.add_argument("--ocr-min-visual-tokens", type=int, default=256)
    ocr.add_argument("--ocr-max-visual-tokens", type=int, default=512)

    answer = parser.add_argument_group("Qwen3-VL answer settings")
    answer.add_argument("--topk", type=int, default=32)
    answer.add_argument("--min-answer-frames", type=int, default=12)
    answer.add_argument("--answer-decode-max-side", type=int, default=1024)
    answer.add_argument("--answer-min-visual-tokens", type=int, default=256)
    answer.add_argument("--answer-max-visual-tokens", type=int, default=512)
    answer.add_argument("--max-text-evidence-items", type=int, default=64)
    answer.add_argument("--max-evidence-chars", type=int, default=20000)
    answer.add_argument("--evidence-temporal-window-seconds", type=float, default=20.0)
    answer.add_argument("--answer-max-new-tokens", type=int, default=128)

    parser.add_argument("--quantization", choices=("4bit", "8bit", "none"), default="4bit")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--attention", choices=("sdpa", "flash_attention_2", "eager"), default="sdpa"
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and coverage without loading a model or writing outputs.",
    )
    args = parser.parse_args()

    defaults = split_defaults(args.split)
    for name, default in defaults.items():
        value = getattr(args, name)
        setattr(args, name, (value or default).expanduser().resolve())
    args.qwen_path = args.qwen_path.expanduser().resolve()
    if args.lora_path is not None:
        args.lora_path = args.lora_path.expanduser().resolve()

    positive = {
        "--ocr-batch-size": args.ocr_batch_size,
        "--ocr-max-frames-per-video": args.ocr_max_frames_per_video,
        "--ocr-max-new-tokens": args.ocr_max_new_tokens,
        "--ocr-decode-max-side": args.ocr_decode_max_side,
        "--ocr-min-visual-tokens": args.ocr_min_visual_tokens,
        "--ocr-max-visual-tokens": args.ocr_max_visual_tokens,
        "--topk": args.topk,
        "--min-answer-frames": args.min_answer_frames,
        "--answer-decode-max-side": args.answer_decode_max_side,
        "--answer-min-visual-tokens": args.answer_min_visual_tokens,
        "--answer-max-visual-tokens": args.answer_max_visual_tokens,
        "--max-text-evidence-items": args.max_text_evidence_items,
        "--max-evidence-chars": args.max_evidence_chars,
        "--evidence-temporal-window-seconds": args.evidence_temporal_window_seconds,
        "--answer-max-new-tokens": args.answer_max_new_tokens,
    }
    if args.max_samples is not None:
        positive["--max-samples"] = args.max_samples
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.ocr_min_visual_tokens > args.ocr_max_visual_tokens:
        parser.error("OCR min visual tokens cannot exceed max")
    if args.answer_min_visual_tokens > args.answer_max_visual_tokens:
        parser.error("Answer min visual tokens cannot exceed max")
    if args.min_answer_frames > args.topk:
        parser.error("--min-answer-frames cannot exceed --topk")

    required = [args.questions, args.video_dir, args.selections]
    if args.stage in {"all", "ocr"}:
        required.extend((args.source_evidence, args.qwen_path))
    if args.stage == "clean":
        required.append(args.raw_evidence)
    if args.stage == "answer":
        required.extend((args.cleaned_evidence, args.qwen_path))
    for path in required:
        if not path.exists():
            parser.error(f"Required input is missing: {path}")
    inference.require_qwen3_checkpoint(args.qwen_path, parser.error)
    if args.lora_path is not None:
        inference.require_adapter(args.lora_path, parser.error)
    inference.configure_visible_gpus(args.gpu_ids, parser.error)
    return args


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved.is_file():
        stat = resolved.stat()
        return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    config = resolved / "config.json"
    stat = config.stat() if config.is_file() else resolved.stat()
    return {"path": str(resolved), "config_mtime_ns": stat.st_mtime_ns}


def read_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    with args.questions.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"question_id", "video_id", "question"}
    if not rows or not required <= rows[0].keys():
        raise ValueError(f"Questions must contain {sorted(required)}")
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    missing = [
        row["video_id"] for row in rows
        if not (args.video_dir / f'{row["video_id"]}.mp4').is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing videos: {len(set(missing))}; first={missing[0]}")
    return rows


def read_jsonl_by(path: Path, key: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            value = record.get(key)
            if value is not None:
                records[str(value)] = record
    return records


def atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def group_by_video(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)
    return grouped


def validate_coverage(
    rows: list[dict[str, str]],
    selections: dict[str, dict[str, Any]],
    source: dict[str, dict[str, Any]],
) -> None:
    missing_mmr = [
        row["question_id"] for row in rows
        if not selections.get(row["question_id"], {}).get("mmr_ranked_frame_indices")
    ]
    missing_source = sorted({row["video_id"] for row in rows if row["video_id"] not in source})
    if missing_mmr:
        raise RuntimeError(f"Missing MMR selections: {len(missing_mmr)}; first={missing_mmr[0]}")
    if missing_source:
        raise RuntimeError(f"Missing source evidence: {len(missing_source)}; first={missing_source[0]}")


def choose_ocr_indices(
    questions: list[dict[str, str]],
    selections: dict[str, dict[str, Any]],
    limit: int,
) -> list[int]:
    best_scores: dict[int, float] = {}
    for row in questions:
        record = selections[row["question_id"]]
        frames = record.get("mmr_ranked_frame_indices", [])
        scores = record.get("mmr_ranked_scores", [])
        for offset, frame in enumerate(frames):
            score = float(scores[offset]) if offset < len(scores) else 0.0
            frame = int(frame)
            best_scores[frame] = max(score, best_scores.get(frame, float("-inf")))
    ranked = sorted(best_scores, key=lambda frame: (-best_scores[frame], frame))[:limit]
    return sorted(ranked)


def clean_markup(text: str) -> str:
    text = re.sub(r"\{\\[^}]+\}|<[^>]+>", " ", text)
    return " ".join(text.replace("\\N", " ").replace("\\n", " ").split()).strip()


def parse_ocr_response(
    text: str, batch: list[tuple[int, float, Any]]
) -> list[dict[str, Any]]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    start, end = cleaned.find("["), cleaned.rfind("]")
    parsed: Any = None
    if start >= 0 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
    if parsed is None and start >= 0:
        decoder = json.JSONDecoder()
        position = start + 1
        prefix = []
        while position < len(cleaned):
            while position < len(cleaned) and (
                cleaned[position].isspace() or cleaned[position] == ","
            ):
                position += 1
            if position >= len(cleaned) or cleaned[position] == "]":
                break
            try:
                item, position = decoder.raw_decode(cleaned, position)
            except json.JSONDecodeError:
                break
            prefix.append(item)
        parsed = prefix or None

    by_label = {
        offset + 1: (frame, timestamp)
        for offset, (frame, timestamp, _) in enumerate(batch)
    }
    records = []
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                label = int(item.get("id", item.get("frame")))
            except (TypeError, ValueError):
                continue
            value = item.get("text")
            if label not in by_label or not isinstance(value, str):
                continue
            value = clean_markup(value)
            if not value or value.lower() in {"none", "no text", "n/a"}:
                continue
            frame, timestamp = by_label[label]
            records.append(
                {
                    "start": round(timestamp, 3),
                    "end": round(timestamp, 3),
                    "frame_index": frame,
                    "text": value,
                    "source": "ocr",
                    "ocr_model": "qwen3-vl-8b",
                }
            )
    return records


def ocr_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": "qwen3vl-ocr-replace-only-v1",
        "qwen3": file_identity(args.qwen_path),
        "selections": file_identity(args.selections),
        "source_evidence": file_identity(args.source_evidence),
        "asr_reused": True,
        "subtitles_reused": True,
        "old_ocr_reused": False,
        "ocr_batch_size": args.ocr_batch_size,
        "ocr_max_frames_per_video": args.ocr_max_frames_per_video,
        "ocr_max_new_tokens": args.ocr_max_new_tokens,
        "decode_max_side": args.ocr_decode_max_side,
        "min_visual_tokens": args.ocr_min_visual_tokens,
        "max_visual_tokens": args.ocr_max_visual_tokens,
        "quantization": args.quantization,
        "dtype": args.dtype,
        "attention": args.attention,
        "max_samples": args.max_samples,
    }


def load_qwen3_ocr(args: argparse.Namespace, torch: Any) -> tuple[Any, Any]:
    model_args = argparse.Namespace(
        qwen_path=args.qwen_path,
        dtype=args.dtype,
        quantization=args.quantization,
        attention=args.attention,
        min_visual_tokens=args.ocr_min_visual_tokens,
        max_visual_tokens=args.ocr_max_visual_tokens,
    )
    return inference.load_qwen3vl(model_args, torch)


def ocr_batch(
    batch: list[tuple[int, float, Any]],
    model: Any,
    processor: Any,
    process_vision_info: Any,
    torch: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for label, (_, timestamp, image) in enumerate(batch, start=1):
        content.append({"type": "text", "text": f"Frame ID {label}, time {timestamp:.1f}s:"})
        content.append({"type": "image", "image": image})
    content.append(
        {
            "type": "text",
            "text": (
                "Perform OCR only. Transcribe text visibly present in each frame, "
                "including burned-in subtitles, signs, labels, and screens. Do not "
                "describe images and do not infer hidden text. Limit each frame's "
                "transcription to 160 characters, preserving names and key wording. "
                "Return a JSON array with objects {\"id\": integer, \"text\": string}. "
                "Use \"NONE\" when a frame has no readable text."
            ),
        }
    )
    messages = [{"role": "user", "content": content}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(inference.model_input_device(model, torch))
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.ocr_max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    new_tokens = generated[0, inputs.input_ids.shape[1] :]
    response = processor.decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    records = parse_ocr_response(response, batch)
    if not records and "none" not in response.lower():
        print(f"Warning: unparsed Qwen3 OCR response: {response[:240]!r}", file=sys.stderr)
    del inputs, generated, new_tokens, image_inputs, video_inputs
    return records


def ocr_batch_with_fallback(
    batch: list[tuple[int, float, Any]],
    model: Any,
    processor: Any,
    process_vision_info: Any,
    torch: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    try:
        return ocr_batch(batch, model, processor, process_vision_info, torch, args)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower() or len(batch) == 1:
            raise
        print(f"OCR batch OOM at batch={len(batch)}; retrying one image at a time", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        records = []
        for item in batch:
            records.extend(
                ocr_batch([item], model, processor, process_vision_info, torch, args)
            )
        return records


def run_ocr(args: argparse.Namespace) -> None:
    import cv2
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    rows = read_rows(args)
    selections = read_jsonl_by(args.selections, "question_id")
    source = read_jsonl_by(args.source_evidence, "video_id")
    validate_coverage(rows, selections, source)
    grouped = group_by_video(rows)
    config = ocr_config(args)
    existing = read_jsonl_by(args.raw_evidence, "video_id") if args.resume and args.raw_evidence.is_file() else {}
    records: dict[str, dict[str, Any]] = {}
    indices_by_video: dict[str, list[int]] = {}
    pending = []
    for video_id, questions in grouped.items():
        indices = choose_ocr_indices(questions, selections, args.ocr_max_frames_per_video)
        indices_by_video[video_id] = indices
        cached = existing.get(video_id)
        if (
            cached
            and cached.get("qwen3_ocr_config") == config
            and cached.get("ocr_source_frame_indices") == indices
        ):
            records[video_id] = cached
        else:
            pending.append(video_id)
    print(
        f"Qwen3 OCR: split={args.split} videos={len(grouped)} "
        f"cached={len(records)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        atomic_jsonl(args.raw_evidence, [records[video_id] for video_id in grouped])
        return

    model, processor = load_qwen3_ocr(args, torch)
    errors = []
    for position, video_id in enumerate(pending, start=1):
        video_path = args.video_dir / f"{video_id}.mp4"
        try:
            indices = indices_by_video[video_id]
            frame_map, fps = inference.decode_frame_map(
                video_path, indices, args.ocr_decode_max_side, cv2, Image
            )
            available = [
                (frame, frame / fps, frame_map[frame])
                for frame in indices if frame in frame_map
            ]
            ocr_records = []
            for start in range(0, len(available), args.ocr_batch_size):
                ocr_records.extend(
                    ocr_batch_with_fallback(
                        available[start : start + args.ocr_batch_size],
                        model, processor, process_vision_info, torch, args,
                    )
                )
            base = dict(source[video_id])
            records[video_id] = {
                **base,
                "ocr": ocr_records,
                "ocr_source_frame_indices": indices,
                "evidence_config": config,
                "qwen3_ocr_config": config,
            }
            atomic_jsonl(
                args.raw_evidence,
                [records[value] for value in grouped if value in records],
            )
            print(
                f"OCR [{position}/{len(pending)}] {video_path.name}: "
                f"frames={len(available)} records={len(ocr_records)}",
                flush=True,
            )
            frame_map.clear()
        except Exception as error:
            errors.append(video_id)
            print(f"ERROR OCR {video_path.name}: {type(error).__name__}: {error}", file=sys.stderr)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    if errors:
        raise RuntimeError(f"Qwen3 OCR failed for {len(errors)} videos; first={errors[0]}")
    atomic_jsonl(args.raw_evidence, [records[video_id] for video_id in grouped])
    print(f"Qwen3 raw OCR evidence written to {args.raw_evidence}", flush=True)


def inferred_duration(record: dict[str, Any]) -> float:
    return max(
        (
            float(item.get("end", item.get("start", 0.0)))
            for group in ("asr", "subtitles", "ocr")
            for item in record.get(group, [])
        ),
        default=0.0,
    )


def selection_durations(path: Path) -> dict[str, float]:
    return {
        video_id: float(record.get("duration_seconds", 0.0))
        for video_id, record in read_jsonl_by(path, "video_id").items()
        if float(record.get("duration_seconds", 0.0)) > 0
    }


def run_clean(args: argparse.Namespace) -> None:
    records = list(read_jsonl_by(args.raw_evidence, "video_id").values())
    durations = selection_durations(args.selections)
    cleaned = []
    reasons: Counter[str] = Counter()
    per_video = []
    raw_total = kept_total = 0
    for record in records:
        video_id = str(record["video_id"])
        raw_ocr = list(record.get("ocr", []))
        kept, dropped = clean_ocr_only(
            raw_ocr, durations.get(video_id, inferred_duration(record))
        )
        cleaned.append({**record, "ocr": kept})
        raw_total += len(raw_ocr)
        kept_total += len(kept)
        reasons.update(compact_text(item.get("filter_reason")) for item in dropped)
        per_video.append(
            {
                "video_id": video_id,
                "raw_ocr": len(raw_ocr),
                "kept_ocr": len(kept),
                "dropped_ocr": len(dropped),
            }
        )
    config = {
        "method": CLEANING_METHOD,
        "source_evidence": file_identity(args.raw_evidence),
        "selections": file_identity(args.selections),
        "cleaner": file_identity(HERE / "ocr_clean_core.py"),
        "changed_channel": "ocr",
        "asr_changed": False,
        "subtitles_changed": False,
        "ocr_model": "qwen3-vl-8b",
    }
    atomic_jsonl(args.cleaned_evidence, cleaned)
    atomic_json(args.cleaned_evidence.with_suffix(".meta.json"), config)
    atomic_json(
        args.cleaned_evidence.with_name(f"{args.cleaned_evidence.stem}.report.json"),
        {
            "config": config,
            "videos": len(cleaned),
            "raw_ocr": raw_total,
            "kept_ocr": kept_total,
            "dropped_ocr": raw_total - kept_total,
            "drop_reasons": dict(reasons),
            "per_video": per_video,
        },
    )
    print(
        f"Qwen3 OCR clean: videos={len(cleaned)} raw={raw_total} "
        f"kept={kept_total} dropped={raw_total-kept_total}",
        flush=True,
    )


def run_answer(args: argparse.Namespace) -> None:
    command = [
        str(PYTHON), str(HERE / "infer_public.py"),
        "--split", args.split,
        "--gpu-ids", args.gpu_ids,
        "--questions", str(args.questions),
        "--video-dir", str(args.video_dir),
        "--selections", str(args.selections),
        "--evidence", str(args.cleaned_evidence),
        "--qwen-path", str(args.qwen_path),
        "--output", str(args.output),
        "--topk", str(args.topk),
        "--min-answer-frames", str(args.min_answer_frames),
        "--decode-max-side", str(args.answer_decode_max_side),
        "--min-visual-tokens", str(args.answer_min_visual_tokens),
        "--max-visual-tokens", str(args.answer_max_visual_tokens),
        "--max-text-evidence-items", str(args.max_text_evidence_items),
        "--max-evidence-chars", str(args.max_evidence_chars),
        "--evidence-temporal-window-seconds", str(args.evidence_temporal_window_seconds),
        "--quantization", args.quantization,
        "--dtype", args.dtype,
        "--attention", args.attention,
        "--max-new-tokens", str(args.answer_max_new_tokens),
        "--resume" if args.resume else "--no-resume",
    ]
    if args.split == "public":
        command.append("--allow-custom-public-inputs")
    if args.max_samples is not None:
        command.extend(("--max-samples", str(args.max_samples)))
    if args.lora_path is not None:
        command.extend(("--lora-path", str(args.lora_path)))
    subprocess.run(command, cwd=HERE, check=True)


def dry_run(args: argparse.Namespace) -> None:
    rows = read_rows(args)
    selections = read_jsonl_by(args.selections, "question_id")
    source = read_jsonl_by(args.source_evidence, "video_id")
    validate_coverage(rows, selections, source)
    print(
        json.dumps(
            {
                "split": args.split,
                "questions": len(rows),
                "videos": len({row["video_id"] for row in rows}),
                "gpu_ids": args.gpu_ids,
                "qwen3_ocr": str(args.qwen_path),
                "selections": str(args.selections),
                "source_evidence": str(args.source_evidence),
                "raw_evidence": str(args.raw_evidence),
                "cleaned_evidence": str(args.cleaned_evidence),
                "output": str(args.output),
                "ocr_images": {
                    "decode_max_side": args.ocr_decode_max_side,
                    "visual_tokens": [args.ocr_min_visual_tokens, args.ocr_max_visual_tokens],
                    "batch_size": args.ocr_batch_size,
                    "max_frames_per_video": args.ocr_max_frames_per_video,
                },
                "answer_images": {
                    "decode_max_side": args.answer_decode_max_side,
                    "visual_tokens": [args.answer_min_visual_tokens, args.answer_max_visual_tokens],
                },
                "asr_reused": True,
                "subtitles_reused": True,
                "base_answer": args.lora_path is None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.dry_run:
        dry_run(args)
        return
    if args.stage in {"all", "ocr"}:
        run_ocr(args)
    if args.stage in {"all", "clean"}:
        run_clean(args)
    if args.stage in {"all", "answer"}:
        run_answer(args)


if __name__ == "__main__":
    main()
