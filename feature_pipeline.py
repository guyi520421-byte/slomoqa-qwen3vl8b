#!/usr/bin/env python3
"""SloMo-QA: VideoITG + DINOv2/time MMR + ASR/subtitle/OCR + Qwen QA."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DATA_ROOT = Path("/data/wen1/datasets/slomo_qa")
# Keep the retrieval implementation folder-local. Model checkpoints and raw
# datasets remain external inputs, but no Python code is imported elsewhere.
OLD_PIPELINE = HERE / "videoitg_base.py"
DEFAULT_SELECTOR = Path("/data/wen1/MLLMs/VideoITG-8B")
DEFAULT_DINO = Path("/data/wen1/MLLMs/dinov2_vitb14_reg4_pretrain.pth")
DEFAULT_QWEN = Path("/data/wen1/MLLMs/Qwen_25vl_7b")
VIDEOITG_PYTHON = Path("/data/miniconda3/envs/videoitg/bin/python")
QWEN_PYTHON = Path("/data/miniconda3/envs/egoVqa/bin/python")


def configure_visible_gpus() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-ids")
    known, _ = parser.parse_known_args()
    if known.gpu_ids:
        values = [value.strip() for value in known.gpu_ids.split(",")]
        if not values or any(not value.isdigit() for value in values):
            parser.error("--gpu-ids must look like 0 or 0,1")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(values)


configure_visible_gpus()
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_old_pipeline() -> Any:
    if not OLD_PIPELINE.is_file():
        raise FileNotFoundError(f"Base pipeline not found: {OLD_PIPELINE}")
    spec = importlib.util.spec_from_file_location("slomoqa_hard_dedup_base", OLD_PIPELINE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base pipeline: {OLD_PIPELINE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_old_pipeline()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Question-aware VideoITG retrieval, DINOv2/time MMR, multimodal "
            "text evidence, and open-ended SloMo-QA."
        )
    )
    parser.add_argument("--split", choices=("private", "public"), default="private")
    parser.add_argument(
        "--stage",
        choices=("all", "select", "mmr", "dino", "evidence", "answer"),
        default="all",
    )
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--selector-path", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--dino-path", type=Path, default=DEFAULT_DINO)
    parser.add_argument("--qwen-path", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-ids")
    parser.add_argument("--selector-device", default="cuda:0")
    parser.add_argument("--dino-device", default="cuda:0")
    parser.add_argument("--candidate-frames", type=int, default=512)
    parser.add_argument("--min-candidate-frames", type=int, default=128)
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--dino-pool-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--min-answer-frames", type=int, default=12)
    parser.add_argument("--dino-batch-size", type=int, default=8)
    parser.add_argument(
        "--dino-dtype", choices=("fp16", "bf16", "fp32"), default="fp16"
    )

    parser.add_argument("--mmr-relevance-weight", type=float, default=1.0)
    parser.add_argument("--mmr-visual-weight", type=float, default=0.30)
    parser.add_argument("--mmr-temporal-weight", type=float, default=0.15)
    parser.add_argument("--mmr-global-weight", type=float, default=0.60)
    parser.add_argument("--mmr-patch-weight", type=float, default=0.40)
    parser.add_argument("--mmr-temporal-tau-seconds", type=float, default=3.0)
    parser.add_argument("--mmr-patch-grid-size", type=int, default=7)

    parser.add_argument("--asr-model-path", type=Path)
    parser.add_argument("--asr-device", default="cuda:0")
    parser.add_argument(
        "--asr-dtype", choices=("fp16", "bf16", "fp32"), default="fp16"
    )
    parser.add_argument("--asr-language", default="en")
    parser.add_argument(
        "--asr-task", choices=("transcribe", "translate"), default="transcribe"
    )
    parser.add_argument("--asr-chunk-seconds", type=float, default=30.0)
    parser.add_argument("--subtitle-dir", type=Path)
    parser.add_argument(
        "--ocr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ocr-model-path", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--ocr-batch-size", type=int, default=3)
    parser.add_argument("--ocr-max-frames-per-video", type=int, default=96)
    parser.add_argument("--ocr-max-new-tokens", type=int, default=512)
    parser.add_argument("--max-text-evidence-items", type=int, default=12)
    parser.add_argument("--max-evidence-chars", type=int, default=5000)
    parser.add_argument("--evidence-temporal-window-seconds", type=float, default=20.0)

    parser.add_argument("--decode-max-side", type=int, default=740)
    parser.add_argument("--min-visual-tokens", type=int, default=128)
    parser.add_argument("--max-visual-tokens", type=int, default=256)
    parser.add_argument(
        "--quantization", choices=("4bit", "8bit", "none"), default="4bit"
    )
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--attention",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    args.questions = args.questions or (
        DATA_ROOT / f"sf20k_{args.split}_test_questions.csv"
    )
    args.video_dir = args.video_dir or (
        DATA_ROOT / f"sf20k_{args.split}_test_videos"
    )
    args.selections = args.selections or (
        HERE / "inputs" / f"{args.split}_selections.jsonl"
    )
    args.evidence = args.evidence or (
        HERE / "inputs" / f"{args.split}_evidence_raw.jsonl"
    )
    args.output = args.output or (
        HERE / "outputs" / f"{args.split}_qwen25vl_predictions.json"
    )

    positive = {
        "--candidate-frames": args.candidate_frames,
        "--min-candidate-frames": args.min_candidate_frames,
        "--target-fps": args.target_fps,
        "--dino-pool-size": args.dino_pool_size,
        "--topk": args.topk,
        "--min-answer-frames": args.min_answer_frames,
        "--dino-batch-size": args.dino_batch_size,
        "--mmr-temporal-tau-seconds": args.mmr_temporal_tau_seconds,
        "--mmr-patch-grid-size": args.mmr_patch_grid_size,
        "--asr-chunk-seconds": args.asr_chunk_seconds,
        "--ocr-batch-size": args.ocr_batch_size,
        "--ocr-max-frames-per-video": args.ocr_max_frames_per_video,
        "--ocr-max-new-tokens": args.ocr_max_new_tokens,
        "--max-text-evidence-items": args.max_text_evidence_items,
        "--max-evidence-chars": args.max_evidence_chars,
        "--min-visual-tokens": args.min_visual_tokens,
        "--max-visual-tokens": args.max_visual_tokens,
        "--max-new-tokens": args.max_new_tokens,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.min_candidate_frames > args.candidate_frames:
        parser.error("--min-candidate-frames cannot exceed --candidate-frames")
    if not args.min_answer_frames <= args.topk <= args.dino_pool_size:
        parser.error("require min-answer-frames <= topk <= dino-pool-size")
    if args.dino_pool_size > args.candidate_frames:
        parser.error("--dino-pool-size cannot exceed --candidate-frames")
    if args.max_visual_tokens < args.min_visual_tokens:
        parser.error("--max-visual-tokens cannot be smaller than min")
    weights = (
        args.mmr_relevance_weight,
        args.mmr_visual_weight,
        args.mmr_temporal_weight,
        args.mmr_global_weight,
        args.mmr_patch_weight,
    )
    if any(value < 0 for value in weights):
        parser.error("MMR weights must be non-negative")
    if args.mmr_global_weight + args.mmr_patch_weight <= 0:
        parser.error("at least one DINO global/patch weight must be positive")
    return args


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


def mmr_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": "videoitg-dinov2-temporal-mmr-v1",
        "dino": file_identity(args.dino_path),
        "pool_size": args.dino_pool_size,
        "topk": args.topk,
        "relevance_weight": args.mmr_relevance_weight,
        "visual_weight": args.mmr_visual_weight,
        "temporal_weight": args.mmr_temporal_weight,
        "global_weight": args.mmr_global_weight,
        "patch_weight": args.mmr_patch_weight,
        "temporal_tau_seconds": args.mmr_temporal_tau_seconds,
        "patch_grid_size": args.mmr_patch_grid_size,
        "dtype": args.dino_dtype,
    }


def evidence_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": "timestamped-asr-subtitle-qwen-ocr-v2",
        "asr_model": file_identity(args.asr_model_path),
        "asr_chunk_seconds": args.asr_chunk_seconds,
        "asr_language": args.asr_language,
        "asr_task": args.asr_task,
        "subtitle_dir": str(args.subtitle_dir.resolve()) if args.subtitle_dir else None,
        "ocr": args.ocr,
        "ocr_model": file_identity(args.ocr_model_path) if args.ocr else None,
        "ocr_max_frames_per_video": args.ocr_max_frames_per_video,
        "ocr_batch_size": args.ocr_batch_size,
        "ocr_max_new_tokens": args.ocr_max_new_tokens,
        # These settings affect OCR pixels/tokens and therefore must be part of
        # the resume identity. The historical runner omitted them, which could
        # incorrectly reuse an OCR cache after changing image resolution.
        "decode_max_side": args.decode_max_side,
        "min_visual_tokens": args.min_visual_tokens,
        "max_visual_tokens": args.max_visual_tokens,
        "quantization": args.quantization,
        "dtype": args.dtype,
        "attention": args.attention,
        "mmr_config": mmr_config(args),
    }


def answer_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": "qwen-mmr-timestamped-multimodal-evidence-v1",
        "qwen": file_identity(args.qwen_path),
        "selections_file": file_identity(args.selections),
        "evidence_file": file_identity(args.evidence),
        "quantization": args.quantization,
        "dtype": args.dtype,
        "attention": args.attention,
        "topk": args.topk,
        "min_visual_tokens": args.min_visual_tokens,
        "max_visual_tokens": args.max_visual_tokens,
        "max_new_tokens": args.max_new_tokens,
        "max_text_evidence_items": args.max_text_evidence_items,
        "max_evidence_chars": args.max_evidence_chars,
        "evidence_temporal_window_seconds": args.evidence_temporal_window_seconds,
        "evidence_config": evidence_config(args),
    }


def load_jsonl_by(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: ignoring invalid line {path}:{line_number}")
                continue
            value = record.get(key)
            if value is not None:
                records[str(value)] = record
    return records


def write_jsonl_by(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(records):
            handle.write(json.dumps(records[key], ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def spatially_pool_patches(patches: Any, grid_size: int, torch: Any) -> Any:
    """Reduce patch-token cost while retaining coarse spatial correspondence."""
    count = int(patches.shape[1])
    side = math.isqrt(count)
    if side * side == count:
        spatial = patches.transpose(1, 2).reshape(
            patches.shape[0], patches.shape[2], side, side
        )
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            spatial, (grid_size, grid_size)
        )
        pooled = pooled.flatten(2).transpose(1, 2)
    else:
        pooled = torch.nn.functional.adaptive_avg_pool1d(
            patches.transpose(1, 2), grid_size * grid_size
        ).transpose(1, 2)
    return torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)


def mmr_rerank_cached_features(
    ranked: list[int],
    ranked_scores: list[float],
    fps: float,
    all_indices: list[int],
    features: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    if not ranked:
        raise ValueError("MMR received no ranked frames")
    if fps <= 0:
        raise ValueError(f"FPS must be positive, got {fps}")
    position = {index: offset for offset, index in enumerate(all_indices)}
    feature_offsets = [position[index] for index in ranked]
    global_features = features["global"][feature_offsets].float()
    patch_features = spatially_pool_patches(
        features["patch"][feature_offsets], args.mmr_patch_grid_size, torch
    )

    global_similarity = global_features @ global_features.T
    flat_patch = patch_features.reshape(patch_features.shape[0], -1)
    patch_similarity = (flat_patch @ flat_patch.T) / patch_features.shape[1]
    visual_denominator = args.mmr_global_weight + args.mmr_patch_weight
    visual_similarity = (
        args.mmr_global_weight * global_similarity
        + args.mmr_patch_weight * patch_similarity
    ) / visual_denominator
    visual_similarity = visual_similarity.clamp(min=0.0, max=1.0)

    timestamps = torch.tensor([index / fps for index in ranked], dtype=torch.float32)
    temporal_similarity = torch.exp(
        -torch.abs(timestamps[:, None] - timestamps[None, :])
        / args.mmr_temporal_tau_seconds
    )
    raw_scores = torch.tensor(
        [ranked_scores[i] if i < len(ranked_scores) else 0.0 for i in range(len(ranked))],
        dtype=torch.float32,
    )
    score_range = raw_scores.max() - raw_scores.min()
    if float(score_range) > 1e-8:
        normalized_scores = (raw_scores - raw_scores.min()) / score_range
    else:
        normalized_scores = torch.ones_like(raw_scores)
    if len(ranked) == 1:
        rank_prior = torch.ones(1)
    else:
        rank_prior = torch.linspace(1.0, 0.0, len(ranked))
    relevance = 0.85 * normalized_scores + 0.15 * rank_prior

    budget = min(args.topk, len(ranked))
    selected = [0]
    diagnostics = [
        {
            "frame_index": ranked[0],
            "videoitg_rank": 1,
            "videoitg_score": round(float(raw_scores[0]), 6),
            "normalized_relevance": round(float(relevance[0]), 6),
            "max_visual_similarity": 0.0,
            "max_temporal_similarity": 0.0,
            "mmr_utility": round(float(args.mmr_relevance_weight * relevance[0]), 6),
        }
    ]
    available = torch.ones(len(ranked), dtype=torch.bool)
    available[0] = False
    while len(selected) < budget:
        selected_tensor = torch.tensor(selected, dtype=torch.long)
        visual_redundancy = visual_similarity[:, selected_tensor].max(dim=1).values
        temporal_redundancy = temporal_similarity[:, selected_tensor].max(dim=1).values
        utility = (
            args.mmr_relevance_weight * relevance
            - args.mmr_visual_weight * visual_redundancy
            - args.mmr_temporal_weight * temporal_redundancy
        )
        utility = utility.masked_fill(~available, float("-inf"))
        chosen = int(utility.argmax().item())
        selected.append(chosen)
        available[chosen] = False
        diagnostics.append(
            {
                "frame_index": ranked[chosen],
                "videoitg_rank": chosen + 1,
                "videoitg_score": round(float(raw_scores[chosen]), 6),
                "normalized_relevance": round(float(relevance[chosen]), 6),
                "max_visual_similarity": round(float(visual_redundancy[chosen]), 6),
                "max_temporal_similarity": round(float(temporal_redundancy[chosen]), 6),
                "mmr_utility": round(float(utility[chosen]), 6),
            }
        )

    chosen_frames = [ranked[index] for index in selected]
    chronological = sorted(chosen_frames)
    score_map = dict(zip(ranked, raw_scores.tolist()))
    excluded = [ranked[index] for index in range(len(ranked)) if index not in set(selected)]
    return {
        "mmr_ranked_frame_indices": chosen_frames,
        "mmr_ranked_scores": [round(float(score_map[index]), 6) for index in chosen_frames],
        "mmr_selected_frame_indices": chronological,
        "mmr_selected_scores": [round(float(score_map[index]), 6) for index in chronological],
        "mmr_selected_timestamps": [round(index / fps, 3) for index in chronological],
        "mmr_diagnostics": diagnostics,
        "mmr_budget_excluded_frame_indices": excluded,
        "mmr_source_pool_size": len(ranked),
        "mmr_patch_count": features.get("patch_count"),
        "mmr_pooled_patch_count": int(patch_features.shape[1]),
    }


def run_mmr(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    BASE.require_reference_root()
    from dinov2_frame_filter import DINOv2RedundancyFilter

    config = mmr_config(args)
    selections = BASE.load_selections(args.selections)
    pending = [
        row
        for row in rows
        if row["question_id"] in selections
        and (
            selections[row["question_id"]].get("mmr_config") != config
            or selections[row["question_id"]].get("mmr_source_frame_indices")
            != selections[row["question_id"]].get("ranked_frame_indices")
        )
    ]
    print(
        f"MMR: questions={len(rows)} pending={len(pending)} "
        f"missing_select={len(rows)-len(selections)}"
    )
    if not pending:
        return
    filterer = DINOv2RedundancyFilter(
        args.dino_path,
        device=args.dino_device,
        batch_size=args.dino_batch_size,
        dtype=args.dino_dtype,
    )
    grouped = BASE.group_by_video(pending)
    errors = 0
    for video_position, (video_id, questions) in enumerate(grouped.items(), start=1):
        video_path = args.video_dir / f"{video_id}.mp4"
        all_indices = sorted(
            {
                int(index)
                for row in questions
                for index in selections[row["question_id"]]["ranked_frame_indices"]
            }
        )
        try:
            features = filterer.encode_video_frames(video_path, all_indices)
            for row in questions:
                record = selections[row["question_id"]]
                ranked = [int(value) for value in record["ranked_frame_indices"]]
                result = mmr_rerank_cached_features(
                    ranked,
                    [float(value) for value in record.get("ranked_scores", [])],
                    float(record["fps"]),
                    all_indices,
                    features,
                    args,
                )
                record.update(result)
                record["mmr_source_frame_indices"] = ranked
                record["mmr_config"] = config
                BASE.write_selections(args.selections, rows, selections)
            print(
                f"[{video_position}/{len(grouped)}] {video_path.name}: "
                f"questions={len(questions)} union_frames={len(all_indices)}",
                flush=True,
            )
            del features
        except Exception as error:
            errors += len(questions)
            print(
                f"ERROR MMR {video_path.name}: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
        finally:
            gc.collect()
            filterer.torch.cuda.empty_cache()
    print(f"MMR selections written to {args.selections}; errors={errors}")


def clean_text_markup(text: str) -> str:
    text = re.sub(r"\{\\[^}]+\}", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\\N", " ").replace("\\n", " ")
    return " ".join(text.split()).strip()


def parse_clock(value: str) -> float:
    parts = value.strip().replace(",", ".").split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = "0", parts[0], parts[1]
    else:
        return float(parts[0])
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_sidecar_subtitle(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    suffix = path.suffix.lower()
    records: list[dict[str, Any]] = []
    if suffix in {".ass", ".ssa"}:
        for line in text.splitlines():
            if not line.lower().startswith("dialogue:"):
                continue
            fields = line.split(",", 9)
            if len(fields) < 10:
                continue
            cleaned = clean_text_markup(fields[9])
            if cleaned:
                records.append(
                    {
                        "start": round(parse_clock(fields[1]), 3),
                        "end": round(parse_clock(fields[2]), 3),
                        "text": cleaned,
                        "source": f"sidecar:{path.name}",
                    }
                )
        return records
    if suffix == ".txt":
        cleaned = clean_text_markup(text)
        return (
            [{"start": 0.0, "end": 0.0, "text": cleaned, "source": f"sidecar:{path.name}"}]
            if cleaned
            else []
        )

    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    timestamp_pattern = re.compile(
        r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?[,.]\d+)\s*-->\s*"
        r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?[,.]\d+)"
    )
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        match_index = next(
            (index for index, line in enumerate(lines) if timestamp_pattern.search(line)),
            None,
        )
        if match_index is None:
            continue
        match = timestamp_pattern.search(lines[match_index])
        assert match is not None
        cleaned = clean_text_markup(" ".join(lines[match_index + 1 :]))
        if cleaned:
            records.append(
                {
                    "start": round(parse_clock(match.group("start")), 3),
                    "end": round(parse_clock(match.group("end")), 3),
                    "text": cleaned,
                    "source": f"sidecar:{path.name}",
                }
            )
    return records


def find_sidecar_subtitles(
    video_path: Path, subtitle_dir: Path | None
) -> list[Path]:
    roots = [video_path.parent]
    if subtitle_dir is not None and subtitle_dir.resolve() != video_path.parent.resolve():
        roots.append(subtitle_dir)
    paths: list[Path] = []
    for root in roots:
        for suffix in (".srt", ".vtt", ".ass", ".ssa", ".txt"):
            candidate = root / f"{video_path.stem}{suffix}"
            if candidate.is_file():
                paths.append(candidate)
    return paths


def extract_embedded_subtitles(video_path: Path) -> list[dict[str, Any]]:
    try:
        import av
    except ImportError:
        return []
    records: list[dict[str, Any]] = []
    try:
        with av.open(str(video_path)) as container:
            streams = [stream for stream in container.streams if stream.type == "subtitle"]
            for stream in streams:
                for packet in container.demux(stream):
                    packet_time = float(packet.pts * stream.time_base) if packet.pts is not None else 0.0
                    for subtitle in packet.decode():
                        start = packet_time + float(getattr(subtitle, "start_display_time", 0)) / 1000.0
                        end = packet_time + float(getattr(subtitle, "end_display_time", 0)) / 1000.0
                        pieces = []
                        for rect in getattr(subtitle, "rects", []):
                            value = getattr(rect, "text", None) or getattr(rect, "ass", None)
                            if value:
                                pieces.append(str(value).split(",", 9)[-1])
                        cleaned = clean_text_markup(" ".join(pieces))
                        if cleaned:
                            records.append(
                                {
                                    "start": round(start, 3),
                                    "end": round(max(start, end), 3),
                                    "text": cleaned,
                                    "source": "embedded_subtitle",
                                }
                            )
    except Exception as error:
        print(
            f"Warning: embedded subtitles unavailable for {video_path.name}: {error}",
            file=sys.stderr,
        )
    return records


def subtitle_records(video_path: Path, subtitle_dir: Path | None) -> list[dict[str, Any]]:
    records = extract_embedded_subtitles(video_path)
    for path in find_sidecar_subtitles(video_path, subtitle_dir):
        try:
            records.extend(parse_sidecar_subtitle(path))
        except Exception as error:
            print(f"Warning: failed to parse subtitle {path}: {error}", file=sys.stderr)
    seen: set[tuple[int, str]] = set()
    unique = []
    for record in sorted(records, key=lambda item: (item["start"], item["end"])):
        key = (round(float(record["start"]) * 10), record["text"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


def decode_audio_mono_16k(video_path: Path) -> Any:
    import av
    import numpy as np

    arrays = []
    with av.open(str(video_path)) as container:
        if not container.streams.audio:
            return np.zeros(0, dtype=np.float32)
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        for frame in container.decode(stream):
            converted = resampler.resample(frame)
            if converted is None:
                continue
            if not isinstance(converted, list):
                converted = [converted]
            for audio_frame in converted:
                array = audio_frame.to_ndarray().astype(np.float32, copy=False).reshape(-1)
                arrays.append(array)
    return np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float32)


def load_asr(args: argparse.Namespace) -> tuple[Any, Any, Any] | None:
    if args.asr_model_path is None:
        print(
            "ASR: skipped because --asr-model-path was not supplied; "
            "subtitle and OCR evidence remain enabled.",
            flush=True,
        )
        return None
    if not args.asr_model_path.is_dir():
        raise FileNotFoundError(f"ASR checkpoint not found: {args.asr_model_path}")
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.asr_dtype]
    processor = AutoProcessor.from_pretrained(
        args.asr_model_path, local_files_only=True
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.asr_model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(args.asr_device).eval()
    return model, processor, torch


def transcribe_video(
    video_path: Path, runtime: tuple[Any, Any, Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
    model, processor, torch = runtime
    audio = decode_audio_mono_16k(video_path)
    if audio.size == 0:
        return []
    chunk_samples = round(args.asr_chunk_seconds * 16000)
    records = []
    for start_sample in range(0, audio.size, chunk_samples):
        chunk = audio[start_sample : start_sample + chunk_samples]
        if chunk.size < 1600:
            continue
        inputs = processor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        model_inputs = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                continue
            if value.is_floating_point():
                # Whisper processors intentionally return FP32 log-mel features.
                # Match them to FP16/BF16 model weights before the encoder convs.
                value = value.to(device=model.device, dtype=model.dtype)
            else:
                value = value.to(device=model.device)
            model_inputs[key] = value
        generation_kwargs = {
            "task": args.asr_task,
            "language": args.asr_language,
        }
        with torch.inference_mode():
            generated = model.generate(**model_inputs, **generation_kwargs)
        text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if text:
            records.append(
                {
                    "start": round(start_sample / 16000, 3),
                    "end": round(min(audio.size, start_sample + chunk_samples) / 16000, 3),
                    "text": " ".join(text.split()),
                    "source": "asr",
                }
            )
        del inputs, model_inputs, generated
    return records


def parse_ocr_response(text: str, batch: list[tuple[int, float, Any]]) -> list[dict[str, Any]]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    start, end = cleaned.find("["), cleaned.rfind("]")
    parsed: Any = None
    if start >= 0 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            parsed = None
    recovered_prefix = False
    if parsed is None and start >= 0:
        # If generation stops midway through a later object, keep every
        # complete leading object instead of dropping the entire OCR batch.
        decoder = json.JSONDecoder()
        position = start + 1
        prefix: list[Any] = []
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
        if prefix:
            parsed = prefix
            recovered_prefix = True
    by_label = {offset + 1: (frame, timestamp) for offset, (frame, timestamp, _) in enumerate(batch)}
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
            value = clean_text_markup(value)
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
                }
            )
    if recovered_prefix:
        print(
            f"Warning: OCR output was truncated; recovered "
            f"{len(parsed)} complete entries from the batch.",
            file=sys.stderr,
        )
    return records


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
                "Return a JSON array "
                "with objects {\"id\": integer, \"text\": string}. Use \"NONE\" "
                "when a frame has no readable text."
            ),
        }
    )
    messages = [{"role": "user", "content": content}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(BASE.model_input_device(model, torch))
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
        print(f"Warning: could not parse OCR response: {response[:240]!r}", file=sys.stderr)
    del inputs, generated, new_tokens, image_inputs, video_inputs
    return records


def choose_ocr_indices(
    questions: list[dict[str, Any]],
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
            best_scores[int(frame)] = max(score, best_scores.get(int(frame), float("-inf")))
    ranked = sorted(best_scores, key=lambda frame: (-best_scores[frame], frame))[:limit]
    return sorted(ranked)


def load_ocr_model(args: argparse.Namespace, torch: Any) -> tuple[Any, Any]:
    ocr_args = argparse.Namespace(**vars(args))
    ocr_args.qwen_path = args.ocr_model_path
    ocr_args.quantization = "4bit"
    ocr_args.dtype = "bf16"
    ocr_args.attention = args.attention
    return BASE.load_qwen(ocr_args, torch)


def run_evidence(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    config = evidence_config(args)
    selections = BASE.load_selections(args.selections)
    grouped = BASE.group_by_video(
        [row for row in rows if selections.get(row["question_id"], {}).get("mmr_ranked_frame_indices")]
    )
    records = load_jsonl_by(args.evidence, "video_id") if args.resume else {}
    pending: dict[str, list[dict[str, Any]]] = {}
    source_indices: dict[str, list[int]] = {}
    for video_id, questions in grouped.items():
        indices = choose_ocr_indices(
            questions, selections, args.ocr_max_frames_per_video
        )
        source_indices[video_id] = indices
        cached = records.get(video_id, {})
        if cached.get("evidence_config") != config or cached.get("ocr_source_frame_indices") != indices:
            pending[video_id] = questions
    print(
        f"Evidence: videos={len(grouped)} cached={len(grouped)-len(pending)} "
        f"pending={len(pending)}"
    )
    if not pending:
        return

    asr_by_video: dict[str, list[dict[str, Any]]] = {}
    asr_failures: set[str] = set()
    asr_runtime = load_asr(args)
    if asr_runtime is not None:
        for position, video_id in enumerate(pending, start=1):
            video_path = args.video_dir / f"{video_id}.mp4"
            try:
                asr_by_video[video_id] = transcribe_video(video_path, asr_runtime, args)
                print(
                    f"ASR [{position}/{len(pending)}] {video_path.name}: "
                    f"chunks={len(asr_by_video[video_id])}",
                    flush=True,
                )
            except Exception as error:
                print(f"ERROR ASR {video_path.name}: {type(error).__name__}: {error}", file=sys.stderr)
                asr_by_video[video_id] = []
                asr_failures.add(video_id)
        del asr_runtime
        gc.collect()
        torch.cuda.empty_cache()

    ocr_model = ocr_processor = None
    if args.ocr:
        if not args.ocr_model_path.is_dir():
            raise FileNotFoundError(f"OCR Qwen checkpoint not found: {args.ocr_model_path}")
        ocr_model, ocr_processor = load_ocr_model(args, torch)

    errors = 0
    for position, (video_id, questions) in enumerate(pending.items(), start=1):
        video_path = args.video_dir / f"{video_id}.mp4"
        try:
            subtitle = subtitle_records(video_path, args.subtitle_dir)
            ocr_records: list[dict[str, Any]] = []
            indices = source_indices[video_id]
            if args.ocr and indices:
                import cv2

                frame_map, fps = BASE.decode_frame_map(
                    video_path, indices, args.decode_max_side, cv2, Image
                )
                available = [
                    (frame, frame / fps, frame_map[frame])
                    for frame in indices
                    if frame in frame_map
                ]
                for start in range(0, len(available), args.ocr_batch_size):
                    ocr_records.extend(
                        ocr_batch(
                            available[start : start + args.ocr_batch_size],
                            ocr_model,
                            ocr_processor,
                            process_vision_info,
                            torch,
                            args,
                        )
                    )
                frame_map.clear()
            records[video_id] = {
                "video_id": video_id,
                "video_path": video_path.name,
                "asr": asr_by_video.get(video_id, []),
                "subtitles": subtitle,
                "ocr": ocr_records,
                "ocr_source_frame_indices": indices,
                # A transient ASR failure must not become a permanent resume hit.
                "evidence_config": None if video_id in asr_failures else config,
            }
            write_jsonl_by(args.evidence, records)
            print(
                f"Evidence [{position}/{len(pending)}] {video_path.name}: "
                f"asr={len(records[video_id]['asr'])} "
                f"subtitle={len(subtitle)} ocr={len(ocr_records)}",
                flush=True,
            )
        except Exception as error:
            errors += 1
            print(
                f"ERROR evidence {video_path.name}: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    print(f"Evidence written to {args.evidence}; errors={errors}")


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "how", "in",
    "is", "it", "of", "on", "or", "she", "that", "the", "they", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "with",
}


def lexical_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9']+", text.lower())
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
            terms = lexical_terms(text)
            overlap = len(query_terms & terms) / max(1, len(query_terms))
            midpoint = (float(item.get("start", 0)) + float(item.get("end", 0))) / 2
            distance = min(
                (abs(midpoint - timestamp) for timestamp in selected_timestamps),
                default=float("inf"),
            )
            proximity = (
                1.0 / (1.0 + distance / 5.0)
                if distance <= args.evidence_temporal_window_seconds
                else 0.0
            )
            score = 2.5 * overlap + proximity + source_bonus[group]
            enriched = dict(item)
            enriched["evidence_type"] = group[:-1] if group.endswith("s") else group
            candidates.append((score, enriched))
    candidates.sort(
        key=lambda pair: (-pair[0], float(pair[1].get("start", 0)))
    )
    selected = []
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
    lines = []
    for item in records:
        source = str(item.get("evidence_type", item.get("source", "text"))).upper()
        start, end = float(item.get("start", 0)), float(item.get("end", 0))
        if end > start:
            stamp = f"{start:.1f}-{end:.1f}s"
        else:
            stamp = f"{start:.1f}s"
        lines.append(f"[{source} {stamp}] {item['text']}")
    return "\n".join(lines)


def question_answer_instruction(question: str) -> str:
    lowered = question.lower().strip()
    if re.match(r"^(who|where|when|which)\b", lowered):
        return "Return only the requested entity, location, time, or choice, normally 1-8 words."
    if re.match(r"^(is|are|was|were|do|does|did|can|could|will|would|has|have|had)\b", lowered):
        return "Begin with Yes or No when applicable, followed by at most one short clause."
    if re.match(r"^(why|how)\b", lowered):
        return "Return one concise causal or procedural clause, normally no more than 15 words."
    return "Return one exact short answer, normally no more than 12 words."


def generate_open_answer(
    row: dict[str, Any],
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
    title = str(row.get("movie_title", "")).strip()
    content.append(
        {
            "type": "text",
            "text": (
                f"Movie title metadata: {title or 'unknown'}\n"
                "Timestamped ASR/subtitle/OCR evidence:\n"
                f"{format_text_evidence(text_records)}\n\n"
                "The visuals are question-relevant frames selected from the complete "
                "video and shown chronologically. Text evidence may contain recognition "
                "errors. Use only mutually supported evidence, answer the exact question, "
                "and do not summarize unrelated events or invent missing facts.\n"
                f"Question: {row['question']}\n"
                f"{question_answer_instruction(row['question'])}\n"
                "Return only the answer with no prefix or explanation."
            ),
        }
    )
    messages = [{"role": "user", "content": content}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(BASE.model_input_device(model, torch))
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    new_tokens = generated[0, inputs.input_ids.shape[1] :]
    answer = processor.decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    answer = re.sub(r"^\s*(?:answer|prediction)\s*:\s*", "", answer, flags=re.I)
    answer = " ".join(answer.replace("```", "").split())
    if not answer:
        raise ValueError("Qwen returned an empty answer")
    return answer


def metadata_path(output: Path) -> Path:
    return output.with_suffix(".meta.json")


def load_cached_predictions(args: argparse.Namespace) -> dict[str, str]:
    if not args.resume:
        return {}
    meta = metadata_path(args.output)
    if not meta.is_file():
        if args.output.is_file():
            print("Answer cache ignored because its metadata sidecar is missing.")
        return {}
    try:
        cached_config = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if cached_config != answer_config(args):
        print("Answer cache ignored because the inference configuration changed.")
        return {}
    return BASE.load_predictions(args.output)


def write_answer_cache(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    predictions: dict[str, str],
) -> None:
    BASE.write_predictions(args.output, rows, predictions)
    meta = metadata_path(args.output)
    temporary = meta.with_suffix(meta.suffix + ".tmp")
    temporary.write_text(
        json.dumps(answer_config(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, meta)


def run_answer(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    import cv2
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    selections = BASE.load_selections(args.selections)
    evidence = load_jsonl_by(args.evidence, "video_id")
    predictions = load_cached_predictions(args)
    valid_ids = {row["question_id"] for row in rows}
    predictions = {key: value for key, value in predictions.items() if key in valid_ids}
    pending = [
        row
        for row in rows
        if row["question_id"] not in predictions
        and selections.get(row["question_id"], {}).get("mmr_ranked_frame_indices")
    ]
    missing = sum(
        not selections.get(row["question_id"], {}).get("mmr_ranked_frame_indices")
        for row in rows
    )
    print(
        f"Answer: questions={len(rows)} cached={len(predictions)} "
        f"pending={len(pending)} missing_mmr={missing} evidence_videos={len(evidence)}"
    )
    if not pending:
        write_answer_cache(args, rows, predictions)
        return
    model, processor = BASE.load_qwen(args, torch)
    grouped = BASE.group_by_video(pending)
    errors = 0
    for video_position, (video_id, questions) in enumerate(grouped.items(), start=1):
        union_indices = sorted(
            {
                int(index)
                for row in questions
                for index in selections[row["question_id"]]["mmr_ranked_frame_indices"][: args.topk]
            }
        )
        video_path = args.video_dir / f"{video_id}.mp4"
        try:
            frame_map, fps = BASE.decode_frame_map(
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
                        predictions[row["question_id"]] = generate_open_answer(
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


def child_arguments(stage: str) -> list[str]:
    forwarded: list[str] = []
    skip = False
    for argument in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if argument == "--stage":
            skip = True
        elif not argument.startswith("--stage="):
            forwarded.append(argument)
    return [str(Path(__file__).resolve()), "--stage", stage, *forwarded]


def run_all() -> None:
    for python, stage in (
        (VIDEOITG_PYTHON, "select"),
        (VIDEOITG_PYTHON, "mmr"),
        (QWEN_PYTHON, "evidence"),
        (QWEN_PYTHON, "answer"),
    ):
        print(f"Launching {stage} with {python}", flush=True)
        subprocess.run([str(python), *child_arguments(stage)], cwd=HERE, check=True)


def main() -> None:
    args = parse_args()
    if args.stage == "all":
        BASE.load_rows(args)
        run_all()
        return
    rows = BASE.load_rows(args)
    if args.stage == "select":
        BASE.run_select(args, rows)
    elif args.stage in {"mmr", "dino"}:
        run_mmr(args, rows)
    elif args.stage == "evidence":
        run_evidence(args, rows)
    else:
        run_answer(args, rows)


if __name__ == "__main__":
    main()
