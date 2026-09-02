#!/usr/bin/env python3
"""SloMo-QA: VideoITG retrieval + DINOv2 global/patch filtering + Qwen QA."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DATA_ROOT = Path("/data/wen1/datasets/slomo_qa")
# Eagle/VideoITG and the DINOv2 frame filter are vendored next to this file.
REFERENCE_ROOT = HERE
DEFAULT_SELECTOR = Path("/data/wen1/MLLMs/VideoITG-8B")
DEFAULT_DINO = Path("/data/wen1/MLLMs/dinov2_vitb14_reg4_pretrain.pth")
DEFAULT_QWEN = Path("/data/wen1/MLLMs/Qwen_25vl_32b")
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
# All three checkpoints and the SigLIP vision tower are local/cached. Avoid a
# slow or failing Hugging Face network probe on compute nodes.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Question-aware frame retrieval and open QA for SloMo-QA."
    )
    parser.add_argument("--split", choices=("private", "public"), default="private")
    parser.add_argument(
        "--stage",
        choices=("all", "select", "deduplicate", "dino", "answer"),
        default="all",
    )
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--selector-path", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--dino-path", type=Path, default=DEFAULT_DINO)
    parser.add_argument("--qwen-path", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--selections", type=Path)
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
    parser.add_argument("--dino-similarity-threshold", type=float, default=0.97)
    parser.add_argument(
        "--dino-patch-similarity-threshold", type=float, default=0.90
    )
    parser.add_argument(
        "--dino-patch-change-similarity-threshold", type=float, default=0.80
    )
    parser.add_argument(
        "--dino-max-changed-patch-ratio", type=float, default=0.10
    )
    parser.add_argument(
        "--dino-temporal-window-seconds", type=float, default=5.0
    )
    parser.add_argument("--dino-batch-size", type=int, default=8)
    parser.add_argument(
        "--dino-dtype", choices=("fp16", "bf16", "fp32"), default="fp16"
    )
    parser.add_argument(
        "--dino-decode-threads",
        type=int,
        default=1,
        help=(
            "Decord video decode threads. One thread avoids VP9/WebM "
            "avcodec_send_packet failures seen in the SloMo-QA downloads."
        ),
    )
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
    parser.add_argument("--max-new-tokens", type=int, default=64)
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
    args.output = args.output or (
        HERE / "outputs" / f"{args.split}_videoitg_base_predictions.json"
    )

    positive = {
        "--candidate-frames": args.candidate_frames,
        "--min-candidate-frames": args.min_candidate_frames,
        "--target-fps": args.target_fps,
        "--dino-pool-size": args.dino_pool_size,
        "--topk": args.topk,
        "--min-answer-frames": args.min_answer_frames,
        "--dino-batch-size": args.dino_batch_size,
        "--dino-decode-threads": args.dino_decode_threads,
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
    thresholds = (
        args.dino_similarity_threshold,
        args.dino_patch_similarity_threshold,
        args.dino_patch_change_similarity_threshold,
        args.dino_max_changed_patch_ratio,
    )
    if any(not 0.0 <= value <= 1.0 for value in thresholds):
        parser.error("DINO similarity and ratio thresholds must be in [0, 1]")
    return args


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.questions.is_file():
        raise FileNotFoundError(f"Questions CSV not found: {args.questions}")
    with args.questions.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"question_id", "video_id", "question"}
    if not rows or not required <= rows[0].keys():
        raise ValueError(f"CSV must contain {sorted(required)}")
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    missing = [
        row["video_id"]
        for row in rows
        if not (args.video_dir / f'{row["video_id"]}.mp4').is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(set(missing))} videos under {args.video_dir}; "
            f"first={missing[0]}.mp4"
        )
    return rows


def group_by_video(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)
    return grouped


def load_selections(path: Path) -> dict[str, dict[str, Any]]:
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
            if record.get("question_id"):
                records[str(record["question_id"])] = record
    return records


def write_selections(
    path: Path,
    rows: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = records.get(row["question_id"])
            if record is not None:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_predictions(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["question_id"]): str(item["prediction"]).strip()
        for item in data
        if item.get("question_id") and str(item.get("prediction", "")).strip()
    }


def write_predictions(
    path: Path, rows: list[dict[str, Any]], predictions: dict[str, str]
) -> None:
    result = [
        {"question_id": row["question_id"], "prediction": predictions[row["question_id"]]}
        for row in rows
        if row["question_id"] in predictions
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def selection_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": "videoitg-topk-v1",
        "selector_path": str(args.selector_path.resolve()),
        "candidate_frames": args.candidate_frames,
        "target_fps": args.target_fps,
        "dino_pool_size": args.dino_pool_size,
    }


def dino_config(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.dino_path.resolve()
    stat = checkpoint.stat() if checkpoint.is_file() else None
    return {
        "method": "dinov2-global-corresponding-patch-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_size": stat.st_size if stat else None,
        "similarity_threshold": args.dino_similarity_threshold,
        "patch_similarity_threshold": args.dino_patch_similarity_threshold,
        "patch_change_similarity_threshold": args.dino_patch_change_similarity_threshold,
        "max_changed_patch_ratio": args.dino_max_changed_patch_ratio,
        "temporal_window_seconds": args.dino_temporal_window_seconds,
        "topk": args.topk,
        "min_answer_frames": args.min_answer_frames,
        "dtype": args.dino_dtype,
    }


def candidate_indices(
    total_frames: int, fps: float, target_fps: float, max_frames: int
) -> list[int]:
    stride = max(1, round(fps / target_fps))
    eligible = list(range(0, total_frames, stride))
    if len(eligible) <= max_frames:
        return eligible
    scale = len(eligible) / max_frames
    positions = [round((index + 1) * scale - 1) for index in range(max_frames)]
    return [eligible[position] for position in positions]


def require_reference_root() -> None:
    if not REFERENCE_ROOT.is_dir():
        raise FileNotFoundError(f"Reference VideoITG checkout not found: {REFERENCE_ROOT}")
    root = str(REFERENCE_ROOT)
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


def selector_runtime() -> tuple[Any, ...]:
    require_reference_root()
    import torch
    from decord import VideoReader, cpu
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    from eagle.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from eagle.mm_utils import get_model_name_from_path, tokenizer_image_token
    from eagle.model.builder import load_pretrained_model

    return (
        torch,
        VideoReader,
        cpu,
        DEFAULT_IMAGE_TOKEN,
        IMAGE_TOKEN_INDEX,
        get_model_name_from_path,
        tokenizer_image_token,
        load_pretrained_model,
    )


def pad_tokens(tokenizer: Any, tokens: Any, torch: Any) -> tuple[Any, int]:
    padding = tokenizer.pad_token_id
    if padding is None:
        padding = tokenizer.eos_token_id
    return tokens.unsqueeze(0), padding


def prepare_video_candidates(
    video_path: Path,
    budget: int,
    args: argparse.Namespace,
    runtime: tuple[Any, ...],
    image_processor: Any,
) -> tuple[Any, list[int], float, int]:
    torch, video_reader, cpu, *_ = runtime
    reader = video_reader(str(video_path), ctx=cpu(0), num_threads=4)
    total = len(reader)
    fps = float(reader.get_avg_fps())
    sampled = candidate_indices(total, fps, args.target_fps, budget)
    try:
        arrays = reader.get_batch(sampled).asnumpy()
    except Exception as error:
        # Decord's threaded FFmpeg decoder can fail on otherwise valid VP9
        # streams with avcodec_send_packet(...)=EAGAIN.  Reopening the stream
        # with one worker avoids the decoder race while keeping the fast path
        # for codecs that support threaded random access reliably.
        print(
            f"Warning: threaded Decord read failed for {video_path.name}; "
            f"retrying with one decode thread ({type(error).__name__}: {error})",
            flush=True,
        )
        del reader
        reader = video_reader(str(video_path), ctx=cpu(0), num_threads=1)
        arrays = reader.get_batch(sampled).asnumpy()
    del reader
    tensor = image_processor.preprocess(arrays, return_tensors="pt")[
        "pixel_values"
    ].half().to(torch.device(args.selector_device))
    del arrays
    return tensor, sampled, fps, total


def score_question(
    row: dict[str, Any],
    video_tensor: Any,
    sampled: list[int],
    fps: float,
    total_frames: int,
    args: argparse.Namespace,
    runtime: tuple[Any, ...],
    selector: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    (
        torch,
        _,
        _,
        image_token,
        image_token_index,
        _,
        tokenizer_image_token,
        _,
    ) = runtime
    prompt = (
        image_token
        + "Question: "
        + row["question"]
        + "\nLocate the video frames most relevant to answering the question.\n"
    )
    tokens = tokenizer_image_token(
        prompt, tokenizer, image_token_index, return_tensors="pt"
    )
    input_ids, padding = pad_tokens(tokenizer, tokens, torch)
    input_ids = input_ids.to(torch.device(args.selector_device))
    attention_mask = input_ids.ne(padding)
    with torch.inference_mode():
        response = selector(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=[video_tensor],
        )
        scores = response.logits[0].sigmoid().view(-1)
        if scores.numel() != len(sampled):
            raise RuntimeError(
                f"VideoITG produced {scores.numel()} scores for {len(sampled)} frames"
            )
        values, order = torch.sort(scores, descending=True)
        keep = min(args.dino_pool_size, len(sampled))
        ranked = [sampled[index] for index in order[:keep].tolist()]
        ranked_scores = [round(float(value), 6) for value in values[:keep].tolist()]
    chronological = sorted(ranked)
    score_map = dict(zip(ranked, ranked_scores))
    return {
        "question_id": row["question_id"],
        "video_id": row["video_id"],
        "video_path": f'{row["video_id"]}.mp4',
        "question": row["question"],
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": total_frames / fps,
        "candidate_frames": len(sampled),
        "ranked_frame_indices": ranked,
        "ranked_scores": ranked_scores,
        "selected_frame_indices": chronological,
        "selected_scores": [score_map[index] for index in chronological],
        "selected_timestamps": [round(index / fps, 3) for index in chronological],
        "selection_config": selection_config(args),
    }


def run_select(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    config = selection_config(args)
    selections = load_selections(args.selections) if args.resume else {}
    selections = {
        key: value
        for key, value in selections.items()
        if value.get("selection_config") == config
    }
    pending = [row for row in rows if row["question_id"] not in selections]
    print(
        f"Select: questions={len(rows)} cached={len(rows)-len(pending)} "
        f"pending={len(pending)}"
    )
    failure_report = args.selections.with_name(
        f"{args.selections.stem}.select_failures.json"
    )
    failure_report.parent.mkdir(parents=True, exist_ok=True)
    empty_report = failure_report.with_suffix(failure_report.suffix + ".tmp")
    empty_report.write_text("[]\n", encoding="utf-8")
    os.replace(empty_report, failure_report)
    if not pending:
        return
    runtime = selector_runtime()
    torch, _, _, _, _, get_model_name, _, load_model = runtime
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VideoITG")
    tokenizer, selector, image_processor, _ = load_model(
        str(args.selector_path),
        None,
        get_model_name(str(args.selector_path)),
        device_map=args.selector_device,
    )
    selector.eval()
    grouped = group_by_video(pending)
    completed = len(rows) - len(pending)
    errors = 0
    decode_failures: list[dict[str, Any]] = []
    for video_position, (video_id, questions) in enumerate(grouped.items(), start=1):
        video_path = args.video_dir / f"{video_id}.mp4"
        budget = args.candidate_frames
        video_tensor = None
        try:
            while True:
                try:
                    video_tensor, sampled, fps, total = prepare_video_candidates(
                        video_path, budget, args, runtime, image_processor
                    )
                    break
                except RuntimeError as error:
                    next_budget = max(args.min_candidate_frames, budget // 2)
                    if "out of memory" not in str(error).lower() or next_budget >= budget:
                        raise
                    print(f"{video_path.name}: OOM, retrying with {next_budget} candidates")
                    budget = next_budget
                    gc.collect()
                    torch.cuda.empty_cache()
        except Exception as error:
            errors += len(questions)
            failure = {
                "video_id": video_id,
                "video_path": str(video_path),
                "question_ids": [row["question_id"] for row in questions],
                "stage": "video_decode",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            decode_failures.append(failure)
            failure_report.parent.mkdir(parents=True, exist_ok=True)
            temporary = failure_report.with_suffix(failure_report.suffix + ".tmp")
            temporary.write_text(
                json.dumps(decode_failures, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, failure_report)
            print(
                f"SKIP select video [{video_position}/{len(grouped)}] "
                f"{video_path.name}: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
            continue
        try:
            for row in questions:
                try:
                    record = score_question(
                        row,
                        video_tensor,
                        sampled,
                        fps,
                        total,
                        args,
                        runtime,
                        selector,
                        tokenizer,
                    )
                    selections[row["question_id"]] = record
                    completed += 1
                    write_selections(args.selections, rows, selections)
                except Exception as error:
                    errors += 1
                    print(
                        f"ERROR select {row['question_id']}: "
                        f"{type(error).__name__}: {error}",
                        file=sys.stderr,
                    )
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(
                f"[{video_position}/{len(grouped)}] {video_path.name}: "
                f"questions={len(questions)} completed={completed}/{len(rows)} "
                f"candidates={len(sampled)} peak_cuda={peak:.2f} GiB",
                flush=True,
            )
        finally:
            del video_tensor
            gc.collect()
            torch.cuda.empty_cache()
    print(
        f"Selections written to {args.selections}; errors={errors} "
        f"skipped_decode_videos={len(decode_failures)}"
    )
    if decode_failures:
        print(f"Select failure report: {failure_report}")


def filter_cached_features(
    ranked: list[int],
    fps: float,
    all_indices: list[int],
    features: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    torch = __import__("torch")
    feature_positions = {index: position for position, index in enumerate(all_indices)}
    global_features = features["global"]
    patch_features = features["patch"]
    selected: list[int] = []
    redundant: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    for position, frame_index in enumerate(ranked):
        comparisons = [
            kept
            for kept in selected
            if args.dino_temporal_window_seconds <= 0
            or abs(frame_index - ranked[kept]) / fps
            <= args.dino_temporal_window_seconds
        ]
        duplicate_position = None
        metrics: dict[str, float] = {}
        if comparisons:
            current = feature_positions[frame_index]
            comparison_features = [feature_positions[ranked[kept]] for kept in comparisons]
            global_similarity = global_features[comparison_features] @ global_features[current]
            patch_similarity = (
                patch_features[comparison_features]
                * patch_features[current].unsqueeze(0)
            ).sum(dim=-1)
            patch_mean = patch_similarity.mean(dim=-1)
            changed_ratio = (
                patch_similarity < args.dino_patch_change_similarity_threshold
            ).float().mean(dim=-1)
            duplicate_mask = (
                (global_similarity >= args.dino_similarity_threshold)
                & (patch_mean >= args.dino_patch_similarity_threshold)
                & (changed_ratio <= args.dino_max_changed_patch_ratio)
            )
            if bool(duplicate_mask.any().item()):
                combined = (global_similarity + patch_mean) / 2
                best = int(
                    combined.masked_fill(~duplicate_mask, float("-inf")).argmax().item()
                )
                duplicate_position = comparisons[best]
                metrics = {
                    "global_cosine_similarity": round(float(global_similarity[best]), 6),
                    "mean_patch_cosine_similarity": round(float(patch_mean[best]), 6),
                    "changed_patch_ratio": round(float(changed_ratio[best]), 6),
                }
            elif bool((global_similarity >= args.dino_similarity_threshold).any().item()):
                best = int(global_similarity.argmax().item())
                protected.append(
                    {
                        "frame_index": frame_index,
                        "compared_to_frame_index": ranked[comparisons[best]],
                        "global_cosine_similarity": round(float(global_similarity[best]), 6),
                        "mean_patch_cosine_similarity": round(float(patch_mean[best]), 6),
                        "changed_patch_ratio": round(float(changed_ratio[best]), 6),
                    }
                )
        if duplicate_position is None:
            selected.append(position)
        else:
            redundant.append(
                {
                    "frame_index": frame_index,
                    "duplicate_of_frame_index": ranked[duplicate_position],
                    "videoitg_rank": position + 1,
                    "backfilled": False,
                    **metrics,
                }
            )
    backfilled: list[int] = []
    selected_set = set(selected)
    if len(selected) < args.min_answer_frames:
        positions = {index: position for position, index in enumerate(ranked)}
        for item in redundant:
            position = positions[item["frame_index"]]
            if position not in selected_set:
                selected.append(position)
                selected_set.add(position)
                backfilled.append(item["frame_index"])
                item["backfilled"] = True
            if len(selected) >= args.min_answer_frames:
                break
    selected.sort()
    budget_excluded = selected[args.topk :]
    selected = selected[: args.topk]
    ranked_selected = [ranked[position] for position in selected]
    chronological = sorted(ranked_selected)
    return {
        "dino_ranked_frame_indices": ranked_selected,
        "dino_selected_frame_indices": chronological,
        "dino_selected_timestamps": [round(index / fps, 3) for index in chronological],
        "dino_redundant_frames": redundant,
        "dino_local_change_protected_frames": protected,
        "dino_backfilled_frame_indices": backfilled,
        "dino_budget_excluded_frame_indices": [ranked[p] for p in budget_excluded],
        "dino_source_pool_size": len(ranked),
        "dino_patch_count": features.get("patch_count"),
    }


def run_deduplicate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    require_reference_root()
    from dinov2_frame_filter import DINOv2RedundancyFilter

    config = dino_config(args)
    selections = load_selections(args.selections)
    pending = [
        row
        for row in rows
        if row["question_id"] in selections
        and (
            selections[row["question_id"]].get("dino_filter_config") != config
            or selections[row["question_id"]].get("dino_source_frame_indices")
            != selections[row["question_id"]].get("ranked_frame_indices")
        )
    ]
    print(
        f"DINO: questions={len(rows)} pending={len(pending)} "
        f"missing_select={len(rows)-len(selections)}"
    )
    if not pending:
        return
    filterer = DINOv2RedundancyFilter(
        args.dino_path,
        device=args.dino_device,
        batch_size=args.dino_batch_size,
        dtype=args.dino_dtype,
        decode_threads=args.dino_decode_threads,
    )
    grouped = group_by_video(pending)
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
                result = filter_cached_features(
                    ranked, float(record["fps"]), all_indices, features, args
                )
                score_map = dict(zip(ranked, record.get("ranked_scores", [])))
                result["dino_selected_scores"] = [
                    score_map.get(index) for index in result["dino_selected_frame_indices"]
                ]
                record.update(result)
                record["dino_source_frame_indices"] = ranked
                record["dino_filter_config"] = config
                write_selections(args.selections, rows, selections)
            print(
                f"[{video_position}/{len(grouped)}] {video_path.name}: "
                f"questions={len(questions)} union_frames={len(all_indices)}",
                flush=True,
            )
            del features
        except Exception as error:
            errors += len(questions)
            print(
                f"ERROR DINO {video_path.name}: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
        finally:
            gc.collect()
            filterer.torch.cuda.empty_cache()
    print(f"DINO selections written to {args.selections}; errors={errors}")


def decode_frame_map(
    video_path: Path, indices: list[int], max_side: int, cv2: Any, image_cls: Any
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
            scale = min(1.0, max_side / max(height, width)) if max_side > 0 else 1.0
            if scale < 1:
                frame = cv2.resize(
                    frame,
                    (round(width * scale), round(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            frames[index] = image_cls.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames, fps


def load_qwen(args: argparse.Namespace, torch: Any) -> tuple[Any, Any]:
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": "auto",
        "attn_implementation": args.attention,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.qwen_path, **kwargs
    ).eval()
    model.generation_config.temperature = None
    processor = AutoProcessor.from_pretrained(
        args.qwen_path,
        min_pixels=args.min_visual_tokens * 28 * 28,
        max_pixels=args.max_visual_tokens * 28 * 28,
        local_files_only=True,
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


def generate_open_answer(
    row: dict[str, Any],
    frames: list[Any],
    timestamps: list[float],
    model: Any,
    processor: Any,
    process_vision_info: Any,
    torch: Any,
    args: argparse.Namespace,
) -> str:
    content: list[dict[str, Any]] = []
    for frame, timestamp in zip(frames, timestamps):
        content.append({"type": "text", "text": f"Timestamp: {timestamp:.1f}s"})
        content.append({"type": "image", "image": frame})
    content.append(
        {
            "type": "text",
            "text": (
                "The images are question-relevant frames from a complete video, "
                "shown chronologically. Answer based only on their visual evidence.\n"
                f"Question: {row['question']}\n"
                "Return only a concise natural-language answer, with no prefix."
            ),
        }
    )
    messages = [{"role": "user", "content": content}]
    chat = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model_input_device(model, torch))
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    new_tokens = generated[0, inputs.input_ids.shape[1] :]
    answer = processor.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    answer = re.sub(r"^\s*(?:answer|prediction)\s*:\s*", "", answer, flags=re.I)
    answer = " ".join(answer.replace("```", "").split())
    if not answer:
        raise ValueError("Qwen returned an empty answer")
    return answer


def run_answer(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    import cv2
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    selections = load_selections(args.selections)
    predictions = load_predictions(args.output) if args.resume else {}
    valid_ids = {row["question_id"] for row in rows}
    predictions = {key: value for key, value in predictions.items() if key in valid_ids}
    pending = [
        row
        for row in rows
        if row["question_id"] not in predictions
        and selections.get(row["question_id"], {}).get("dino_ranked_frame_indices")
    ]
    missing = sum(
        not selections.get(row["question_id"], {}).get("dino_ranked_frame_indices")
        for row in rows
    )
    print(
        f"Answer: questions={len(rows)} cached={len(predictions)} "
        f"pending={len(pending)} missing_dino={missing}"
    )
    if not pending:
        write_predictions(args.output, rows, predictions)
        return
    model, processor = load_qwen(args, torch)
    grouped = group_by_video(pending)
    errors = 0
    for video_position, (video_id, questions) in enumerate(grouped.items(), start=1):
        union_indices = sorted(
            {
                int(index)
                for row in questions
                for index in selections[row["question_id"]][
                    "dino_ranked_frame_indices"
                ][: args.topk]
            }
        )
        video_path = args.video_dir / f"{video_id}.mp4"
        try:
            frame_map, fps = decode_frame_map(
                video_path, union_indices, args.decode_max_side, cv2, Image
            )
            for row in questions:
                ranked = [
                    int(index)
                    for index in selections[row["question_id"]][
                        "dino_ranked_frame_indices"
                    ]
                ]
                budget = min(args.topk, len(ranked))
                while True:
                    chosen = sorted(index for index in ranked[:budget] if index in frame_map)
                    try:
                        predictions[row["question_id"]] = generate_open_answer(
                            row,
                            [frame_map[index] for index in chosen],
                            [index / fps for index in chosen],
                            model,
                            processor,
                            process_vision_info,
                            torch,
                            args,
                        )
                        write_predictions(args.output, rows, predictions)
                        break
                    except RuntimeError as error:
                        next_budget = max(args.min_answer_frames, budget // 2)
                        if (
                            "out of memory" not in str(error).lower()
                            or next_budget >= budget
                        ):
                            raise
                        budget = next_budget
                        gc.collect()
                        torch.cuda.empty_cache()
                print(
                    f"[{len(predictions)}/{len(rows)}] {row['question_id']}: "
                    f"frames={len(chosen)}",
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
    write_predictions(args.output, rows, predictions)
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
        (VIDEOITG_PYTHON, "deduplicate"),
        (QWEN_PYTHON, "answer"),
    ):
        print(f"Launching {stage} with {python}", flush=True)
        subprocess.run([str(python), *child_arguments(stage)], cwd=HERE, check=True)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.stage == "all":
        load_rows(args)
        run_all()
        return
    rows = load_rows(args)
    if args.stage == "select":
        run_select(args, rows)
    elif args.stage in {"deduplicate", "dino"}:
        run_deduplicate(args, rows)
    else:
        run_answer(args, rows)


if __name__ == "__main__":
    main()
