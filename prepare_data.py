#!/usr/bin/env python3
"""Build the folder-local Qwen3-VL stable-profile training manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCE_MANIFEST = HERE / "inputs/train_manifest_best43_source.jsonl"
OUTPUT_MANIFEST = HERE / "artifacts/train_manifest_qwen3vl8b.jsonl"
OUTPUT_REPORT = HERE / "artifacts/train_manifest_qwen3vl8b.report.json"
EXPECTED_SOURCE_SHA256 = (
    "325961ae1b861c172c2285b26ee340e73f0516dd0e0cb2469a3bd331989cd56e"
)

TOPK = 32
MIN_FRAMES = 12
DECODE_MAX_SIDE = 840
MIN_VISUAL_TOKENS = 128
MAX_VISUAL_TOKENS = 256
MAX_TEXT_EVIDENCE_ITEMS = 25
MAX_EVIDENCE_CHARS = 15000
EVIDENCE_TEMPORAL_WINDOW_SECONDS = 20.0
PIPELINE_METHOD = "legacy-exact-discrete-mmr-frames-v1"
# This name matches the image profile recorded by the adapter that actually
# completed training on 2026-08-29. An older README incorrectly said 740.
PROFILE = "qwen3vl8b-reproduced-decode840-visual128-256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-source-change",
        action="store_true",
        help="Allow a changed source snapshot after manually reviewing it.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing local source snapshot: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(record)
    return records


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_source_record(record: dict[str, Any], seen: set[str]) -> None:
    required = (
        "question_id", "video_id", "video_path", "question", "answer", "fps",
        "frame_indices", "text_evidence", "pipeline",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(
            f"Incomplete source record {record.get('question_id')}: missing={missing}"
        )
    question_id = str(record["question_id"])
    if question_id in seen:
        raise ValueError(f"Duplicate question_id: {question_id}")
    seen.add(question_id)
    frames = [int(value) for value in record["frame_indices"]]
    if frames != sorted(set(frames)) or not MIN_FRAMES <= len(frames) <= TOPK:
        raise ValueError(f"Invalid frame indices for {question_id}")
    if float(record["fps"]) <= 0:
        raise ValueError(f"Invalid FPS for {question_id}")
    if not Path(record["video_path"]).is_file():
        raise FileNotFoundError(record["video_path"])
    answer_words = len(str(record["answer"]).split())
    if not 8 <= answer_words <= 11:
        raise ValueError(f"Answer is not 8-11 words for {question_id}")
    if len(record["text_evidence"]) > MAX_TEXT_EVIDENCE_ITEMS:
        raise ValueError(f"Too much text evidence for {question_id}")
    source_pipeline = record["pipeline"]
    expected_source = {
        "method": PIPELINE_METHOD,
        "topk": TOPK,
        "max_text_evidence_items": MAX_TEXT_EVIDENCE_ITEMS,
        "max_evidence_chars": MAX_EVIDENCE_CHARS,
        "evidence_temporal_window_seconds": EVIDENCE_TEMPORAL_WINDOW_SECONDS,
        "prompt": "legacy-exact-inline-v1",
    }
    mismatches = {
        key: (expected, source_pipeline.get(key))
        for key, expected in expected_source.items()
        if source_pipeline.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Source profile mismatch for {question_id}: {mismatches}")


def main() -> None:
    args = parse_args()
    actual_hash = sha256(SOURCE_MANIFEST) if SOURCE_MANIFEST.is_file() else "missing"
    if not args.allow_source_change and actual_hash != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"Local best43 source snapshot changed: {SOURCE_MANIFEST}\n"
            f"expected sha256={EXPECTED_SOURCE_SHA256}\nactual sha256={actual_hash}"
        )
    source = load_jsonl(SOURCE_MANIFEST)
    if len(source) != 634:
        raise RuntimeError(f"Expected 634 QA records, found {len(source)}")

    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    frame_histogram: Counter[int] = Counter()
    text_histogram: Counter[int] = Counter()
    for original in source:
        validate_source_record(original, seen)
        record = dict(original)
        record["pipeline"] = {
            "method": PIPELINE_METHOD,
            "profile": PROFILE,
            "topk": TOPK,
            "decode_max_side": DECODE_MAX_SIDE,
            "min_visual_tokens": MIN_VISUAL_TOKENS,
            "max_visual_tokens": MAX_VISUAL_TOKENS,
            "max_text_evidence_items": MAX_TEXT_EVIDENCE_ITEMS,
            "max_evidence_chars": MAX_EVIDENCE_CHARS,
            "evidence_temporal_window_seconds": EVIDENCE_TEMPORAL_WINDOW_SECONDS,
            "prompt": "legacy-exact-inline-v1",
            "cache_policy": "folder-local-best43-snapshot-no-feature-recompute",
        }
        result.append(record)
        frame_histogram[len(record["frame_indices"])] += 1
        text_histogram[len(record["text_evidence"])] += 1

    videos = {str(record["video_id"]) for record in result}
    if len(videos) != 415:
        raise RuntimeError(f"Expected 415 videos, found {len(videos)}")
    atomic_write_jsonl(OUTPUT_MANIFEST, result)
    report = {
        "profile": PROFILE,
        "samples": len(result),
        "videos": len(videos),
        "source_snapshot": str(SOURCE_MANIFEST),
        "source_sha256": actual_hash,
        "feature_stages_rerun": [],
        "frame_and_text_evidence_reused": True,
        "frame_count_histogram": dict(sorted(frame_histogram.items())),
        "text_evidence_count_histogram": dict(sorted(text_histogram.items())),
        "pipeline": result[0]["pipeline"],
        "output_manifest": str(OUTPUT_MANIFEST),
    }
    atomic_write_json(OUTPUT_REPORT, report)
    print(
        f"Training manifest written: samples={len(result)} videos={len(videos)} "
        f"profile={PROFILE}"
    )
    print("Reused the local best43 frame/text snapshot; no feature model was run.")
    print(OUTPUT_MANIFEST)


if __name__ == "__main__":
    main()
