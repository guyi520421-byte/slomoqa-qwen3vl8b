#!/usr/bin/env python3
"""Validate SloMoQA selection, evidence, and prediction coverage."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATA_ROOT = Path("/data/wen1/datasets/slomo_qa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("public", "private"), required=True)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    args.questions = args.questions or (
        DATA_ROOT / f"sf20k_{args.split}_test_questions.csv"
    )
    if not any((args.selections, args.evidence, args.predictions)):
        parser.error("supply at least one artifact to validate")
    return args


def load_questions(path: Path) -> tuple[list[str], set[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"question_id", "video_id"} <= set(rows[0]):
        raise RuntimeError(f"Invalid question CSV: {path}")
    question_ids = [str(row["question_id"]) for row in rows]
    if len(question_ids) != len(set(question_ids)):
        raise RuntimeError(f"Duplicate question_id in {path}")
    return question_ids, {str(row["video_id"]) for row in rows}


def load_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from error
            value = str(record.get(key, ""))
            if not value:
                raise RuntimeError(f"Missing {key} at {path}:{line_number}")
            if value in records:
                raise RuntimeError(f"Duplicate {key}={value} in {path}")
            records[value] = record
    return records


def assert_exact_coverage(label: str, expected: set[str], actual: set[str]) -> None:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(
            f"{label} coverage mismatch: missing={len(missing)} "
            f"extra={len(extra)} first_missing={missing[:1]} first_extra={extra[:1]}"
        )


def validate_selections(path: Path, question_ids: list[str]) -> None:
    records = load_jsonl(path, "question_id")
    assert_exact_coverage("selections", set(question_ids), set(records))
    bad = [
        question_id
        for question_id in question_ids
        if not records[question_id].get("mmr_selected_frame_indices")
        or not records[question_id].get("mmr_ranked_frame_indices")
    ]
    if bad:
        raise RuntimeError(
            f"Selections without completed MMR fields: {len(bad)}; first={bad[0]}"
        )
    print(f"Selections: PASS ({len(records)} questions)")


def validate_evidence(path: Path, video_ids: set[str]) -> None:
    records = load_jsonl(path, "video_id")
    assert_exact_coverage("evidence", video_ids, set(records))
    invalid = [
        video_id
        for video_id, record in records.items()
        if not all(
            isinstance(record.get(channel), list)
            for channel in ("asr", "subtitles", "ocr")
        )
    ]
    if invalid:
        raise RuntimeError(
            f"Evidence with invalid channels: {len(invalid)}; first={invalid[0]}"
        )
    print(f"Evidence: PASS ({len(records)} videos)")


def validate_predictions(path: Path, question_ids: list[str]) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid prediction JSON: {path}") from error
    if not isinstance(payload, list):
        raise RuntimeError(f"Predictions must be a JSON array: {path}")
    records: dict[str, str] = {}
    for offset, item in enumerate(payload):
        if not isinstance(item, dict):
            raise RuntimeError(f"Prediction item {offset} is not an object")
        question_id = str(item.get("question_id", ""))
        prediction = str(item.get("prediction", "")).strip()
        if not question_id or not prediction:
            raise RuntimeError(f"Blank prediction item at offset {offset}")
        if question_id in records:
            raise RuntimeError(f"Duplicate prediction for {question_id}")
        records[question_id] = prediction
    assert_exact_coverage("predictions", set(question_ids), set(records))
    print(f"Predictions: PASS ({len(records)} questions, all non-empty)")


def main() -> None:
    args = parse_args()
    question_ids, video_ids = load_questions(args.questions)
    if args.selections:
        validate_selections(args.selections, question_ids)
    if args.evidence:
        validate_evidence(args.evidence, video_ids)
    if args.predictions:
        validate_predictions(args.predictions, question_ids)


if __name__ == "__main__":
    main()
