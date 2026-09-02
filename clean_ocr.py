#!/usr/bin/env python3
"""Strictly clean only the OCR channel while preserving ASR and subtitles."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from ocr_clean_core import clean_ocr_only, compact_text

HERE = Path(__file__).resolve().parent
CLEANING_METHOD = "strict-obvious-ocr-noise-only-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("clean",), default="clean")
    parser.add_argument("--split", choices=("public", "private"), default="public")
    parser.add_argument("--source-evidence", type=Path)
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--cleaned-evidence", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    # Accepted for compatibility with the common stage wrappers; unused here.
    parser.add_argument("--questions", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--video-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source_evidence = args.source_evidence or (
        HERE / "inputs" / f"{args.split}_evidence_raw.jsonl"
    )
    args.selections = args.selections or (
        HERE / "inputs" / f"{args.split}_selections.jsonl"
    )
    args.cleaned_evidence = args.cleaned_evidence or (
        HERE / "inputs" / f"{args.split}_evidence_ocr_clean.jsonl"
    )
    for path in (args.source_evidence, args.selections):
        if not path.is_file():
            parser.error(f"Required input is missing: {path}")
    return args


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def metadata_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def report_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.report.json")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def selection_durations(path: Path) -> dict[str, float]:
    durations = {}
    for record in read_jsonl(path):
        video_id = str(record.get("video_id", ""))
        duration = float(record.get("duration_seconds", 0.0))
        if video_id and duration > 0:
            durations[video_id] = duration
    return durations


def inferred_duration(record: dict[str, Any]) -> float:
    return max(
        (
            float(item.get("end", item.get("start", 0.0)))
            for group in ("asr", "subtitles", "ocr")
            for item in record.get(group, [])
        ),
        default=0.0,
    )


def cleaning_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": CLEANING_METHOD,
        "source_evidence": file_identity(args.source_evidence),
        "selections": file_identity(args.selections),
        "cleaner": file_identity(HERE / "ocr_clean_core.py"),
        "changed_channel": "ocr",
        "asr_changed": False,
        "subtitles_changed": False,
    }


def run_clean(args: argparse.Namespace) -> None:
    config = cleaning_config(args)
    meta = metadata_path(args.cleaned_evidence)
    if args.resume and args.cleaned_evidence.is_file() and meta.is_file():
        try:
            cached = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached = None
        if cached == config:
            print(f"OCR-clean cache is complete: {args.cleaned_evidence}")
            return

    durations = selection_durations(args.selections)
    cleaned_records = []
    reasons: Counter[str] = Counter()
    per_video = []
    raw_total = kept_total = 0
    for record in read_jsonl(args.source_evidence):
        video_id = str(record.get("video_id", ""))
        raw_ocr = list(record.get("ocr", []))
        duration = durations.get(video_id, inferred_duration(record))
        kept, dropped = clean_ocr_only(raw_ocr, duration)
        cleaned_records.append({**record, "ocr": kept})
        raw_total += len(raw_ocr)
        kept_total += len(kept)
        reasons.update(compact_text(item.get("filter_reason")) for item in dropped)
        per_video.append({
            "video_id": video_id,
            "raw_ocr": len(raw_ocr),
            "kept_ocr": len(kept),
            "dropped_ocr": len(dropped),
        })

    write_jsonl(args.cleaned_evidence, cleaned_records)
    write_json(meta, config)
    write_json(report_path(args.cleaned_evidence), {
        "config": config,
        "videos": len(cleaned_records),
        "raw_ocr": raw_total,
        "kept_ocr": kept_total,
        "dropped_ocr": raw_total - kept_total,
        "drop_reasons": dict(reasons),
        "per_video": per_video,
    })
    print(
        f"OCR clean: videos={len(cleaned_records)} raw={raw_total} "
        f"kept={kept_total} dropped={raw_total-kept_total}"
    )
    print(f"Cleaned evidence written to {args.cleaned_evidence}")


def main() -> None:
    run_clean(parse_args())


if __name__ == "__main__":
    main()
