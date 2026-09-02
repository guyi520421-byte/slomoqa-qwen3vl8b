#!/usr/bin/env python3
"""Folder-local strict OCR cleaner shared by Qwen3-VL OCR experiments."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable


HARD_OCR_NOISE = (
    re.compile(r"^frame\s*id\s*\d+\s*,?\s*time\s*[\d.]+s?\s*:?$", re.I),
    re.compile(r"^(?:omeleto|short of the week|filmshortage)\s*$", re.I),
    re.compile(r"^(?:subscribe|like and subscribe|youtube)\s*$", re.I),
)

CREDIT_PATTERN = re.compile(
    r"\b(?:directed|produced|executive producer|screenplay|written|edited|"
    r"cinematography|music by|casting|starring|cast|production|copyright|"
    r"all rights reserved)\b",
    re.I,
)


def compact_text(value: Any) -> str:
    text = re.sub(r"\{\\[^}]+\}|<[^>]+>", " ", str(value or ""))
    return " ".join(text.replace("\\N", " ").replace("\\n", " ").split()).strip()


def normalized_text(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", compact_text(value).lower()))


def clean_ocr_records(
    records: Iterable[dict[str, Any]],
    *,
    duration: float,
    repetition_threshold: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove obvious OCR noise while retaining meaningful text and credits."""

    raw = [dict(item) for item in records if compact_text(item.get("text"))]
    counts = Counter(normalized_text(item.get("text")) for item in raw)
    spans: dict[str, list[float]] = {}
    for item in raw:
        key = normalized_text(item.get("text"))
        spans.setdefault(key, []).append(float(item.get("start", 0.0)))

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    seen_nearby: set[tuple[str, int]] = set()
    repeat_floor = max(repetition_threshold, math.ceil(len(raw) * 0.08))
    for item in sorted(raw, key=lambda value: float(value.get("start", 0.0))):
        text = compact_text(item.get("text"))
        norm = normalized_text(text)
        timestamp = float(item.get("start", 0.0))
        reason: str | None = None
        if any(pattern.search(text) for pattern in HARD_OCR_NOISE):
            reason = "hard_noise_rule"
        repeated_span = max(spans.get(norm, [timestamp])) - min(
            spans.get(norm, [timestamp])
        )
        if (
            reason is None
            and counts[norm] >= repeat_floor
            and repeated_span >= max(20.0, duration * 0.12)
            and len(text) <= 48
        ):
            reason = "repeated_watermark"
        nearby_key = (norm, round(timestamp / 2.0))
        if reason is None and nearby_key in seen_nearby:
            reason = "near_duplicate"
        if reason is not None:
            dropped.append({**item, "text": text, "filter_reason": reason})
            continue
        seen_nearby.add(nearby_key)
        is_credit = bool(CREDIT_PATTERN.search(text))
        kept.append(
            {
                **item,
                "text": text,
                "evidence_type": "ocr",
                "ocr_category": "credit" if is_credit else "plot_text",
                "retrieval_weight": 0.12 if is_credit else 0.40,
                "relative_time": round(timestamp / max(duration, 1e-6), 4),
            }
        )
    return kept, dropped


def ocr_identity(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        round(float(item.get("start", 0.0)), 6),
        round(float(item.get("end", item.get("start", 0.0))), 6),
        item.get("frame_index"),
        normalized_text(item.get("text")),
    )


def clean_ocr_only(
    records: list[dict[str, Any]], duration: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return original OCR objects minus rejected records."""

    enriched, dropped = clean_ocr_records(records, duration=duration)
    remaining = Counter(ocr_identity(item) for item in enriched)
    kept_original = []
    for item in records:
        identity = ocr_identity(item)
        if remaining[identity] > 0:
            kept_original.append(dict(item))
            remaining[identity] -= 1
    if any(remaining.values()):
        raise RuntimeError("Could not map cleaned OCR records back to source evidence")
    return kept_original, dropped
