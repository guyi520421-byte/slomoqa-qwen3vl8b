#!/usr/bin/env python3
"""Offline integrity checks for the organized independent pipeline."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = (
    "feature_pipeline.py", "videoitg_base.py", "dinov2_frame_filter.py",
    "clean_ocr.py", "ocr_clean_core.py", "infer_public.py",
    "eagle/model/builder.py",
    "inputs/public_selections.jsonl", "inputs/private_selections.jsonl",
    "inputs/public_evidence_ocr_clean.jsonl",
    "inputs/private_evidence_ocr_clean.jsonl",
    "inputs/private_evidence_ocr_clean_fullregen_gpu1.jsonl",
    "preflight.sh",
    "run_all.sh",
    "run_best_cached.sh",
    "validate_artifacts.py",
)
EXPECTED_SHA256 = {
    "inputs/public_selections.jsonl": "8bf79b8f11287c14f6fd93bf259ee29637a98eb22a0b680219249e5daf35d5e5",
    "inputs/private_selections.jsonl": "e9691bcb50b664534041eb0d44c48e670afddb4903b78e752d1e762cb1a62085",
    "inputs/public_evidence_ocr_clean.jsonl": "20a20f417927402a108638605914a8e5b6731bfad4773d70e1d0e5a2a8fc7798",
    "inputs/private_evidence_ocr_clean.jsonl": "b41b8cbb26e4745c0815fa3d130247f6397f1cb6f11ad86ba0396ee8d61a5dcf",
    "inputs/private_evidence_ocr_clean_fullregen_gpu1.jsonl": "1a9900a4147d146b82d7c066e155864bba6cb507b80d61f18e7abdf05d84bf0e",
}
FORBIDDEN_CODE_ROOTS = (
    "/data/wen1/slomoQa/ideoITG_dinov2MMR_OCR",
    "/data/wen1/slomoQa/ideoITG_dinov2MMR_OCRClean",
    "/data/wen1/slomoQa/videoITG_dinov2GloabalPatch",
    "/data/wen1/wearable/egolongqa/VideoITG_timeStamps_DINOv2_patch_edit",
)

def jsonl_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing package files: {missing}")
    violations = []
    for path in ROOT.rglob("*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_CODE_ROOTS:
            if forbidden in text:
                violations.append((str(path.relative_to(ROOT)), forbidden))
    if violations:
        raise RuntimeError(f"Cross-project Python dependencies remain: {violations}")
    expected_counts = {
        "inputs/public_selections.jsonl": 538,
        "inputs/private_selections.jsonl": 441,
        "inputs/public_evidence_ocr_clean.jsonl": 50,
        "inputs/private_evidence_ocr_clean.jsonl": 45,
        "inputs/private_evidence_ocr_clean_fullregen_gpu1.jsonl": 45,
    }
    for name, expected in expected_counts.items():
        actual = jsonl_count(ROOT / name)
        if actual != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {actual}")
        actual_sha256 = sha256(ROOT / name)
        if actual_sha256 != EXPECTED_SHA256[name]:
            raise RuntimeError(
                f"{name}: golden cache SHA256 mismatch; "
                f"expected {EXPECTED_SHA256[name]}, got {actual_sha256}"
            )
    print("Package integrity: PASS")
    print("Local Python code has no imports from prior project directories.")
    for name in expected_counts:
        print(f"{name}: {expected_counts[name]} records, sha256={EXPECTED_SHA256[name]}")

if __name__ == "__main__":
    main()
