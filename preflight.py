#!/usr/bin/env python3
"""Decode and tokenize high-cost training samples without loading Qwen3-VL."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoProcessor

import train_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=train_lora.DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-sequence-length", type=int, default=32768)
    args = parser.parse_args()
    if args.samples <= 0 or args.max_sequence_length <= 0:
        parser.error("limits must be positive")
    return args


def main() -> None:
    args = parse_args()
    records = train_lora.read_manifest(train_lora.DEFAULT_MANIFEST, 8, 11)
    config = AutoConfig.from_pretrained(
        args.model, local_files_only=True
    )
    vision = config.vision_config
    token_side = int(vision.patch_size) * int(vision.spatial_merge_size)
    pixel_area = token_side**2
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=train_lora.MIN_VISUAL_TOKENS * pixel_area,
        max_pixels=train_lora.MAX_VISUAL_TOKENS * pixel_area,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    collator = train_lora.LegacyExactFrameCollator(
        processor, process_vision_info, args.max_sequence_length
    )
    ranked = sorted(
        records,
        key=lambda record: (
            len(record["frame_indices"]),
            sum(
                len(str(item.get("text", ""))) + 40
                for item in record["text_evidence"]
            ),
        ),
        reverse=True,
    )
    selected = ranked[:args.samples]
    maximum = 0
    for position, record in enumerate(selected, start=1):
        batch = collator([record])
        tokens = int(batch["input_ids"].shape[1])
        targets = int((batch["labels"] != -100).sum().item())
        grids = batch["image_grid_thw"]
        merge = int(processor.image_processor.merge_size)
        visual_tokens = sum(
            int(grid.prod().item()) // (merge * merge) for grid in grids
        )
        pixel_mib = batch["pixel_values"].numel() * batch["pixel_values"].element_size()
        pixel_mib /= 1024**2
        maximum = max(maximum, tokens)
        print(
            f"Token preflight [{position}/{len(selected)}] "
            f"{record['question_id']}: frames={len(record['frame_indices'])} "
            f"text_evidence={len(record['text_evidence'])} tokens={tokens} "
            f"visual_tokens={visual_tokens} pixel_tensor={pixel_mib:.1f} MiB "
            f"answer_targets={targets}",
            flush=True,
        )
        del batch
        gc.collect()
    print(f"Token preflight passed; maximum observed length={maximum}")


if __name__ == "__main__":
    main()
