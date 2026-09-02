#!/usr/bin/env python3
"""Offline checks for the independent Qwen3-VL training pipeline."""

from __future__ import annotations

import json
import unittest

import prepare_data
import train_lora


class Qwen3VLTrainingPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = prepare_data.load_jsonl(prepare_data.SOURCE_MANIFEST)
        cls.prepared = prepare_data.load_jsonl(prepare_data.OUTPUT_MANIFEST)

    def test_all_best43_examples_are_reused(self) -> None:
        self.assertEqual(len(self.source), 634)
        self.assertEqual(len(self.prepared), 634)
        for source, prepared in zip(self.source, self.prepared):
            for key in (
                "question_id", "video_id", "video_path", "question", "answer",
                "fps", "frame_indices", "text_evidence",
            ):
                self.assertEqual(source[key], prepared[key])

    def test_stable_image_profile(self) -> None:
        for record in self.prepared:
            pipeline = record["pipeline"]
            self.assertEqual(pipeline["profile"], prepare_data.PROFILE)
            self.assertEqual(pipeline["topk"], 32)
            self.assertEqual(pipeline["decode_max_side"], 840)
            self.assertEqual(pipeline["min_visual_tokens"], 128)
            self.assertEqual(pipeline["max_visual_tokens"], 256)
            self.assertEqual(pipeline["max_text_evidence_items"], 25)
            self.assertEqual(pipeline["max_evidence_chars"], 15000)

    def test_training_reader_and_video_split(self) -> None:
        records = train_lora.read_manifest(prepare_data.OUTPUT_MANIFEST, 8, 11)
        self.assertEqual(len(records), 634)
        self.assertEqual(len({record["video_id"] for record in records}), 415)
        training, validation = train_lora.split_by_video(records, 0.10, 42)
        self.assertFalse(
            {record["video_id"] for record in training}
            & {record["video_id"] for record in validation}
        )

    def test_report_records_no_feature_recompute(self) -> None:
        report = json.loads(
            prepare_data.OUTPUT_REPORT.read_text(encoding="utf-8")
        )
        self.assertEqual(report["feature_stages_rerun"], [])
        self.assertTrue(report["frame_and_text_evidence_reused"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
