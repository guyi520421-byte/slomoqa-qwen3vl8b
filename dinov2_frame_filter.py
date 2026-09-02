"""DINOv2-based near-duplicate filtering for VideoITG-ranked frames.

Heavy dependencies are imported lazily so the Qwen-only answer environment can
import the main pipeline without installing timm.

Two frames are considered duplicates only when both their global class tokens
and their spatially corresponding patch tokens are sufficiently similar. This
protects frames whose overall scene is unchanged but whose local object/action
state differs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DINOv2RedundancyFilter:
    """Keep high-ranked frames using global and local DINOv2 features."""

    MODEL_NAME = "vit_base_patch14_reg4_dinov2"

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "cuda:0",
        batch_size: int = 8,
        dtype: str = "fp16",
        decode_threads: int = 4,
    ) -> None:
        try:
            import timm
            import torch
            from timm.data import create_transform, resolve_data_config
            from timm.models.vision_transformer import checkpoint_filter_fn
        except ImportError as exc:
            raise RuntimeError(
                "DINOv2 filtering requires torch and timm. Run this stage in "
                "the videoitg environment."
            ) from exc

        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"DINOv2 checkpoint not found: {checkpoint_path}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for DINOv2 device {device}")
        if batch_size <= 0:
            raise ValueError("DINOv2 batch_size must be positive")
        if decode_threads <= 0:
            raise ValueError("DINOv2 decode_threads must be positive")

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.decode_threads = decode_threads
        if self.device.type == "cpu":
            dtype = "fp32"
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported DINOv2 dtype: {dtype}")
        self.dtype_name = dtype
        self.dtype = dtype_map[dtype]

        model = timm.create_model(
            self.MODEL_NAME,
            pretrained=False,
            num_classes=0,
        )
        checkpoint = self._load_checkpoint(checkpoint_path)
        # Official DINOv2 register-token checkpoints include the class-token
        # position in pos_embed. timm stores it separately; this filter performs
        # the exact official-to-timm conversion.
        checkpoint = checkpoint_filter_fn(checkpoint, model)
        incompatible = model.load_state_dict(checkpoint, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "DINOv2 checkpoint does not match "
                f"{self.MODEL_NAME}: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        data_config = resolve_data_config(model.pretrained_cfg, model=model)
        self.transform = create_transform(**data_config, is_training=False)

    def _load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        torch = self.torch
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f"Unsupported DINOv2 checkpoint type: {type(checkpoint).__name__}"
            )
        if not checkpoint or all(torch.is_tensor(value) for value in checkpoint.values()):
            state_dict = checkpoint
        else:
            state_dict = None
            for key in ("model", "state_dict", "teacher", "student"):
                candidate = checkpoint.get(key)
                if isinstance(candidate, dict):
                    state_dict = candidate
                    break
            if state_dict is None:
                raise RuntimeError(
                    "Could not locate a state dict in DINOv2 checkpoint "
                    f"{checkpoint_path}"
                )

        for prefix in ("module.", "backbone."):
            if state_dict and all(key.startswith(prefix) for key in state_dict):
                state_dict = {
                    key[len(prefix) :]: value for key, value in state_dict.items()
                }
        return state_dict

    def encode_video_frames(
        self,
        video_path: Path,
        frame_indices: list[int],
    ) -> dict[str, Any]:
        """Return normalized class- and patch-token features in index order."""
        try:
            from decord import VideoReader, cpu
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "DINOv2 filtering requires decord and Pillow in the current environment."
            ) from exc

        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not frame_indices:
            raise ValueError("No frame indices supplied to DINOv2")

        reader = VideoReader(
            str(video_path), ctx=cpu(0), num_threads=self.decode_threads
        )
        total_frames = len(reader)
        invalid = [
            index for index in frame_indices if index < 0 or index >= total_frames
        ]
        if invalid:
            raise ValueError(
                f"Frame indices outside [0, {total_frames}): {invalid[:8]}"
            )

        torch = self.torch
        global_embeddings = []
        patch_embeddings = []
        expected_patch_count: int | None = None
        with torch.inference_mode():
            for start in range(0, len(frame_indices), self.batch_size):
                batch_indices = frame_indices[start : start + self.batch_size]
                arrays = reader.get_batch(batch_indices).asnumpy()
                tensors = [
                    self.transform(Image.fromarray(array).convert("RGB"))
                    for array in arrays
                ]
                batch = torch.stack(tensors).to(
                    device=self.device, dtype=self.dtype, non_blocking=True
                )
                features = self.model.forward_features(batch)
                if isinstance(features, dict):
                    global_output = features.get("x_norm_clstoken")
                    patch_output = features.get("x_norm_patchtokens")
                elif torch.is_tensor(features) and features.ndim == 3:
                    num_prefix_tokens = int(
                        getattr(self.model, "num_prefix_tokens", 1)
                    )
                    if features.shape[1] <= num_prefix_tokens:
                        raise RuntimeError(
                            "DINOv2 token output does not contain patch tokens: "
                            f"shape={tuple(features.shape)}, "
                            f"num_prefix_tokens={num_prefix_tokens}"
                        )
                    global_output = features[:, 0]
                    patch_output = features[:, num_prefix_tokens:]
                else:
                    raise RuntimeError(
                        "Unexpected DINOv2 feature output: "
                        f"{getattr(features, 'shape', type(features).__name__)}"
                    )

                if (
                    not torch.is_tensor(global_output)
                    or global_output.ndim != 2
                    or not torch.is_tensor(patch_output)
                    or patch_output.ndim != 3
                    or global_output.shape[0] != patch_output.shape[0]
                ):
                    raise RuntimeError(
                        "Unexpected DINOv2 class/patch feature shapes: "
                        f"global={getattr(global_output, 'shape', None)}, "
                        f"patch={getattr(patch_output, 'shape', None)}"
                    )
                patch_count = int(patch_output.shape[1])
                if expected_patch_count is None:
                    expected_patch_count = patch_count
                elif patch_count != expected_patch_count:
                    raise RuntimeError(
                        "DINOv2 patch count changed between batches: "
                        f"{expected_patch_count} vs {patch_count}"
                    )

                global_output = torch.nn.functional.normalize(
                    global_output.float(), p=2, dim=-1
                )
                patch_output = torch.nn.functional.normalize(
                    patch_output.float(), p=2, dim=-1
                )
                global_embeddings.append(global_output.cpu())
                patch_embeddings.append(patch_output.cpu())
                del batch, features, global_output, patch_output
        return {
            "global": torch.cat(global_embeddings, dim=0),
            "patch": torch.cat(patch_embeddings, dim=0),
            "patch_count": expected_patch_count,
        }

    def filter_ranked_frames(
        self,
        video_path: Path,
        ranked_frame_indices: list[int],
        fps: float,
        max_frames: int,
        min_frames: int,
        similarity_threshold: float,
        temporal_window_seconds: float,
        patch_similarity_threshold: float,
        patch_change_similarity_threshold: float,
        max_changed_patch_ratio: float,
    ) -> dict[str, Any]:
        """Evaluate the full ranked pool, then keep up to ``max_frames``.

        All supplied candidates are encoded and checked for redundancy before
        the non-redundant candidates are truncated by their original VideoITG
        rank. This lets a redundant frame inside the first ``max_frames`` be
        replaced by a lower-ranked, non-redundant candidate from the rest of
        the pool.
        """
        if fps <= 0:
            raise ValueError(f"FPS must be positive, got {fps}")
        thresholds = {
            "similarity_threshold": similarity_threshold,
            "patch_similarity_threshold": patch_similarity_threshold,
            "patch_change_similarity_threshold": patch_change_similarity_threshold,
            "max_changed_patch_ratio": max_changed_patch_ratio,
        }
        for name, value in thresholds.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if max_frames <= 0 or min_frames <= 0 or min_frames > max_frames:
            raise ValueError("Require 0 < min_frames <= max_frames")

        ranked = list(dict.fromkeys(int(index) for index in ranked_frame_indices))
        if not ranked:
            raise ValueError("No VideoITG-ranked frames to filter")
        features = self.encode_video_frames(video_path, ranked)
        global_embeddings = features["global"]
        patch_embeddings = features["patch"]

        torch = self.torch
        selected_positions: list[int] = []
        redundant: list[dict[str, Any]] = []
        locally_protected: list[dict[str, Any]] = []
        evaluated = 0
        for position, frame_index in enumerate(ranked):
            evaluated = position + 1
            timestamp = frame_index / fps
            comparison_positions = [
                kept
                for kept in selected_positions
                if temporal_window_seconds <= 0
                or abs(timestamp - ranked[kept] / fps) <= temporal_window_seconds
            ]
            duplicate_of_position = None
            duplicate_metrics: dict[str, float] = {}
            if comparison_positions:
                global_similarities = (
                    global_embeddings[comparison_positions]
                    @ global_embeddings[position]
                )
                corresponding_patch_similarities = (
                    patch_embeddings[comparison_positions]
                    * patch_embeddings[position].unsqueeze(0)
                ).sum(dim=-1)
                mean_patch_similarities = corresponding_patch_similarities.mean(
                    dim=-1
                )
                changed_patch_ratios = (
                    corresponding_patch_similarities
                    < patch_change_similarity_threshold
                ).float().mean(dim=-1)
                combined_similarities = (
                    global_similarities + mean_patch_similarities
                ) / 2.0
                duplicate_mask = (
                    (global_similarities >= similarity_threshold)
                    & (mean_patch_similarities >= patch_similarity_threshold)
                    & (changed_patch_ratios <= max_changed_patch_ratio)
                )

                if bool(duplicate_mask.any().item()):
                    eligible_scores = combined_similarities.masked_fill(
                        ~duplicate_mask, float("-inf")
                    )
                    best_offset = int(eligible_scores.argmax().item())
                    duplicate_of_position = comparison_positions[best_offset]
                    patch_values = corresponding_patch_similarities[best_offset]
                    duplicate_metrics = {
                        "global": float(global_similarities[best_offset].item()),
                        "patch_mean": float(
                            mean_patch_similarities[best_offset].item()
                        ),
                        "patch_p10": float(
                            torch.quantile(patch_values.float(), 0.1).item()
                        ),
                        "changed_ratio": float(
                            changed_patch_ratios[best_offset].item()
                        ),
                        "combined": float(
                            combined_similarities[best_offset].item()
                        ),
                    }
                else:
                    global_duplicate_mask = global_similarities >= similarity_threshold
                    if bool(global_duplicate_mask.any().item()):
                        global_scores = global_similarities.masked_fill(
                            ~global_duplicate_mask, float("-inf")
                        )
                        best_offset = int(global_scores.argmax().item())
                        compared_to_position = comparison_positions[best_offset]
                        patch_mean = float(
                            mean_patch_similarities[best_offset].item()
                        )
                        changed_ratio = float(
                            changed_patch_ratios[best_offset].item()
                        )
                        reasons = []
                        if patch_mean < patch_similarity_threshold:
                            reasons.append("low_mean_patch_similarity")
                        if changed_ratio > max_changed_patch_ratio:
                            reasons.append("too_many_changed_patches")
                        locally_protected.append(
                            {
                                "frame_index": frame_index,
                                "timestamp": round(timestamp, 3),
                                "compared_to_frame_index": ranked[
                                    compared_to_position
                                ],
                                "compared_to_timestamp": round(
                                    ranked[compared_to_position] / fps, 3
                                ),
                                "global_cosine_similarity": round(
                                    float(global_similarities[best_offset].item()), 6
                                ),
                                "mean_patch_cosine_similarity": round(
                                    patch_mean, 6
                                ),
                                "patch_p10_cosine_similarity": round(
                                    float(
                                        torch.quantile(
                                            corresponding_patch_similarities[
                                                best_offset
                                            ].float(),
                                            0.1,
                                        ).item()
                                    ),
                                    6,
                                ),
                                "changed_patch_ratio": round(changed_ratio, 6),
                                "protection_reasons": reasons,
                                "videoitg_rank": position + 1,
                            }
                        )

            if duplicate_of_position is not None:
                duplicate_of = ranked[duplicate_of_position]
                redundant.append(
                    {
                        "frame_index": frame_index,
                        "timestamp": round(timestamp, 3),
                        "duplicate_of_frame_index": duplicate_of,
                        "duplicate_of_timestamp": round(duplicate_of / fps, 3),
                        # Preserve the legacy field for downstream compatibility.
                        "cosine_similarity": round(duplicate_metrics["global"], 6),
                        "global_cosine_similarity": round(
                            duplicate_metrics["global"], 6
                        ),
                        "mean_patch_cosine_similarity": round(
                            duplicate_metrics["patch_mean"], 6
                        ),
                        "patch_p10_cosine_similarity": round(
                            duplicate_metrics["patch_p10"], 6
                        ),
                        "changed_patch_ratio": round(
                            duplicate_metrics["changed_ratio"], 6
                        ),
                        "combined_similarity": round(
                            duplicate_metrics["combined"], 6
                        ),
                        "videoitg_rank": position + 1,
                        "backfilled": False,
                    }
                )
            else:
                selected_positions.append(position)

        backfilled_positions: list[int] = []
        if len(selected_positions) < min_frames:
            selected_set = set(selected_positions)
            frame_to_position = {
                frame_index: position
                for position, frame_index in enumerate(ranked)
            }
            for item in redundant:
                position = frame_to_position[int(item["frame_index"])]
                if position in selected_set:
                    continue
                selected_positions.append(position)
                selected_set.add(position)
                backfilled_positions.append(position)
                item["backfilled"] = True
                if len(selected_positions) >= min_frames:
                    break

        selected_positions.sort()
        budget_excluded_positions = selected_positions[max_frames:]
        selected_positions = selected_positions[:max_frames]
        selected_ranked = [ranked[position] for position in selected_positions]
        selected_chronological = sorted(selected_ranked)
        return {
            "dino_ranked_frame_indices": selected_ranked,
            "dino_selected_frame_indices": selected_chronological,
            "dino_selected_timestamps": [
                round(index / fps, 3) for index in selected_chronological
            ],
            "dino_redundant_frames": redundant,
            "dino_local_change_protected_frames": locally_protected,
            "dino_backfilled_frame_indices": [
                ranked[position] for position in backfilled_positions
            ],
            "dino_budget_excluded_frame_indices": [
                ranked[position] for position in budget_excluded_positions
            ],
            "dino_source_pool_size": len(ranked),
            "dino_evaluated_candidates": evaluated,
            "dino_patch_count": features["patch_count"],
        }
