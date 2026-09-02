"""Deterministic readiness eval for 4096-token preprocessing/training."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root")


_add_project_to_path()

from dataloaders.masks import make_sdpa_self_attention_mask
from pipeline.vectorization import vectorize_image_with_stats
from scripts.sketchformer.config import compose_training_config
from scripts.sketchformer.parity_check import parity_metrics
from utils.tokenizer import (
    ErrorFeedbackQuantizer,
    decode_tokens,
    encode_stroke5,
    quantization_metrics,
)


def main() -> int:
    config = compose_training_config(
        "configs/train.yaml",
        experiment="anime_tok_dict_long",
    )
    if int(config["model"]["input"]["max_seq_len"]) != 4096:
        raise AssertionError("Long-sequence model max_seq_len is not 4096")
    if bool(config["data"]["sequence"]["truncate_long_sequences"]):
        raise AssertionError("Long-sequence data must refuse truncation")

    image = np.full((64, 64), 255, dtype=np.uint8)
    cv2.line(image, (8, 32), (56, 32), 0, thickness=7)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "thick.png"
        cv2.imwrite(str(path), image)
        strokes, stats = vectorize_image_with_stats(
            path,
            method="centerline",
            threshold_profile="legacy",
        )
    if len(strokes) != 1 or stats.simplified_point_count > 4:
        raise AssertionError("thick line was not reduced to one compact centerline")

    codebook = np.asarray(
        [[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]],
        dtype=np.float32,
    )
    rows = [[0.03, 0.0, 1.0, 0.0, 0.0] for _ in range(100)]
    rows[-1][2:4] = [0.0, 1.0]
    rows.append([0.0, 0.0, 0.0, 0.0, 1.0])
    stroke5 = np.asarray(rows, dtype=np.float32)
    legacy_tokens = encode_stroke5(stroke5, codebook)
    corrected_tokens = encode_stroke5(stroke5, codebook, error_feedback=True)
    legacy_error = quantization_metrics(stroke5, legacy_tokens, codebook)
    corrected_error = quantization_metrics(stroke5, corrected_tokens, codebook)
    if corrected_error.mean_point_error >= legacy_error.mean_point_error:
        raise AssertionError("error-feedback quantization did not reduce path drift")
    if corrected_error.endpoint_error > 0.011:
        raise AssertionError(
            f"endpoint correction exceeded tolerance: {corrected_error.endpoint_error}"
        )

    import torch

    compact_mask = make_sdpa_self_attention_mask(
        torch.tensor([[True, True, False, False]])
    )
    if compact_mask.shape != (1, 1, 1, 4):
        raise AssertionError(f"attention mask is not broadcast-sized: {compact_mask.shape}")

    pair_codebook = np.asarray(
        [[0.12, 0.0], [0.08, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    pair_stroke5 = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pair_tokens = ErrorFeedbackQuantizer(pair_codebook).encode(pair_stroke5)
    pair_decoded = decode_tokens(pair_tokens, pair_codebook)
    pair_endpoint = float(pair_decoded[:-1, 0].sum())
    if not np.isclose(pair_endpoint, 0.2):
        raise AssertionError("pair-token residual search did not recover the endpoint")

    parity_cosine, parity_agreement = parity_metrics(
        torch.ones(1, 3, 4),
        torch.ones(1, 3, 4),
    )
    if parity_cosine < 0.999 or parity_agreement < 0.99:
        raise AssertionError("checkpoint parity metric gate failed")

    report = {
        "eval": "long_sequence",
        "status": "pass",
        "max_seq_len": 4096,
        "centerline_strokes": len(strokes),
        "centerline_points": stats.simplified_point_count,
        "legacy_quantization_mean_error": legacy_error.mean_point_error,
        "feedback_quantization_mean_error": corrected_error.mean_point_error,
        "feedback_quantization_endpoint_error": corrected_error.endpoint_error,
        "attention_mask_shape": list(compact_mask.shape),
        "pair_quantization_endpoint": pair_endpoint,
        "parity_cosine": parity_cosine,
        "parity_argmax_agreement": parity_agreement,
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
