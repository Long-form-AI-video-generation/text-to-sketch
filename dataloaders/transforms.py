"""Preprocessing transforms for stroke3 and tok-dict sequence data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


def _config_get(config: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if not config:
        return default
    return config.get(key, default)


def validate_stroke3(stroke: np.ndarray) -> np.ndarray:
    """Validate and return a float32 ``(N, 3)`` stroke3 array."""

    array = np.asarray(stroke, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected stroke3 array with shape (N, 3), got {array.shape}")
    if len(array) == 0:
        raise ValueError("Stroke sequence is empty")
    if not np.isfinite(array).all():
        raise ValueError("Stroke sequence contains NaN or infinite values")
    return array


def validate_token_sequence(tokens: np.ndarray) -> np.ndarray:
    """Validate and return an int64 ``(N,)`` token sequence."""

    array = np.asarray(tokens, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"Expected token array with shape (N,), got {array.shape}")
    if len(array) == 0:
        raise ValueError("Token sequence is empty")
    if np.any(array < 0):
        raise ValueError("Token sequence contains negative token IDs")
    return array


def clip_deltas(stroke: np.ndarray, limit: float) -> np.ndarray:
    """Clip only dx/dy values while preserving pen-state values."""

    if limit <= 0:
        return stroke
    result = np.array(stroke, copy=True, dtype=np.float32)
    result[:, :2] = np.clip(result[:, :2], -limit, limit)
    return result


def normalize_by_bounds(stroke: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize relative dx/dy values by the absolute sketch bounding box."""

    points = np.cumsum(stroke[:, :2], axis=0)
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    scale = float(max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1], eps))

    result = np.array(stroke, copy=True, dtype=np.float32)
    result[:, :2] = result[:, :2] / scale
    return result


def random_scale(stroke: np.ndarray, rng: np.random.Generator, min_value: float, max_value: float) -> np.ndarray:
    """Randomly scale x and y deltas independently."""

    result = np.array(stroke, copy=True, dtype=np.float32)
    scale = rng.uniform(min_value, max_value, size=(2,)).astype(np.float32)
    result[:, :2] *= scale
    return result


def stroke_jitter(stroke: np.ndarray, rng: np.random.Generator, std: float) -> np.ndarray:
    """Add Gaussian noise to dx/dy values only."""

    if std <= 0:
        return stroke
    result = np.array(stroke, copy=True, dtype=np.float32)
    result[:, :2] += rng.normal(0.0, std, size=result[:, :2].shape).astype(np.float32)
    return result


@dataclass
class Stroke3Transform:
    """Callable transform used by ``StrokeSequenceDataset``."""

    split: str
    max_length: int | None = None
    truncate_long_sequences: bool = True
    normalize: bool = True
    delta_clip: float = 1000.0
    augmentation: Mapping[str, Any] | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        stroke = validate_stroke3(sample["stroke3"])
        stroke = clip_deltas(stroke, self.delta_clip)

        if self.split == "train" and _config_get(self.augmentation, "enabled", False):
            stroke = self._augment(stroke)

        if self.normalize:
            stroke = normalize_by_bounds(stroke)

        if self.max_length is not None and len(stroke) > self.max_length:
            if not self.truncate_long_sequences:
                raise ValueError(
                    f"Sequence length {len(stroke)} exceeds max_length={self.max_length}"
                )
            stroke = stroke[: self.max_length]

        transformed = dict(sample)
        transformed["stroke3"] = stroke.astype(np.float32, copy=False)
        transformed["length"] = int(len(stroke))
        return transformed

    def _augment(self, stroke: np.ndarray) -> np.ndarray:
        random_scale_cfg = _config_get(self.augmentation, "random_scale", {})
        if _config_get(random_scale_cfg, "enabled", False):
            stroke = random_scale(
                stroke,
                self.rng,
                float(_config_get(random_scale_cfg, "min", 0.9)),
                float(_config_get(random_scale_cfg, "max", 1.1)),
            )

        jitter_cfg = _config_get(self.augmentation, "stroke_jitter", {})
        if _config_get(jitter_cfg, "enabled", False):
            stroke = stroke_jitter(
                stroke,
                self.rng,
                float(_config_get(jitter_cfg, "std", 0.0)),
            )

        shuffle_cfg = _config_get(self.augmentation, "shuffle_strokes", {})
        if _config_get(shuffle_cfg, "enabled", False):
            raise NotImplementedError("Stroke-level shuffling is not implemented yet")

        return stroke


@dataclass
class TokenSequenceTransform:
    """Callable transform used for tok-dict sequence samples."""

    split: str
    max_length: int | None = None
    truncate_long_sequences: bool = True
    add_start_token: bool = True
    add_end_token: bool = True
    sos_token_id: int | None = None
    sep_token_id: int | None = None
    eos_token_id: int | None = None

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        tokens = validate_token_sequence(sample["tokens"])

        if self.add_start_token and self.sos_token_id is not None:
            if int(tokens[0]) != int(self.sos_token_id):
                tokens = np.concatenate(
                    [np.asarray([self.sos_token_id], dtype=np.int64), tokens]
                )

        if self.add_end_token and self.eos_token_id is not None:
            if int(tokens[-1]) != int(self.eos_token_id):
                tokens = np.concatenate(
                    [tokens, np.asarray([self.eos_token_id], dtype=np.int64)]
                )

        if self.max_length is not None and len(tokens) > self.max_length:
            if not self.truncate_long_sequences:
                raise ValueError(
                    f"Sequence length {len(tokens)} exceeds max_length={self.max_length}"
                )
            tokens = np.array(tokens[: self.max_length], copy=True, dtype=np.int64)
            if self.add_end_token and self.eos_token_id is not None:
                if len(tokens) > 1 and self.sep_token_id is not None:
                    tokens[-2:] = [int(self.sep_token_id), int(self.eos_token_id)]
                else:
                    tokens[-1] = int(self.eos_token_id)

        transformed = dict(sample)
        transformed["tokens"] = tokens.astype(np.int64, copy=False)
        transformed["length"] = int(len(tokens))
        return transformed
