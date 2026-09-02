"""Sketch token encoding helpers.

Encodes a stroke-5 array into a sequence of discrete token indices using a
K-means codebook built by ``prep_data.sketch_token.create_token_dict``.

Special tokens match the released TensorFlow Sketchformer tokenizer:
    0       → padding
    1..K    → codebook motion tokens
    K + 1   → stroke separator / pen lift
    K + 2   → start of sketch
    K + 3   → end of sketch
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class QuantizationMetrics:
    """Cumulative spatial error after a token encode/decode round trip."""

    mean_point_error: float
    max_point_error: float
    endpoint_error: float


class ErrorFeedbackQuantizer:
    """Encode geometry with one- or two-token residual vector search.

    The released dictionary has sparse coverage close to zero. Selecting one
    token independently for every source point therefore makes short segments
    oscillate. This encoder searches codebook-vector pairs and rejects pair
    orderings whose intermediate point leaves the source segment.
    """

    def __init__(
        self,
        codebook: np.ndarray,
        *,
        pair_candidates: int = 64,
        max_greedy_step: float = 0.5,
        max_prefix_tokens: int = 16,
    ) -> None:
        centers = np.asarray(codebook, dtype=np.float32)
        if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) == 0:
            raise ValueError(f"Expected non-empty codebook shape (K, 2), got {centers.shape}")
        if pair_candidates <= 0:
            raise ValueError("pair_candidates must be positive")
        self.codebook = centers
        self.size = len(centers)
        self.pair_candidates = min(int(pair_candidates), self.size * self.size)
        self.max_greedy_step = float(max_greedy_step)
        self.max_prefix_tokens = int(max_prefix_tokens)
        self.index = cKDTree(centers)
        pair_sums = (centers[:, None, :] + centers[None, :, :]).reshape(-1, 2)
        self.pair_index = cKDTree(pair_sums)

    def encode(
        self,
        stroke5: np.ndarray,
        *,
        motion_token_offset: int = 1,
        sep_token_id: int | None = None,
        sos_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> np.ndarray:
        """Encode a stroke-5 sequence while correcting cumulative drift."""

        rows = np.asarray(stroke5, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != 5:
            raise ValueError(f"Expected stroke-5 shape (N, 5), got {rows.shape}")
        sep = self.size + 1 if sep_token_id is None else int(sep_token_id)
        sos = self.size + 2 if sos_token_id is None else int(sos_token_id)
        eos = self.size + 3 if eos_token_id is None else int(eos_token_id)
        offset = int(motion_token_offset)
        tokens: list[int] = [sos]
        desired = np.zeros(2, dtype=np.float32)
        decoded = np.zeros(2, dtype=np.float32)
        previous_desired = desired.copy()
        stroke_start = True

        for row in rows:
            if row[4] >= 0.5:
                tokens.append(eos)
                break
            desired += row[:2]

            prefix, decoded = self._greedy_prefix(desired, decoded)
            for cluster_id in prefix:
                tokens.append(cluster_id + offset)
                if stroke_start:
                    # Pen-up relocation prefixes must remain invisible.
                    tokens.append(sep)

            cluster_ids, decoded = self._best_final_step(
                desired,
                decoded,
                previous_desired,
                stroke_start=stroke_start,
            )
            for index, cluster_id in enumerate(cluster_ids):
                tokens.append(cluster_id + offset)
                if stroke_start and index < len(cluster_ids) - 1:
                    tokens.append(sep)

            if row[3] >= 0.5:
                tokens.append(sep)
                stroke_start = True
            else:
                stroke_start = False
            previous_desired = desired.copy()

        if tokens[-1] != eos:
            tokens.append(eos)
        return np.asarray(tokens, dtype=np.int32)

    def _greedy_prefix(
        self,
        desired: np.ndarray,
        decoded: np.ndarray,
    ) -> tuple[list[int], np.ndarray]:
        """Bring long residuals into the range covered by a two-token sum."""

        output: list[int] = []
        current = decoded.copy()
        for _ in range(max(0, self.max_prefix_tokens)):
            residual = desired - current
            norm = float(np.linalg.norm(residual))
            if norm <= 1.0:
                break
            target_step = residual * min(1.0, self.max_greedy_step / norm)
            cluster_id = int(self.index.query(target_step)[1])
            candidate = current + self.codebook[cluster_id]
            if np.linalg.norm(desired - candidate) >= norm:
                break
            output.append(cluster_id)
            current = candidate
        return output, current

    def _best_final_step(
        self,
        desired: np.ndarray,
        decoded: np.ndarray,
        previous_desired: np.ndarray,
        *,
        stroke_start: bool,
    ) -> tuple[list[int], np.ndarray]:
        residual = desired - decoded
        one_id = int(self.index.query(residual)[1])
        one_end = decoded + self.codebook[one_id]
        best_score = float(np.linalg.norm(desired - one_end))
        best_ids = [one_id]
        best_end = one_end

        pair_ids = np.atleast_1d(
            self.pair_index.query(residual, k=self.pair_candidates)[1]
        ).astype(np.int64, copy=False)
        first_ids = pair_ids // self.size
        second_ids = pair_ids % self.size
        pair_ends = (
            decoded[None, :]
            + self.codebook[first_ids]
            + self.codebook[second_ids]
        )
        endpoint_errors = np.linalg.norm(desired[None, :] - pair_ends, axis=1)
        if stroke_start:
            scores_first = endpoint_errors
            scores_second = endpoint_errors
        else:
            first_points = decoded[None, :] + self.codebook[first_ids]
            second_points = decoded[None, :] + self.codebook[second_ids]
            scores_first = np.maximum(
                endpoint_errors,
                _point_segment_distances(first_points, previous_desired, desired),
            )
            scores_second = np.maximum(
                endpoint_errors,
                _point_segment_distances(second_points, previous_desired, desired),
            )

        first_index = int(np.argmin(scores_first))
        second_index = int(np.argmin(scores_second))
        if float(scores_first[first_index]) < best_score:
            best_score = float(scores_first[first_index])
            best_ids = [int(first_ids[first_index]), int(second_ids[first_index])]
            best_end = pair_ends[first_index]
        if float(scores_second[second_index]) < best_score:
            best_ids = [int(second_ids[second_index]), int(first_ids[second_index])]
            best_end = pair_ends[second_index]
        return best_ids, np.asarray(best_end, dtype=np.float32)


def _point_segment_distances(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator == 0.0:
        return np.linalg.norm(points - end[None, :], axis=1)
    positions = ((points - start[None, :]) @ segment) / denominator
    projections = start[None, :] + np.clip(positions, 0.0, 1.0)[:, None] * segment
    return np.linalg.norm(points - projections, axis=1)


def encode_stroke5(
    stroke5: np.ndarray,
    codebook: np.ndarray,
    *,
    motion_token_offset: int = 1,
    sep_token_id: int | None = None,
    sos_token_id: int | None = None,
    eos_token_id: int | None = None,
    error_feedback: bool = False,
    endpoint_tolerance: float = 0.01,
    max_endpoint_corrections: int = 3,
) -> np.ndarray:
    """Map each stroke-5 row to a discrete token index.

    Pen-lift points (p2=1) emit two tokens: a shifted codebook motion token
    preserving the (dx, dy) displacement, followed by SEP.
    """
    K = len(codebook)
    sep = K + 1 if sep_token_id is None else int(sep_token_id)
    sos = K + 2 if sos_token_id is None else int(sos_token_id)
    eos = K + 3 if eos_token_id is None else int(eos_token_id)
    tokens: list[int] = [sos]
    index = cKDTree(np.asarray(codebook, dtype=np.float32)) if error_feedback else None
    desired_position = np.zeros(2, dtype=np.float32)
    decoded_position = np.zeros(2, dtype=np.float32)

    for row in stroke5:
        p3 = row[4]
        p2 = row[3]

        if p3 == 1.0:
            tokens.append(eos)
            break
        else:
            # Quantize (dx, dy) for ALL points, including pen-lift.
            if error_feedback:
                assert index is not None
                desired_position += np.asarray(row[:2], dtype=np.float32)
                needed_delta = desired_position - decoded_position
                _, cluster_id = index.query(needed_delta)
                cluster_id = int(cluster_id)
                decoded_position += np.asarray(codebook[cluster_id], dtype=np.float32)
            else:
                delta = codebook - row[:2]
                cluster_id = int(np.argmin((delta * delta).sum(axis=1)))
            motion_token = cluster_id + int(motion_token_offset)
            tokens.append(motion_token)

            if p2 == 1.0:
                if error_feedback and endpoint_tolerance > 0:
                    assert index is not None
                    for _ in range(max(0, int(max_endpoint_corrections))):
                        residual = desired_position - decoded_position
                        residual_norm = float(np.linalg.norm(residual))
                        if residual_norm <= endpoint_tolerance:
                            break
                        _, correction_id = index.query(residual)
                        correction_id = int(correction_id)
                        corrected_position = decoded_position + np.asarray(
                            codebook[correction_id],
                            dtype=np.float32,
                        )
                        if float(np.linalg.norm(desired_position - corrected_position)) >= residual_norm:
                            break
                        tokens.append(correction_id + int(motion_token_offset))
                        decoded_position = corrected_position
                tokens.append(sep)

    if tokens[-1] != eos:
        tokens.append(eos)

    return np.array(tokens, dtype=np.int32)


def quantization_metrics(
    stroke5: np.ndarray,
    tokens: np.ndarray,
    codebook: np.ndarray,
) -> QuantizationMetrics:
    """Measure cumulative path drift introduced by a token sequence."""

    original = np.asarray(stroke5, dtype=np.float32)
    decoded = decode_tokens(tokens, codebook)
    original_rows = original[original[:, 4] < 0.5]
    decoded_rows = decoded[decoded[:, 4] < 0.5]
    if len(original_rows) == 0 or len(decoded_rows) == 0:
        return QuantizationMetrics(0.0, 0.0, 0.0)
    original_points = np.cumsum(original_rows[:, :2], axis=0)
    decoded_points = np.cumsum(decoded_rows[:, :2], axis=0)
    decoded_index = cKDTree(decoded_points)
    errors, _ = decoded_index.query(original_points)
    original_endpoints = original_points[original_rows[:, 3] >= 0.5]
    decoded_endpoints = decoded_points[decoded_rows[:, 3] >= 0.5]
    if len(original_endpoints) and len(decoded_endpoints):
        endpoint_error = float(
            np.linalg.norm(original_endpoints[-1] - decoded_endpoints[-1])
        )
    else:
        endpoint_error = float(np.linalg.norm(original_points[-1] - decoded_points[-1]))
    return QuantizationMetrics(
        mean_point_error=float(np.mean(errors)),
        max_point_error=float(np.max(errors)),
        endpoint_error=endpoint_error,
    )


def decode_tokens(
    tokens: np.ndarray,
    codebook: np.ndarray,
    *,
    motion_token_offset: int = 1,
    pad_token_id: int = 0,
    sep_token_id: int | None = None,
    sos_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> np.ndarray:
    """Approximate inverse of ``encode_stroke5``.

    Handles the TensorFlow-compatible token IDs produced by ``encode_stroke5``.
    """
    K = len(codebook)
    sep = K + 1 if sep_token_id is None else int(sep_token_id)
    sos = K + 2 if sos_token_id is None else int(sos_token_id)
    eos = K + 3 if eos_token_id is None else int(eos_token_id)
    motion_start = int(motion_token_offset)
    motion_end = motion_start + K
    rows: list[list[float]] = []

    for tok in tokens:
        tok = int(tok)
        if tok in {int(pad_token_id), sos}:
            continue
        if tok == eos:
            rows.append([0.0, 0.0, 0.0, 0.0, 1.0])
            break
        if tok == sep:
            if rows:
                rows[-1][2] = 0.0
                rows[-1][3] = 1.0
        elif motion_start <= tok < motion_end:
            dx, dy = codebook[tok - motion_start]
            rows.append([float(dx), float(dy), 1.0, 0.0, 0.0])

    if not rows or rows[-1][4] != 1.0:
        rows.append([0.0, 0.0, 0.0, 0.0, 1.0])

    return np.array(rows, dtype=np.float32)
