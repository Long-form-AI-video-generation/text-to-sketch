"""

Converts timed stroke data  ``[[x, y, t], ...]``  produced by Step 3 into
the stroke-5 format expected by Sketchformer:

    [Δx, Δy, p1, p2, p3]

where (Δx, Δy) are relative displacements to the previous point and the
three pen-state flags are mutually exclusive:

    p1 = 1  →  pen is drawing (mid-stroke)
    p2 = 1  →  last point of the current stroke; pen lifts next
    p3 = 1  →  end-of-sketch sentinel (appended once at the very end)

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Stroke5Transform:
    """Spatial metadata required to compare normalized strokes to the source."""

    canvas_height: int
    canvas_width: int
    start_x: float
    start_y: float
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    scale: float
    normalization_extent: float


def to_stroke5(
    timed_strokes: list[list[list[float]]],
    canvas_size: int = 256,
) -> np.ndarray:
    """Convert timed strokes to stroke-5 format.

    Parameters
    ----------
    timed_strokes : list[list[list[float]]]
        Output of Step D.  Each element is a stroke; each point is
        ``[x, y, timestamp]`` in absolute pixel coordinates.
    canvas_size : int
        Unused directly — normalization is data-driven (bounding box).
        Kept as a parameter for future fixed-canvas workflows.

    Returns
    -------
    np.ndarray, shape (N + 1, 5), dtype float32
        N = total number of points across all strokes.
        The last row is always the end-of-sketch token ``[0, 0, 0, 0, 1]``.
    """
    stroke5, _ = to_stroke5_with_metadata(timed_strokes, canvas_size=canvas_size)
    return stroke5


def strokes_to_stroke5(
    strokes: list[list[tuple[int, int]]],
    *,
    canvas_shape: tuple[int, int],
    normalization_extent: float = 1.0,
) -> tuple[np.ndarray, Stroke5Transform]:
    """Convert static vector paths directly, without synthetic resampling."""

    timed = [
        [[float(x), float(y), 0.0] for x, y in stroke]
        for stroke in strokes
        if stroke
    ]
    return to_stroke5_with_metadata(
        timed,
        canvas_size=canvas_shape,
        normalization_extent=normalization_extent,
    )


def stroke5_to_canvas_strokes(
    stroke5: np.ndarray,
    transform: Stroke5Transform,
) -> list[list[tuple[int, int]]]:
    """Restore normalized stroke-5 geometry to its source-image coordinates."""

    rows = np.asarray(stroke5, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 5:
        raise ValueError(f"Expected stroke-5 shape (N, 5), got {rows.shape}")

    x = float(transform.start_x)
    y = float(transform.start_y)
    strokes: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for row in rows:
        if row[4] >= 0.5:
            break
        x += float(row[0]) * transform.scale
        y += float(row[1]) * transform.scale
        point = (int(round(x)), int(round(y)))
        if not current or point != current[-1]:
            current.append(point)
        if row[3] >= 0.5:
            if current:
                strokes.append(current)
            current = []
    if current:
        strokes.append(current)
    return strokes


def to_stroke5_with_metadata(
    timed_strokes: list[list[list[float]]],
    canvas_size: int | tuple[int, int] = 256,
    *,
    normalization_extent: float = 1.0,
) -> tuple[np.ndarray, Stroke5Transform]:
    """Convert timed strokes and return the normalization transform."""

    if normalization_extent <= 0:
        raise ValueError("normalization_extent must be positive")

    if isinstance(canvas_size, tuple):
        canvas_height, canvas_width = int(canvas_size[0]), int(canvas_size[1])
    else:
        canvas_height = canvas_width = int(canvas_size)

    nonempty = [stroke for stroke in timed_strokes if stroke]
    if not nonempty:
        transform = Stroke5Transform(
            canvas_height=canvas_height,
            canvas_width=canvas_width,
            start_x=0.0,
            start_y=0.0,
            bbox_x=0.0,
            bbox_y=0.0,
            bbox_width=0.0,
            bbox_height=0.0,
            scale=1.0,
            normalization_extent=float(normalization_extent),
        )
        sentinel = np.asarray([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        return sentinel, transform

    # Collect all absolute (x, y) to compute a global bounding box.
    all_x: list[float] = []
    all_y: list[float] = []
    for stroke in nonempty:
        for pt in stroke:
            all_x.append(pt[0])
            all_y.append(pt[1])

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    x_range = max(x_max - x_min, 1.0)   # guard against zero-width sketches
    y_range = max(y_max - y_min, 1.0)
    scale = max(x_range, y_range) / float(normalization_extent)

    def _norm(x: float, y: float) -> tuple[float, float]:
        return (x - x_min) / scale, (y - y_min) / scale

    # Build stroke-5 rows.
    rows: list[list[float]] = []
    prev_x = prev_y = 0.0
    first  = True

    for stroke in nonempty:
        n_pts = len(stroke)
        for idx, pt in enumerate(stroke):
            nx, ny = _norm(pt[0], pt[1])

            # First point ever → delta is (0, 0).
            dx = 0.0 if first else nx - prev_x
            dy = 0.0 if first else ny - prev_y
            first = False

            is_last_in_stroke = (idx == n_pts - 1)
            if is_last_in_stroke:
                p1, p2, p3 = 0, 1, 0   # pen lifts after this point
            else:
                p1, p2, p3 = 1, 0, 0   # pen is drawing

            rows.append([dx, dy, float(p1), float(p2), float(p3)])
            prev_x, prev_y = nx, ny

    # End-of-sketch sentinel.
    rows.append([0.0, 0.0, 0.0, 0.0, 1.0])

    transform = Stroke5Transform(
        canvas_height=canvas_height,
        canvas_width=canvas_width,
        start_x=float(nonempty[0][0][0]),
        start_y=float(nonempty[0][0][1]),
        bbox_x=float(x_min),
        bbox_y=float(y_min),
        bbox_width=float(x_max - x_min),
        bbox_height=float(y_max - y_min),
        scale=float(scale),
        normalization_extent=float(normalization_extent),
    )
    return np.array(rows, dtype=np.float32), transform
