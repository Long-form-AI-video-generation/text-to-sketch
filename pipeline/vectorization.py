"""Raster line-art vectorization for centerline and legacy contour pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.filters import apply_hysteresis_threshold, threshold_otsu, threshold_sauvola
from skimage.morphology import remove_small_objects, skeletonize

DEFAULT_RDP_EPSILON = 0.5
DEFAULT_MAX_GEOMETRY_ERROR = 2.0
VECTORIZERS = ("contour", "centerline")
THRESHOLD_PROFILES = ("legacy", "otsu", "sauvola", "hysteresis")

Point = tuple[int, int]
Stroke = list[Point]


@dataclass(frozen=True)
class CenterlineBranch:
    """A single skeleton branch with its graph topology metadata."""

    branch_id: int
    points: list[Point]
    start_node_id: int
    end_node_id: int
    component_id: int
    is_loop: bool


@dataclass(frozen=True)
class VectorizationStats:
    """Geometry and point-count summary for one vectorized sketch."""

    epsilon: float
    raw_stroke_count: int
    raw_point_count: int
    simplified_stroke_count: int
    simplified_point_count: int
    method: str = "contour"
    threshold_profile: str = "legacy"
    foreground_point_count: int = 0
    image_height: int = 0
    image_width: int = 0

    @property
    def removed_point_count(self) -> int:
        return self.raw_point_count - self.simplified_point_count

    @property
    def point_retention_ratio(self) -> float:
        if self.raw_point_count == 0:
            return 0.0
        return self.simplified_point_count / self.raw_point_count


@dataclass(frozen=True)
class CenterlineMetrics:
    """Tolerance-aware raster agreement between two one-pixel centerlines."""

    precision: float
    recall: float
    f1: float
    symmetric_chamfer: float


def vectorize_image(
    image_path: str | Path,
    epsilon: float = DEFAULT_RDP_EPSILON,
    *,
    method: Literal["contour", "centerline"] = "contour",
    threshold_profile: str = "hysteresis",
    min_object_size: int = 4,
) -> list[Stroke]:
    """Vectorize line art with the requested legacy or centerline method."""

    strokes, _ = vectorize_image_with_stats(
        image_path,
        epsilon=epsilon,
        method=method,
        threshold_profile=threshold_profile,
        min_object_size=min_object_size,
    )
    return strokes


def vectorize_image_with_stats(
    image_path: str | Path,
    epsilon: float = DEFAULT_RDP_EPSILON,
    *,
    method: Literal["contour", "centerline"] = "contour",
    threshold_profile: str = "hysteresis",
    min_object_size: int = 4,
    structured: bool = False,
) -> tuple[list[Stroke], VectorizationStats] | tuple[list[CenterlineBranch], VectorizationStats]:
    """Vectorize a sketch and report the geometry retained by simplification."""

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if method not in VECTORIZERS:
        raise ValueError(f"Unknown vectorizer {method!r}; expected one of {VECTORIZERS}")

    image = read_grayscale_image(image_path)
    if method == "contour":
        return _vectorize_contours(image, epsilon)
    return _vectorize_centerline(
        image,
        epsilon,
        threshold_profile=threshold_profile,
        min_object_size=min_object_size,
        structured=structured,
    )


def read_grayscale_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    return image


def foreground_mask(
    image: np.ndarray,
    *,
    profile: str = "hysteresis",
    min_object_size: int = 4,
) -> np.ndarray:
    """Convert grayscale line confidence into a cleaned foreground mask."""

    if profile not in THRESHOLD_PROFILES:
        raise ValueError(
            f"Unknown threshold profile {profile!r}; expected one of {THRESHOLD_PROFILES}"
        )
    gray = np.asarray(image, dtype=np.uint8)
    if gray.ndim != 2:
        raise ValueError(f"Expected a grayscale image, got shape {gray.shape}")

    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    light_background = float(np.median(border)) >= 128.0
    confidence = (255.0 - gray.astype(np.float32)) / 255.0
    if not light_background:
        confidence = 1.0 - confidence

    if profile == "legacy":
        mask = confidence > (127.0 / 255.0)
    elif profile == "otsu":
        cutoff = _safe_otsu(confidence)
        mask = confidence > cutoff
    elif profile == "sauvola":
        local_cutoff = threshold_sauvola(confidence, window_size=31, k=0.15)
        mask = confidence > local_cutoff
    else:
        high = float(np.clip(_safe_otsu(confidence), 0.08, 0.80))
        low = max(0.02, high * 0.5)
        mask = apply_hysteresis_threshold(confidence, low, high)

    mask = np.asarray(mask, dtype=bool)
    if min_object_size > 1 and mask.any():
        try:
            mask = remove_small_objects(mask, max_size=int(min_object_size) - 1)
        except TypeError:  # scikit-image < 0.26
            mask = remove_small_objects(mask, min_size=int(min_object_size))
    return mask


def simplify_strokes(
    strokes: list[Stroke],
    epsilon: float,
) -> list[Stroke]:
    """Simplify stroke paths with an absolute pixel-space RDP tolerance."""

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    simplified: list[Stroke] = []
    for stroke in strokes:
        simplified_points = _simplify_single(stroke, epsilon)
        if simplified_points is not None:
            simplified.append(simplified_points)
    return simplified


def simplify_branches(
    branches: list[CenterlineBranch],
    epsilon: float,
) -> list[CenterlineBranch]:
    """Simplify branch points via RDP while preserving topology metadata."""

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    simplified: list[CenterlineBranch] = []
    for branch in branches:
        simplified_points = _simplify_single(branch.points, epsilon)
        if simplified_points is None:
            continue
        first_point = branch.points[0]
        last_point = branch.points[-1]
        simplified_points[0] = first_point
        simplified_points[-1] = last_point
        simplified.append(
            CenterlineBranch(
                branch_id=branch.branch_id,
                points=simplified_points,
                start_node_id=branch.start_node_id,
                end_node_id=branch.end_node_id,
                component_id=branch.component_id,
                is_loop=branch.is_loop,
            )
        )
    return simplified


def _simplify_single(
    stroke: list[Point],
    epsilon: float,
) -> list[Point] | None:
    """RDP-simplify one coordinate list. Returns None if result is too short."""

    if len(stroke) < 2:
        return None
    closed = len(stroke) > 2 and _points_touch(stroke[0], stroke[-1])
    if epsilon == 0:
        points = list(stroke)
    else:
        contour = np.asarray(stroke, dtype=np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(contour, float(epsilon), closed=closed)
        points = [(int(round(point[0][0])), int(round(point[0][1]))) for point in approx]
    points = _deduplicate_consecutive(points)
    if closed and len(points) > 2 and points[0] != points[-1]:
        points.append(points[0])
    if len(points) > 1:
        return points
    return None


def rasterize_strokes(
    strokes: list[Stroke],
    image_shape: tuple[int, int],
    *,
    line_width: int = 1,
) -> np.ndarray:
    """Rasterize vector strokes into a boolean canvas."""

    canvas = np.zeros(image_shape, dtype=np.uint8)
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        points = np.asarray(stroke, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], False, 1, thickness=int(line_width), lineType=cv2.LINE_8)
    return canvas.astype(bool)


def centerline_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    tolerance_px: float = 2.0,
) -> CenterlineMetrics:
    """Measure centerline agreement without penalizing small raster offsets."""

    reference = np.asarray(reference, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have the same shape")
    if not reference.any() or not candidate.any():
        exact_empty = not reference.any() and not candidate.any()
        value = 1.0 if exact_empty else 0.0
        chamfer = 0.0 if exact_empty else float("inf")
        return CenterlineMetrics(value, value, value, chamfer)

    distance_to_reference = distance_transform_edt(~reference)
    distance_to_candidate = distance_transform_edt(~candidate)
    precision = float(np.mean(distance_to_reference[candidate] <= tolerance_px))
    recall = float(np.mean(distance_to_candidate[reference] <= tolerance_px))
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    chamfer = 0.5 * (
        float(np.mean(distance_to_reference[candidate]))
        + float(np.mean(distance_to_candidate[reference]))
    )
    return CenterlineMetrics(precision, recall, f1, chamfer)


def source_centerline(image: np.ndarray, *, threshold_profile: str = "hysteresis") -> np.ndarray:
    """Return the one-pixel topology used as the centerline geometry reference."""

    return skeletonize(foreground_mask(image, profile=threshold_profile))


def _vectorize_contours(
    image: np.ndarray,
    epsilon: float,
) -> tuple[list[Stroke], VectorizationStats]:
    _, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    raw = [
        [(int(point[0][0]), int(point[0][1])) for point in contour]
        for contour in contours
        if len(contour) >= 2
    ]
    strokes = simplify_strokes(raw, epsilon)
    return strokes, VectorizationStats(
        epsilon=float(epsilon),
        raw_stroke_count=len(raw),
        raw_point_count=sum(len(stroke) for stroke in raw),
        simplified_stroke_count=len(strokes),
        simplified_point_count=sum(len(stroke) for stroke in strokes),
        method="contour",
        threshold_profile="legacy",
        foreground_point_count=int((thresholded > 0).sum()),
        image_height=int(image.shape[0]),
        image_width=int(image.shape[1]),
    )


def _vectorize_centerline(
    image: np.ndarray,
    epsilon: float,
    *,
    threshold_profile: str,
    min_object_size: int,
    structured: bool = False,
) -> tuple[list[Stroke], VectorizationStats] | tuple[list[CenterlineBranch], VectorizationStats]:
    mask = foreground_mask(
        image,
        profile=threshold_profile,
        min_object_size=min_object_size,
    )
    skeleton = skeletonize(mask)
    raw_branches = _skeleton_paths(skeleton)
    raw_count = len(raw_branches)

    if structured:
        branches = simplify_branches(raw_branches, epsilon)
        return branches, VectorizationStats(
            epsilon=float(epsilon),
            raw_stroke_count=raw_count,
            raw_point_count=int(skeleton.sum()),
            simplified_stroke_count=len(branches),
            simplified_point_count=sum(len(b.points) for b in branches),
            method="centerline",
            threshold_profile=threshold_profile,
            foreground_point_count=int(mask.sum()),
            image_height=int(image.shape[0]),
            image_width=int(image.shape[1]),
        )

    raw_strokes = [branch.points for branch in raw_branches]
    strokes = simplify_strokes(raw_strokes, epsilon)
    return strokes, VectorizationStats(
        epsilon=float(epsilon),
        raw_stroke_count=raw_count,
        raw_point_count=int(skeleton.sum()),
        simplified_stroke_count=len(strokes),
        simplified_point_count=sum(len(stroke) for stroke in strokes),
        method="centerline",
        threshold_profile=threshold_profile,
        foreground_point_count=int(mask.sum()),
        image_height=int(image.shape[0]),
        image_width=int(image.shape[1]),
    )


def _skeleton_paths(skeleton: np.ndarray) -> list[CenterlineBranch]:
    if not np.asarray(skeleton, dtype=bool).any():
        return []
    try:
        from skan import Skeleton
    except ImportError as exc:
        raise RuntimeError(
            "Centerline vectorization requires skan>=0.13.1; install project requirements"
        ) from exc

    import networkx as nx

    graph = Skeleton(np.asarray(skeleton, dtype=np.uint8), keep_images=False)

    node_to_component: dict[int, int] = {}
    nx_graph = nx.from_scipy_sparse_array(graph.graph)
    for component_id, component_nodes in enumerate(nx.connected_components(nx_graph)):
        for node in component_nodes:
            node_to_component[node] = component_id

    branches: list[CenterlineBranch] = []
    for index in range(graph.n_paths):
        coordinates = np.asarray(graph.path_coordinates(index))
        if len(coordinates) < 2:
            continue
        points = [
            (int(round(column)), int(round(row)))
            for row, column in coordinates[:, :2]
        ]
        points = _deduplicate_consecutive(points)
        if len(points) < 2:
            continue

        node_ids = graph.path(index)
        start_node = int(node_ids[0])
        end_node = int(node_ids[-1])
        is_loop = start_node == end_node
        component_id = node_to_component.get(start_node, 0)

        branches.append(
            CenterlineBranch(
                branch_id=index,
                points=points,
                start_node_id=start_node,
                end_node_id=end_node,
                component_id=component_id,
                is_loop=is_loop,
            )
        )

    branches.sort(
        key=lambda b: (
            min(y for _, y in b.points),
            min(x for x, _ in b.points),
            -len(b.points),
        )
    )
    return branches


def _safe_otsu(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float32).ravel()
    if len(flat) == 0 or float(flat.min()) == float(flat.max()):
        return float(flat[0]) if len(flat) else 0.5
    return float(threshold_otsu(flat))


def _points_touch(first: Point, second: Point) -> bool:
    return abs(first[0] - second[0]) <= 1 and abs(first[1] - second[1]) <= 1


def _deduplicate_consecutive(points: list[Point]) -> list[Point]:
    result: list[Point] = []
    for point in points:
        if not result or point != result[-1]:
            result.append(point)
    return result
