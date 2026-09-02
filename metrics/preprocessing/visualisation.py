"""Plot preprocessing sketches beside vectorized stroke output."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.vectorization import VectorizationStats


def plot_strokes(
    strokes: list[list[tuple[int, int]]],
    ax: plt.Axes,
    image_shape: tuple[int, int],
) -> None:
    """Draw simplified strokes on a matplotlib axis."""
    height, width = image_shape

    if not strokes:
        ax.text(0.5, 0.5, "No strokes", ha="center", va="center")
    else:
        colormap = plt.get_cmap("viridis")
        denominator = max(1, len(strokes) - 1)

        for index, stroke in enumerate(strokes):
            if len(stroke) < 2:
                continue
            color = colormap(index / denominator)
            xs = [point[0] for point in stroke]
            ys = [point[1] for point in stroke]
            ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.85)

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def save_original_vs_simplified(
    image_path: Path,
    strokes: list[list[tuple[int, int]]],
    stats: VectorizationStats,
    output_path: Path,
    rank: int,
) -> None:
    """Save a side-by-side visual comparison for one sketch."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150)
    fig.suptitle(f"Rank {rank:02d}: {image_path.name}", fontsize=11)

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(
        f"Original sketch\n"
        f"raw strokes: {stats.raw_stroke_count:,} | raw points: {stats.raw_point_count:,}",
        fontsize=9,
    )
    axes[0].axis("off")

    plot_strokes(strokes, axes[1], image.shape)
    axes[1].set_title(
        f"RDP epsilon={stats.epsilon:g}\n"
        f"strokes: {stats.simplified_stroke_count:,} | points: "
        f"{stats.simplified_point_count:,}",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
