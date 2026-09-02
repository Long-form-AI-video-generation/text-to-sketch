"""Select dense sketches and compare original images with RDP simplification.

The default run scans all processed sketch PNGs, ranks them by the number of
pre-simplification contour points, keeps the top 20, and writes:

- side-by-side visualizations
- a CSV summary of raw and simplified stroke/point counts
- optional stroke-5 files generated from the simplified strokes
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tqdm import tqdm

from utils.paths import DEFAULT_FILTERED_SKETCHES_DIR, PROCESSED_DATA_DIR
from pipeline.kinematics import generate_kinematics
from pipeline.ordering import (
    order_directional_bias,
    order_greedy_nearest_neighbor,
    order_tsp,
)
from pipeline.stroke5 import to_stroke5
from pipeline.vectorization import (
    DEFAULT_RDP_EPSILON,
    VectorizationStats,
    vectorize_image_with_stats,
)
from utils.io import save_stroke5
from metrics.visualisation import save_original_vs_simplified


_ORDER_FN_MAP = {
    "directional": order_directional_bias,
    "greedy": order_greedy_nearest_neighbor,
    "tsp": order_tsp,
}


@dataclass(frozen=True)
class SketchCandidate:
    """A sketch plus its vectorization statistics."""

    path: Path
    stats: VectorizationStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank sketches by raw contour points and compare the densest "
            "examples against RDP-simplified strokes."
        )
    )
    parser.add_argument(
        "--sketches-dir",
        type=Path,
        default=DEFAULT_FILTERED_SKETCHES_DIR,
        help="Directory containing processed sketch PNG files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to data/processed/rdp_epsilon_<value>.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_RDP_EPSILON,
        help="Absolute RDP epsilon in image pixels.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of highest-point sketches to process.",
    )
    parser.add_argument(
        "--ordering",
        choices=sorted(_ORDER_FN_MAP),
        default="directional",
        help="Stroke ordering used when exporting stroke-5 files.",
    )
    parser.add_argument(
        "--skip-stroke5",
        action="store_true",
        help="Only write visualizations and reports.",
    )
    return parser.parse_args()


def iter_sketches(sketches_dir: Path) -> list[Path]:
    extensions = ("*.png", "*.jpg", "*.jpeg")
    paths: list[Path] = []
    for pattern in extensions:
        paths.extend(sketches_dir.rglob(pattern))
    return sorted(paths)


def collect_candidates(
    sketch_paths: list[Path],
    epsilon: float,
) -> tuple[list[SketchCandidate], int]:
    candidates: list[SketchCandidate] = []
    skipped = 0

    for image_path in tqdm(sketch_paths, desc="Scanning sketches", unit="sketch"):
        try:
            _, stats = vectorize_image_with_stats(image_path, epsilon=epsilon)
        except Exception as exc:
            tqdm.write(f"[skip] {image_path}: {exc}")
            skipped += 1
            continue

        if stats.raw_point_count == 0:
            skipped += 1
            continue

        candidates.append(SketchCandidate(path=image_path, stats=stats))

    candidates.sort(key=lambda item: (-item.stats.raw_point_count, str(item.path)))
    return candidates, skipped


def process_candidate(
    candidate: SketchCandidate,
    sketches_dir: Path,
    output_dir: Path,
    rank: int,
    ordering: str,
    write_stroke5: bool,
) -> dict[str, str | int | float]:
    strokes, stats = vectorize_image_with_stats(
        candidate.path,
        epsilon=candidate.stats.epsilon,
    )

    slug = slugify_relative_path(candidate.path, sketches_dir)
    visualization_path = output_dir / "visualizations" / f"{rank:02d}_{slug}.png"
    save_original_vs_simplified(
        image_path=candidate.path,
        strokes=strokes,
        stats=stats,
        output_path=visualization_path,
        rank=rank,
    )

    stroke5_path = ""
    stroke5_rows = 0
    if write_stroke5:
        ordered = _ORDER_FN_MAP[ordering](strokes)
        timed = generate_kinematics(ordered)
        stroke5 = to_stroke5(timed)
        stroke5_rows = int(len(stroke5))
        stroke5_out = output_dir / "stroke5" / f"{rank:02d}_{slug}.npz"
        save_stroke5(stroke5, stroke5_out)
        stroke5_path = str(stroke5_out.relative_to(output_dir))

    return {
        "rank": rank,
        "sketch": str(candidate.path.relative_to(sketches_dir)),
        "epsilon": stats.epsilon,
        "raw_stroke_count": stats.raw_stroke_count,
        "raw_point_count": stats.raw_point_count,
        "simplified_stroke_count": stats.simplified_stroke_count,
        "simplified_point_count": stats.simplified_point_count,
        "removed_point_count": stats.removed_point_count,
        "point_retention_percent": round(stats.point_retention_ratio * 100, 2),
        "stroke5_rows": stroke5_rows,
        "visualization": str(visualization_path.relative_to(output_dir)),
        "stroke5_file": stroke5_path,
    }


def write_report(rows: list[dict[str, str | int | float]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "sketch",
        "epsilon",
        "raw_stroke_count",
        "raw_point_count",
        "simplified_stroke_count",
        "simplified_point_count",
        "removed_point_count",
        "point_retention_percent",
        "stroke5_rows",
        "visualization",
        "stroke5_file",
    ]

    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    metadata_path: Path,
    sketches_dir: Path,
    output_dir: Path,
    epsilon: float,
    top_k: int,
    input_count: int,
    skipped_count: int,
    ordering: str,
    wrote_stroke5: bool,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sketches_dir": str(sketches_dir),
        "output_dir": str(output_dir),
        "epsilon": epsilon,
        "top_k": top_k,
        "input_count": input_count,
        "skipped_count": skipped_count,
        "selection_metric": "raw_point_count",
        "ordering": ordering,
        "wrote_stroke5": wrote_stroke5,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def slugify_relative_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    raw_slug = "__".join(relative.parts)
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in raw_slug)


def default_output_dir(epsilon: float) -> Path:
    epsilon_slug = f"{epsilon:g}".replace(".", "_")
    return PROCESSED_DATA_DIR / f"rdp_epsilon_{epsilon_slug}"


def main() -> None:
    args = parse_args()

    sketches_dir = args.sketches_dir.resolve()
    output_dir = (args.output_dir or default_output_dir(args.epsilon)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sketch_paths = iter_sketches(sketches_dir)
    if not sketch_paths:
        raise SystemExit(f"No sketch images found in {sketches_dir}")

    print(
        f"[rdp] scanning {len(sketch_paths):,} sketches from {sketches_dir} "
        f"with epsilon={args.epsilon:g}"
    )
    candidates, skipped = collect_candidates(sketch_paths, epsilon=args.epsilon)
    if not candidates:
        raise SystemExit("No sketches with contour points were found.")

    selected = candidates[: min(args.top_k, len(candidates))]
    print(f"[rdp] selected top {len(selected)} sketches by raw contour point count")

    rows: list[dict[str, str | int | float]] = []
    for rank, candidate in enumerate(tqdm(selected, desc="Writing outputs", unit="sketch"), 1):
        rows.append(
            process_candidate(
                candidate=candidate,
                sketches_dir=sketches_dir,
                output_dir=output_dir,
                rank=rank,
                ordering=args.ordering,
                write_stroke5=not args.skip_stroke5,
            )
        )

    report_path = output_dir / "reports" / "top_stroke_point_sketches.csv"
    metadata_path = output_dir / "reports" / "run_metadata.json"
    write_report(rows, report_path)
    write_metadata(
        metadata_path=metadata_path,
        sketches_dir=sketches_dir,
        output_dir=output_dir,
        epsilon=args.epsilon,
        top_k=len(selected),
        input_count=len(sketch_paths),
        skipped_count=skipped,
        ordering=args.ordering,
        wrote_stroke5=not args.skip_stroke5,
    )

    print(f"[rdp] report: {report_path}")
    print(f"[rdp] visuals: {output_dir / 'visualizations'}")
    if not args.skip_stroke5:
        print(f"[rdp] stroke5: {output_dir / 'stroke5'}")

    print("\nTop selections:")
    for row in rows:
        print(
            f"  #{row['rank']:>2} {row['sketch']} | "
            f"raw points={row['raw_point_count']:,} | "
            f"simplified strokes={row['simplified_stroke_count']:,} | "
            f"simplified points={row['simplified_point_count']:,}"
        )


if __name__ == "__main__":
    main()
