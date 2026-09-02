"""Benchmark centerline preprocessing on a fixed sketch sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np

from pipeline.ordering import (
    order_continuity_greedy,
    order_continuity_topology,
    order_directional_bias,
    order_greedy_nearest_neighbor,
    order_tsp,
)

_ORDER_FN_MAP = {
    "continuity": order_continuity_greedy,
    "directional": order_directional_bias,
    "greedy": order_greedy_nearest_neighbor,
    "tsp": order_tsp,
    "continuity-topology": order_continuity_topology,
}
from pipeline.stroke5 import Stroke5Transform, stroke5_to_canvas_strokes
from pipeline.vectorization import (
    THRESHOLD_PROFILES,
    rasterize_strokes,
    read_grayscale_image,
)
from pipeline.workflow import fit_centerline_sequence
from utils.paths import DEFAULT_FILTERED_SKETCHES_DIR, DEFAULT_SKETCH_TOKEN_DIR
from utils.tokenizer import ErrorFeedbackQuantizer, decode_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sketches-dir", type=Path, default=DEFAULT_FILTERED_SKETCHES_DIR)
    parser.add_argument(
        "--extractor-dir",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Compare named extractor outputs on common relative paths; repeat as needed.",
    )
    parser.add_argument("--token-dict-dir", type=Path, default=DEFAULT_SKETCH_TOKEN_DIR)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ordering",
        choices=sorted(_ORDER_FN_MAP),
        default="continuity",
        help="Stroke-ordering method used to create the review reconstruction.",
    )
    parser.add_argument("--rdp-epsilon", type=float, default=0.5)
    parser.add_argument("--max-geometry-error", type=float, default=2.0)
    parser.add_argument("--max-token-length", type=int, default=4096)
    parser.add_argument(
        "--threshold-profiles",
        nargs="+",
        choices=THRESHOLD_PROFILES,
        default=["otsu", "sauvola", "hysteresis"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/evaluations/centerline_preprocessing.json"),
    )
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=None,
        help="Write source/token-decoded pairs and a manual feature checklist.",
    )
    parser.add_argument("--enforce", action="store_true")
    return parser.parse_args()


def evaluate_profile(
    paths: list[Path],
    codebook: np.ndarray,
    quantizer: ErrorFeedbackQuantizer,
    *,
    profile: str,
    epsilon: float,
    max_geometry_error: float,
    max_token_length: int,
    ordering: str = "continuity",
) -> dict[str, object]:
    rows: list[dict[str, float | int | str | bool]] = []
    for path in paths:
        try:
            result = fit_centerline_sequence(
                image_path=path,
                codebook=codebook,
                quantizer=quantizer,
                order_fn=_ORDER_FN_MAP[ordering],
                ordering=ordering,
                initial_epsilon=epsilon,
                max_epsilon=max_geometry_error,
                threshold_profile=profile,
                max_token_length=max_token_length,
            )
        except Exception as exc:
            rows.append(
                {
                    "sample": path.name,
                    "status": "error",
                    "error": str(exc),
                    "token_length": max_token_length + 1,
                    "within_limit": False,
                    "rdp_epsilon": epsilon,
                    "normalization_extent": 0.0,
                    "stroke_count": 0,
                    "point_count": 0,
                    "geometry_f1_2px": 0.0,
                    "geometry_chamfer_px": 1.0e9,
                    "vector_geometry_f1_2px": 0.0,
                    "vector_geometry_chamfer_px": 1.0e9,
                    "quantization_mean_error": 1.0,
                    "quantization_endpoint_error": 1.0,
                }
            )
            continue
        record = result["record"]
        rows.append(
            {
                "sample": path.name,
                "status": "accepted" if result["accepted"] else "overlength",
                "token_length": int(record["token_length"]),
                "within_limit": bool(result["accepted"]),
                "rdp_epsilon": float(record["rdp_epsilon"]),
                "normalization_extent": float(record["normalization_extent"]),
                "stroke_count": int(record["stroke_count"]),
                "point_count": int(record["point_count"]),
                "geometry_f1_2px": float(record["geometry_f1_2px"]),
                "geometry_chamfer_px": float(record["geometry_chamfer_px"]),
                "vector_geometry_f1_2px": float(record["vector_geometry_f1_2px"]),
                "vector_geometry_chamfer_px": float(
                    record["vector_geometry_chamfer_px"]
                ),
                "quantization_mean_error": float(record["quantization_mean_error"]),
                "quantization_endpoint_error": float(
                    record["quantization_endpoint_error"]
                ),
            }
        )

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=np.float64)

    summary = {
        "samples": len(rows),
        "accepted": int(sum(bool(row["within_limit"]) for row in rows)),
        "overlength": int(sum(row["status"] == "overlength" for row in rows)),
        "errors": int(sum(row["status"] == "error" for row in rows)),
        "token_length_p50": float(np.quantile(values("token_length"), 0.5)),
        "token_length_p95": float(np.quantile(values("token_length"), 0.95)),
        "geometry_f1_2px_p50": float(np.quantile(values("geometry_f1_2px"), 0.5)),
        "geometry_f1_2px_p10": float(np.quantile(values("geometry_f1_2px"), 0.1)),
        "geometry_chamfer_px_p95": float(
            np.quantile(values("geometry_chamfer_px"), 0.95)
        ),
        "quantization_endpoint_error_p95": float(
            np.quantile(values("quantization_endpoint_error"), 0.95)
        ),
    }
    return {"profile": profile, "summary": summary, "samples": rows}


def parse_extractor_dirs(
    specifications: list[str],
    fallback: Path,
) -> list[tuple[str, Path]]:
    if not specifications:
        return [(fallback.name or "sketches", fallback)]
    parsed: list[tuple[str, Path]] = []
    names: set[str] = set()
    for specification in specifications:
        name, separator, path_text = specification.partition("=")
        if not separator or not name.strip() or not path_text.strip():
            raise ValueError("--extractor-dir must use NAME=PATH")
        name = name.strip()
        if name in names:
            raise ValueError(f"Duplicate extractor name: {name}")
        names.add(name)
        parsed.append((name, Path(path_text).expanduser()))
    return parsed


def select_common_samples(
    extractors: list[tuple[str, Path]],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[Path]]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    path_maps: dict[str, dict[str, Path]] = {}
    for name, root in extractors:
        mapping = {
            path.relative_to(root).as_posix(): path
            for path in sorted(root.rglob("*.png"))
        }
        if not mapping:
            raise FileNotFoundError(f"No PNG sketches found under {root}")
        path_maps[name] = mapping
    common = set.intersection(*(set(mapping) for mapping in path_maps.values()))
    if not common:
        raise ValueError("Extractor directories have no common relative PNG paths")
    if len(common) < samples:
        raise ValueError(
            f"Requested {samples} common samples, but only {len(common)} are available"
        )
    selected_keys = random.Random(seed).sample(sorted(common), int(samples))
    return {
        name: [path_maps[name][key] for key in selected_keys]
        for name, _ in extractors
    }


def write_manual_review_set(
    paths: list[Path],
    *,
    output_dir: Path,
    codebook: np.ndarray,
    quantizer: ErrorFeedbackQuantizer,
    threshold_profile: str,
    initial_epsilon: float,
    max_geometry_error: float,
    max_token_length: int,
    ordering: str = "continuity",
) -> Path:
    """Render the exact benchmark targets and a feature-preservation checklist."""

    output_dir.mkdir(parents=True, exist_ok=True)
    checklist: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        result = fit_centerline_sequence(
            image_path=path,
            codebook=codebook,
            quantizer=quantizer,
            order_fn=_ORDER_FN_MAP[ordering],
            ordering=ordering,
            initial_epsilon=initial_epsilon,
            max_epsilon=max_geometry_error,
            threshold_profile=threshold_profile,
            max_token_length=max_token_length,
        )
        record = result["record"]
        transform = Stroke5Transform(**record["transform"])
        source = read_grayscale_image(path)
        decoded = decode_tokens(result["tokens"], codebook)
        strokes = stroke5_to_canvas_strokes(decoded, transform)
        reconstruction = np.where(
            rasterize_strokes(strokes, source.shape),
            0,
            255,
        ).astype(np.uint8)
        stem = f"{index:03d}_{path.stem}"
        source_path = output_dir / f"{stem}_source.png"
        reconstruction_path = output_dir / f"{stem}_token.png"
        comparison_path = output_dir / f"{stem}_comparison.png"
        cv2.imwrite(str(source_path), source)
        cv2.imwrite(str(reconstruction_path), reconstruction)
        divider = np.full((source.shape[0], 2), 127, dtype=np.uint8)
        cv2.imwrite(
            str(comparison_path),
            np.concatenate((source, divider, reconstruction), axis=1),
        )
        checklist.append(
            {
                "sample": path.name,
                "comparison_path": str(comparison_path),
                "geometry_f1_2px": float(record["geometry_f1_2px"]),
                "eyes_preserved": None,
                "mouth_preserved": None,
                "face_outline_preserved": None,
                "hair_contours_preserved": None,
                "major_accessories_preserved": None,
                "manual_pass": None,
            }
        )
    checklist_path = output_dir / "manual_review_checklist.json"
    checklist_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "required_passes": min(95, len(checklist)),
                "samples": checklist,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return checklist_path


def main() -> int:
    args = parse_args()
    extractors = parse_extractor_dirs(args.extractor_dir, args.sketches_dir)
    selected_paths = select_common_samples(
        extractors,
        samples=args.samples,
        seed=args.seed,
    )
    codebook_path = args.token_dict_dir / "codebook.npy"
    codebook = np.load(codebook_path)
    quantizer = ErrorFeedbackQuantizer(codebook)
    extractor_results = []
    for name, root in extractors:
        profiles = [
            evaluate_profile(
                selected_paths[name],
                codebook,
                quantizer,
                profile=profile,
                epsilon=args.rdp_epsilon,
                max_geometry_error=args.max_geometry_error,
                max_token_length=args.max_token_length,
                ordering=args.ordering,
            )
            for profile in args.threshold_profiles
        ]
        selected_profile = max(profiles, key=_profile_selection_key)["profile"]
        extractor_results.append(
            {
                "extractor": name,
                "directory": str(root),
                "selected_profile": selected_profile,
                "profiles": profiles,
            }
        )
    selected_extractor_result = max(
        extractor_results,
        key=lambda result: _profile_selection_key(
            next(
                profile
                for profile in result["profiles"]
                if profile["profile"] == result["selected_profile"]
            )
        ),
    )
    selected_extractor = str(selected_extractor_result["extractor"])
    selected_profile = str(selected_extractor_result["selected_profile"])
    profiles = selected_extractor_result["profiles"]
    report = {
        "schema_version": 2,
        "seed": args.seed,
        "sample_count": len(selected_paths[selected_extractor]),
        "rdp_epsilon": args.rdp_epsilon,
        "max_geometry_error": args.max_geometry_error,
        "max_token_length": args.max_token_length,
        "ordering": args.ordering,
        "token_dictionary_path": str(codebook_path),
        "token_dictionary_sha256": hashlib.sha256(codebook_path.read_bytes()).hexdigest(),
        "selected_extractor": selected_extractor,
        "selected_profile": selected_profile,
        "extractors": extractor_results,
        # Retained for single-extractor report consumers.
        "profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"selected_extractor={selected_extractor}")
    print(f"selected_profile={selected_profile}")
    print(f"report={args.output}")
    if args.review_dir:
        checklist_path = write_manual_review_set(
            selected_paths[selected_extractor],
            output_dir=args.review_dir,
            codebook=codebook,
            quantizer=quantizer,
            threshold_profile=selected_profile,
            initial_epsilon=args.rdp_epsilon,
            max_geometry_error=args.max_geometry_error,
            max_token_length=args.max_token_length,
            ordering=args.ordering,
        )
        print(f"manual_review={checklist_path}")
    if args.enforce:
        selected_result = next(item for item in profiles if item["profile"] == selected_profile)
        summary = selected_result["summary"]
        failures = []
        if int(summary["overlength"]) != 0:
            failures.append(f"overlength={summary['overlength']}")
        if int(summary["errors"]) != 0:
            failures.append(f"errors={summary['errors']}")
        if float(summary["geometry_f1_2px_p50"]) < 0.95:
            failures.append(f"f1_p50={summary['geometry_f1_2px_p50']:.4f}")
        if float(summary["geometry_f1_2px_p10"]) < 0.90:
            failures.append(f"f1_p10={summary['geometry_f1_2px_p10']:.4f}")
        if float(summary["quantization_endpoint_error_p95"]) >= 0.01:
            failures.append(
                "endpoint_error_p95="
                f"{summary['quantization_endpoint_error_p95']:.6f}"
            )
        if failures:
            raise SystemExit("centerline quality gate failed: " + ", ".join(failures))
    return 0


def _profile_selection_key(result: dict[str, object]) -> tuple[float, float]:
    summary = result["summary"]
    return (
        float(summary["geometry_f1_2px_p50"]),
        -float(summary["token_length_p95"]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
