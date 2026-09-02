"""Deterministic serial/parallel centerline preprocessing equivalence eval."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "pipeline").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root")


_add_project_to_path()

from pipeline.workflow import run_pipeline
from utils.io import load_stroke5, load_token_sequence


def _write_line_sketch(path: Path, *, index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((64, 64), 255, dtype=np.uint8)
    if index % 2:
        cv2.line(
            image,
            (8, 8 + index),
            (55, max(12, 52 - index)),
            0,
            thickness=3,
        )
    else:
        cv2.line(image, (8, 12 + index), (55, 12 + index), 0, thickness=3)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write fixture image: {path}")


def _write_codebook(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    axes = (-0.5, 0.0, 0.5)
    codebook = np.asarray([(x, y) for x in axes for y in axes], dtype=np.float32)
    np.save(path / "codebook.npy", codebook)


def _load_manifest(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalize_manifest(records: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized = []
    for record in records:
        item = dict(record)
        item.pop("stroke5_path", None)
        item.pop("tokens_path", None)
        normalized.append(item)
    return normalized


def _assert_archives_equal(
    serial_dir: Path,
    parallel_dir: Path,
    *,
    loader,
) -> int:
    serial_paths = sorted(serial_dir.rglob("*.npz"))
    parallel_paths = sorted(parallel_dir.rglob("*.npz"))
    serial_relative = [path.relative_to(serial_dir) for path in serial_paths]
    parallel_relative = [path.relative_to(parallel_dir) for path in parallel_paths]
    if serial_relative != parallel_relative:
        raise AssertionError(
            f"Archive paths differ: {serial_relative} != {parallel_relative}"
        )
    for serial_path, parallel_path in zip(serial_paths, parallel_paths):
        np.testing.assert_array_equal(loader(serial_path), loader(parallel_path))
    return len(serial_paths)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sketches = root / "sketches"
        codebook = root / "codebook"
        _write_codebook(codebook)
        for index in range(6):
            directory = sketches / ("a" if index % 2 == 0 else "b")
            _write_line_sketch(directory / f"sketch_{index}.png", index=index)
        (sketches / "broken.png").write_bytes(b"not an image")

        common = {
            "sketches_dir": sketches,
            "sketch_token_dir": codebook,
            "n_sketches": 7,
            "ordering": "continuity",
            "rdp_epsilon": 2.0,
            "seed": 42,
            "vectorizer": "centerline",
            "threshold_profile": "hysteresis",
            "max_token_length": 4096,
            "max_geometry_error": 2.0,
            "token_dict_dir": codebook,
            "fail_on_overlength": False,
            "extractor_name": "parallel-eval",
        }
        serial_root = root / "serial"
        parallel_root = root / "parallel"

        serial_start = time.perf_counter()
        run_pipeline(
            **common,
            stroke5_dir=serial_root / "stroke5",
            manifest_path=serial_root / "manifest.jsonl",
            num_workers=1,
        )
        serial_seconds = time.perf_counter() - serial_start

        parallel_start = time.perf_counter()
        run_pipeline(
            **common,
            stroke5_dir=parallel_root / "stroke5",
            manifest_path=parallel_root / "manifest.jsonl",
            num_workers=2,
        )
        parallel_seconds = time.perf_counter() - parallel_start

        serial_records = _load_manifest(serial_root / "manifest.jsonl")
        parallel_records = _load_manifest(parallel_root / "manifest.jsonl")
        if _normalize_manifest(serial_records) != _normalize_manifest(parallel_records):
            raise AssertionError("Serial and parallel manifests differ")
        stroke_count = _assert_archives_equal(
            serial_root / "stroke5",
            parallel_root / "stroke5",
            loader=load_stroke5,
        )
        token_count = _assert_archives_equal(
            serial_root / "tokens",
            parallel_root / "tokens",
            loader=load_token_sequence,
        )
        statuses = [str(record["status"]) for record in serial_records]
        if statuses.count("accepted") != 6 or statuses.count("error") != 1:
            raise AssertionError(f"Unexpected manifest statuses: {statuses}")

        report = {
            "eval": "parallel_preprocessing",
            "status": "pass",
            "attempted": len(serial_records),
            "accepted": statuses.count("accepted"),
            "errors": statuses.count("error"),
            "stroke_archives": stroke_count,
            "token_archives": token_count,
            "serial_seconds": serial_seconds,
            "parallel_seconds": parallel_seconds,
            "observed_speedup": serial_seconds / max(parallel_seconds, 1.0e-9),
        }
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
