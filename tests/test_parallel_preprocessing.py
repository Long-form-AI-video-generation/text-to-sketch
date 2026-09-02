from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline import workflow


class _FakeProcessPoolExecutor:
    instance: "_FakeProcessPoolExecutor | None" = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.map_function = None
        self.map_chunksize = None
        self.shutdown_called = False
        _FakeProcessPoolExecutor.instance = self

    def map(self, function, samples, *, chunksize):
        self.map_function = function
        self.map_chunksize = chunksize
        return iter(
            {
                "status": "accepted",
                "source_path": str(path),
                "sample_id": Path(path).stem,
            }
            for path in samples
        )

    def shutdown(self) -> None:
        self.shutdown_called = True


class ParallelPreprocessingDispatchTest(unittest.TestCase):
    def test_worker_count_is_cpu_capped_and_pool_results_keep_input_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sketches = root / "sketches"
            sketches.mkdir()
            samples = [
                sketches / "first.png",
                sketches / "second.png",
                sketches / "third.png",
            ]
            for sample in samples:
                sample.write_bytes(b"")
            token_dict = root / "codebook"
            token_dict.mkdir()
            np.save(token_dict / "codebook.npy", np.zeros((3, 2), dtype=np.float32))
            manifest = root / "manifest.jsonl"

            with (
                patch.object(workflow, "_available_cpu_count", return_value=2),
                patch.object(
                    workflow,
                    "ProcessPoolExecutor",
                    _FakeProcessPoolExecutor,
                ),
            ):
                workflow._run_centerline_pipeline(
                    samples=samples,
                    sketches_dir=sketches,
                    stroke5_dir=root / "stroke5",
                    token_dict_dir=token_dict,
                    ordering="continuity",
                    rdp_epsilon=2.0,
                    threshold_profile="hysteresis",
                    max_token_length=4096,
                    max_geometry_error=2.0,
                    manifest_path=manifest,
                    fail_on_overlength=False,
                    extractor_name="test",
                    num_workers=8,
                )

            pool = _FakeProcessPoolExecutor.instance
            assert pool is not None
            self.assertEqual(pool.kwargs["max_workers"], 2)
            self.assertTrue(pool.kwargs["initargs"][0].limit_library_threads)
            self.assertIs(pool.map_function, workflow._process_centerline_sample)
            self.assertEqual(pool.map_chunksize, 1)
            self.assertTrue(pool.shutdown_called)
            records = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["sample_id"] for record in records],
                ["first", "second", "third"],
            )

    def test_direct_api_rejects_invalid_parallel_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sketches = root / "sketches"
            sketches.mkdir()
            (sketches / "sample.png").write_bytes(b"")
            common = {
                "sketches_dir": sketches,
                "stroke5_dir": root / "stroke5",
                "sketch_token_dir": root / "codebook",
                "n_sketches": 1,
                "ordering": "continuity",
            }

            with self.assertRaisesRegex(ValueError, "at least 1"):
                workflow.run_pipeline(**common, num_workers=0)
            with self.assertRaisesRegex(ValueError, "only supported for centerline"):
                workflow.run_pipeline(
                    **common,
                    vectorizer="contour",
                    num_workers=2,
                )

    def test_manifest_replacement_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.jsonl"
            path.write_text("existing\n", encoding="utf-8")
            calls = 0

            def fail_on_second_record(*_args, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("simulated manifest failure")
                return "{}"

            with (
                patch.object(workflow.json, "dumps", side_effect=fail_on_second_record),
                self.assertRaisesRegex(RuntimeError, "simulated manifest failure"),
            ):
                workflow._write_manifest(path, [{"id": 1}, {"id": 2}])

            self.assertEqual(path.read_text(encoding="utf-8"), "existing\n")
            self.assertFalse(list(Path(tmpdir).glob(".manifest.jsonl.*")))


if __name__ == "__main__":
    unittest.main()
