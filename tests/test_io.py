from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from utils import io


class AtomicIoTest(unittest.TestCase):
    def test_failed_token_replacement_preserves_previous_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tokens.npz"
            original = np.asarray([1, 2, 3], dtype=np.int32)
            io.save_token_sequence(original, path)

            def fail_after_partial_write(target, **_arrays) -> None:
                if hasattr(target, "write"):
                    target.write(b"partial archive")
                else:
                    Path(target).write_bytes(b"partial archive")
                raise RuntimeError("simulated write failure")

            with (
                patch.object(
                    io.np,
                    "savez_compressed",
                    side_effect=fail_after_partial_write,
                ),
                self.assertRaisesRegex(RuntimeError, "simulated write failure"),
            ):
                io.save_token_sequence(
                    np.asarray([4, 5, 6], dtype=np.int32),
                    path,
                )

            np.testing.assert_array_equal(io.load_token_sequence(path), original)
            self.assertFalse(list(Path(tmpdir).glob(".tokens.npz.*")))

    def test_stroke5_round_trip_uses_atomic_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "stroke5.npz"
            expected = np.asarray(
                [[0.1, 0.2, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]],
                dtype=np.float32,
            )

            io.save_stroke5(expected, path)

            np.testing.assert_array_equal(io.load_stroke5(path), expected)
            self.assertFalse(list(path.parent.glob(".stroke5.npz.*")))


if __name__ == "__main__":
    unittest.main()
