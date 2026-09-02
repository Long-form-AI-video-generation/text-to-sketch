from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from metrics.sketchformer.reconstruction import (
    ReconstructionExample,
    attach_reconstruction_render_metadata,
    collect_reconstruction_examples,
    load_reconstruction_render_metadata,
)
from metrics.sketchformer.visualisation import (
    rasterize_decoded_strokes,
    save_reconstruction_pair,
    stroke3_to_points,
)
from pipeline.stroke5 import Stroke5Transform


class SketchformerVisualisationTest(unittest.TestCase):
    @staticmethod
    def _transform() -> Stroke5Transform:
        return Stroke5Transform(
            canvas_height=64,
            canvas_width=64,
            start_x=0.0,
            start_y=0.0,
            bbox_x=0.0,
            bbox_y=0.0,
            bbox_width=6.0,
            bbox_height=6.0,
            scale=10.0,
            normalization_extent=1.0,
        )

    def test_stroke3_to_points_accepts_decoded_stroke5(self) -> None:
        stroke5 = np.asarray(
            [
                [1.0, 2.0, 1.0, 0.0, 0.0],
                [3.0, 4.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        points = stroke3_to_points(stroke5)

        np.testing.assert_allclose(
            points,
            np.asarray(
                [
                    [1.0, 2.0],
                    [4.0, 6.0],
                    [4.0, 6.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_save_reconstruction_pair_accepts_decoded_stroke5(self) -> None:
        stroke5 = np.asarray(
            [
                [1.0, 2.0, 1.0, 0.0, 0.0],
                [3.0, 4.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        example = ReconstructionExample(
            target=stroke5,
            prediction=stroke5,
            length=len(stroke5),
            source_file="sample.npz",
            source_index=0,
            label=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "pair.png"
            saved = save_reconstruction_pair(example, output_path)

            self.assertEqual(saved, output_path)
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_rasterizer_ignores_single_point_pen_up_relocations(self) -> None:
        decoded = np.asarray(
            [
                [1.0, 1.0, 0.0, 1.0, 0.0],  # Invisible singleton relocation.
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        image = rasterize_decoded_strokes(
            decoded,
            self._transform(),
            (64, 64),
        )

        self.assertEqual(int(image[10, 10]), 255)
        self.assertEqual(int(image[10, 25]), 0)

    def test_manifest_context_adds_original_source_and_transform(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_root = root / "images"
            image_root.mkdir()
            source_path = image_root / "sample.png"
            cv2.imwrite(
                str(source_path),
                np.full((64, 64), 255, dtype=np.uint8),
            )
            manifest_path = root / "preprocessing_manifest.jsonl"
            record = {
                "sample_id": "sample",
                "source_relative_path": "sample.png",
                "source_path": "old/location/sample.png",
                "status": "accepted",
                "transform": self._transform().__dict__,
            }
            manifest_path.write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )

            metadata = load_reconstruction_render_metadata(
                manifest_path,
                project_root=root,
                source_images_root=image_root,
            )
            example = ReconstructionExample(
                target=np.asarray(
                    [
                        [1.0, 1.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                prediction=np.asarray(
                    [
                        [1.0, 1.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                length=3,
                source_file="test.npz",
                source_index=0,
                sample_id="sample",
            )

            enriched = attach_reconstruction_render_metadata(
                [example],
                metadata,
            )[0]
            self.assertEqual(enriched.source_image_path, str(source_path))
            self.assertEqual(enriched.canvas_transform, self._transform())

            output_path = root / "three-panel.png"
            save_reconstruction_pair(enriched, output_path)
            rendered = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(rendered)
            assert rendered is not None
            self.assertGreater(rendered.shape[1], 2 * rendered.shape[0])

    def test_teacher_forced_visual_target_uses_complete_batch_target(self) -> None:
        codebook = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        batch_targets = torch.tensor([[4, 1, 2, 3, 5]], dtype=torch.long)
        predicted_tokens = torch.tensor([[1, 2, 3, 5]], dtype=torch.long)
        token_logits = torch.full((1, 4, 6), -10.0)
        token_logits.scatter_(2, predicted_tokens.unsqueeze(-1), 10.0)
        output = SimpleNamespace(
            reconstruction=SimpleNamespace(token_logits=token_logits),
            # Deliberately different to prove plotting does not use loss targets.
            loss_targets=torch.tensor([[2, 3, 5, 0]], dtype=torch.long),
            loss_valid_mask=torch.tensor([[True, True, True, True]]),
        )
        batch = {
            "targets": batch_targets,
            "lengths": torch.tensor([5]),
            "valid_mask": torch.tensor([[True, True, True, True, True]]),
            "source_files": ["valid.npz"],
            "source_indices": torch.tensor([0]),
            "sample_ids": ["sample"],
            "labels": torch.tensor([0]),
        }

        example = collect_reconstruction_examples(
            output,
            batch,
            max_examples=1,
            codebook=codebook,
        )[0]

        self.assertEqual(example.length, 5)
        np.testing.assert_allclose(example.target[0, :2], [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
