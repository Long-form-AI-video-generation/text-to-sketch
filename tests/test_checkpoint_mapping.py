from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from models.sketchformer.checkpoint_mapping import (
    convert_tok_dict_tensorflow_state,
    load_torch_checkpoint,
    resize_vector,
    resize_learned_position_embedding,
    tensorflow_kernel_to_linear_weight,
)
from models.sketchformer.pretrained import inspect_tensorflow_checkpoint


class CheckpointMappingTest(unittest.TestCase):
    def test_resize_learned_position_embedding_changes_length_only(self) -> None:
        source = torch.arange(12, dtype=torch.float32).view(3, 4)

        resized = resize_learned_position_embedding(source, target_length=6)

        self.assertEqual(resized.shape, (6, 4))
        self.assertEqual(resized.dtype, source.dtype)

    def test_resize_vector_changes_length_only(self) -> None:
        source = torch.arange(4, dtype=torch.float32)

        resized = resize_vector(source, target_length=8)

        self.assertEqual(resized.shape, (8,))
        self.assertEqual(resized.dtype, source.dtype)

    def test_tensorflow_kernel_to_linear_weight_transposes(self) -> None:
        source = torch.arange(6, dtype=torch.float32).view(2, 3)

        converted = tensorflow_kernel_to_linear_weight(source)

        self.assertEqual(converted.shape, (3, 2))
        self.assertEqual(converted.tolist(), source.T.tolist())

    def test_convert_tok_dict_tensorflow_state_maps_and_skips_keys(self) -> None:
        target_state = {
            "input_embedding.token_embedding.weight": torch.zeros(8, 4),
            "target_embedding.token_embedding.weight": torch.zeros(8, 4),
            "encoder.layers.0.self_attn.q_proj.weight": torch.zeros(4, 4),
            "encoder.layers.0.self_attn.q_proj.bias": torch.zeros(4),
            "pool.W_attn": torch.zeros(4, 6),
            "pool.b_attn": torch.zeros(6),
            "pool.V_attn": torch.zeros(6, 1),
            "latent_expander.expand_layer.weight": torch.zeros(8, 1),
            "latent_expander.expand_layer.bias": torch.zeros(8),
            "latent_expander.long_weight": torch.zeros(16, 1),
            "latent_expander.long_bias": torch.zeros(16),
            "reconstruction_head.projection.weight": torch.zeros(8, 4),
            "reconstruction_head.projection.bias": torch.zeros(8),
        }
        tf_state = {
            "transformer/encoder/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE": torch.ones(8, 4),
            "transformer/decoder/embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE": torch.ones(8, 4) * 2,
            "transformer/encoder/enc_layers/0/mha/wq/kernel/.ATTRIBUTES/VARIABLE_VALUE": torch.ones(4, 4),
            "transformer/encoder/enc_layers/0/mha/wq/bias/.ATTRIBUTES/VARIABLE_VALUE": torch.ones(4),
            "transformer/bottleneck_layer/W_attn/.ATTRIBUTES/VARIABLE_VALUE": torch.ones(4, 6),
            "transformer/bottleneck_layer/b_attn/.ATTRIBUTES/VARIABLE_VALUE": torch.ones(6),
            "transformer/bottleneck_layer/V_attn/.ATTRIBUTES/VARIABLE_VALUE": torch.ones(6, 1),
            "transformer/expand_layer/expand_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE": torch.ones(1, 4),
            "transformer/expand_layer/expand_layer/bias/.ATTRIBUTES/VARIABLE_VALUE": torch.ones(4),
            "transformer/output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE": torch.ones(4, 8),
            "transformer/output_layer/bias/.ATTRIBUTES/VARIABLE_VALUE": torch.ones(8),
            "transformer/classify_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE": torch.ones(4, 2),
            "transformer/output_layer/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE": torch.ones(4, 8),
        }

        converted, report = convert_tok_dict_tensorflow_state(
            tf_state,
            target_state,
            target_seq_len=8,
        )

        self.assertIn("input_embedding.token_embedding.weight", converted)
        self.assertIn("reconstruction_head.projection.weight", converted)
        self.assertEqual(converted["latent_expander.expand_layer.weight"].shape, (8, 1))
        self.assertEqual(report.initialized_keys, [
            "latent_expander.long_bias",
            "latent_expander.long_weight",
        ])
        self.assertTrue(any("classify_layer" in key for key in report.skipped_keys))
        self.assertTrue(any("OPTIMIZER_SLOT" in key for key in report.skipped_keys))

    def test_load_torch_checkpoint_reports_missing_and_unexpected_keys(self) -> None:
        model = nn.Linear(2, 2)
        checkpoint = {
            "state_dict": {
                "weight": torch.ones_like(model.weight),
                "unexpected.weight": torch.zeros(1),
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save(checkpoint, path)

            report = load_torch_checkpoint(model, path, strict=False)

        self.assertIn("bias", report.missing_keys)
        self.assertIn("unexpected.weight", report.unexpected_keys)

    def test_tensorflow_checkpoint_inspection_accepts_index_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "ckpt-1"
            prefix.with_suffix(".index").write_text("", encoding="utf-8")
            (Path(tmp) / "ckpt-1.data-00000-of-00001").write_text("", encoding="utf-8")

            checkpoint = inspect_tensorflow_checkpoint(prefix.with_suffix(".index"))

            self.assertEqual(checkpoint.prefix, prefix)
            self.assertTrue(checkpoint.exists)
            self.assertEqual(len(checkpoint.data_files), 1)


if __name__ == "__main__":
    unittest.main()
