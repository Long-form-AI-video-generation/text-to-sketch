from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from dataloaders.masks import build_sequence_masks
from models.sketchformer.config import (
    PositionalEncodingConfig,
    ReconstructionHeadConfig,
    SketchformerConfig,
    TokenDictionaryConfig,
)
from models.sketchformer.model import SketchformerModel


def _checkpointed_token_model() -> SketchformerModel:
    return SketchformerModel(
        SketchformerConfig(
            name="sketchformer_tok_dict",
            input_mode="tok_dict",
            max_seq_len=16,
            token_dictionary=TokenDictionaryConfig(
                codebook_size=4,
                motion_token_offset=1,
                pad_token_id=0,
                sep_token_id=5,
                sos_token_id=6,
                eos_token_id=7,
                vocab_size=8,
            ),
            d_model=16,
            latent_dim=16,
            pool_hidden_dim=8,
            pooling_mode="tf_self_attn_v1",
            latent_expander_mode="tf_dense",
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            dim_feedforward=32,
            dropout=0.1,
            activation="relu",
            norm_first=False,
            use_final_norm=False,
            gradient_checkpointing=True,
            positional_encoding=PositionalEncodingConfig(
                type="sinusoidal",
                max_length=16,
            ),
            decoder_autoregressive=True,
            blind_decoder_mask=True,
            reconstruction=ReconstructionHeadConfig(
                enabled=True,
                target="tok_dict",
            ),
        )
    )


class SketchformerAttentionRegressionTest(unittest.TestCase):
    def test_checkpointed_training_combines_causal_and_padding_visibility(self) -> None:
        """The SDPA call must never receive both an explicit mask and is_causal."""

        torch.manual_seed(11)
        model = _checkpointed_token_model().train()
        tokens = torch.tensor(
            [
                [6, 1, 5, 7, 0, 0],
                [6, 2, 3, 5, 4, 7],
            ],
            dtype=torch.long,
        )
        masks = build_sequence_masks([4, 6], max_length=6)
        batch = {"tokens": tokens, "targets": tokens.clone(), **masks}

        real_sdpa = F.scaled_dot_product_attention
        attention_calls: list[tuple[torch.Tensor | None, bool]] = []

        def recording_sdpa(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
        ) -> torch.Tensor:
            attention_calls.append(
                (
                    None if attn_mask is None else attn_mask.detach().clone(),
                    bool(is_causal),
                )
            )
            return real_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

        with patch.object(F, "scaled_dot_product_attention", side_effect=recording_sdpa):
            output = model(batch)
            assert output.reconstruction is not None
            assert output.loss_targets is not None
            logits = output.reconstruction.token_logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                output.loss_targets.reshape(-1),
                ignore_index=0,
            )
            loss.backward()

        self.assertTrue(attention_calls)
        self.assertFalse(
            any(mask is not None and is_causal for mask, is_causal in attention_calls)
        )

        decoder_valid_mask = masks["valid_mask"][:, :-1]
        causal_visibility = torch.ones((5, 5), dtype=torch.bool).tril()
        expected_decoder_mask = (
            decoder_valid_mask[:, None, None, :] & causal_visibility
        ).contiguous()
        decoder_masks = [
            mask
            for mask, _ in attention_calls
            if mask is not None and mask.shape == expected_decoder_mask.shape
        ]
        self.assertTrue(decoder_masks)
        for decoder_mask in decoder_masks:
            self.assertTrue(torch.equal(decoder_mask, expected_decoder_mask))

        decoder_grad = model.decoder.layers[0].self_attn.q_proj.weight.grad
        self.assertIsNotNone(decoder_grad)
        assert decoder_grad is not None
        self.assertTrue(torch.isfinite(decoder_grad).all())


if __name__ == "__main__":
    unittest.main()
