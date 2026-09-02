"""Mask helpers for variable-length stroke sequences."""

from __future__ import annotations

from typing import Any

import torch


def lengths_to_valid_mask(
    lengths: torch.Tensor | list[int],
    max_length: int | None = None,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a boolean mask where True marks a real sequence element."""

    lengths_tensor = torch.as_tensor(lengths, dtype=torch.long, device=device)
    if lengths_tensor.ndim != 1:
        raise ValueError("lengths must be a 1D tensor or list")

    if max_length is None:
        max_length = int(lengths_tensor.max().item()) if len(lengths_tensor) else 0
    positions = torch.arange(max_length, device=lengths_tensor.device)
    return positions.unsqueeze(0) < lengths_tensor.unsqueeze(1)


def valid_to_padding_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    """Return a key-padding mask where True marks padding positions."""

    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    return ~valid_mask


def causal_mask(size: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Return a lower-triangular boolean mask where True means attention is allowed."""

    return torch.ones((size, size), dtype=torch.bool, device=device).tril()


def make_sdpa_self_attention_mask(
    valid_mask: torch.Tensor,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """Build a broadcast SDPA key mask with shape ``(batch, 1, 1, source)``.

    PyTorch ``scaled_dot_product_attention`` uses True to mean the attention
    element is allowed. This is the inverse of ``nn.MultiheadAttention`` key
    padding mask semantics, where True means the token is masked out.
    """

    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must have shape (batch, sequence)")

    del causal  # Causality is passed to SDPA via is_causal without an LxL tensor.
    return valid_mask[:, None, None, :].contiguous()


def build_sequence_masks(
    lengths: torch.Tensor | list[int],
    max_length: int | None = None,
    *,
    causal: bool = False,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Return all standard masks used by the in-repo Sketchformer path."""

    valid_mask = lengths_to_valid_mask(lengths, max_length=max_length, device=device)
    return {
        "valid_mask": valid_mask,
        "padding_mask": valid_to_padding_mask(valid_mask),
        "sdpa_mask": make_sdpa_self_attention_mask(valid_mask, causal=causal),
        "is_causal": bool(causal),
    }
