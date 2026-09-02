"""Positional encodings for long stroke sequences."""

from __future__ import annotations

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding matching the TensorFlow baseline."""

    def __init__(self, max_length: int, d_model: int) -> None:
        super().__init__()
        self.max_length = int(max_length)
        positions = torch.arange(self.max_length, dtype=torch.float32).unsqueeze(1)
        dimensions = torch.arange(d_model, dtype=torch.float32).unsqueeze(0)
        angle_rates = torch.pow(10000.0, -(2 * torch.div(dimensions, 2, rounding_mode="floor")) / d_model)
        angles = positions * angle_rates
        encoding = torch.empty(self.max_length, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(angles[:, 0::2])
        encoding[:, 1::2] = torch.cos(angles[:, 1::2])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: torch.Tensor, *, offset: int = 0) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected input shape (batch, sequence, channels)")
        sequence_length = x.shape[1]
        if offset < 0 or offset + sequence_length > self.max_length:
            raise ValueError(
                f"Position range [{offset}, {offset + sequence_length}) exceeds "
                f"capacity {self.max_length}"
            )
        positions = self.encoding[offset : offset + sequence_length]
        return x + positions.to(dtype=x.dtype, device=x.device).unsqueeze(0)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embedding for variable-length stroke sequences."""

    def __init__(self, max_length: int, d_model: int) -> None:
        super().__init__()
        self.max_length = int(max_length)
        self.embedding = nn.Embedding(self.max_length, d_model)

    def forward(self, x: torch.Tensor, *, offset: int = 0) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected input shape (batch, sequence, channels)")
        sequence_length = x.shape[1]
        if offset < 0 or offset + sequence_length > self.max_length:
            raise ValueError(
                f"Position range [{offset}, {offset + sequence_length}) exceeds "
                f"capacity {self.max_length}"
            )
        positions = torch.arange(offset, offset + sequence_length, device=x.device)
        return x + self.embedding(positions).unsqueeze(0)


def build_positional_encoding(kind: str, max_length: int, d_model: int) -> nn.Module:
    """Build the configured positional encoding module."""

    if kind == "learned":
        return LearnedPositionalEncoding(max_length, d_model)
    if kind == "sinusoidal":
        return SinusoidalPositionalEncoding(max_length, d_model)
    raise ValueError("positional encoding must be one of: learned, sinusoidal")
