"""Output heads for reconstruction and optional classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.sketchformer.config import SketchformerConfig


@dataclass
class ReconstructionOutput:
    """Structured continuous reconstruction output."""

    raw: torch.Tensor
    xy: torch.Tensor
    pen_logits: torch.Tensor
    mixture_logits: torch.Tensor | None = None
    mu: torch.Tensor | None = None
    log_sigma: torch.Tensor | None = None
    rho: torch.Tensor | None = None


@dataclass
class TokenReconstructionOutput:
    """Structured token-dictionary reconstruction output."""

    raw: torch.Tensor
    token_logits: torch.Tensor


class ContinuousReconstructionHead(nn.Module):
    """Predict continuous xy deltas and pen-state logits."""

    def __init__(self, config: SketchformerConfig) -> None:
        super().__init__()
        self.config = config.reconstruction
        self.pen_classes = config.pen_classes
        self.num_mixtures = int(self.config.num_mixtures)

        if self.config.xy_distribution == "gaussian_mixture":
            output_dim = self.num_mixtures * 6 + self.pen_classes
        elif self.config.xy_distribution == "deterministic":
            output_dim = 2 + self.pen_classes
        else:
            raise ValueError(f"Unsupported xy distribution: {self.config.xy_distribution}")

        self.projection = nn.Linear(config.d_model, output_dim)

    def forward(self, x: torch.Tensor) -> ReconstructionOutput:
        raw = self.projection(x)
        if self.config.xy_distribution == "deterministic":
            xy = raw[..., :2]
            pen_logits = raw[..., 2 : 2 + self.pen_classes]
            return ReconstructionOutput(raw=raw, xy=xy, pen_logits=pen_logits)

        mixture_dim = self.num_mixtures
        cursor = 0
        mixture_logits = raw[..., cursor : cursor + mixture_dim]
        cursor += mixture_dim

        mu = raw[..., cursor : cursor + mixture_dim * 2].view(*raw.shape[:-1], mixture_dim, 2)
        cursor += mixture_dim * 2

        log_sigma = raw[..., cursor : cursor + mixture_dim * 2].view(
            *raw.shape[:-1],
            mixture_dim,
            2,
        )
        cursor += mixture_dim * 2

        rho = torch.tanh(raw[..., cursor : cursor + mixture_dim])
        cursor += mixture_dim

        pen_logits = raw[..., cursor : cursor + self.pen_classes]
        mixture_weights = torch.softmax(mixture_logits, dim=-1)
        xy = torch.sum(mixture_weights.unsqueeze(-1) * mu, dim=-2)

        return ReconstructionOutput(
            raw=raw,
            xy=xy,
            pen_logits=pen_logits,
            mixture_logits=mixture_logits,
            mu=mu,
            log_sigma=log_sigma,
            rho=rho,
        )


class TokenReconstructionHead(nn.Module):
    """Predict one tok-dict vocabulary distribution per sequence position."""

    def __init__(self, config: SketchformerConfig) -> None:
        super().__init__()
        self.projection = nn.Linear(config.d_model, config.token_dictionary.vocab_size)

    def forward(self, x: torch.Tensor) -> TokenReconstructionOutput:
        logits = self.projection(x)
        return TokenReconstructionOutput(raw=logits, token_logits=logits)


class ClassificationHead(nn.Module):
    """Optional classifier over the pooled sketch embedding."""

    def __init__(self, config: SketchformerConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(config.classification.dropout),
            nn.Linear(config.latent_dim, config.classification.num_classes),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)
