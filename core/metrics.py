"""Metrics for Sketchformer reconstruction validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from core.losses import masked_mean


@dataclass
class ReconstructionMetrics:
    """Common validation metrics for continuous stroke reconstruction."""

    xy_mse: torch.Tensor
    xy_l1: torch.Tensor
    pen_accuracy: torch.Tensor
    valid_tokens: torch.Tensor

    def as_log_dict(self, prefix: str = "val") -> dict[str, torch.Tensor]:
        return {
            f"{prefix}/xy_mse": self.xy_mse.detach(),
            f"{prefix}/xy_l1": self.xy_l1.detach(),
            f"{prefix}/pen_accuracy": self.pen_accuracy.detach(),
            f"{prefix}/valid_tokens": self.valid_tokens.detach(),
        }


@dataclass
class TokenReconstructionMetrics:
    """Common validation metrics for tok-dict reconstruction."""

    token_loss: torch.Tensor
    token_accuracy: torch.Tensor
    token_perplexity: torch.Tensor
    valid_tokens: torch.Tensor

    def as_log_dict(self, prefix: str = "val") -> dict[str, torch.Tensor]:
        return {
            f"{prefix}/token_loss": self.token_loss.detach(),
            f"{prefix}/token_accuracy": self.token_accuracy.detach(),
            f"{prefix}/token_perplexity": self.token_perplexity.detach(),
            f"{prefix}/valid_tokens": self.valid_tokens.detach(),
        }


@torch.no_grad()
def reconstruction_metrics(
    output: Any,
    batch: Mapping[str, torch.Tensor],
) -> ReconstructionMetrics | TokenReconstructionMetrics:
    """Compute mask-aware reconstruction metrics."""

    if output.reconstruction is None:
        raise ValueError("Model output does not include reconstruction predictions")

    targets = output.loss_targets if getattr(output, "loss_targets", None) is not None else batch["targets"]
    valid_mask = batch.get("valid_mask")
    if getattr(output, "loss_valid_mask", None) is not None:
        valid_mask = output.loss_valid_mask
    elif valid_mask is None:
        valid_mask = torch.ones(targets.shape[:2], dtype=torch.bool, device=targets.device)
    elif valid_mask.shape[:2] != targets.shape[:2]:
        valid_mask = valid_mask[:, : targets.shape[1]]
    else:
        valid_mask = valid_mask.to(device=targets.device, dtype=torch.bool)

    token_logits = getattr(output.reconstruction, "token_logits", None)
    if token_logits is not None:
        token_targets = targets.to(device=token_logits.device, dtype=torch.long)
        token_mask = valid_mask.to(device=token_logits.device, dtype=torch.bool)
        loss = F.cross_entropy(
            token_logits.reshape(-1, token_logits.shape[-1]),
            token_targets.reshape(-1),
            reduction="none",
        ).view_as(token_targets)
        token_loss = masked_mean(loss, token_mask)
        token_predictions = torch.argmax(token_logits, dim=-1)
        token_accuracy = masked_mean(
            (token_predictions == token_targets).to(dtype=torch.float32),
            token_mask,
        )
        return TokenReconstructionMetrics(
            token_loss=token_loss,
            token_accuracy=token_accuracy,
            token_perplexity=torch.exp(token_loss.detach().clamp(max=50.0)),
            valid_tokens=token_mask.sum().to(dtype=torch.float32),
        )

    target_xy = targets[..., :2]
    pred_xy = output.reconstruction.xy
    xy_mse = masked_mean(F.mse_loss(pred_xy, target_xy, reduction="none"), valid_mask)
    xy_l1 = masked_mean(F.l1_loss(pred_xy, target_xy, reduction="none"), valid_mask)

    pen_target = targets[..., 2].round().long().clamp(
        min=0,
        max=output.reconstruction.pen_logits.shape[-1] - 1,
    )
    pen_pred = torch.argmax(output.reconstruction.pen_logits, dim=-1)
    correct = (pen_pred == pen_target).to(dtype=torch.float32)
    pen_accuracy = masked_mean(correct, valid_mask)
    valid_tokens = valid_mask.sum().to(dtype=torch.float32)

    return ReconstructionMetrics(
        xy_mse=xy_mse,
        xy_l1=xy_l1,
        pen_accuracy=pen_accuracy,
        valid_tokens=valid_tokens,
    )
