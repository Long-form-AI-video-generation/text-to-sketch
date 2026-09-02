"""Lightning-style training module without requiring Lightning as a dependency."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from builders import build_optimizer, build_scheduler
from core.losses import SketchformerLoss
from core.trainer import validation_step


class SketchformerTrainingModule(nn.Module):
    """Thin training wrapper around model, loss, optimizer, and scheduler config."""

    def __init__(
        self,
        model: nn.Module,
        *,
        loss_weights: Any,
        optimizer_config: Mapping[str, Any],
        total_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = SketchformerLoss(loss_weights)
        self.optimizer_config = dict(optimizer_config)
        self.total_steps = total_steps

    def forward(self, batch: Mapping[str, Any]) -> Any:
        return self.model(batch)

    def training_step(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        output = self.model(batch)
        losses = self.loss_fn(output, batch)
        return {
            "loss": losses.total,
            "logs": losses.as_log_dict(prefix="train"),
            "output": output,
        }

    def validation_step(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        result = validation_step(self.model, batch, self.loss_fn)
        return {
            "loss": result.loss,
            "logs": result.logs,
            "output": result.output,
        }

    def configure_optimizers(self):
        optimizer = build_optimizer(self.model, self.optimizer_config)
        scheduler = build_scheduler(
            optimizer,
            self.optimizer_config,
            total_steps=self.total_steps,
        )
        return optimizer, scheduler
