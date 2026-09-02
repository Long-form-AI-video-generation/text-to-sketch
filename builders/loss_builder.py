"""Loss configuration builders for Sketchformer fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LossWeights:
    """Weights for the loss terms used by the training loop."""

    reconstruction: float = 1.0
    token: float = 1.0
    pen_state: float = 1.0
    classification: float = 0.0
    kl: float = 0.0


def build_loss_weights(config: Mapping[str, Any]) -> LossWeights:
    """Build structured loss weights from optimizer/training config."""

    weights = config.get("loss_weights", config)
    return LossWeights(
        reconstruction=float(weights.get("reconstruction", 1.0)),
        token=float(weights.get("token", weights.get("reconstruction", 1.0))),
        pen_state=float(weights.get("pen_state", 1.0)),
        classification=float(weights.get("classification", 0.0)),
        kl=float(weights.get("kl", 0.0)),
    )


def build_loss(config: Mapping[str, Any]):
    """Build the concrete loss object once ``core.losses`` is available."""

    try:
        from core.losses import SketchformerLoss
    except ImportError as exc:
        raise ImportError(
            "core.losses.SketchformerLoss is required before build_loss can "
            "create the training loss object."
        ) from exc

    return SketchformerLoss(build_loss_weights(config))
