"""Lightweight PyTorch training helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class StepResult:
    """Result of one train or validation step."""

    loss: torch.Tensor
    logs: dict[str, torch.Tensor]
    output: Any


def move_to_device(value: Any, device: torch.device | str) -> Any:
    """Recursively move tensors in a nested batch to a device."""

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def _scheduler_object(scheduler: Any) -> Any:
    return getattr(scheduler, "scheduler", scheduler)


def train_step(
    model: nn.Module,
    batch: Mapping[str, Any],
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Any | None = None,
    device: torch.device | str | None = None,
    grad_clip_norm: float | None = None,
) -> StepResult:
    """Run one optimization step."""

    model.train()
    if device is not None:
        batch = move_to_device(batch, device)

    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss_output = loss_fn(output, batch)
    loss_output.total.backward()

    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

    optimizer.step()
    scheduler_obj = _scheduler_object(scheduler)
    if scheduler_obj is not None:
        scheduler_obj.step()

    return StepResult(
        loss=loss_output.total.detach(),
        logs=loss_output.as_log_dict(prefix="train"),
        output=output,
    )


@torch.no_grad()
def validation_step(
    model: nn.Module,
    batch: Mapping[str, Any],
    loss_fn: nn.Module,
    *,
    device: torch.device | str | None = None,
) -> StepResult:
    """Run one validation step."""

    model.eval()
    if device is not None:
        batch = move_to_device(batch, device)

    output = model(batch)
    loss_output = loss_fn(output, batch)
    return StepResult(
        loss=loss_output.total.detach(),
        logs=loss_output.as_log_dict(prefix="val"),
        output=output,
    )


def average_logs(
    logs: list[dict[str, torch.Tensor]],
    *,
    weight_key: str | None = None,
) -> dict[str, torch.Tensor]:
    """Average scalar logs, optionally weighting variable-size batches."""

    if not logs:
        return {}
    keys = sorted({key for entry in logs for key in entry})
    averaged: dict[str, torch.Tensor] = {}
    for key in keys:
        entries = [entry for entry in logs if key in entry]
        values = [entry[key].detach().float() for entry in entries]
        if values:
            if weight_key is None or key == weight_key:
                averaged[key] = torch.stack(values).mean()
            else:
                weights = torch.stack(
                    [entry[weight_key].detach().float() for entry in entries]
                )
                averaged[key] = torch.sum(torch.stack(values) * weights) / weights.sum().clamp_min(1.0)
    return averaged
