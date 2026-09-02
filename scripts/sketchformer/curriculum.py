"""Length curriculum contracts for long-sequence Sketchformer fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    max_length: int
    epochs: int
    learning_rate: float
    trainable: str = "all"


def parse_curriculum(
    trainer_config: Mapping[str, Any],
    *,
    default_max_length: int,
) -> list[CurriculumStage]:
    """Parse configured stages or return one conventional all-model stage."""

    curriculum = trainer_config.get("curriculum", {})
    if not bool(curriculum.get("enabled", False)):
        training = trainer_config.get("training", {})
        return [
            CurriculumStage(
                name="full",
                max_length=int(default_max_length),
                epochs=int(training.get("max_epochs", 1)),
                learning_rate=float("nan"),
                trainable="all",
            )
        ]

    raw_stages = curriculum.get("stages", [])
    if not raw_stages:
        raise ValueError("trainer.curriculum.enabled requires at least one stage")
    stages = [
        CurriculumStage(
            name=str(stage.get("name", f"stage-{index + 1}")),
            max_length=int(stage["max_length"]),
            epochs=int(stage["epochs"]),
            learning_rate=float(stage["learning_rate"]),
            trainable=str(stage.get("trainable", "all")),
        )
        for index, stage in enumerate(raw_stages)
    ]
    previous_length = 0
    for stage in stages:
        if stage.max_length <= previous_length:
            raise ValueError("curriculum max_length values must increase strictly")
        if stage.max_length > default_max_length:
            raise ValueError("curriculum stage exceeds model/data max_length")
        if stage.epochs <= 0 or stage.learning_rate <= 0:
            raise ValueError("curriculum epochs and learning_rate must be positive")
        if stage.trainable not in {"expander", "decoder", "all"}:
            raise ValueError("curriculum trainable must be expander, decoder, or all")
        previous_length = stage.max_length
    if stages[-1].max_length != default_max_length:
        raise ValueError("final curriculum stage must reach configured max_length")
    return stages


def set_trainable_scope(model: torch.nn.Module, scope: str) -> int:
    """Freeze model parameters outside the configured curriculum scope."""

    if scope not in {"expander", "decoder", "all"}:
        raise ValueError("scope must be expander, decoder, or all")
    for parameter in model.parameters():
        parameter.requires_grad = scope == "all"
    if scope == "expander":
        _set_module_trainable(model.latent_expander)
    elif scope == "decoder":
        for module_name in (
            "target_embedding",
            "latent_expander",
            "decoder",
            "reconstruction_head",
        ):
            module = getattr(model, module_name, None)
            if module is not None:
                _set_module_trainable(module)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if count == 0:
        raise ValueError(f"curriculum scope {scope!r} selected no trainable parameters")
    return count


def resume_epoch_for_stage(
    stages: list[CurriculumStage],
    stage_index: int,
    completed_epochs: int,
) -> int:
    """Return the first local epoch to run for a resumed curriculum stage."""

    if stage_index < 0 or stage_index >= len(stages):
        raise IndexError("stage_index is outside the curriculum")
    before = sum(stage.epochs for stage in stages[:stage_index])
    return min(stages[stage_index].epochs, max(0, int(completed_epochs) - before))


def _set_module_trainable(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = True
