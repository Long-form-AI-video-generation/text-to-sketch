"""Shared config loading for Sketchformer training scripts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch

from builders.config_utils import deep_merge, get_nested, load_yaml


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from a script location."""

    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        if (path / "pyproject.toml").exists() and (path / "configs").exists():
            return path
    raise RuntimeError("Could not find project root with pyproject.toml and configs/")


def compose_training_config(
    config_path: str | Path,
    *,
    experiment: str | None = None,
) -> dict[str, Any]:
    """Compose the lightweight Hydra-style config files used by this repo."""

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = find_project_root(Path(__file__)) / config_path
    config_dir = config_path.parent

    root_config = load_yaml(config_path)
    composed: dict[str, Any] = {}

    for entry in root_config.get("defaults", []):
        if entry == "_self_":
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"Unsupported defaults entry: {entry}")
        for group, name in entry.items():
            selected = experiment if group == "experiment" and experiment else name
            composed[group] = load_yaml(config_dir / group / f"{selected}.yaml")

    root_without_defaults = {
        key: value for key, value in root_config.items() if key != "defaults"
    }
    composed = deep_merge(composed, root_without_defaults)

    overrides = get_nested(composed, "experiment.overrides", {})
    for section, section_override in overrides.items():
        composed[section] = deep_merge(composed.get(section, {}), section_override)

    _sync_token_dictionary_config(composed)
    return composed


def _sync_token_dictionary_config(config: dict[str, Any]) -> None:
    """Keep tok-dict data/model vocabulary IDs from drifting apart."""

    if str(get_nested(config, "data.format.type", "stroke3")) not in {
        "tok_dict",
        "token",
        "tokens",
    }:
        return

    token_dictionary = get_nested(config, "data.format.token_dictionary", {})
    if not token_dictionary:
        return

    model = config.setdefault("model", {})
    model_input = model.setdefault("input", {})
    existing = model_input.get("token_dictionary", {})
    model_input["token_dictionary"] = deep_merge(existing, token_dictionary)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a user-requested device string."""

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_path(project_root: Path, path: str | Path) -> Path:
    """Resolve repo-relative paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def batch_limit(loader: Iterable[Any], limit: int | float | None) -> int:
    """Resolve a trainer limit value into an integer batch count."""

    loader_len = len(loader)  # type: ignore[arg-type]
    if limit is None:
        return loader_len
    if isinstance(limit, float):
        if limit <= 0:
            return 0
        if limit <= 1:
            return max(1, int(loader_len * limit))
        return min(loader_len, int(limit))
    return min(loader_len, int(limit))


def parse_batch_limit(value: str) -> int | float:
    """Parse CLI batch limits while preserving integer semantics."""

    if "." in value:
        return float(value)
    return int(value)


def limited(loader: Iterable[Any], limit: int | float | None) -> Iterator[Any]:
    """Yield at most ``limit`` batches from a loader."""

    max_batches = batch_limit(loader, limit)
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        yield batch


def format_logs(logs: dict[str, torch.Tensor | float], *, precision: int = 4) -> str:
    """Format scalar logs for terminal output."""

    parts = []
    for key, value in sorted(logs.items()):
        scalar = float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        parts.append(f"{key}={scalar:.{precision}f}")
    return " ".join(parts)
