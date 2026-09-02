"""Export native Sketchformer weights from a training checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root directory.")


PROJECT_ROOT = _add_project_to_path()

import torch

from builders import build_model
from core import load_checkpoint
from scripts.sketchformer.config import compose_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = compose_training_config(args.config, experiment=args.experiment)
    model = build_model(config["model"])
    load_checkpoint(PROJECT_ROOT / args.checkpoint, model, strict=False)

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()

    if output_path.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file(state_dict, str(output_path))
    else:
        torch.save({"state_dict": state_dict, "config": config}, output_path)

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
