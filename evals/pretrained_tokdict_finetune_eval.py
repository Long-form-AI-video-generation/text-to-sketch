"""Readiness eval for released tok-dict checkpoint fine-tuning."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root directory.")


PROJECT_ROOT = _add_project_to_path()

from builders import build_model
from dataloaders.masks import build_sequence_masks
from scripts.sketchformer.config import compose_training_config


def main() -> int:
    config = compose_training_config(
        "configs/train.yaml",
        experiment="anime_tok_dict_finetune",
    )
    token_config = config["model"]["input"]["token_dictionary"]
    expected = {
        "pad_token_id": 0,
        "motion_token_offset": 1,
        "sep_token_id": 1001,
        "sos_token_id": 1002,
        "eos_token_id": 1003,
        "vocab_size": 1004,
    }
    for key, value in expected.items():
        if int(token_config[key]) != value:
            raise AssertionError(f"{key}={token_config[key]} != {value}")

    codebook_path = PROJECT_ROOT / config["data"]["format"]["token_dictionary"]["codebook_path"]
    if not codebook_path.exists():
        raise AssertionError(
            "Missing released codebook export. Run: python -m "
            "prep_data.sketch_token.create_token_dict --source-token-dict-pkl "
            "sketchformer/prep_data/sketch_token/token_dict.pkl --output-dir "
            "data/processed/sketch_token"
        )
    codebook = np.load(codebook_path)
    if codebook.shape != (1000, 2):
        raise AssertionError(f"Expected codebook shape (1000, 2), got {codebook.shape}")

    model = build_model(config["model"])
    model.eval()
    tokens = torch.tensor([[1002, 1, 1001, 1003, 0, 0]], dtype=torch.long)
    masks = build_sequence_masks([4], max_length=6)
    with torch.no_grad():
        output = model({"tokens": tokens, "targets": tokens.clone(), **masks})
    if output.reconstruction is None:
        raise AssertionError("Missing reconstruction output")
    if output.reconstruction.token_logits.shape != (1, 5, 1004):
        raise AssertionError(
            f"Unexpected token logits shape: {output.reconstruction.token_logits.shape}"
        )
    print("pretrained_tokdict_finetune_eval=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
