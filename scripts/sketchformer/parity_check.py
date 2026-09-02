"""Verify that the 4096-position extension preserves 200-token outputs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root")


PROJECT_ROOT = _add_project_to_path()

import numpy as np
import torch
import torch.nn.functional as F

from builders import build_model
from dataloaders.masks import build_sequence_masks
from scripts.sketchformer.config import compose_training_config, resolve_device
from scripts.sketchformer.train import _load_state_dict_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiment", default="anime_tok_dict_long")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-npz", default=None)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--min-argmax-agreement", type=float, default=0.99)
    return parser.parse_args()


def parity_metrics(
    candidate_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> tuple[float, float]:
    """Return flattened logit cosine similarity and token argmax agreement."""

    if candidate_logits.shape != reference_logits.shape:
        raise ValueError(
            f"Logit shapes differ: {tuple(candidate_logits.shape)} != "
            f"{tuple(reference_logits.shape)}"
        )
    candidate = candidate_logits.detach().float().reshape(1, -1)
    reference = reference_logits.detach().float().reshape(1, -1)
    cosine = float(F.cosine_similarity(candidate, reference).item())
    agreement = float(
        (candidate_logits.argmax(dim=-1) == reference_logits.argmax(dim=-1))
        .float()
        .mean()
        .item()
    )
    return cosine, agreement


def _random_tokens(
    model_config: dict,
    *,
    samples: int,
    sequence_length: int,
    seed: int,
) -> torch.Tensor:
    token_config = model_config["input"]["token_dictionary"]
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        int(token_config["motion_token_offset"]),
        int(token_config["motion_token_offset"]) + int(token_config["codebook_size"]),
        (samples, sequence_length),
        generator=generator,
    )
    tokens[:, 0] = int(token_config["sos_token_id"])
    tokens[:, -1] = int(token_config["eos_token_id"])
    return tokens


def _reference_model_config(model_config: dict, sequence_length: int) -> dict:
    reference = copy.deepcopy(model_config)
    reference["input"]["max_seq_len"] = sequence_length
    reference["architecture"]["latent_expander_base_length"] = sequence_length
    reference["embedding"]["positional_encoding"]["max_length"] = sequence_length
    return reference


def _token_logits(model, tokens: torch.Tensor) -> torch.Tensor:
    batch = {
        "tokens": tokens,
        "targets": tokens.clone(),
        **build_sequence_masks(
            [tokens.shape[1]] * tokens.shape[0],
            max_length=tokens.shape[1],
            device=tokens.device,
        ),
    }
    output = model(batch)
    if output.reconstruction is None or output.reconstruction.token_logits is None:
        raise AssertionError("tok-dict model did not return token logits")
    return output.reconstruction.token_logits


def main() -> int:
    args = parse_args()
    if args.sequence_length != 200:
        raise ValueError("Checkpoint parity must be measured at the released length 200")
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    config = compose_training_config(args.config, experiment=args.experiment)
    model_config = config["model"]
    base_length = int(model_config["architecture"]["latent_expander_base_length"])
    if base_length != args.sequence_length:
        raise ValueError(
            f"Configured latent expander base length is {base_length}, expected 200"
        )
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    state = _load_state_dict_file(checkpoint_path)

    candidate_model = build_model(model_config)
    candidate_model.load_state_dict(state, strict=True)
    candidate_model = candidate_model.to(device).eval()

    if args.reference_npz:
        reference_path = Path(args.reference_npz)
        if not reference_path.is_absolute():
            reference_path = PROJECT_ROOT / reference_path
        payload = np.load(reference_path)
        tokens = torch.as_tensor(payload["tokens"], dtype=torch.long)
        reference_logits = torch.as_tensor(payload["logits"], dtype=torch.float32)
        reference_kind = "external_npz"
    else:
        tokens = _random_tokens(
            model_config,
            samples=args.samples,
            sequence_length=args.sequence_length,
            seed=args.seed,
        )
        reference_model = build_model(
            _reference_model_config(model_config, args.sequence_length)
        )
        reference_state = {
            key: value for key, value in state.items() if key in reference_model.state_dict()
        }
        reference_model.load_state_dict(reference_state, strict=True)
        reference_model = reference_model.to(device).eval()
        with torch.no_grad():
            reference_logits = _token_logits(reference_model, tokens.to(device)).cpu()
        reference_kind = "native_exact_200"

    if tokens.ndim != 2 or tokens.shape[1] != args.sequence_length:
        raise ValueError(
            f"Reference tokens must have shape (batch, 200), got {tuple(tokens.shape)}"
        )
    with torch.no_grad():
        candidate_logits = _token_logits(candidate_model, tokens.to(device)).cpu()
    cosine, agreement = parity_metrics(candidate_logits, reference_logits.cpu())
    passed = cosine >= args.min_cosine and agreement >= args.min_argmax_agreement
    report = {
        "checkpoint": str(checkpoint_path),
        "reference": reference_kind,
        "sequence_length": args.sequence_length,
        "samples": int(tokens.shape[0]),
        "logit_cosine_similarity": cosine,
        "argmax_agreement": agreement,
        "min_cosine": args.min_cosine,
        "min_argmax_agreement": args.min_argmax_agreement,
        "status": "pass" if passed else "fail",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("200-token checkpoint parity gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
