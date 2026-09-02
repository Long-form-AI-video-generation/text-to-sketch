"""Run a real long-sequence forward/backward memory gate on CUDA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_project_to_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            sys.path.insert(0, str(parent))
            return parent
    raise RuntimeError("Could not find project root")


_add_project_to_path()

import torch

from builders import build_model
from dataloaders.masks import build_sequence_masks
from scripts.sketchformer.config import compose_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiment", default="anime_tok_dict_long")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-memory-gb", type=float, default=22.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the RTX 3090 memory gate")
    config = compose_training_config(args.config, experiment=args.experiment)
    max_length = int(config["model"]["input"]["max_seq_len"])
    if args.sequence_length > max_length:
        raise ValueError(
            f"sequence_length={args.sequence_length} exceeds model max_seq_len={max_length}"
        )

    device = torch.device("cuda")
    model = build_model(config["model"]).to(device).train()
    token_config = config["model"]["input"]["token_dictionary"]
    tokens = torch.randint(
        int(token_config["motion_token_offset"]),
        int(token_config["motion_token_offset"]) + int(token_config["codebook_size"]),
        (args.batch_size, args.sequence_length),
        dtype=torch.long,
        device=device,
    )
    tokens[:, 0] = int(token_config["sos_token_id"])
    tokens[:, -1] = int(token_config["eos_token_id"])
    batch = {
        "tokens": tokens,
        "targets": tokens.clone(),
        **build_sequence_masks(
            [args.sequence_length] * args.batch_size,
            max_length=args.sequence_length,
            device=device,
        ),
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(batch)
        if output.reconstruction is None:
            raise AssertionError("model did not produce reconstruction logits")
        loss = output.reconstruction.token_logits.float().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    report = {
        "device": torch.cuda.get_device_name(device),
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "peak_allocated_gb": allocated,
        "peak_reserved_gb": reserved,
        "limit_gb": args.max_memory_gb,
        "status": "pass" if allocated < args.max_memory_gb else "fail",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if allocated >= args.max_memory_gb:
        raise SystemExit(
            f"memory gate failed: allocated={allocated:.2f}GB limit={args.max_memory_gb:.2f}GB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
