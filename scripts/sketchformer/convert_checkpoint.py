"""Convert released TensorFlow Sketchformer checkpoints to native weights."""

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

from builders import build_model
from models.sketchformer.checkpoint_mapping import convert_tok_dict_tensorflow_state
from models.sketchformer.pretrained import inspect_tensorflow_checkpoint
from scripts.sketchformer.config import compose_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-format", choices=["tensorflow", "torch"], default="tensorflow")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_tensorflow_checkpoint(prefix: Path) -> dict[str, object]:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required to read the released checkpoint. Install a "
            "TensorFlow build in this environment or run conversion in the original "
            "Sketchformer environment."
        ) from exc

    reader = tf.train.load_checkpoint(str(prefix))
    return {
        key: reader.get_tensor(key)
        for key in reader.get_variable_to_shape_map()
    }


def main() -> int:
    args = parse_args()
    config = compose_training_config(args.config, experiment=args.experiment)
    source = PROJECT_ROOT / args.source
    output = PROJECT_ROOT / args.output
    checkpoint = inspect_tensorflow_checkpoint(source)

    print(f"model={config['model']['name']}")
    print(f"source={source}")
    print(f"output={output}")
    print(f"source_format={args.source_format}")
    if args.source_format == "tensorflow":
        print(f"tensorflow_index={checkpoint.index_file}")
        print(f"tensorflow_data_shards={len(checkpoint.data_files)}")
        print(f"tensorflow_checkpoint_complete={checkpoint.exists}")

    if args.dry_run:
        return 0

    if args.source_format == "tensorflow":
        if not checkpoint.exists:
            raise SystemExit(f"Incomplete TensorFlow checkpoint source: {checkpoint.prefix}")
        tf_state = _load_tensorflow_checkpoint(checkpoint.prefix)
        model = build_model(config["model"])
        converted, report = convert_tok_dict_tensorflow_state(
            tf_state,
            model.state_dict(),
            target_seq_len=int(config["model"]["input"]["max_seq_len"]),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix == ".safetensors":
            from safetensors.torch import save_file

            save_file(converted, str(output))
        else:
            import torch

            torch.save({"state_dict": converted, "conversion_report": report.__dict__}, output)
        print(f"converted_keys={len(report.converted_keys)}")
        print(f"skipped_keys={len(report.skipped_keys)}")
        print(f"missing_target_keys={len(report.missing_target_keys)}")
        print(f"initialized_long_sequence_keys={len(report.initialized_keys)}")
        return 0

    raise SystemExit(
        "Torch-to-native conversion is not needed yet; use scripts/sketchformer/export.py "
        "for PyTorch checkpoints."
    )


if __name__ == "__main__":
    raise SystemExit(main())
