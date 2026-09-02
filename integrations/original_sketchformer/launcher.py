"""Launcher for fine-tuning with the original Sketchformer codebase.

This module intentionally does not import or reimplement the Sketchformer model.
It only prepares Docker commands that run the original code under ``sketchformer/``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = "sketchformer-tf2-cpu"
DEFAULT_DOCKERFILE = "integrations/original_sketchformer/docker/Dockerfile.cpu"
DEFAULT_GPU_IMAGE = "sketchformer-tf2-gpu"
DEFAULT_GPU_DOCKERFILE = "integrations/original_sketchformer/docker/Dockerfile.gpu"
DEFAULT_DATASET = "data/processed/sketchformer-ready-data/stroke3"
DEFAULT_SOURCE_STROKE5 = "data/processed/stroke5"
DEFAULT_PRETRAINED_OUTPUT = "weights/pretrained"
DEFAULT_FINETUNE_OUTPUT = "weights/finetuned"
DEFAULT_CONTINUOUS_RESUME = (
    "weights/pretrained/"
    "sketch-transformer-tf2-cvpr_tform_cont/weights/ckpt-12"
)
DEFAULT_LEGACY_MAX_SEQ_LEN = 200


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def container_path(path: str) -> str:
    if path in {"latest", "none", "None", ""}:
        return path
    if path.startswith("/workspace/"):
        return path

    host_path = project_path(path).resolve()
    try:
        relative = host_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise SystemExit(
            "Sketchformer codebase fine-tuning only supports paths inside "
            "the project root: "
            "{}".format(path)
        ) from exc
    return "/workspace/{}".format(relative.as_posix())


def docker_base(args: argparse.Namespace) -> list[str]:
    command = []
    if args.sudo:
        command.append("sudo")
    command.append(args.docker_bin)
    return command


def docker_run(args: argparse.Namespace, workdir: str) -> list[str]:
    command = docker_base(args) + [
        "run",
        "--rm",
        "-v",
        "{}:/workspace".format(PROJECT_ROOT),
        "-w",
        workdir,
    ]
    if args.gpus:
        command += ["--gpus", args.gpus]
    command.append(args.image)
    return command


def run_or_print(command: list[str], dry_run: bool) -> int:
    printable = shlex.join(command)
    if dry_run:
        print(printable)
        return 0
    print("[Sketchformer codebase fine-tuning] {}".format(printable))
    return subprocess.call(command)


def compose_legacy_data_hparams(args: argparse.Namespace) -> str:
    if args.data_hparams:
        return args.data_hparams
    return "use_continuous_data=True,max_seq_len={}".format(args.max_seq_len)


def maybe_warn_sequence_checkpoint_mismatch(args: argparse.Namespace) -> None:
    if args.dry_run:
        return
    if args.resume in {"none", "None", ""}:
        return
    if args.data_hparams:
        return
    if int(args.max_seq_len) != DEFAULT_LEGACY_MAX_SEQ_LEN:
        print(
            "[warning] The original continuous Sketchformer checkpoint was "
            "trained with max_seq_len=200. Changing --max-seq-len while "
            "resuming that checkpoint may fail because TensorFlow layer "
            "shapes are sequence-length dependent."
        )


def build_image(args: argparse.Namespace) -> int:
    command = docker_base(args) + [
        "build",
        "-t",
        args.image,
        "-f",
        str(project_path(args.dockerfile)),
        str(PROJECT_ROOT / "integrations" / "original_sketchformer" / "docker"),
    ]
    return run_or_print(command, args.dry_run)


def prepare_data(args: argparse.Namespace) -> int:
    if not args.dry_run:
        project_path(args.target_dir).mkdir(parents=True, exist_ok=True)
    command = docker_run(args, "/workspace") + [
        "python",
        "scripts/prepare_data/prepare_sketchformer_legacy.py",
        "--source-dir",
        container_path(args.source_dir),
        "--target-dir",
        container_path(args.target_dir),
        "--n-chunks",
        str(args.n_chunks),
        "--seed",
        str(args.seed),
        "--train-frac",
        str(args.train_frac),
        "--valid-frac",
        str(args.valid_frac),
        "--test-frac",
        str(args.test_frac),
        "--n-classes",
        str(args.n_classes),
        "--min-valid-size",
        str(args.min_valid_size),
    ]
    return run_or_print(command, args.dry_run)


def evaluate_reconstruction(args: argparse.Namespace) -> int:
    command = docker_run(args, "/workspace/sketchformer") + [
        "python",
        "evaluate-metrics.py",
        "sketch-transformer-tf2",
        "--dataset",
        container_path(args.dataset),
        "-o",
        container_path(args.output_dir),
        "--id",
        args.model_id,
        "--gpu",
        str(args.gpu),
        "--resume",
        container_path(args.resume),
        "--hparams",
        args.hparams,
        "--metrics",
    ] + args.metrics
    return run_or_print(command, args.dry_run)


def finetune_continuous(args: argparse.Namespace) -> int:
    if not args.dry_run:
        project_path(args.output_dir).mkdir(parents=True, exist_ok=True)
    maybe_warn_sequence_checkpoint_mismatch(args)
    command = docker_run(args, "/workspace/sketchformer") + [
        "python",
        "train.py",
        "sketch-transformer-tf2",
        "--dataset",
        container_path(args.dataset),
        "-o",
        container_path(args.output_dir),
        "--id",
        args.run_id,
        "--gpu",
        str(args.gpu),
        "--data-hparams",
        compose_legacy_data_hparams(args),
        "--base-hparams",
        args.base_hparams,
        "--hparams",
        args.model_hparams,
    ]
    if args.resume not in {"none", "None", ""}:
        command += ["--resume", container_path(args.resume)]
    return run_or_print(command, args.dry_run)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", default=DEFAULT_IMAGE,
                        help="Docker image name used for original Sketchformer.")
    parser.add_argument("--dockerfile", default=DEFAULT_DOCKERFILE,
                        help="Dockerfile used by build-image.")
    parser.add_argument("--docker-bin", default="docker",
                        help="Docker executable to call.")
    parser.add_argument("--gpus", default=None,
                        help="Optional Docker --gpus value, for example 'all'.")
    parser.add_argument("--sudo", action="store_true",
                        help="Prefix Docker commands with sudo.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the Docker command without running it.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launcher for Sketchformer codebase fine-tuning: "
            "Text-to-Sketch prepares data, and the original Sketchformer "
            "checkout trains/evaluates inside Docker."
        )
    )
    add_common(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-image",
                                  help="Build the CPU TensorFlow 2.1 Docker image.")
    build.set_defaults(func=build_image)

    build_gpu = subparsers.add_parser(
        "build-gpu-image",
        help="Build the TensorFlow GPU image intended for RTX 3090 servers.")
    build_gpu.set_defaults(
        func=build_image,
        image=DEFAULT_GPU_IMAGE,
        dockerfile=DEFAULT_GPU_DOCKERFILE,
    )

    prep = subparsers.add_parser(
        "prepare-data",
        help="Convert stroke5 data to legacy Sketchformer stroke3 chunks.")
    prep.add_argument("--source-dir", default=DEFAULT_SOURCE_STROKE5)
    prep.add_argument("--target-dir", default=DEFAULT_DATASET)
    prep.add_argument("--n-chunks", type=int, default=10)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--train-frac", type=float, default=0.8)
    prep.add_argument("--valid-frac", type=float, default=0.1)
    prep.add_argument("--test-frac", type=float, default=0.1)
    prep.add_argument("--n-classes", type=int, default=345)
    prep.add_argument("--min-valid-size", type=int, default=0)
    prep.set_defaults(func=prepare_data)

    eval_parser = subparsers.add_parser(
        "evaluate-reconstruction",
        help="Run original Sketchformer reconstruction evaluation.")
    eval_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    eval_parser.add_argument("--output-dir", default=DEFAULT_PRETRAINED_OUTPUT)
    eval_parser.add_argument("--model-id", default="cvpr_tform_cont")
    eval_parser.add_argument("--resume", default=DEFAULT_CONTINUOUS_RESUME)
    eval_parser.add_argument("--gpu", default=0, type=int)
    eval_parser.add_argument("--hparams", default="batch_size=18,slack_config=")
    eval_parser.add_argument("--metrics", nargs="+",
                             default=["sketch-reconstruction"])
    eval_parser.set_defaults(func=evaluate_reconstruction)

    finetune = subparsers.add_parser(
        "finetune-continuous",
        help="Fine-tune the original continuous Sketchformer checkpoint.")
    finetune.add_argument("--dataset", default=DEFAULT_DATASET)
    finetune.add_argument("--output-dir", default=DEFAULT_FINETUNE_OUTPUT)
    finetune.add_argument("--run-id", default="anime-continuous-finetune")
    finetune.add_argument("--resume", default=DEFAULT_CONTINUOUS_RESUME)
    finetune.add_argument("--gpu", default=0, type=int)
    finetune.add_argument(
        "--max-seq-len",
        default=DEFAULT_LEGACY_MAX_SEQ_LEN,
        type=int,
        help=(
            "Legacy dataloader sequence length. Keep 200 when resuming the "
            "released continuous checkpoint; larger values are for from-scratch "
            "legacy experiments."
        ),
    )
    finetune.add_argument(
        "--data-hparams",
        default=None)
    finetune.add_argument(
        "--base-hparams",
        default=(
            "batch_size=8,num_epochs=1,save_every=1.0,safety_save=1.0,"
            "log_every=10,notify_every=100000,slack_config="
        ))
    finetune.add_argument(
        "--model-hparams",
        default="class_weight=0.0",
        help=(
            "Original Sketchformer --hparams. The default keeps the "
            "classifier layer for checkpoint compatibility but removes "
            "classification pressure for unlabeled anime data."
        ))
    finetune.set_defaults(func=finetune_continuous)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
