"""Inspect pretrained Sketchformer checkpoint assets."""

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
    raise RuntimeError("Could not find project root directory.")


PROJECT_ROOT = _add_project_to_path()

from models.sketchformer.pretrained import (
    DEFAULT_PRETRAINED_ROOT,
    format_validation_report,
    validate_pretrained_assets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_PRETRAINED_ROOT))
    parser.add_argument("--recommended-checkpoint", default="weights/ckpt-12")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if validation reports an error.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = validate_pretrained_assets(
        args.root,
        project_root=PROJECT_ROOT,
        recommended_checkpoint=args.recommended_checkpoint,
    )

    if args.json:
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_validation_report(manifest))

    return 1 if args.strict and not manifest.ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
