"""Compatibility wrapper for ``prep_data.extract_sketches``."""

from __future__ import annotations

import sys
from pathlib import Path


def _add_project_to_path() -> None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pipeline").exists() and (parent / "prep_data").exists():
            sys.path.insert(0, str(parent))
            return
    raise RuntimeError("Could not find project root directory.")


_add_project_to_path()

from prep_data.extract_sketches import main


if __name__ == "__main__":
    main()
