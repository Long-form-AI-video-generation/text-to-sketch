"""Shared project paths.

Keeping path discovery in one place prevents every script from guessing where
``data/`` lives.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DEFAULT_RAW_IMAGE_DIR = RAW_DATA_DIR / "portraits"
DEFAULT_SKETCHES_DIR = PROCESSED_DATA_DIR / "sketches"
DEFAULT_FILTERED_SKETCHES_DIR = PROCESSED_DATA_DIR / "sketches_filtered"
DEFAULT_STROKE5_DIR = PROCESSED_DATA_DIR / "stroke5"
DEFAULT_SKETCH_TOKEN_DIR = PROCESSED_DATA_DIR / "sketch_token"
DEFAULT_TOKENS_DIR = PROCESSED_DATA_DIR / "tokens"
DEFAULT_EVALUATIONS_DIR = PROCESSED_DATA_DIR / "evaluations"


def project_path(path: str | Path) -> Path:
    """Resolve relative paths from the repository root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path
