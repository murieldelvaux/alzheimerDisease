"""Project configuration utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_DIR = PROJECT_ROOT / "venv_alzheimer" / "share"/ "dataset"
DEFAULT_DATA_FILENAME = "ADNIMERGE_11Nov2025.csv"
DEFAULT_MERGE_CSV = DEFAULT_DATA_DIR / DEFAULT_DATA_FILENAME
MERGE_CSV_ENV_VAR = "ALZHEIMER_MERGE_CSV"


def _normalise_path(path_like: os.PathLike[str] | str) -> Path:
    """Return an absolute path located from the project root."""
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def get_merge_csv_path(cli_override: Optional[str] = None) -> Path:
    """Resolve the path to the ADNIMERGE merge table.

    The resolution order is:
    1. explicit CLI argument (``cli_override``)
    2. environment variable ``ALZHEIMER_MERGE_CSV``
    3. default ``data/ADNIMERGE_11Nov2025.csv`` relative to the project root.
    """

    if cli_override:
        return _normalise_path(cli_override)

    env_override = os.getenv(MERGE_CSV_ENV_VAR)
    if env_override:
        return _normalise_path(env_override)

    return DEFAULT_MERGE_CSV
