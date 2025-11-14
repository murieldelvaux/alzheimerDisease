"""Data loading helpers for the Alzheimer disease merge table."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

STRING_COLUMNS = ("DX", "DX_bl", "VISCODE")


def _trim_column_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove surrounding whitespace from the dataframe column labels."""
    frame.columns = frame.columns.str.strip()
    return frame


def _strip_string_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Trim whitespace from selected string columns when present."""
    for column in columns:
        if column in frame.columns:
            frame[column] = frame[column].astype(str).str.strip()
    return frame


def _coerce_month_to_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure the ``Month`` column is a numeric dtype when present."""
    if "Month" in frame.columns:
        frame["Month"] = pd.to_numeric(frame["Month"], errors="coerce")
    return frame


def load_merge_table(path: str | Path) -> pd.DataFrame:
    """Load the ADNIMERGE table applying a light cleaning pass."""
    csv_path = Path(path).expanduser()

    if not csv_path.is_file():
        raise FileNotFoundError(f"Merge table not found at '{csv_path}'")

    frame = pd.read_csv(csv_path, low_memory=False)
    frame = _trim_column_labels(frame)
    frame = _strip_string_columns(frame, STRING_COLUMNS)
    frame = _coerce_month_to_numeric(frame)
    return frame


__all__ = ["load_merge_table"]
