"""Load and validate OHLCV CSV input."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


class DataValidationError(ValueError):
    """Raised when input CSV violates the data contract."""


REQUIRED_COLS = {"open", "high", "low", "close", "volume"}
TIME_ALIASES = ("time", "timestamps")


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load an OHLCV CSV and return a DatetimeIndex-keyed frame.

    Contract:
      - One column out of {"time", "timestamps"} as the timestamp.
      - Columns open, high, low, close, volume (float64).
      - Strictly monotonic increasing time, no duplicates.
      - All ISO 8601 UTC; tz-aware index returned.
    """
    path = Path(path)
    df = pd.read_csv(path)

    time_col = next((c for c in TIME_ALIASES if c in df.columns), None)
    if time_col is None:
        raise DataValidationError(f"No time column. Need one of {TIME_ALIASES}, got {list(df.columns)}")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required column(s): {sorted(missing)}")

    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    if not df[time_col].is_monotonic_increasing:
        raise DataValidationError("time/timestamps column is not monotonic increasing")
    if df[time_col].duplicated().any():
        raise DataValidationError("Duplicate timestamps detected")

    df = df.set_index(time_col)[["open", "high", "low", "close", "volume"]]
    df = df.astype("float64")
    return df
