"""SPDR GLD daily holdings loader for phase5_gld_holdings primary (B0015a).

Reads cache/alt_data/gld_holdings.parquet (populated by scripts/ingest_gld_holdings.py)
and aligns the daily holdings series to the caller's target_index.

PIT discipline: SPDR publishes end-of-day after the trading day closes. A market
bar at calendar date t must see the holdings stamped t-1 or earlier — never t.
We enforce via .shift(1) on the daily series BEFORE the reindex+ffill.

This matches the pipeline/macro_fetch.py FRED .shift(1) convention.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CACHE_PATH = Path("cache/alt_data/gld_holdings.parquet")


class GldHoldingsCacheMissing(FileNotFoundError):
    """Raised when the parquet cache is absent. Run scripts/ingest_gld_holdings.py."""


def load_gld_holdings(
    target_index: pd.DatetimeIndex,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> pd.DataFrame:
    """Return GLD daily holdings aligned to target_index with PIT shift applied.

    Output columns: gld_oz_held (float, may be NaN before warm-up at GLD inception
    2004-11-18 or after the cache's last update).

    .shift(1) discipline:
        Holdings stamped at calendar date t in the cache (SPDR's official
        end-of-day value) are visible to market bars at date >= t+1 only.

    For target_index entries before GLD inception (2004-11-18) or in pre-cache
    gap regions, returns NaN (callers handle as no-fire).

    Raises GldHoldingsCacheMissing if the parquet is absent.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise GldHoldingsCacheMissing(
            f"GLD holdings cache not found at {cache_path}. "
            "Run: uv run python scripts/ingest_gld_holdings.py"
        )
    target_index = pd.DatetimeIndex(target_index)
    if len(target_index) == 0:
        return pd.DataFrame(columns=["gld_oz_held"], index=target_index)

    holdings = pd.read_parquet(cache_path)
    if not isinstance(holdings.index, pd.DatetimeIndex):
        holdings.index = pd.to_datetime(holdings.index, utc=True)
    # Defensive: cache should already be UTC-tz, sorted, dedup.
    if holdings.index.tz is None:
        holdings.index = holdings.index.tz_localize("UTC")
    holdings = holdings.sort_index()

    # PIT calendar shift: value stamped at date t becomes visible at date t+1.
    # Critical to use a CALENDAR-day shift (Timedelta(days=1)) not a row shift
    # (.shift(1)) — the cache has only trading days, so a row-shift would
    # leave weekend bars without coverage. Calendar shift + reindex+ffill
    # correctly propagates Friday's value through Saturday/Sunday/Monday
    # (where Monday's own stamp is also lag-shifted to Tuesday visibility).
    shifted = holdings.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)

    # Reindex with ffill: target bars between non-trading days hold the last
    # available calendar-shifted holdings value.
    aligned = shifted.reindex(target_index, method="ffill")
    aligned = aligned[["gld_oz_held"]]
    aligned.index = target_index
    return aligned
