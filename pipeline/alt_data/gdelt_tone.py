"""GDELT daily tone loader for phase5_gdelt_tone primary (B0015c).

Reads cache/alt_data/gdelt_tone_{theme}.parquet (populated by
scripts/ingest_gdelt_tone.py) and aligns the daily tone series to the
caller's target_index.

PIT discipline: GDELT V2 publishes near-realtime (15-min lag). The daily
aggregated tone value stamped at calendar date t was generated from all
articles posted between t-00:00 UTC and t-23:59 UTC. A market bar at
calendar date t (UTC midnight = start of day) sees only stamps with date
strictly < t. We enforce this via calendar-day shift (same convention as
pipeline/alt_data/gld_holdings.py).

Coverage: ECON_INFLATION theme starts ~2017 in GDELT V2. Pre-2017 bars
return NaN.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_THEME = "ECON_INFLATION"


class GdeltToneCacheMissing(FileNotFoundError):
    """Raised when the parquet cache is absent. Run scripts/ingest_gdelt_tone.py."""


def _default_cache_path(theme: str) -> Path:
    return Path(f"cache/alt_data/gdelt_tone_{theme.lower()}.parquet")


def load_gdelt_tone(
    target_index: pd.DatetimeIndex,
    theme: str = DEFAULT_THEME,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return GDELT daily tone aligned to target_index with PIT shift applied.

    Output columns: tone (float, daily average tone; may be NaN before
    ~2017 or in gap regions).

    .shift(1) calendar-day discipline:
        Tone stamped at calendar date t in the cache is visible only at
        market bars with date >= t+1.

    For target_index entries before GDELT V2/theme coverage, returns NaN
    (callers handle as no-fire).

    Raises GdeltToneCacheMissing if the parquet is absent.
    """
    cache_path = Path(cache_path) if cache_path else _default_cache_path(theme)
    if not cache_path.exists():
        raise GdeltToneCacheMissing(
            f"GDELT tone cache not found at {cache_path}. "
            f"Run: uv run python scripts/ingest_gdelt_tone.py --theme {theme}"
        )
    target_index = pd.DatetimeIndex(target_index)
    if len(target_index) == 0:
        return pd.DataFrame(columns=["tone"], index=target_index)

    tone = pd.read_parquet(cache_path)
    if not isinstance(tone.index, pd.DatetimeIndex):
        tone.index = pd.to_datetime(tone.index, utc=True)
    if tone.index.tz is None:
        tone.index = tone.index.tz_localize("UTC")
    tone = tone.sort_index()

    # Calendar-day PIT shift (same semantic as pipeline/alt_data/gld_holdings.py).
    shifted = tone.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)

    aligned = shifted.reindex(target_index, method="ffill")
    aligned = aligned[["tone"]]
    aligned.index = target_index
    return aligned
