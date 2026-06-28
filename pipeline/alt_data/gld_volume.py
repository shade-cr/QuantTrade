"""GLD real-volume features for XAU bars (B0147).

Reads cache/alt_data/gld_volume.parquet (populated by
scripts/ingest_gld_volume.py) and emits two volatility/regime liquidity
features computed on GLD's OWN trading-day series, then aligned to the
caller's target index with the calendar-day PIT shift:

  * gld_dvol_z42    — z-score (trailing 42 trading days) of log dollar volume
                      (close x shares). Participation surges in the
                      exchange-traded gold vehicle, on REAL volume — the
                      dimension the CFD's tick count cannot see.
  * gld_amihud_z252 — z-score (trailing 252d) of the 21d trailing mean Amihud
                      illiquidity |log ret| / dollar_volume. Legitimate here
                      (real volume), unlike the CFD tick-count version
                      (B0136/B0141 caveats). Higher = gold liquidity drying up.

Expert verdict baked in (B0147 history, 2026-06-03): these are VOLATILITY /
REGIME features, NOT directional alpha — OFI directional content is sub-second
and gone by D1; keep only if MDA/CFI shows orthogonality to rv_regime.

PIT discipline: GLD daily volume is final after the US close. A market bar at
calendar date t may only see GLD values stamped <= t-1: the features are
computed on GLD's native series FIRST (trailing windows), then the stamped
index is shifted +1 CALENDAR day and reindex+ffill'd onto the target index —
identical to pipeline/alt_data/gld_holdings.py.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CACHE_PATH = Path("cache/alt_data/gld_volume.parquet")

GLD_VOLUME_FEATURES = ("gld_dvol_z42", "gld_amihud_z252")


class GldVolumeCacheMissing(FileNotFoundError):
    """Raised when the parquet cache is absent. Run scripts/ingest_gld_volume.py."""


def _zscore(s: pd.Series, window: int) -> pd.Series:
    return (s - s.rolling(window).mean()) / s.rolling(window).std()


def load_gld_volume_features(
    target_index: pd.DatetimeIndex,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> pd.DataFrame:
    """Return the GLD volume feature block aligned to target_index, PIT-shifted.

    NaN before GLD inception (2004-11-18) + warmup, and beyond the cache's
    last stamp +1 day; callers' dropna handles warmup like every other
    rolling feature.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise GldVolumeCacheMissing(
            f"GLD volume cache not found at {cache_path}. "
            "Run: uv run python scripts/ingest_gld_volume.py"
        )
    target_index = pd.DatetimeIndex(target_index)
    if len(target_index) == 0:
        return pd.DataFrame(columns=list(GLD_VOLUME_FEATURES), index=target_index)

    gld = pd.read_parquet(cache_path)
    if not isinstance(gld.index, pd.DatetimeIndex):
        gld.index = pd.to_datetime(gld.index, utc=True)
    if gld.index.tz is None:
        gld.index = gld.index.tz_localize("UTC")
    gld = gld.sort_index()

    # Features on GLD's native trading-day series (trailing windows only).
    dollar_vol = gld["gld_close"] * gld["gld_volume"]
    feats = pd.DataFrame(index=gld.index)
    feats["gld_dvol_z42"] = _zscore(np.log(dollar_vol), 42)
    amihud_21 = (np.log(gld["gld_close"]).diff().abs() / dollar_vol).rolling(21).mean()
    feats["gld_amihud_z252"] = _zscore(amihud_21, 252)
    feats = feats.replace([np.inf, -np.inf], np.nan)

    # PIT calendar shift (+1 day), then reindex+ffill: Friday's stamp covers
    # the weekend; Monday's own stamp becomes visible Tuesday.
    shifted = feats.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    aligned = shifted.reindex(target_index, method="ffill")
    aligned.index = target_index
    return aligned[list(GLD_VOLUME_FEATURES)]
