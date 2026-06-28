"""Sentiment-as-feature builder (Tier 1 Phase 2).

Loads sentiment time series from `cache/sentiment/` and aligns them to a
target H4 bar index for meta-labeler feature augmentation. Strict shift(1)
on each source's native frequency before reindexing — no look-ahead.

Sources currently supported:
  - alternative.me Fear & Greed Index (daily, crypto-relevant)
    Columns added: sent__fgi_value, sent__fgi_pct_chg_5d
  - (future) GDELT GKG daily tone aggregates
  - (future) FRED UMCSENT monthly consumer sentiment

Per-asset gating: FGI is meaningful only for crypto majors (BTC/ETH/SOL).
For non-crypto assets, the column is still emitted but as NaN (gradient-
boosted models handle NaN natively). MDA importance will surface dead
features post-run.

NO PRIMARY ENGINE — this module produces FEATURES only.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


FGI_CACHE_FILE = "fear_greed.parquet"


def load_fear_greed_index(cache_dir: str | Path) -> pd.DataFrame:
    """Load FGI daily parquet. Returns df with `fgi_value` column.

    Index is UTC-aware DatetimeIndex (daily timestamps).
    Raises FileNotFoundError if the parquet is missing — run
    `scripts/ingest_fear_greed.py` to populate the cache.
    """
    path = Path(cache_dir) / FGI_CACHE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"FGI cache not found at {path}. Run scripts/ingest_fear_greed.py."
        )
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"FGI parquet {path}: index must be DatetimeIndex")
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    return df


def build_sentiment_features(
    primary_index: pd.DatetimeIndex,
    cache_dir: str | Path,
    asset_class: str,
) -> pd.DataFrame:
    """Build sentiment features aligned to `primary_index` (target H4 bars).

    For each source: shift(1) on its NATIVE frequency BEFORE reindexing to
    H4 with ffill. This guarantees that an H4 bar at time T can only use
    sentiment values stamped at dates <= T-1 (no look-ahead).

    Args:
      primary_index: H4 bar index of the target asset.
      cache_dir: directory containing sentiment parquets (e.g. cache/sentiment).
      asset_class: 'fx', 'metal', or 'crypto'. FGI is gated to crypto only.

    Returns:
      DataFrame with same index as primary_index, columns prefixed `sent__`.
      Non-applicable columns are emitted as all-NaN (downstream models
      handle NaN; MDA will flag dead columns).
    """
    out = pd.DataFrame(index=primary_index)

    # FGI — crypto-relevant only
    try:
        fgi = load_fear_greed_index(cache_dir)
        # shift(1) on daily NATIVE frequency
        fgi_shifted = fgi.shift(1)
        fgi_h4 = fgi_shifted["fgi_value"].reindex(primary_index, method="ffill")
        # 5-day momentum (computed on daily, then ffilled — captures sentiment shifts)
        fgi_chg5d = fgi_shifted["fgi_value"].pct_change(periods=5).reindex(
            primary_index, method="ffill"
        )
    except FileNotFoundError:
        fgi_h4 = pd.Series(np.nan, index=primary_index)
        fgi_chg5d = pd.Series(np.nan, index=primary_index)

    # Gate by asset class: emit FGI only for crypto, NaN for others
    if asset_class == "crypto":
        out["sent__fgi_value"] = fgi_h4.values
        out["sent__fgi_pct_chg_5d"] = fgi_chg5d.values
    else:
        # Emit columns with NaN so the schema is consistent across assets
        out["sent__fgi_value"] = np.nan
        out["sent__fgi_pct_chg_5d"] = np.nan

    return out
