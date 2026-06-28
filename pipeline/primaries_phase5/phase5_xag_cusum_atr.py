"""Phase 5 custom primary for Loop A proposal 20260529-XAGUSD-D1-BULL_STR-B004v3.

Pseudocode (verbatim from proposal):
  LONG (+1) when ALL hold:
    (1) CUSUM positive-side event fires (threshold = 1.0 * ATR, 42-bar ATR)
    (2) real_yield_5y_z252d <= 40th percentile over trailing 84-bar window
    (3) volume >= 50th percentile over trailing 84-bar window
  No short side.

Inputs read: real_yield_5y_z252d (from features frame, FRED .shift(1) applied upstream).
INPUT_COLUMNS is disjoint from the meta-labeler feature set (real_yield read here as
a gate condition; it remains available to the meta as a conditioning feature).

Causal-window discipline:
  ATR lookback = 42 bars prior to t (shift(1) convention).
  Percentile ranks use shift(1).rolling(84) — only bars strictly before t.
  CUSUM via cusum_filter_signal which is strict-causal by construction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pipeline.labels import cusum_filter_signal

INPUT_COLUMNS: tuple[str, ...] = ("real_yield_5y_z252d",)

ATR_LOOKBACK = 42
CUSUM_ATR_MULT = 1.0
YIELD_WINDOW = 84
YIELD_QUANTILE_CEIL = 0.40
VOLUME_WINDOW = 84
VOLUME_QUANTILE_FLOOR = 0.50


def _atr(ohlcv: pd.DataFrame, window: int) -> pd.Series:
    high, low, close = (ohlcv[c].astype(float) for c in ("high", "low", "close"))
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.shift(1).rolling(window, min_periods=window).mean()


def _rolling_rank(s: pd.Series, window: int) -> pd.Series:
    """Rank of s[t] within the trailing `window` bars before t (causal)."""
    shifted = s.shift(1)
    def _rank(x):
        cur = x.iloc[-1]
        prior = x.iloc[:-1].dropna()
        if len(prior) < window // 2:
            return np.nan
        return float((prior <= cur).sum()) / len(prior)
    return shifted.rolling(window + 1, min_periods=window // 2 + 1).apply(_rank, raw=False)


def signal(ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: dict) -> pd.Series:
    """Long on CUSUM event when real-yield low AND volume sufficient."""
    atr = _atr(ohlcv, ATR_LOOKBACK)
    cusum_sig = cusum_filter_signal(
        close=ohlcv["close"].astype(float),
        atr=atr,
        threshold_atr=CUSUM_ATR_MULT,
    )
    cusum_up = (cusum_sig == 1.0)

    # Real-yield gate: <= 40th pct of trailing 84-bar window
    if "real_yield_5y_z252d" in features.columns:
        ry = features["real_yield_5y_z252d"].reindex(ohlcv.index).astype(float)
    else:
        ry = pd.Series(np.nan, index=ohlcv.index)
    ry_rank = _rolling_rank(ry, YIELD_WINDOW)
    yield_low = ry_rank <= YIELD_QUANTILE_CEIL

    # Volume gate: >= 50th pct of trailing 84-bar window
    vol = ohlcv["volume"].astype(float)
    vol_rank = _rolling_rank(vol, VOLUME_WINDOW)
    vol_ok = vol_rank >= VOLUME_QUANTILE_FLOOR

    out = pd.Series(0, index=ohlcv.index, dtype="int8")
    out[cusum_up & yield_low & vol_ok] = 1
    return out
