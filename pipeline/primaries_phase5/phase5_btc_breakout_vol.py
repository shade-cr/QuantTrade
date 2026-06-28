"""Phase 5 custom primary for Loop A proposal 20260529-BTCUSD-H4-BEAR_QUI-B001v1.

Pseudocode (verbatim from proposal):
  SHORT signal (-1) when:
    (1) close < rolling_low  (fresh 21-bar low breakdown)
    (2) volume < 0.7 * vol_ma  (participation contracted, no absorption)
  No long signal: 0 otherwise.
  All thresholds relative (rolling low, volume vs trailing mean).

Causal-window discipline:
  rolling_low uses shift(1).rolling(21).min() — trailing 21 bars BEFORE bar t.
  vol_ma uses shift(1).rolling(42).mean() — trailing 42 bars BEFORE bar t.
  No reference to dates, named events, or absolute price levels.
"""
from __future__ import annotations
import pandas as pd

INPUT_COLUMNS: tuple[str, ...] = ()  # reads only ohlcv

BREAKOUT_LOOKBACK = 21   # bars (primary_params.breakout_lookback)
VOLUME_MA_LOOKBACK = 42  # bars (primary_params.volume_ma_lookback)
VOLUME_CONTRACTION_PCT = 0.7


def signal(ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: dict) -> pd.Series:
    """SHORT when close breaks 21-bar low AND volume is contracted vs 42-bar MA."""
    close = ohlcv["close"].astype(float)
    low   = ohlcv["low"].astype(float)
    vol   = ohlcv["volume"].astype(float)

    # Trailing 21-bar low BEFORE current bar (shift(1) = no lookahead).
    rolling_low = low.shift(1).rolling(BREAKOUT_LOOKBACK, min_periods=BREAKOUT_LOOKBACK).min()
    # Trailing 42-bar volume MA BEFORE current bar.
    vol_ma = vol.shift(1).rolling(VOLUME_MA_LOOKBACK, min_periods=VOLUME_MA_LOOKBACK).mean()

    breakdown = close < rolling_low
    contracted = vol < (VOLUME_CONTRACTION_PCT * vol_ma)

    out = pd.Series(0, index=ohlcv.index, dtype="int8")
    out[breakdown & contracted] = -1
    return out
