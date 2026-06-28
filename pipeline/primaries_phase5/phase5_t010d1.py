"""Phase 5 custom primary for Loop A proposal 20260526-XAUUSD-BULL_STR-T010D1.

Pseudocode (verbatim from proposal):
  Symmetric CUSUM on log-close, threshold h = 1.5 * ATR(14).
  On a positive CUSUM event, emit LONG (+1) if:
    COT rank (trailing 252 bars) <= 0.5
    OR real-yield rank (trailing 252 bars) <= 0.5
  (OR logic: either macro fuel channel suffices)
  Negative CUSUM events -> 0. No short side.

Inputs read: real_yield_5y_z252d (from features frame, FRED .shift(1) applied upstream).
INPUT_COLUMNS is disjoint from the meta-labeler feature set (real_yield is read here
but not dropped from the meta — it is a conditioning input, not a blacklisted column).

Causal-window discipline:
  ATR uses trailing window strictly before t via shift(1) convention.
  Rolling quantile ranks use shift(1) so rank at t uses only bars t-1...t-252.
  COT loaded via load_cot_net_noncomm_z (publication-lag-aware by construction).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pipeline.labels import cusum_filter_signal
from phase5.regime_stats import load_cot_net_noncomm_z

INPUT_COLUMNS: tuple[str, ...] = ("real_yield_5y_z252d",)

CUSUM_ATR_MULT = 1.5
ATR_WINDOW = 14
RANK_WINDOW = 252
RANK_THRESHOLD = 0.5


def _atr(ohlcv: pd.DataFrame, window: int) -> pd.Series:
    high, low, close = (ohlcv[c].astype(float) for c in ("high", "low", "close"))
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def _rolling_rank(s: pd.Series, window: int) -> pd.Series:
    """Rank of s[t] within trailing `window` bars strictly before t."""
    n = len(s)
    out = np.full(n, np.nan, dtype=float)
    vals = s.values.astype(float)
    for t in range(window, n):
        cur = vals[t]
        if not np.isfinite(cur):
            continue
        prior = vals[t - window:t]
        prior_v = prior[np.isfinite(prior)]
        if prior_v.size < window // 2:
            continue
        out[t] = float((prior_v <= cur).sum()) / float(prior_v.size)
    return pd.Series(out, index=s.index)


def signal(ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: dict) -> pd.Series:
    """Long on positive CUSUM when COT or real-yield is in lower half."""
    atr = _atr(ohlcv, ATR_WINDOW)
    cusum_sig = cusum_filter_signal(
        close=ohlcv["close"].astype(float),
        atr=atr,
        threshold_atr=CUSUM_ATR_MULT,
    )
    cusum_up = (cusum_sig == 1.0)

    # COT rank (trailing 252 bars)
    cot_z = load_cot_net_noncomm_z(asset="XAUUSD", target_index=ohlcv.index)
    cot_rank = _rolling_rank(cot_z, RANK_WINDOW)
    cot_low = (cot_rank <= RANK_THRESHOLD).fillna(False)

    # Real-yield rank (trailing 252 bars)
    if "real_yield_5y_z252d" in features.columns:
        ry = features["real_yield_5y_z252d"].reindex(ohlcv.index).astype(float)
    else:
        ry = pd.Series(np.nan, index=ohlcv.index)
    ry_rank = _rolling_rank(ry, RANK_WINDOW)
    ry_low = (ry_rank <= RANK_THRESHOLD).fillna(False)

    out = pd.Series(0, index=ohlcv.index, dtype="int8")
    out[cusum_up & (cot_low | ry_low)] = 1
    return out
