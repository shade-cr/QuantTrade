"""Phase 5 custom primary for Loop A proposal 20260529-XAGUSD-D1-BULL_QUI-B003v1.

Pseudocode (verbatim from proposal):
  Compute a symmetric CUSUM filter on log-close with a per-bar threshold of
  cusum_threshold_atr_mult (=1.0) times the trailing ATR(atr_lookback=14)
  normalized by close (i.e. cumulative positive/negative deviations reset to
  zero when they cross +/- threshold, emitting an event on the crossing bar).
  On each UP CUSUM event bar t (the positive-drift crossing): compute
  participation = volume[t] relative to its trailing median over
  volume_median_lookback (=42) bars, i.e.
  vol_ratio = volume[t] / median(volume[t-42 .. t-1]).
  LONG signal (+1) when vol_ratio < 1.0 (below-median participation: cheap
  absorption markup). NO signal (0) when vol_ratio >= 1.0 (above-median
  participation: climactic / churn leg - explicitly skipped, do NOT short).
  On DOWN CUSUM events: NO signal (0) - this hypothesis is long-only within
  the bull regime. All non-event bars: NO signal (0).

Causal-window discipline:
  - ATR(14) is Wilder-style (ewm alpha=1/14) over true range, then shift(1):
    the threshold applied at bar t uses only bars <= t-1. Normalization by
    close.shift(1) for the same reason.
  - The volume median window is t-42..t-1 (shift(1).rolling(42).median()),
    exclusive of bar t, exactly as the pseudocode specifies.
  - The CUSUM accumulates log-returns; the return at bar t (close[t] vs
    close[t-1]) is known at the close of bar t, when the signal is stamped.
  - No reference to dates, named events, or absolute price levels.

Frozen interpretation decisions (DA review 2026-06-10, PROCEED_WITH_CAVEAT —
locked BEFORE any M3 audit run; none of these may be revisited if the audit
underperforms, each is otherwise a silent degree of freedom on the event set):
  1. FULL RESET: any crossing (up or down) zeroes BOTH accumulators. The
     pseudocode's "reset to zero when they cross" is read as a full state
     reset, not the canonical per-side reset (LdP 2.4). A DOWN crossing
     therefore erases accumulated positive drift.
  2. THRESHOLD DOUBLE-SHIFT: thr[t] = (ATR/close) computed through t-1
     (single .shift(1) applied to the ratio). The contemporaneous reading
     (ATR through t / close[t]) would also be lookahead-free; the lagged
     reading is the stricter one and its event-set sensitivity was NOT
     explored — it must not be tuned later.
  3. WARMUP DROP: bars with non-finite return/threshold contribute NOTHING
     to the accumulators (skipped, not merely non-emitting). Emission is
     additionally blocked until the 42-bar volume median matures.
  4. CROSS = STRICTLY EXCEED: s_pos > thr (not >=). Ties are measure-zero
     on float log-returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INPUT_COLUMNS: tuple[str, ...] = ()  # reads only ohlcv

CUSUM_THRESHOLD_ATR_MULT = 1.0   # primary_params.cusum_threshold_atr_mult
VOLUME_MEDIAN_LOOKBACK = 42      # primary_params.volume_median_lookback
ATR_LOOKBACK = 14                # primary_params.atr_lookback


def _trailing_atr_norm(ohlcv: pd.DataFrame) -> pd.Series:
    """Wilder ATR(14) normalized by close, shifted so bar t sees only <= t-1."""
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / ATR_LOOKBACK, adjust=False).mean()
    return (atr / close).shift(1)


def signal(ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: dict) -> pd.Series:
    """LONG on an UP CUSUM crossing with below-median participation; else 0."""
    close = ohlcv["close"].astype(float)
    vol = ohlcv["volume"].astype(float)

    log_ret = np.log(close / close.shift(1))
    thr = CUSUM_THRESHOLD_ATR_MULT * _trailing_atr_norm(ohlcv)
    vol_med = vol.shift(1).rolling(
        VOLUME_MEDIAN_LOOKBACK, min_periods=VOLUME_MEDIAN_LOOKBACK
    ).median()

    s_pos = 0.0
    s_neg = 0.0
    ret_arr = log_ret.to_numpy()
    thr_arr = thr.to_numpy()
    med_arr = vol_med.to_numpy()
    vol_arr = vol.to_numpy()
    out_arr = np.zeros(len(ret_arr), dtype="int8")

    for i in range(len(ret_arr)):
        r = ret_arr[i]
        t = thr_arr[i]
        if not np.isfinite(r) or not np.isfinite(t) or t <= 0:
            # Warmup (NaN return/threshold): accumulate nothing, emit nothing.
            continue
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if s_pos > t:
            s_pos = 0.0
            s_neg = 0.0
            # UP event: gate on below-median participation (cheap markup).
            m = med_arr[i]
            if np.isfinite(m) and m > 0 and (vol_arr[i] / m) < 1.0:
                out_arr[i] = 1
        elif s_neg < -t:
            s_pos = 0.0
            s_neg = 0.0
            # DOWN event: explicitly no signal (long-only hypothesis).

    return pd.Series(out_arr, index=ohlcv.index, dtype="int8")
