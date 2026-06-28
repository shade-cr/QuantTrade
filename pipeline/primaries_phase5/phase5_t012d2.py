"""Phase 5 custom primary for Loop A proposal 20260526-XAUUSD-BEAR_QUI-T012D2.

Pseudocode (verbatim from proposal):
  For each bar t in BEAR_QUIET:
    (1) Weekly sign-flip: cot_net_noncomm_z52w changed sign from negative to
        positive within the last 10 daily bars (cot_z.shift(7) < 0 AND cot_z >= 0,
        using a 7-day window to approximate "prior weekly observation").
    (2) Real-yield gate: real_yield_5y_z252d at bar t >= its rolling 55th
        percentile over a trailing 252-bar window (yield not in deep-negative tail).
    (3) First-bar deduplication: emit only on the first bar where the sign-flip
        condition fires after a reset (5-bar cooldown after any emission).
    Emit side=+1 when all three hold. No short side.

Inputs read: real_yield_5y_z252d (from features frame, FRED .shift(1) applied upstream).
INPUT_COLUMNS is disjoint from the meta-labeler feature set (real_yield read here as
a gate condition; it remains available to the meta as a conditioning feature).

Causal-window discipline:
  COT shifted 3 daily bars for publication lag (provided by load_cot_net_noncomm_z).
  All rolling windows exclude current bar (shift(1) convention).
  Rolling rank for real-yield gate is trailing-252 on prior bars only.

Simplification: the pseudocode's "two most recent distinct weekly COT observations"
is implemented as cot_z.shift(7) vs cot_z (the prior-week value), a conservative
approximation for D1 data that guarantees causality. The 5-bar cooldown enforces
the "first bar per sign-flip event" de-duplication.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from phase5.regime_stats import load_cot_net_noncomm_z

INPUT_COLUMNS: tuple[str, ...] = ("real_yield_5y_z252d",)

WEEKLY_SHIFT = 7       # bars (approx 1 week on D1) for prior-week COT value
YIELD_WINDOW = 252
YIELD_QUANTILE_FLOOR = 0.55  # must be >= 55th pct (NOT in deep-negative tail)
COOLDOWN_BARS = 5      # de-dup: suppress re-entry for 5 bars after emission


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
    """Long on COT weekly sign-flip AND real-yield not in deep-negative tail."""
    cot_z = load_cot_net_noncomm_z(asset="XAUUSD", target_index=ohlcv.index)

    # Sign-flip: prior-week COT was negative, current is positive.
    cot_prior = cot_z.shift(WEEKLY_SHIFT)
    sign_flip = (cot_prior < 0) & (cot_z >= 0)

    # Real-yield gate: >= 55th percentile of trailing 252-bar window.
    if "real_yield_5y_z252d" in features.columns:
        ry = features["real_yield_5y_z252d"].reindex(ohlcv.index).astype(float)
    else:
        ry = pd.Series(np.nan, index=ohlcv.index)
    ry_rank = _rolling_rank(ry, YIELD_WINDOW)
    yield_ok = (ry_rank >= YIELD_QUANTILE_FLOOR).fillna(False)

    # Raw signal (pre-dedup)
    raw = sign_flip & yield_ok

    # Cooldown de-duplication: suppress re-entry for COOLDOWN_BARS bars after emission.
    out_arr = np.zeros(len(ohlcv), dtype="int8")
    last_emit = -COOLDOWN_BARS - 1
    raw_arr = raw.values
    for i in range(len(raw_arr)):
        if raw_arr[i] and (i - last_emit) > COOLDOWN_BARS:
            out_arr[i] = 1
            last_emit = i

    return pd.Series(out_arr, index=ohlcv.index)
