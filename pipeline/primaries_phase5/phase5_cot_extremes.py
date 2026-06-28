"""Phase 5 custom primary for proposal `20260526-XAU-BULLQ-COT-001` (B0015b).

Long-only confluence signal — "smart money + USD weakness" thesis:

  Long entry (+1) when ALL of:
    (1) commercials_net_long_z52 > 1.5   — strict-causal weekly z-score of
        commercials (Producer/Merchant + Swap Dealers) net long pct of OI
    (2) dxy_20d_chg < -0.02               — features['dtwexbgs_close']
        .pct_change(20).shift(1) at bar t
  Short entry (-1): never (long-only by design)
  No signal (0): all other bars

Min-separation: 0.20 ATR between consecutive entries (same convention as
phase5_xau_bearq_conjunction).

Inputs read:
  - ohlcv["close"], ohlcv["high"], ohlcv["low"]    # raw OHLCV; universal substrate
  - features["dtwexbgs_close"]                       # FROM the features arg (orchestrator-loaded)
  - pipeline.cot_features.build_cot_commercials_raw  # raw weekly CFTC commercials
                                                       # via direct call (NOT in features)

  Computed inline:
    - commercials_net_long_pct  = net_long / total_oi
    - commercials_net_long_z52  = strict_causal_zscore(pct, window=52 weeks)
    - dxy_20d_chg               = features['dtwexbgs_close'].pct_change(20).shift(1)
    - atr_14                    = standard true-range rolling mean

  All inputs from `features` namespace are listed in INPUT_COLUMNS below.
  The orchestrator (scripts/run_backtest.py::_run_one_primary) applies
  apply_primary_feature_blacklist AFTER calling this signal() — the meta
  sees the features frame MINUS the blacklist. The primary sees the
  unfiltered features arg, so it can still read dtwexbgs_close. All inputs
  are disjoint from the meta's view by construction (blacklist filters cot_*,
  dxy_*, dtwexbgs_* and the existing non-commercials cot_features outputs).

Causal-window discipline (mirrors phase5_xau_bearq_conjunction):
  All rolling windows use strict-prior conventions. Tests in
  tests/phase5/test_phase5_cot_extremes.py verify this property.

Per docs/superpowers/specs/2026-05-26-cot-extremes-primary.md (Draft v2)
and docs/superpowers/plans/2026-05-26-cot-extremes-primary.md (Task 7).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.cot_features import build_cot_commercials_raw


# Declared inputs from the `features` namespace per Layer (c) contract
# (docs/superpowers/specs/2026-05-26-edge-search-scope-decision.md §Precondición).
# Reads dtwexbgs_close FROM features — orchestrator loads it via build_macro_features
# and applies the blacklist filter (apply_primary_feature_blacklist) BEFORE the
# meta-labeler sees the features frame; the primary still receives the unfiltered
# features arg here.
INPUT_COLUMNS: tuple[str, ...] = ("dtwexbgs_close",)


# Configuration constants — proposal-locked. Do not parameterize from cfg
# unless they become tuned hyperparameters in a future proposal.
Z_THRESHOLD = 1.5
DXY_CHG_THRESHOLD = -0.02
DXY_CHG_LOOKBACK = 20
# 52-week-equivalent on daily-aligned (ffilled-from-weekly) commercials series.
# 52 weeks * 5 trading days/week = 260 D1 bars. The underlying values change
# weekly so the rolling stats effectively integrate 52 distinct weekly
# observations even though the window is in daily-bar count. Documented
# divergence from the spec wording "52 weeks" — the implementation uses 260
# trading-day bars which is mathematically equivalent in expectation.
Z_WINDOW_DAILY_BARS = 260
MIN_SEPARATION_ATR_MULT = 0.20
ATR_LOOKBACK = 14


def _strict_causal_zscore(s: pd.Series, window: int) -> pd.Series:
    """Strict-prior z-score: rolling stats at bar t use rows t-window..t-1
    (exclusive of t). Numerator is (s[t] - mu[t]). Standard no-self-reference
    convention mirroring phase5_xau_bearq_conjunction._strict_causal_zscore.
    """
    shifted = s.shift(1)
    mu = shifted.rolling(window=window, min_periods=window).mean()
    sd = shifted.rolling(window=window, min_periods=window).std(ddof=0)
    return (s - mu) / sd


def _compute_atr_14(ohlcv: pd.DataFrame) -> pd.Series:
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=ATR_LOOKBACK, min_periods=ATR_LOOKBACK).mean()


def _apply_min_separation_atr(
    raw_signals: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    min_atr_mult: float,
) -> pd.Series:
    """Drop fires that occur within min_atr_mult * ATR(t) of the previous fire."""
    out = raw_signals.copy()
    last_entry_price: float | None = None
    close_arr = close.values
    atr_arr = atr.values
    sig_arr = out.values.copy()  # avoid mutating shared array if out is a view
    for i in range(len(out)):
        if sig_arr[i] == 0:
            continue
        if last_entry_price is None:
            last_entry_price = float(close_arr[i])
            continue
        atr_i = atr_arr[i] if i < len(atr_arr) else np.nan
        if not np.isfinite(atr_i) or atr_i <= 0:
            last_entry_price = float(close_arr[i])
            continue
        gap = abs(float(close_arr[i]) - last_entry_price)
        if gap < min_atr_mult * float(atr_i):
            sig_arr[i] = 0
        else:
            last_entry_price = float(close_arr[i])
    out = pd.Series(sig_arr, index=out.index, dtype=out.dtype)
    return out


def signal(ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: dict) -> pd.Series:
    """Return side in {0, +1} for the COT-extremes confluence (long-only).

    Per the SKILL.md contract: pure function, deterministic given (ohlcv,
    features, cfg). NaN treated as 0 (no signal). Strict-causal rolling
    windows (no centered, no future bars).
    """
    if "dtwexbgs_close" not in features.columns:
        raise KeyError(
            "phase5_cot_extremes requires features['dtwexbgs_close'] (DTWEXBGS daily). "
            "Ensure pipeline.macro_fetch includes DTWEXBGS and the orchestrator "
            "passes the unfiltered features frame to signal() (the blacklist filter "
            "is applied AFTER signal() returns)."
        )

    # 1. Load raw commercials COT aligned to ohlcv index. NOT from features
    #    by design — keeps commercials off the meta's view by construction
    #    (orchestrator never adds commercials_* to build_tier2_features outputs).
    commercials = build_cot_commercials_raw(asset="XAUUSD", target_index=ohlcv.index)

    # 2. Compute commercials_net_long z strict-causally on the daily-aligned
    #    (ffilled-from-weekly) series. Window=260 D1 bars ≈ 52 weeks.
    pct = commercials["net_long"] / commercials["total_oi"].replace(0, np.nan)
    z = _strict_causal_zscore(pct, window=Z_WINDOW_DAILY_BARS)

    # 3. Read DXY from features (orchestrator-loaded). The blacklist filter
    #    runs AFTER this signal() returns, so the primary sees the column even
    #    though the meta won't.
    dxy = features["dtwexbgs_close"]
    dxy_chg = dxy.pct_change(DXY_CHG_LOOKBACK).shift(1)

    # 4. Build the confluence mask.
    cond_cot = z > Z_THRESHOLD
    cond_dxy = dxy_chg < DXY_CHG_THRESHOLD

    raw = pd.Series(0, index=ohlcv.index, dtype="int8")
    long_mask = (cond_cot & cond_dxy).fillna(False)
    raw[long_mask] = 1

    # 5. ATR-based min-separation gate.
    atr_14 = _compute_atr_14(ohlcv)
    sig = _apply_min_separation_atr(
        raw, ohlcv["close"], atr_14, MIN_SEPARATION_ATR_MULT,
    )
    return sig
