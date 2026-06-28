"""Phase 5 custom primary for proposal `20260526-XAU-GLD-FLOW-001` (B0015a).

Long-only single-condition signal — "GLD ETF accumulation flow precedes XAU rally":

  Long entry (+1) when:
    gld_holdings_5d_chg_z252 > +1.0  (strict-causal on daily GLD ETF holdings)
  Short entry (-1): never (long-only by design)
  No signal (0): all other bars

Single-condition by design (corrects the B0015b multi-condition AND
under-firing failure mode). Threshold +1.0 is the Gaussian 84th percentile,
locked a-priori from external literature — NOT calibrated to within-sample
fire rate.

Min-separation: 0.20 ATR between consecutive entries.

Inputs read:
  - ohlcv["close"], ohlcv["high"], ohlcv["low"]    # raw OHLCV; universal substrate
  - pipeline.alt_data.gld_holdings.load_gld_holdings   # raw SPDR daily holdings
                                                         # via direct call (NOT in features)

  Computed inline:
    - holdings_5d_chg     = gld_oz_held.pct_change(5)
    - holdings_z252       = strict_causal_zscore(holdings_5d_chg, window=252)
    - atr_14              = standard true-range rolling mean

  All inputs are disjoint from build_tier2_features() outputs reaching the meta
  by construction: the GLD holdings series is NEVER added to
  build_tier2_features outputs (separate alt_data module), AND the blacklist
  filter removes any cot_*/dxy_*/dtwexbgs_*/gld_* columns from the meta's view.

INPUT_COLUMNS=() because the primary reads from ohlcv (universal substrate)
and pipeline.alt_data.gld_holdings (separate alt_data path) — neither is in
the `features` namespace.

Per docs/superpowers/specs/2026-05-26-gld-holdings-primary.md (Draft v2).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.alt_data.gld_holdings import load_gld_holdings


# Declared inputs from the `features` namespace. Empty: primary reads only raw
# ohlcv + alt-data via separate path, NOT from features.
INPUT_COLUMNS: tuple[str, ...] = ()

# Configuration constants — proposal-locked.
Z_THRESHOLD = 1.0   # Gaussian 84th percentile, a-priori from positioning literature
HOLDINGS_CHG_LOOKBACK = 5
Z_WINDOW_DAILY_BARS = 252  # ~1 calendar year on D1 (~252 trading days)
MIN_SEPARATION_ATR_MULT = 0.20
ATR_LOOKBACK = 14

# Default cache path; overridable via monkeypatch in tests or by passing
# a different path via the orchestrator config.
DEFAULT_CACHE_PATH = Path("cache/alt_data/gld_holdings.parquet")


def _strict_causal_zscore(s: pd.Series, window: int) -> pd.Series:
    """Strict-prior z-score: rolling stats at bar t use rows t-window..t-1
    (exclusive of t). Mirrors phase5_xau_bearq_conjunction._strict_causal_zscore
    and phase5_cot_extremes._strict_causal_zscore.
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
    """Drop fires within min_atr_mult * ATR(t) of the previous fire."""
    out = raw_signals.copy()
    last_entry_price: float | None = None
    close_arr = close.values
    atr_arr = atr.values
    sig_arr = out.values.copy()
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
    """Return side in {0, +1} for the GLD accumulation-flow primary (long-only).

    Per SKILL.md: pure function, deterministic given (ohlcv, features, cfg).
    NaN treated as 0 (no signal). Strict-causal rolling windows.

    Raises GldHoldingsCacheMissing if cache/alt_data/gld_holdings.parquet
    is absent (run scripts/ingest_gld_holdings.py first).
    """
    # 1. Load raw GLD holdings aligned to ohlcv index (with .shift(1)
    #    publication-lag applied inside load_gld_holdings).
    holdings = load_gld_holdings(target_index=ohlcv.index, cache_path=DEFAULT_CACHE_PATH)
    oz = holdings["gld_oz_held"]

    # 2. Compute strict-causal 5d-pct-change z-score over 252 D1 bars.
    pct_5d = oz.pct_change(HOLDINGS_CHG_LOOKBACK)
    z = _strict_causal_zscore(pct_5d, window=Z_WINDOW_DAILY_BARS)

    # 3. Build the single-condition mask.
    cond = (z > Z_THRESHOLD).fillna(False)

    raw = pd.Series(0, index=ohlcv.index, dtype="int8")
    raw[cond] = 1

    # 4. ATR-based min-separation gate.
    atr_14 = _compute_atr_14(ohlcv)
    sig = _apply_min_separation_atr(
        raw, ohlcv["close"], atr_14, MIN_SEPARATION_ATR_MULT,
    )
    return sig
