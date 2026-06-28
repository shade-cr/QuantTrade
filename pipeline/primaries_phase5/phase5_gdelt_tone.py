"""Phase 5 custom primary for proposal `20260526-XAU-GDELT-TONE-001` (B0015c).

Long-only single-condition signal — "extreme inflation-anxiety narrative
precedes XAU rally":

  Long entry (+1) when:
    gdelt_inflation_tone_z252 < -1.0  (strict-causal on daily GDELT tone)
  Short entry (-1): never (long-only by design)
  No signal (0): all other bars

Single-condition by design (lesson from B0015b/a). Threshold -1.0 is the
Gaussian 16th percentile (one-sided), locked a-priori — NOT calibrated to
within-sample fire rate.

Min-separation: 0.20 ATR between consecutive entries.

Inputs read:
  - ohlcv["close"], ohlcv["high"], ohlcv["low"]    # raw OHLCV; universal substrate
  - pipeline.alt_data.gdelt_tone.load_gdelt_tone     # daily aggregated tone via direct call

  Computed inline:
    - tone_z252  = strict_causal_zscore(tone, window=252)
    - atr_14     = standard true-range rolling mean

  All inputs are disjoint from build_tier2_features outputs reaching the meta.
  GDELT data is NEVER added to features by design.

INPUT_COLUMNS=() because the primary reads from ohlcv + pipeline.alt_data.gdelt_tone
(separate alt_data path) — neither is in the features namespace.

Per docs/superpowers/specs/2026-05-26-gdelt-tone-primary.md.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.alt_data.gdelt_tone import load_gdelt_tone


# Declared inputs from the `features` namespace. Empty: primary reads only raw
# ohlcv + alt-data via separate path.
INPUT_COLUMNS: tuple[str, ...] = ()

# Configuration constants — proposal-locked.
Z_THRESHOLD = -1.0         # Gaussian 16th percentile (one-sided); a-priori
Z_WINDOW_DAILY_BARS = 252  # ~1 calendar year of trading days
THEME = "ECON_INFLATION"
MIN_SEPARATION_ATR_MULT = 0.20
ATR_LOOKBACK = 14

# Default cache path; overridable in tests via monkeypatch.
DEFAULT_CACHE_PATH: Path | None = None  # None -> use load_gdelt_tone's default


def _strict_causal_zscore(s: pd.Series, window: int) -> pd.Series:
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
    """Return side in {0, +1} for the GDELT inflation-tone primary.

    Per SKILL.md: pure function, NaN treated as 0, strict-causal windows.
    Raises GdeltToneCacheMissing if cache absent.
    """
    tone = load_gdelt_tone(
        target_index=ohlcv.index,
        theme=THEME,
        cache_path=DEFAULT_CACHE_PATH,
    )
    z = _strict_causal_zscore(tone["tone"], window=Z_WINDOW_DAILY_BARS)
    cond = (z < Z_THRESHOLD).fillna(False)

    raw = pd.Series(0, index=ohlcv.index, dtype="int8")
    raw[cond] = 1

    atr_14 = _compute_atr_14(ohlcv)
    sig = _apply_min_separation_atr(
        raw, ohlcv["close"], atr_14, MIN_SEPARATION_ATR_MULT,
    )
    return sig
