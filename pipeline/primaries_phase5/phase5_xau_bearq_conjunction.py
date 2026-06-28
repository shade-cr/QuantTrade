"""Phase 5 custom primary for proposal `20260526-XAU-BEARQ-002`.

Short-side multi-feature conjunction signal:

  Short entry (-1) when ALL of:
    (1) roc_63 in lower quantile band [0.0, 0.35] over trailing 252 bars
    (2) rv_20 in lower quantile band [0.0, 0.40] over trailing 252 bars
    (3) close < ma_200
    (4) volume z-score over trailing 63 bars >= -0.5  (participation not collapsed)

  Long entry (+1): never (short-only).
  No signal (0): all other bars.

  Min-separation: 0.20 ATR between consecutive entries.

Causal-window discipline (per Day-2 skeptic caveat C/D):
  All rolling windows use `.shift(1).rolling(N)` so the value at time t is
  evaluated against a band/threshold derived from STRICTLY-PRIOR data
  (bars t-N through t-1, exclusive of t). This is the no-self-reference
  rolling-quantile convention. Unit tests in tests/phase5/test_phase5_xau_bearq_conjunction.py
  verify this property.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Declared inputs from the `features` namespace per Layer (c) contract
# (docs/superpowers/specs/2026-05-26-edge-search-scope-decision.md §Precondición).
# This primary reads only raw ohlcv (close/high/low/volume), so the tuple is
# empty. Raw OHLCV is universal substrate exempt from the blacklist check.
INPUT_COLUMNS: tuple[str, ...] = ()


# Configuration constants — proposal-locked, do not parameterize from cfg
# unless those become tuned hyperparameters in a future proposal.
ROC_QUANTILE_UPPER = 0.35   # roc_63 must be in the lower 35% band
RV_QUANTILE_UPPER = 0.40    # rv_20 must be in the lower 40% band
QUANTILE_WINDOW = 252       # trailing window for band computation
VOLUME_Z_WINDOW = 63        # trailing window for volume z-score
VOLUME_Z_MIN = -0.5         # volume z-score floor (participation not collapsed)
MIN_SEPARATION_ATR_MULT = 0.20  # minimum bar separation = 0.20 * ATR(14)
ROC_LOOKBACK = 63
RV_LOOKBACK = 20
MA_SLOW_LOOKBACK = 200


def _strict_causal_quantile(s: pd.Series, window: int, q: float) -> pd.Series:
    """Rolling quantile that uses ONLY data strictly prior to t.

    Returns the quantile of s[t-window:t-1] at bar t (i.e., the value at t
    is NOT included in the quantile distribution). This is the no-
    self-reference convention required by SKILL.md.
    """
    return s.shift(1).rolling(window=window, min_periods=window).quantile(q)


def _strict_causal_zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score using strictly-prior data."""
    shifted = s.shift(1)
    mu = shifted.rolling(window=window, min_periods=window).mean()
    sd = shifted.rolling(window=window, min_periods=window).std(ddof=0)
    return (s - mu) / sd


def _compute_features(ohlcv: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute the four features the conjunction depends on, all PIT-clean.

    Returns: {roc_63, rv_20, ma_200, volume_zscore_63, atr_14}.
    """
    close = ohlcv["close"]
    log_close = np.log(close.astype(float))
    log_ret = log_close.diff()

    roc_63 = log_close.diff(ROC_LOOKBACK)
    rv_20 = log_ret.rolling(window=RV_LOOKBACK, min_periods=RV_LOOKBACK).std(ddof=0) * np.sqrt(252)
    ma_200 = close.rolling(window=MA_SLOW_LOOKBACK, min_periods=MA_SLOW_LOOKBACK).mean()
    vol_z = _strict_causal_zscore(ohlcv["volume"].astype(float), VOLUME_Z_WINDOW)

    # ATR(14) for min-separation gate
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()

    return {
        "roc_63": roc_63,
        "rv_20": rv_20,
        "ma_200": ma_200,
        "volume_zscore_63": vol_z,
        "atr_14": atr_14,
    }


def _apply_min_separation_atr(raw_signals: pd.Series, close: pd.Series, atr: pd.Series,
                               min_atr_mult: float) -> pd.Series:
    """Drop signals that fire within min_atr_mult * ATR(t) of the previous signal."""
    out = raw_signals.copy()
    last_entry_price: float | None = None
    last_entry_idx: int | None = None
    close_arr = close.values
    atr_arr = atr.values
    sig_arr = out.values
    for i in range(len(out)):
        if sig_arr[i] == 0:
            continue
        if last_entry_price is None:
            last_entry_idx = i
            last_entry_price = close_arr[i]
            continue
        atr_i = atr_arr[i] if i < len(atr_arr) else np.nan
        if not np.isfinite(atr_i) or atr_i <= 0:
            # ATR not computable yet — keep the signal but don't update separation tracker
            last_entry_idx = i
            last_entry_price = close_arr[i]
            continue
        gap = abs(close_arr[i] - last_entry_price)
        if gap < min_atr_mult * atr_i:
            sig_arr[i] = 0
        else:
            last_entry_idx = i
            last_entry_price = close_arr[i]
    out.iloc[:] = sig_arr
    return out


def signal(ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: dict) -> pd.Series:
    """Return a side series in {-1, 0} (short-only) for the BEARQ conjunction.

    `features` is supplied by the pipeline (Tier-2 features), but for this
    custom primary we compute our four feature streams from `ohlcv` directly
    to keep the function self-contained and easy to unit-test.

    `cfg` is the run-time config (not used here — params are proposal-locked).
    """
    feats = _compute_features(ohlcv)
    roc_63 = feats["roc_63"]
    rv_20 = feats["rv_20"]
    ma_200 = feats["ma_200"]
    vol_z = feats["volume_zscore_63"]
    atr_14 = feats["atr_14"]
    close = ohlcv["close"]

    # Strict-causal quantile bands on trailing QUANTILE_WINDOW
    roc_band_upper = _strict_causal_quantile(roc_63, QUANTILE_WINDOW, ROC_QUANTILE_UPPER)
    rv_band_upper = _strict_causal_quantile(rv_20, QUANTILE_WINDOW, RV_QUANTILE_UPPER)

    cond_roc = roc_63 <= roc_band_upper      # roc_63 in lower band
    cond_rv = rv_20 <= rv_band_upper         # rv_20 in lower band
    cond_close = close < ma_200              # structural downtrend
    cond_vol = vol_z >= VOLUME_Z_MIN         # participation not collapsed

    raw = pd.Series(0, index=ohlcv.index, dtype="int8")
    short_mask = cond_roc & cond_rv & cond_close & cond_vol
    raw[short_mask] = -1

    # Apply min-separation gate
    sig = _apply_min_separation_atr(raw, close, atr_14, MIN_SEPARATION_ATR_MULT)
    return sig
