"""Cross-asset cointegration spread engine (Tier 1 Phase 1).

Generates +1/-1 events on rolling log-spread Z-score thresholds for a
pre-committed pair of cointegrated assets. Per-fold Engle-Granger ADF
p-value gate prevents firing in regimes where the pair has decoupled.

Information axis: cross-asset relative-value. Orthogonal to single-asset
price-derived primaries (EMA / momentum / CUSUM / Bollinger) because the
trigger comes from the SPREAD between two assets, not either asset's own
price autocorrelation. Same axis as VIX-regime (cross-asset / macro) but
a different sub-axis (relative value vs macro vol regime).

LIMITATION (v1): single-leg interpretation. The engine returns side for
the PRIMARY asset only. Triple-barrier exits on the primary's own price,
NOT the spread's mean reversion (Z=0). This means cointegration informs
ENTRY but spread mean-reversion is NOT enforced on EXIT. Edge from the
spread thesis decays after the entry impulse; the barrier may close on
noise unrelated to the cointegration. Acceptable as v1; refactor to
dual-leg in v2 if this engine is promoted.

This is a DIRECTIONAL signal informed by spread, NOT a pairs trade.
"""
from __future__ import annotations
from typing import Optional
import warnings

import numpy as np
import pandas as pd


def coint_spread_signal(
    primary_close: pd.Series,
    other_close: pd.Series,
    fold_train_indices: Optional[list[np.ndarray]] = None,
    zscore_lookback: int = 60,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    coint_pvalue_cutoff: float = 0.10,
) -> pd.Series:
    """Mean-reversion signal on log-spread Z-score with per-fold cointegration gate.

    Spread = log(primary_close) - log(other_close). Z-score computed on a
    rolling window of `zscore_lookback` bars. Side fires when |Z| exceeds
    `entry_threshold` (sign convention below).

    Sign convention (single-leg):
      - z > +entry_threshold → spread too HIGH → primary overvalued → -1 (short primary)
      - z < -entry_threshold → spread too LOW  → primary undervalued → +1 (long primary)
      - else 0

    Per-fold cointegration gate:
      If `fold_train_indices` is provided, for each fold's train window run
      Engle-Granger ADF on the spread restricted to that window. If
      ADF p-value > `coint_pvalue_cutoff`, the signal is zeroed in that
      fold's TEST window (complement of train within the full series).

      If `fold_train_indices` is None, the gate is SKIPPED with a UserWarning.
      Test-time use only — production should always pass fold_train_indices.

    Returns pd.Series of float in {-1.0, 0.0, +1.0} with same index as `primary_close`.
    """
    if not primary_close.index.equals(other_close.index):
        # Align defensively — caller should pre-align, but don't blow up
        other_close = other_close.reindex(primary_close.index)

    log_spread = np.log(primary_close) - np.log(other_close)

    # Rolling Z-score of spread
    rolling_mean = log_spread.rolling(zscore_lookback).mean()
    rolling_std = log_spread.rolling(zscore_lookback).std()
    z = (log_spread - rolling_mean) / rolling_std.replace(0, np.nan)

    # Raw signal
    sig = pd.Series(0.0, index=primary_close.index)
    valid = z.notna()
    sig[valid & (z > entry_threshold)] = -1.0
    sig[valid & (z < -entry_threshold)] = 1.0

    # Per-fold cointegration p-value gate
    if fold_train_indices is None:
        warnings.warn(
            "coint_spread_signal called without fold_train_indices — "
            "p-value gate skipped. For production, always pass fold indices.",
            stacklevel=2,
        )
        return sig

    from statsmodels.tsa.stattools import adfuller

    n = len(primary_close)
    fold_diagnostics: list[dict] = []
    for fold_k, train_idx in enumerate(fold_train_indices):
        if len(train_idx) < zscore_lookback:
            # Not enough data to test cointegration
            continue
        train_spread = log_spread.iloc[train_idx].dropna()
        if len(train_spread) < zscore_lookback:
            continue
        try:
            adf_result = adfuller(train_spread.values, autolag="AIC", regression="c")
            pvalue = float(adf_result[1])
        except (ValueError, RuntimeError):
            pvalue = 1.0   # treat as decoupled on error

        fold_diagnostics.append({"fold": fold_k, "adfuller_pvalue": pvalue})

        if pvalue > coint_pvalue_cutoff:
            # Determine this fold's test window as complement of train indices
            # within the full range. With expanding-window WF, test window
            # follows immediately after train_idx.max().
            test_start = int(train_idx.max()) + 1
            # If there's a next fold, the test window extends to its train start;
            # otherwise to the end of the series.
            if fold_k + 1 < len(fold_train_indices):
                test_end = int(fold_train_indices[fold_k + 1][0])
            else:
                test_end = n
            sig.iloc[test_start:test_end] = 0.0

    # Attach diagnostics as a series attribute for downstream logging.
    # (pd.Series does have .attrs since pandas 1.1)
    try:
        sig.attrs["cointegration_fold_diagnostics"] = fold_diagnostics
    except Exception:
        pass

    return sig


def load_pairs_config(path) -> list[dict]:
    """Load configs/cointegration_pairs.yaml. Returns list of {primary, other} dicts."""
    from pathlib import Path
    import yaml
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"pairs config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    pairs = raw.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError(f"pairs config {p}: 'pairs' must be a list of {{primary, other}}")
    return pairs


def lookup_pair_for_asset(asset: str, pairs: list[dict]) -> dict | None:
    """Find the pair where `asset` is the primary. Returns dict or None."""
    for pair in pairs:
        if pair.get("primary") == asset:
            return pair
    return None
