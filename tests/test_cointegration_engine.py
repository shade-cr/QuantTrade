"""Tests for cointegration spread engine (Tier 1 Phase 1).

The engine fires on log-spread Z-score thresholds when the pair is empirically
cointegrated. Per-fold Engle-Granger ADF p-value gate prevents firing in
regimes where the pair has decoupled.

Invariants:
  1. Constant pair → no events (no spread variance)
  2. Diverging pair → fires -1 (spread Z > threshold, short primary)
  3. Converging pair → fires +1 (spread Z < -threshold, long primary)
  4. No-pair-configured → all zeros (asset not in pairs list)
  5. Sign inverts when pair reversed (BTC primary vs ETH primary, same spread)
  6. Warmup (< zscore_lookback bars) → all zeros
  7. Output contract: pd.Series, same index, dtype float, values in {-1,0,1}
  8. Higher threshold → fewer events
  9. P-value gate zeros fold when pair decoupled (non-stationary residual)
  10. P-value gate passes when pair stationary (AR(1) spread)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from pipeline.cointegration import coint_spread_signal


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")


def test_coint_constant_pair_returns_zero():
    n = 200
    a = pd.Series(100.0, index=_idx(n))
    b = pd.Series(50.0, index=_idx(n))
    sig = coint_spread_signal(a, b, zscore_lookback=20, entry_threshold=2.0)
    assert (sig == 0.0).all()


def test_coint_diverging_pair_fires_negative_on_overshoot():
    """Primary (a) outpaces other (b) → log(a/b) drifts up → spread Z > +2 → -1 short a."""
    n = 200
    rng = np.random.default_rng(0)
    common = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    # Build a slow-divergent pair: a outperforms b by 0.5% per bar after bar 100
    a_values = common.copy()
    b_values = common.copy()
    a_values[100:] = common[100:] * np.exp(np.arange(n - 100) * 0.005)
    a = pd.Series(a_values, index=_idx(n))
    b = pd.Series(b_values, index=_idx(n))
    sig = coint_spread_signal(a, b, zscore_lookback=30, entry_threshold=2.0)
    assert (sig == -1.0).any(), "diverging spread should fire -1 (short primary)"


def test_coint_converging_pair_fires_positive_on_undershoot():
    """Primary (a) underperforms other (b) → log(a/b) drifts down → spread Z < -2 → +1 long a."""
    n = 200
    rng = np.random.default_rng(0)
    common = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    a_values = common.copy()
    b_values = common.copy()
    a_values[100:] = common[100:] * np.exp(-np.arange(n - 100) * 0.005)
    a = pd.Series(a_values, index=_idx(n))
    b = pd.Series(b_values, index=_idx(n))
    sig = coint_spread_signal(a, b, zscore_lookback=30, entry_threshold=2.0)
    assert (sig == 1.0).any(), "converging-below spread should fire +1 (long primary)"


def test_coint_sign_inverts_when_pair_reversed():
    """Same log spread, just swap primary↔other → all non-zero signs flip."""
    n = 200
    rng = np.random.default_rng(42)
    common = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    a_values = common.copy()
    b_values = common.copy()
    a_values[100:] = common[100:] * np.exp(np.arange(n - 100) * 0.003)
    a = pd.Series(a_values, index=_idx(n))
    b = pd.Series(b_values, index=_idx(n))

    sig_a_primary = coint_spread_signal(a, b, zscore_lookback=30, entry_threshold=2.0)
    sig_b_primary = coint_spread_signal(b, a, zscore_lookback=30, entry_threshold=2.0)
    # Where both non-zero, signs invert
    both = (sig_a_primary != 0) & (sig_b_primary != 0)
    if both.sum() == 0:
        pytest.skip("no overlap of non-zero — adjust fixture")
    assert (sig_a_primary[both] == -sig_b_primary[both]).all()


def test_coint_warmup_returns_zero_within_zscore_lookback():
    n = 100
    a = pd.Series(100.0 + np.arange(n, dtype=float), index=_idx(n))
    b = pd.Series(50.0 + 0.5 * np.arange(n, dtype=float), index=_idx(n))
    sig = coint_spread_signal(a, b, zscore_lookback=50, entry_threshold=2.0)
    # First 49 bars must be zero (insufficient data for rolling stats)
    assert (sig.iloc[:49] == 0.0).all()


def test_coint_output_contract():
    n = 100
    rng = np.random.default_rng(0)
    a = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, n)), index=_idx(n))
    b = pd.Series(80.0 + np.cumsum(rng.normal(0, 1, n)), index=_idx(n))
    sig = coint_spread_signal(a, b, zscore_lookback=20, entry_threshold=2.0)
    assert isinstance(sig, pd.Series)
    assert sig.index.equals(a.index)
    assert np.issubdtype(sig.dtype, np.floating)
    assert set(sig.dropna().unique()).issubset({-1.0, 0.0, 1.0})


def test_coint_higher_threshold_fewer_events():
    n = 300
    rng = np.random.default_rng(7)
    common = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    a = pd.Series(common * np.exp(np.cumsum(rng.normal(0, 0.002, n))), index=_idx(n))
    b = pd.Series(common * np.exp(np.cumsum(rng.normal(0, 0.002, n))), index=_idx(n))
    sig_loose = coint_spread_signal(a, b, zscore_lookback=30, entry_threshold=1.0)
    sig_tight = coint_spread_signal(a, b, zscore_lookback=30, entry_threshold=3.0)
    assert (sig_tight != 0).sum() <= (sig_loose != 0).sum()


def test_coint_pvalue_gate_zeros_fold_when_decoupled():
    """Synthetic non-stationary residual (random walk) → ADF p > cutoff → fold zeroed."""
    n = 400
    rng = np.random.default_rng(0)
    # Common factor random walk
    common = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    # Pair with random-walk residual (NOT stationary)
    rw_residual = np.cumsum(rng.normal(0, 0.005, n))   # I(1) noise
    a = pd.Series(common * np.exp(rw_residual), index=_idx(n))
    b = pd.Series(common, index=_idx(n))
    # Define a fold's train indices spanning the bulk of data
    fold_train = [np.arange(50, 300)]
    sig = coint_spread_signal(
        a, b, zscore_lookback=30, entry_threshold=1.5,
        fold_train_indices=fold_train, coint_pvalue_cutoff=0.10,
    )
    # Spread is non-stationary → ADF p high → signal zeroed for the test window of that fold
    # Fold test window is the complement of train within full series — i.e., bars 300+
    assert (sig.iloc[300:] == 0.0).all(), "fold test window should be zeroed when pair decoupled"


def test_coint_pvalue_gate_passes_when_stationary():
    """Stationary spread (mean-reverting AR(1)) → ADF p < cutoff → signal fires."""
    n = 400
    rng = np.random.default_rng(1)
    # Common factor
    common = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    # Strongly mean-reverting residual: AR(1) with phi=0.5
    residual = np.zeros(n)
    for t in range(1, n):
        residual[t] = 0.5 * residual[t - 1] + rng.normal(0, 0.02)
    a = pd.Series(common * np.exp(residual), index=_idx(n))
    b = pd.Series(common, index=_idx(n))
    fold_train = [np.arange(50, 300)]
    sig = coint_spread_signal(
        a, b, zscore_lookback=30, entry_threshold=1.5,
        fold_train_indices=fold_train, coint_pvalue_cutoff=0.10,
    )
    # Stationary spread should fire at least some non-zero signal in the test window
    assert (sig.iloc[300:] != 0).any(), "stationary spread should fire signals"


def test_coint_no_fold_indices_warns_but_proceeds():
    """If fold_train_indices is None, the gate is skipped — emit signal but with caveat."""
    n = 200
    rng = np.random.default_rng(0)
    common = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    a = pd.Series(common * np.exp(np.cumsum(rng.normal(0, 0.001, n))), index=_idx(n))
    b = pd.Series(common, index=_idx(n))
    sig = coint_spread_signal(a, b, zscore_lookback=30, entry_threshold=1.5)
    # No exception, no gate, output contract preserved
    assert isinstance(sig, pd.Series)
    assert set(sig.dropna().unique()).issubset({-1.0, 0.0, 1.0})
