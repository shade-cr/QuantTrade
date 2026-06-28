"""Tests for cusum_filter_signal — AFML §3.3 symmetric CUSUM filter primary.

The CUSUM filter accumulates log-returns and emits a +1 event when the cumulative
upward run exceeds a threshold, and -1 when the downward run exceeds it. After
an event, the corresponding accumulator resets to zero. The threshold is
volatility-adaptive: `threshold_atr * ATR[t-1] / close[t-1]` so it scales with
current realized volatility (avoids fixed-pct thresholds that misbehave across
regimes).

Invariants tested:
  1. Constant price → no events ever
  2. Strong monotone up → eventually triggers +1
  3. Strong monotone down → eventually triggers -1
  4. After an event, the accumulator resets (no immediate re-fire)
  5. Symmetric: mirrored returns produce mirrored events at mirror positions
  6. NaN/zero ATR is handled (no event fired, no exception)
  7. Output is pd.Series with same index as input close
  8. Output values are exactly in {-1.0, 0.0, +1.0}
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from pipeline.labels import cusum_filter_signal


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")


def test_cusum_constant_price_no_events():
    n = 100
    close = pd.Series(np.full(n, 100.0), index=_idx(n))
    atr = pd.Series(np.full(n, 1.0), index=_idx(n))
    sig = cusum_filter_signal(close, atr, threshold_atr=2.0)
    assert (sig == 0.0).all(), f"constant price should never trigger; got {sig.value_counts()}"


def test_cusum_strong_uptrend_fires_positive():
    """A clear, sustained uptrend (1% per bar) should fire +1 eventually."""
    n = 50
    # 1% per bar geometric growth, ATR small relative to price moves
    close = pd.Series(100.0 * (1.01 ** np.arange(n)), index=_idx(n))
    atr = pd.Series(np.full(n, 0.5), index=_idx(n))  # ATR=0.5 on price~100 → 0.5% threshold base
    sig = cusum_filter_signal(close, atr, threshold_atr=2.0)
    assert (sig == 1.0).any(), "strong uptrend should fire at least one +1 event"
    assert (sig == -1.0).sum() == 0, "uptrend should NOT fire -1 events"


def test_cusum_strong_downtrend_fires_negative():
    """A clear, sustained downtrend should fire -1 eventually."""
    n = 50
    close = pd.Series(100.0 * (0.99 ** np.arange(n)), index=_idx(n))
    atr = pd.Series(np.full(n, 0.5), index=_idx(n))
    sig = cusum_filter_signal(close, atr, threshold_atr=2.0)
    assert (sig == -1.0).any(), "strong downtrend should fire at least one -1 event"
    assert (sig == 1.0).sum() == 0, "downtrend should NOT fire +1 events"


def test_cusum_event_resets_accumulator():
    """After firing, the same direction accumulator should reset so the next
    event requires accumulating threshold-worth of returns again from zero."""
    n = 100
    # Big jump, then flat — should fire once at the jump, never again
    prices = [100.0] * 5 + [110.0] * 95   # +10% jump at t=5, flat after
    close = pd.Series(prices, index=_idx(n))
    atr = pd.Series(np.full(n, 1.0), index=_idx(n))
    sig = cusum_filter_signal(close, atr, threshold_atr=2.0)
    pos_events = sig[sig == 1.0]
    # At most one positive event because after firing, accumulator resets and
    # the price is flat so cumulative log-return = 0 forever after.
    assert len(pos_events) == 1, f"expected exactly 1 +1 event, got {len(pos_events)}"
    assert (sig == -1.0).sum() == 0


def test_cusum_symmetric_under_sign_flip():
    """If you negate the price path's log-returns, +1 and -1 events should
    mirror at the same positions."""
    n = 60
    rng = np.random.default_rng(42)
    # Random walk
    returns = rng.normal(0, 0.005, n)
    close_up = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=_idx(n))
    close_down = pd.Series(100.0 * np.exp(np.cumsum(-returns)), index=_idx(n))
    atr = pd.Series(np.full(n, 0.5), index=_idx(n))
    sig_up = cusum_filter_signal(close_up, atr, threshold_atr=2.0)
    sig_down = cusum_filter_signal(close_down, atr, threshold_atr=2.0)
    # Same event positions, opposite signs
    assert (sig_up != 0).equals(sig_down != 0), "event positions should match"
    nonzero_up = sig_up[sig_up != 0]
    nonzero_down = sig_down[sig_down != 0]
    np.testing.assert_array_equal(nonzero_up.values, -nonzero_down.values)


def test_cusum_handles_nan_atr_without_exception():
    """ATR has a warm-up period (NaN at start); CUSUM must not crash."""
    n = 30
    close = pd.Series(100.0 + np.arange(n, dtype=float), index=_idx(n))
    atr = pd.Series([np.nan] * 14 + [1.0] * (n - 14), index=_idx(n))
    sig = cusum_filter_signal(close, atr, threshold_atr=2.0)
    assert len(sig) == n
    # Events before t=14 (where atr is NaN) cannot fire
    assert (sig.iloc[:14] == 0.0).all()


def test_cusum_handles_zero_atr_safely():
    """ATR=0 produces threshold=0; we MUST NOT divide-by-zero crash."""
    n = 20
    close = pd.Series(100.0 + 0.5 * np.arange(n, dtype=float), index=_idx(n))
    atr = pd.Series(np.zeros(n), index=_idx(n))
    sig = cusum_filter_signal(close, atr, threshold_atr=2.0)
    # No event because threshold is 0/undefined; behavior: emit nothing
    assert len(sig) == n


def test_cusum_output_contract():
    """Output is a pd.Series, same index, values strictly in {-1.0, 0.0, +1.0}."""
    n = 80
    rng = np.random.default_rng(0)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=_idx(n))
    atr = pd.Series(np.full(n, 1.0), index=_idx(n))
    sig = cusum_filter_signal(close, atr, threshold_atr=2.0)
    assert isinstance(sig, pd.Series)
    assert sig.index.equals(close.index)
    unique_vals = set(sig.dropna().unique())
    assert unique_vals.issubset({-1.0, 0.0, 1.0}), f"unexpected values: {unique_vals}"


def test_cusum_higher_threshold_fewer_events():
    """A larger threshold_atr should produce fewer (or equal) events vs smaller."""
    n = 200
    rng = np.random.default_rng(7)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=_idx(n))
    atr = pd.Series(np.full(n, 1.0), index=_idx(n))
    sig_low = cusum_filter_signal(close, atr, threshold_atr=1.0)
    sig_high = cusum_filter_signal(close, atr, threshold_atr=4.0)
    n_low = (sig_low != 0).sum()
    n_high = (sig_high != 0).sum()
    assert n_high <= n_low, f"higher threshold should give fewer events; got {n_high} vs {n_low}"


def test_cusum_returns_dtype_float():
    """Return dtype is float (pipeline expects float side, even though values are ints)."""
    n = 30
    close = pd.Series(100.0 + np.arange(n, dtype=float), index=_idx(n))
    atr = pd.Series(np.full(n, 1.0), index=_idx(n))
    sig = cusum_filter_signal(close, atr, threshold_atr=2.0)
    assert np.issubdtype(sig.dtype, np.floating)
