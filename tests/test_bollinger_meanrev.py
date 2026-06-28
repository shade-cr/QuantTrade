"""Tests for bollinger_meanrev_signal — mean-reversion primary using Bollinger bands.

Signal logic:
  - Compute middle = SMA(close, period), upper/lower = middle +/- k * stdev(close, period)
  - When close touches/crosses BELOW lower band → +1 (oversold, expect mean reversion up)
  - When close touches/crosses ABOVE upper band → -1 (overbought, expect mean reversion down)
  - Otherwise 0

This is the ORTHOGONAL primary to ema_cross/momentum_zscore: it expects price
to revert, not trend. Designed for FX choppy regimes where trend-following dies.

Invariants:
  1. Constant price → no events (no band breach)
  2. Strong uptrend → fires -1 (overbought reversion)
  3. Strong downtrend → fires +1 (oversold reversion)
  4. Output is pd.Series same index, dtype float, values in {-1,0,+1}
  5. Higher k_stdev → fewer events
  6. NaN warm-up period handled (no event before SMA fills)
  7. Symmetric under sign flip of returns
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from pipeline.labels import bollinger_meanrev_signal


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")


def test_bollinger_constant_price_no_events():
    n = 50
    close = pd.Series(np.full(n, 100.0), index=_idx(n))
    sig = bollinger_meanrev_signal(close, period=20, k_stdev=2.0)
    # std=0 → band collapses; no events should fire
    assert (sig == 0.0).all()


def test_bollinger_upward_shock_after_flat_fires_negative():
    """Price flat then sudden spike → upper band breach → fire -1 (overbought revert)."""
    n = 60
    # 40 bars flat at 100, then 20 bars at 120 (sudden +20% spike that exceeds 2*stdev)
    close = pd.Series([100.0] * 40 + [120.0] * 20, index=_idx(n))
    sig = bollinger_meanrev_signal(close, period=20, k_stdev=2.0)
    assert (sig == -1.0).any(), "spike up after flat should fire -1 (mean-rev short)"
    assert (sig == 1.0).sum() == 0, "no longs in pure up shock"


def test_bollinger_downward_shock_after_flat_fires_positive():
    n = 60
    close = pd.Series([100.0] * 40 + [80.0] * 20, index=_idx(n))
    sig = bollinger_meanrev_signal(close, period=20, k_stdev=2.0)
    assert (sig == 1.0).any(), "drop after flat should fire +1 (mean-rev long)"
    assert (sig == -1.0).sum() == 0


def test_bollinger_output_contract():
    n = 100
    rng = np.random.default_rng(0)
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, n)), index=_idx(n))
    sig = bollinger_meanrev_signal(close, period=20, k_stdev=2.0)
    assert isinstance(sig, pd.Series)
    assert sig.index.equals(close.index)
    assert np.issubdtype(sig.dtype, np.floating)
    vals = set(sig.dropna().unique())
    assert vals.issubset({-1.0, 0.0, 1.0}), f"unexpected: {vals}"


def test_bollinger_higher_k_fewer_events():
    n = 200
    rng = np.random.default_rng(7)
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, n)), index=_idx(n))
    sig_tight = bollinger_meanrev_signal(close, period=20, k_stdev=1.0)
    sig_loose = bollinger_meanrev_signal(close, period=20, k_stdev=3.0)
    assert (sig_loose != 0).sum() <= (sig_tight != 0).sum()


def test_bollinger_handles_warmup_nan():
    """No event should fire before period bars filled."""
    n = 30
    close = pd.Series(100.0 + np.arange(n, dtype=float), index=_idx(n))
    sig = bollinger_meanrev_signal(close, period=20, k_stdev=2.0)
    assert (sig.iloc[:19] == 0.0).all(), "events before warm-up should be zero"


def test_bollinger_symmetric_under_sign_flip():
    """Mirror price → mirror signals."""
    n = 80
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.5, n)
    close_up = pd.Series(100.0 + np.cumsum(returns), index=_idx(n))
    close_down = pd.Series(100.0 - np.cumsum(returns), index=_idx(n))
    sig_up = bollinger_meanrev_signal(close_up, period=20, k_stdev=2.0)
    sig_down = bollinger_meanrev_signal(close_down, period=20, k_stdev=2.0)
    # Same event positions, opposite signs
    assert (sig_up != 0).equals(sig_down != 0)
    nz_up = sig_up[sig_up != 0]
    nz_down = sig_down[sig_down != 0]
    np.testing.assert_array_equal(nz_up.values, -nz_down.values)
