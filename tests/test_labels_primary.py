"""Tests for pipeline.labels primary signals + primary state derivation."""
from __future__ import annotations
import numpy as np
import pandas as pd

from pipeline.features import build_technical_features
from pipeline.labels import (
    ema_crossover_signal,
    momentum_zscore_signal,
    compute_primary_state,
)


def test_ema_crossover_signal_emits_pm1_in_trend(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    feats = build_technical_features(df)
    sig = ema_crossover_signal(df["close"], feats["_atr_14"], fast=20, slow=50, dead_zone_atr=0.0)
    assert set(sig.dropna().unique()).issubset({-1.0, 1.0})
    assert (sig == 1).any() and (sig == -1).any()


def test_ema_crossover_dead_zone_skips(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    feats = build_technical_features(df)
    sig = ema_crossover_signal(df["close"], feats["_atr_14"], fast=20, slow=50, dead_zone_atr=100.0)
    assert (sig == 0).sum() > len(sig) * 0.9


def test_momentum_zscore_signal_thresholds():
    idx = pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC")
    # Quadratic log-price (log(close) = t^2): log-returns grow linearly over time,
    # so the 20-bar return climbs well above its rolling mean and (r-mu)/sd > 0.3
    # in the latter half of the series — a genuine acceleration detector test.
    t = np.linspace(0, 3, 300)
    close = pd.Series(np.exp(t**2) * 100, index=idx)
    sig = momentum_zscore_signal(close, lookback=20, threshold=0.3)
    assert (sig.iloc[280:] == 1).all()


def test_bars_since_signal_resets_on_side_change():
    """For side = [+1,+1,+1,-1,-1,+1,+1] bars_since must be [0,1,2,0,1,0,1]."""
    idx = pd.date_range("2020-01-01", periods=7, freq="B", tz="UTC")
    side = pd.Series([1, 1, 1, -1, -1, 1, 1], index=idx)
    state = compute_primary_state(side, cap=60)
    np.testing.assert_array_equal(state["bars_since_signal"].values, [0, 1, 2, 0, 1, 0, 1])
    np.testing.assert_array_equal(state["primary_side"].values, [1, 1, 1, -1, -1, 1, 1])


def test_bars_since_signal_caps_at_60():
    idx = pd.date_range("2020-01-01", periods=100, freq="B", tz="UTC")
    side = pd.Series([1] * 100, index=idx)
    state = compute_primary_state(side, cap=60)
    assert state["bars_since_signal"].max() == 60
    assert state["bars_since_signal"].iloc[60:].eq(60).all()


def test_compute_primary_state_only_nonzero():
    """`valid` (events) only contains rows where side != 0; the function must accept
    an already-filtered series and treat side changes (incl. -1→+1 through implicit 0) correctly."""
    idx = pd.date_range("2020-01-01", periods=4, freq="B", tz="UTC")
    side = pd.Series([1, 1, -1, -1], index=idx)
    state = compute_primary_state(side, cap=60)
    np.testing.assert_array_equal(state["bars_since_signal"].values, [0, 1, 0, 1])


def test_compute_primary_state_handles_empty_series():
    """An empty side series (no primary signals at all) must not crash."""
    side = pd.Series([], dtype=int, index=pd.DatetimeIndex([], tz="UTC"))
    state = compute_primary_state(side, cap=60)
    assert state.empty
    assert list(state.columns) == ["primary_side", "bars_since_signal"]
