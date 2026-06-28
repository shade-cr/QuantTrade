"""Tests for the triple-barrier label producer."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from pipeline.labels import triple_barrier_labels, TripleBarrierEvent


def _make_ohlc(close_path: list[float], spread: float = 0.0):
    n = len(close_path)
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    c = pd.Series(close_path, index=idx)
    h = c + spread
    l = c - spread
    return pd.DataFrame({"open": c.shift(1).fillna(c.iloc[0]), "high": h, "low": l, "close": c})


def test_monotonic_up_all_longs_label_1():
    ohlc = _make_ohlc(list(np.linspace(100.0, 200.0, 100)))
    atr = pd.Series(1.0, index=ohlc.index)
    events = pd.DataFrame({"side": [1] * 50}, index=ohlc.index[:50])
    out = triple_barrier_labels(ohlc, events, atr, horizon=20, tp_mult=2.0, sl_mult=1.0)
    assert (out["label"] == 1).all()


def test_monotonic_down_all_longs_label_0():
    ohlc = _make_ohlc(list(np.linspace(200.0, 100.0, 100)))
    atr = pd.Series(1.0, index=ohlc.index)
    events = pd.DataFrame({"side": [1] * 50}, index=ohlc.index[:50])
    out = triple_barrier_labels(ohlc, events, atr, horizon=20, tp_mult=2.0, sl_mult=1.0)
    assert (out["label"] == 0).all()


def test_short_asymmetry_monotonic_down_all_shorts_label_1():
    """Critical: shorts must use low <= TP_short. A single < instead of <= for shorts
    (or wrong comparator) silences this entire test."""
    ohlc = _make_ohlc(list(np.linspace(200.0, 100.0, 100)))
    atr = pd.Series(1.0, index=ohlc.index)
    events = pd.DataFrame({"side": [-1] * 50}, index=ohlc.index[:50])
    out = triple_barrier_labels(ohlc, events, atr, horizon=20, tp_mult=2.0, sl_mult=1.0)
    assert (out["label"] == 1).all(), "Shorts mislabeled in a monotonic-down market"


def test_no_lookahead_beyond_horizon():
    """A change at t+horizon+1 must not affect the label at t."""
    base = list(np.linspace(100.0, 110.0, 100))
    ohlc_a = _make_ohlc(base)
    base_b = base.copy()
    base_b[50 + 21] = 1e9  # extreme spike one bar PAST horizon for event at index 50
    ohlc_b = _make_ohlc(base_b)
    atr = pd.Series(0.2, index=ohlc_a.index)
    events = pd.DataFrame({"side": [1]}, index=[ohlc_a.index[50]])
    out_a = triple_barrier_labels(ohlc_a, events, atr, horizon=20, tp_mult=2.0, sl_mult=1.0)
    out_b = triple_barrier_labels(ohlc_b, events, atr, horizon=20, tp_mult=2.0, sl_mult=1.0)
    assert out_a.loc[ohlc_a.index[50], "label"] == out_b.loc[ohlc_a.index[50], "label"]


def test_tp_and_sl_in_same_bar_is_label_0():
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    close = pd.Series([100.0] * n, index=idx)
    # On bar 1, the bar's range engulfs both TP (102) and SL (99) for a long entered at bar 0.
    high = close.copy(); high.iloc[1] = 103.0
    low = close.copy(); low.iloc[1] = 98.0
    ohlc = pd.DataFrame({"open": close, "high": high, "low": low, "close": close})
    atr = pd.Series(1.0, index=idx)
    events = pd.DataFrame({"side": [1]}, index=[idx[0]])
    out = triple_barrier_labels(ohlc, events, atr, horizon=10, tp_mult=2.0, sl_mult=1.0)
    assert out.loc[idx[0], "label"] == 0  # conservative tie → SL wins


def test_outcome_window_stored_for_sample_weights():
    """triple_barrier_labels must record the END index of the outcome window per event;
    sample_weights.avg_uniqueness uses this to compute label overlap."""
    ohlc = _make_ohlc(list(np.linspace(100.0, 100.0, 50)))  # flat → all timeouts
    atr = pd.Series(1.0, index=ohlc.index)
    events = pd.DataFrame({"side": [1, 1, 1]}, index=[ohlc.index[0], ohlc.index[5], ohlc.index[10]])
    out = triple_barrier_labels(ohlc, events, atr, horizon=20, tp_mult=2.0, sl_mult=1.0)
    assert "t_end_idx" in out.columns
    assert (out["t_end_idx"] - np.arange(len(out)) * 5 - 20 == 0).all()


def test_exit_price_uses_open_on_gap_through_sl():
    """If the next bar gaps DOWN past the SL of a long, exit_price must be the OPEN of
    that bar (slippage realism), not the SL level itself."""
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    close = pd.Series([100.0] * n, index=idx)
    high = close.copy()
    low = close.copy()
    open_ = close.copy()
    # On bar 1, gap-down: open well below SL (entry=100, SL=99 with ATR=1, sl_mult=1).
    open_.iloc[1] = 95.0; high.iloc[1] = 95.5; low.iloc[1] = 94.0
    ohlc = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    atr = pd.Series(1.0, index=idx)
    events = pd.DataFrame({"side": [1]}, index=[idx[0]])
    out = triple_barrier_labels(ohlc, events, atr, horizon=10, tp_mult=2.0, sl_mult=1.0)
    assert out.loc[idx[0], "label"] == 0
    assert out.loc[idx[0], "exit_price"] == 95.0  # open of the gap bar


def test_exit_price_at_barrier_for_intrabar_touch():
    """When TP is touched intrabar (not via gap), exit_price equals the TP level."""
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    close = pd.Series([100.0] * n, index=idx)
    high = close.copy()
    low = close.copy()
    open_ = close.copy()
    high.iloc[5] = 105.0  # touches TP=102 from within
    ohlc = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    atr = pd.Series(1.0, index=idx)
    events = pd.DataFrame({"side": [1]}, index=[idx[0]])
    out = triple_barrier_labels(ohlc, events, atr, horizon=10, tp_mult=2.0, sl_mult=1.0)
    assert out.loc[idx[0], "label"] == 1
    assert out.loc[idx[0], "exit_price"] == 102.0  # TP level
