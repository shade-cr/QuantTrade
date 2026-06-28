"""Tests for intraday macro fetcher (Phase 2 T2).

Tests are network-free: they monkeypatch `_make_yf_ticker` to return a
fake Ticker whose `.history(...)` returns a pre-built DataFrame. This
exercises the cache logic, the alignment math, and the stale-flag gate
without hitting Yahoo.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

import pipeline.macro_fetch_intraday as mfi
from pipeline.macro_fetch_intraday import (
    IntradayFetchError,
    TICKER_DXY,
    TICKER_VIX,
    fetch_intraday_series,
    get_macro_value_at_bar,
    build_intraday_macro_frame,
    build_intraday_macro_frame_with_daily_fallback,
)


# ---------------------------------------------------------------------------
# Fake Ticker
# ---------------------------------------------------------------------------

class _FakeTicker:
    """Minimal yfinance.Ticker stand-in for tests."""

    call_count = 0  # class-level counter so the cache-hit test can assert

    def __init__(self, symbol: str, df: pd.DataFrame | None = None):
        self.symbol = symbol
        self._df = df

    def history(self, start, end, interval, auto_adjust=False, **kwargs):
        _FakeTicker.call_count += 1
        if self._df is None:
            return pd.DataFrame()  # empty → triggers error path
        # yfinance returns a DataFrame with Close column and tz-aware index
        return self._df.loc[start:end] if not self._df.empty else self._df


def _make_intraday_history(n: int = 100, freq: str = "4h", start: str = "2024-01-01") -> pd.DataFrame:
    """Synthetic OHLCV from yfinance.history() with realistic structure."""
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = 100.0 + np.cumsum(np.random.default_rng(0).standard_normal(n) * 0.5)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.001,
            "Low": close * 0.998,
            "Close": close,
            "Volume": np.zeros(n),
        },
        index=idx,
    )


@pytest.fixture(autouse=True)
def _reset_fake_ticker_counter():
    _FakeTicker.call_count = 0
    yield


# ---------------------------------------------------------------------------
# fetch_intraday_series: cache + network
# ---------------------------------------------------------------------------

def test_fetch_writes_cache_and_returns_close_series(monkeypatch, tmp_path):
    df = _make_intraday_history(n=100)
    monkeypatch.setattr(mfi, "_make_yf_ticker", lambda symbol: _FakeTicker(symbol, df))
    result = fetch_intraday_series(
        "DX-Y.NYB",
        start="2024-01-01", end="2024-01-20",
        interval="4h", cache_dir=tmp_path,
    )
    assert isinstance(result, pd.Series)
    assert result.name == "close"
    assert len(result) > 0
    assert (result > 0).all()
    # Cache file written
    assert any(tmp_path.glob("*.parquet")), "cache parquet not written"


def test_fetch_uses_cache_on_second_call(monkeypatch, tmp_path):
    """A second fetch for the same range must NOT hit `_make_yf_ticker`."""
    df = _make_intraday_history(n=500, start="2023-01-01")
    monkeypatch.setattr(mfi, "_make_yf_ticker", lambda symbol: _FakeTicker(symbol, df))

    r1 = fetch_intraday_series(
        "DX-Y.NYB",
        start="2023-01-01", end="2023-04-01",
        interval="4h", cache_dir=tmp_path,
    )
    calls_after_first = _FakeTicker.call_count
    # Second call within the cached range
    r2 = fetch_intraday_series(
        "DX-Y.NYB",
        start="2023-01-15", end="2023-03-15",
        interval="4h", cache_dir=tmp_path,
    )
    calls_after_second = _FakeTicker.call_count
    assert calls_after_second == calls_after_first, (
        f"cache hit should skip _make_yf_ticker; "
        f"calls increased from {calls_after_first} to {calls_after_second}"
    )
    # r2 should be a subset of r1's index range
    assert r2.index.min() >= pd.Timestamp("2023-01-15", tz="UTC")


def test_fetch_raises_when_yfinance_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(mfi, "_make_yf_ticker", lambda symbol: _FakeTicker(symbol, None))
    with pytest.raises(IntradayFetchError):
        fetch_intraday_series(
            "DX-Y.NYB",
            start="2024-01-01", end="2024-01-20",
            interval="4h", cache_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# get_macro_value_at_bar: alignment without look-ahead
# ---------------------------------------------------------------------------

def test_get_macro_value_returns_last_known_strictly_before():
    """Macro at t=10:00 must use the macro stamp ≤ 09:59. The function
    uses `searchsorted(side='right') - 1`, which finds the last index
    strictly before bar_time_utc."""
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01 00:00", periods=5, freq="h", tz="UTC"))
    macro = pd.Series([10, 20, 30, 40, 50], index=idx, dtype=float)
    # Bar at 02:30 UTC (mid-bar) — should use 02:00 (value 30)
    bar_time = pd.Timestamp("2024-01-01 02:30:00", tz="UTC")
    value, hours_ago = get_macro_value_at_bar(macro, bar_time)
    assert value == 30.0, f"expected 30 (value at 02:00), got {value}"
    assert hours_ago == pytest.approx(0.5), f"expected 0.5h since update, got {hours_ago}"


def test_get_macro_value_returns_nan_when_no_prior_data():
    """If bar_time_utc precedes the first macro stamp, return (NaN, inf)."""
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01 12:00", periods=5, freq="h", tz="UTC"))
    macro = pd.Series([10, 20, 30, 40, 50], index=idx, dtype=float)
    bar_time = pd.Timestamp("2024-01-01 06:00:00", tz="UTC")
    value, hours_ago = get_macro_value_at_bar(macro, bar_time)
    assert np.isnan(value)
    assert hours_ago == float("inf")


def test_get_macro_value_at_exact_stamp_uses_strict_less_than():
    """At exactly t = macro_stamp_t, the function must NOT use that stamp
    (would be lookahead since the stamp publishes AT t). It uses the
    stamp BEFORE."""
    idx = pd.DatetimeIndex(pd.date_range("2024-01-01 00:00", periods=5, freq="h", tz="UTC"))
    macro = pd.Series([10, 20, 30, 40, 50], index=idx, dtype=float)
    # Bar at exactly 03:00 — must use 02:00 stamp (value 30)
    bar_time = pd.Timestamp("2024-01-01 03:00:00", tz="UTC")
    value, hours_ago = get_macro_value_at_bar(macro, bar_time)
    assert value == 30.0, f"expected 30 (stamp at 02:00), got {value} (would be lookahead if 40)"


# ---------------------------------------------------------------------------
# build_intraday_macro_frame: end-to-end
# ---------------------------------------------------------------------------

def test_build_frame_returns_dxy_and_vix_columns(monkeypatch, tmp_path):
    dxy_df = _make_intraday_history(n=200, start="2024-01-01")
    vix_df = _make_intraday_history(n=200, start="2024-01-01")

    def fake_ticker(symbol):
        if symbol == TICKER_DXY:
            return _FakeTicker(symbol, dxy_df)
        if symbol == TICKER_VIX:
            return _FakeTicker(symbol, vix_df)
        return _FakeTicker(symbol, None)

    monkeypatch.setattr(mfi, "_make_yf_ticker", fake_ticker)

    bar_index = pd.date_range("2024-01-05 00:00", periods=24, freq="4h", tz="UTC")
    frame = build_intraday_macro_frame(
        bar_index, start="2024-01-01", end="2024-02-01",
        cache_dir=tmp_path,
    )
    assert {"dxy", "vix", "dxy_stale", "vix_stale"} <= set(frame.columns)
    assert frame.index.equals(bar_index)


def test_stale_flag_set_when_hours_since_update_exceeds_threshold(monkeypatch, tmp_path):
    """If a bar's last-known macro value is > 24h old, dxy_stale must be 1."""
    # Macro stamps every 4h for the first 10 bars, then a 100h gap.
    macro_idx_early = pd.date_range("2024-01-01 00:00", periods=10, freq="4h", tz="UTC")
    last_early = macro_idx_early[-1]
    macro_idx_late = pd.DatetimeIndex([last_early + pd.Timedelta(hours=100)])
    macro_idx = macro_idx_early.append(macro_idx_late)
    n = len(macro_idx)
    dxy_df = pd.DataFrame(
        {
            "Open": np.linspace(100, 110, n),
            "High": np.linspace(100, 110, n),
            "Low": np.linspace(100, 110, n),
            "Close": np.linspace(100, 110, n),
            "Volume": np.zeros(n),
        },
        index=macro_idx,
    )
    vix_df = dxy_df.copy()

    def fake_ticker(symbol):
        return _FakeTicker(symbol, dxy_df if symbol == TICKER_DXY else vix_df)

    monkeypatch.setattr(mfi, "_make_yf_ticker", fake_ticker)

    # Bar 50h after the last "early" macro stamp — well into the gap.
    bar_index = pd.DatetimeIndex([last_early + pd.Timedelta(hours=50)])
    frame = build_intraday_macro_frame(
        bar_index, start="2024-01-01", end="2024-02-01",
        cache_dir=tmp_path,
    )
    assert frame["dxy_stale"].iloc[0] == 1.0, (
        f"dxy_stale should be 1 when last update is 50h ago, got {frame['dxy_stale'].iloc[0]}"
    )


def test_stale_flag_zero_when_recent_update(monkeypatch, tmp_path):
    """If a bar's last-known macro is within 24h, dxy_stale must be 0."""
    dxy_df = _make_intraday_history(n=200, start="2024-01-01")
    vix_df = _make_intraday_history(n=200, start="2024-01-01")

    def fake_ticker(symbol):
        return _FakeTicker(symbol, dxy_df if symbol == TICKER_DXY else vix_df)

    monkeypatch.setattr(mfi, "_make_yf_ticker", fake_ticker)

    # Bar shortly after a known macro stamp.
    bar_index = pd.date_range("2024-01-05 02:00", periods=4, freq="4h", tz="UTC")
    frame = build_intraday_macro_frame(
        bar_index, start="2024-01-01", end="2024-02-01",
        cache_dir=tmp_path,
    )
    # All these bars are well within 24h of the previous macro stamp.
    assert (frame["dxy_stale"] == 0.0).all(), (
        f"dxy_stale should be 0 for recent bars, got {frame['dxy_stale'].tolist()}"
    )


# ---------------------------------------------------------------------------
# build_intraday_macro_frame_with_daily_fallback: covers the >730-day case
# ---------------------------------------------------------------------------

def _daily_macro_frame(start: str, end: str) -> pd.DataFrame:
    """Mimic pipeline.macro_fetch.build_macro_frame output (DTWEXBGS + VIXCLS
    columns, daily UTC index, already shifted by 1 day)."""
    idx = pd.date_range(start, end, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "DTWEXBGS": np.linspace(100, 110, len(idx)),
            "DFII5": np.linspace(0.5, 1.5, len(idx)),
            "VIXCLS": np.linspace(15, 25, len(idx)),
        },
        index=idx,
    )


def test_daily_fallback_fills_dxy_when_intraday_empty(monkeypatch, tmp_path):
    """When yfinance returns no data (typical for ranges > 730 days), the
    fallback path fills dxy + vix from the daily FRED frame and marks
    every fallback row as stale=1."""
    monkeypatch.setattr(mfi, "_make_yf_ticker", lambda symbol: _FakeTicker(symbol, None))

    bar_index = pd.date_range("2021-01-01", periods=24, freq="4h", tz="UTC")
    daily = _daily_macro_frame("2020-12-01", "2021-02-01")

    frame = build_intraday_macro_frame_with_daily_fallback(
        bar_index, start="2021-01-01", end="2021-01-05",
        daily_macro_frame=daily, cache_dir=tmp_path,
    )

    # All dxy + vix values must be filled from daily (none NaN), and stale=1.
    assert frame["dxy"].notna().all(), "fallback should fill all dxy values from daily"
    assert frame["vix"].notna().all(), "fallback should fill all vix values from daily"
    assert (frame["dxy_stale"] == 1.0).all(), "daily-sourced rows must be stale=1"
    assert (frame["vix_stale"] == 1.0).all()
    # Values must match DTWEXBGS / VIXCLS forward-fill at each bar time
    # (the last daily stamp strictly before each H4 bar).
    expected_dxy_first = float(daily["DTWEXBGS"].loc[:bar_index[0]].iloc[-2])  # strict-before
    # `get_macro_value_at_bar` uses side='left' - 1 → strictly before bar_index[0]
    # which is 2021-01-01 00:00 UTC. The last daily stamp strictly before is
    # 2020-12-31 (daily series at midnight UTC by convention).
    assert frame["dxy"].iloc[0] == pytest.approx(expected_dxy_first, abs=1e-9)


def test_daily_fallback_preserves_intraday_when_available(monkeypatch, tmp_path):
    """When yfinance DOES return data for some bars, those bars must NOT
    be overwritten by the daily fallback — the intraday values win."""
    # Intraday covers 2024-01-01 to 2024-01-10.
    intraday_df = _make_intraday_history(n=60, start="2024-01-01")
    monkeypatch.setattr(mfi, "_make_yf_ticker", lambda symbol: _FakeTicker(symbol, intraday_df))

    bar_index = pd.date_range("2024-01-02", periods=24, freq="4h", tz="UTC")
    daily = _daily_macro_frame("2023-12-01", "2024-02-01")

    frame = build_intraday_macro_frame_with_daily_fallback(
        bar_index, start="2024-01-01", end="2024-01-10",
        daily_macro_frame=daily, cache_dir=tmp_path,
    )

    # Values should be from intraday (Close prices ≈ 100), NOT daily (linspace 100-110).
    # Sample a few bars: intraday should be < 120 (within Close range) and != linspace value.
    daily_at_bar = daily["DTWEXBGS"].iloc[10]
    intraday_at_bar = frame["dxy"].iloc[5]  # mid-range bar — should be from intraday
    assert intraday_at_bar != daily_at_bar, (
        f"intraday should win when available; got {intraday_at_bar} vs daily {daily_at_bar}"
    )
    assert frame["dxy_stale"].iloc[5] == 0.0, "intraday-sourced bar should have stale=0"
