"""Shared pytest fixtures for QuantHack pipeline tests."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synth_ohlcv() -> pd.DataFrame:
    """500 bars of synthetic D1 OHLCV with realistic structure."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2010-01-04", periods=n, freq="B", tz="UTC")
    log_returns = rng.normal(0.0001, 0.012, size=n)
    close = 1000.0 * np.exp(np.cumsum(log_returns))
    spread = np.abs(rng.normal(0.0, 0.008, size=n)) * close
    high = close + spread
    low = close - spread
    open_ = np.concatenate([[close[0]], close[:-1] * (1 + rng.normal(0, 0.002, size=n - 1))])
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    volume = rng.integers(50_000, 500_000, size=n).astype(float)
    return pd.DataFrame(
        {"time": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


@pytest.fixture
def tmp_csv(tmp_path, synth_ohlcv):
    """synth_ohlcv written to a temp CSV, returns the path."""
    p = tmp_path / "test_ohlcv.csv"
    synth_ohlcv.to_csv(p, index=False)
    return p


@pytest.fixture
def synth_ohlcv_long() -> pd.DataFrame:
    """9000 bars of synthetic daily OHLCV — long enough to clear the 1260-bar
    D1 regime burn-in (5y vol-percentile window) and produce labeled regimes."""
    rng = np.random.default_rng(7)
    n = 9000
    dates = pd.date_range("1995-01-02", periods=n, freq="B", tz="UTC")
    # Mild regime drift so trend/vol axes both flip across the series.
    drift = 0.0002 * np.sin(np.linspace(0, 12 * np.pi, n))
    vol = 0.010 + 0.006 * (np.sin(np.linspace(0, 5 * np.pi, n)) > 0)
    log_returns = rng.normal(0, 1, size=n) * vol + drift
    close = 1000.0 * np.exp(np.cumsum(log_returns))
    spread = np.abs(rng.normal(0.0, 0.006, size=n)) * close
    open_ = np.concatenate([[close[0]], close[:-1] * (1 + rng.normal(0, 0.002, size=n - 1))])
    high = np.maximum.reduce([close + spread, open_, close])
    low = np.minimum.reduce([close - spread, open_, close])
    volume = rng.integers(50_000, 500_000, size=n).astype(float)
    return pd.DataFrame(
        {"time": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


@pytest.fixture
def tmp_csv_long(tmp_path, synth_ohlcv_long) -> "Path":
    """synth_ohlcv_long written to a temp CSV, returns the path."""
    from pathlib import Path
    p = tmp_path / "LONGTEST_D1.csv"
    synth_ohlcv_long.to_csv(p, index=False)
    return p
