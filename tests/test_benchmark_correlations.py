"""B0003 — compute_benchmark_correlations (maxdama §4.10 hidden-beta check)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.reporting import compute_benchmark_correlations


def _daily_index(n: int, tz="UTC") -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="D", tz=tz)


def test_perfect_positive_correlation_recovered():
    idx = _daily_index(120)
    # Benchmark level whose daily returns equal the strategy daily pnl exactly.
    rng = np.random.default_rng(0)
    daily_ret = rng.normal(0, 0.01, size=120)
    level = pd.Series(100 * np.cumprod(1 + daily_ret), index=idx)
    # Strategy pnl == the benchmark's same-day return (so corr should be ~+1).
    pnl = pd.Series(np.r_[np.nan, daily_ret[1:]], index=idx).dropna()
    out = compute_benchmark_correlations(pnl, {"BENCH": level})
    assert out["BENCH"] is not None
    assert out["BENCH"] > 0.99


def test_negative_correlation_flags_riskoff_play():
    idx = _daily_index(120)
    rng = np.random.default_rng(1)
    daily_ret = rng.normal(0, 0.01, size=120)
    level = pd.Series(100 * np.cumprod(1 + daily_ret), index=idx)
    # Strategy pnl == NEGATIVE of benchmark return → corr ~ -1 (hidden risk-off).
    pnl = pd.Series(np.r_[np.nan, -daily_ret[1:]], index=idx).dropna()
    out = compute_benchmark_correlations(pnl, {"BENCH": level})
    assert out["BENCH"] < -0.99


def test_insufficient_overlap_returns_none():
    idx = _daily_index(10)
    pnl = pd.Series(np.arange(10, dtype=float), index=idx)
    level = pd.Series(100 + np.arange(10, dtype=float), index=idx)
    out = compute_benchmark_correlations(pnl, {"BENCH": level}, min_overlap=30)
    assert out["BENCH"] is None


def test_empty_strategy_pnl_returns_none_for_all():
    out = compute_benchmark_correlations(
        pd.Series([], dtype=float, index=pd.DatetimeIndex([], tz="UTC")),
        {"VIX": pd.Series([1.0, 2.0]), "SP500": pd.Series([1.0, 2.0])},
    )
    assert out == {"VIX": None, "SP500": None}


def test_tz_naive_benchmark_is_localized():
    """Benchmark cached as tz-naive (FRED parquet convention) must still align."""
    idx_utc = _daily_index(120, tz="UTC")
    idx_naive = _daily_index(120, tz=None)
    rng = np.random.default_rng(2)
    daily_ret = rng.normal(0, 0.01, size=120)
    level = pd.Series(100 * np.cumprod(1 + daily_ret), index=idx_naive)  # tz-naive
    pnl = pd.Series(np.r_[np.nan, daily_ret[1:]], index=idx_utc).dropna()  # tz-aware
    out = compute_benchmark_correlations(pnl, {"BENCH": level})
    assert out["BENCH"] is not None
    assert out["BENCH"] > 0.99


def test_sparse_event_pnl_aggregated_by_day():
    """Multiple same-day events are summed before correlating."""
    base = _daily_index(120)
    rng = np.random.default_rng(3)
    # Two events on each of the first 60 days, none after — exercises groupby-sum.
    idx = base[:60].append(base[:60])
    pnl = pd.Series(rng.normal(0, 0.001, size=120), index=idx).sort_index()
    level = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, size=120)), index=base)
    out = compute_benchmark_correlations(pnl, {"BENCH": level})
    # 60 overlapping days >= min_overlap; result is a finite float.
    assert out["BENCH"] is not None
    assert np.isfinite(out["BENCH"])
