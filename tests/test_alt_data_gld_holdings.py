"""Tests for pipeline.alt_data.gld_holdings.load_gld_holdings (B0015a).

Invariants:
  1. Returns DataFrame indexed to target_index with column gld_oz_held.
  2. Publication-lag: value stamped at SPDR cache date t is visible ONLY at
     market bars >= t+1. A bar at the SAME date t must NOT see that value.
  3. Pre-GLD-inception (before 2004-11-18) bars have NaN.
  4. Missing cache raises GldHoldingsCacheMissing with a helpful message.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.alt_data.gld_holdings import (
    GldHoldingsCacheMissing,
    load_gld_holdings,
)


def _write_synth_cache(tmp_path: Path, rows: dict[str, list]) -> Path:
    """Write a synthetic gld_holdings.parquet at tmp_path."""
    idx = pd.to_datetime(rows["dates"], utc=True)
    df = pd.DataFrame({"gld_oz_held": rows["oz"]}, index=idx)
    if "aum" in rows:
        df["gld_aum_usd"] = rows["aum"]
    cache_path = tmp_path / "gld_holdings.parquet"
    df.to_parquet(cache_path)
    return cache_path


def test_returns_target_index_with_gld_oz_held_column(tmp_path):
    cache = _write_synth_cache(tmp_path, {
        "dates": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "oz": [1000.0, 1010.0, 1020.0, 1015.0],
    })
    target_idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    out = load_gld_holdings(target_idx, cache_path=cache)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["gld_oz_held"]
    assert out.index.equals(target_idx)


def test_publication_lag_no_leak_same_day(tmp_path):
    """A market bar at calendar date t must NOT see the holdings stamped at t."""
    cache = _write_synth_cache(tmp_path, {
        "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "oz": [1000.0, 2000.0, 3000.0],
    })
    target_idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    out = load_gld_holdings(target_idx, cache_path=cache)
    # 2024-01-01 bar: the cache stamps the FIRST observation at 2024-01-01,
    # but a same-day bar must NOT see it. After .shift(1) on the cache,
    # 2024-01-01 has the PRIOR value (NaN here since 2024-01-01 is the first cache row).
    assert pd.isna(out.loc[pd.Timestamp("2024-01-01", tz="UTC"), "gld_oz_held"])
    # 2024-01-02 bar: sees the .shift(1) of cache, which is the 2024-01-01 value = 1000.
    assert out.loc[pd.Timestamp("2024-01-02", tz="UTC"), "gld_oz_held"] == 1000.0
    # 2024-01-03 bar: sees the 2024-01-02 value = 2000.
    assert out.loc[pd.Timestamp("2024-01-03", tz="UTC"), "gld_oz_held"] == 2000.0


def test_ffill_holds_holdings_across_weekend_gap(tmp_path):
    """SPDR cache has only trading days; market index may have a weekend gap.
    Target bars between cache stamps must ffill the latest visible (post-.shift)
    value."""
    cache = _write_synth_cache(tmp_path, {
        "dates": ["2024-01-05", "2024-01-08"],  # Friday + Monday
        "oz": [1000.0, 1100.0],
    })
    # Target with Saturday 2024-01-06 in between
    target_idx = pd.date_range("2024-01-06", periods=4, freq="D", tz="UTC")
    out = load_gld_holdings(target_idx, cache_path=cache)
    # 2024-01-06 Saturday: after .shift(1), Friday's value (1000) is visible.
    # ffill from latest <= Saturday = the .shift'd Friday value = 1000.
    assert out.loc[pd.Timestamp("2024-01-06", tz="UTC"), "gld_oz_held"] == 1000.0
    # 2024-01-07 Sunday: same value (no new cache stamp).
    assert out.loc[pd.Timestamp("2024-01-07", tz="UTC"), "gld_oz_held"] == 1000.0
    # 2024-01-08 Monday: cache row at 2024-01-08 after .shift(1) becomes
    # visible at 2024-01-09; so Monday still sees Friday's 1000.
    assert out.loc[pd.Timestamp("2024-01-08", tz="UTC"), "gld_oz_held"] == 1000.0
    # 2024-01-09 Tuesday: sees 1100 (Monday's value, post-shift).
    assert out.loc[pd.Timestamp("2024-01-09", tz="UTC"), "gld_oz_held"] == 1100.0


def test_pre_inception_returns_nan(tmp_path):
    """Bars before the first cache row return NaN."""
    cache = _write_synth_cache(tmp_path, {
        "dates": ["2024-01-15", "2024-01-16"],
        "oz": [1000.0, 1010.0],
    })
    target_idx = pd.date_range("2024-01-10", periods=3, freq="D", tz="UTC")
    out = load_gld_holdings(target_idx, cache_path=cache)
    assert out["gld_oz_held"].isna().all()


def test_missing_cache_raises_with_helpful_message(tmp_path):
    """Missing parquet raises GldHoldingsCacheMissing with run command hint."""
    target_idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    with pytest.raises(GldHoldingsCacheMissing, match="ingest_gld_holdings"):
        load_gld_holdings(target_idx, cache_path=tmp_path / "does_not_exist.parquet")


def test_empty_target_index_returns_empty(tmp_path):
    cache = _write_synth_cache(tmp_path, {
        "dates": ["2024-01-01"],
        "oz": [1000.0],
    })
    out = load_gld_holdings(pd.DatetimeIndex([]), cache_path=cache)
    assert len(out) == 0
    assert list(out.columns) == ["gld_oz_held"]


def test_real_cache_loads_correctly_if_present():
    """If the real cache exists, load 100 bars and verify schema + coverage.

    This is a thin smoke test that confirms the ingest script's output is
    consumable. Skipped if the cache is absent (CI without the ingest step)."""
    cache = Path("cache/alt_data/gld_holdings.parquet")
    if not cache.exists():
        pytest.skip("No real GLD holdings cache present; run scripts/ingest_gld_holdings.py first")
    target_idx = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    out = load_gld_holdings(target_idx, cache_path=cache)
    assert len(out) == 100
    assert "gld_oz_held" in out.columns
    # Should have substantial non-NaN coverage in 2020-Q1 (well past GLD 2004 inception)
    assert out["gld_oz_held"].notna().sum() >= 50, (
        f"Real cache has unexpectedly sparse coverage in 2020-Q1: {out['gld_oz_held'].notna().sum()}/100"
    )
