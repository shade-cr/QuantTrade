"""Tests for pipeline.alt_data.gdelt_tone.load_gdelt_tone (B0015c).

Mirrors test_alt_data_gld_holdings.py structure since the PIT semantic is
identical (calendar-day shift before reindex+ffill).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.alt_data.gdelt_tone import (
    GdeltToneCacheMissing,
    load_gdelt_tone,
)


def _write_synth_cache(tmp_path: Path, dates: list[str], tones: list[float]) -> Path:
    idx = pd.to_datetime(dates, utc=True)
    df = pd.DataFrame({"tone": tones}, index=idx)
    cache_path = tmp_path / "gdelt_tone_test.parquet"
    df.to_parquet(cache_path)
    return cache_path


def test_returns_target_index_with_tone_column(tmp_path):
    cache = _write_synth_cache(
        tmp_path,
        ["2024-01-01", "2024-01-02", "2024-01-03"],
        [-1.5, -1.2, -0.8],
    )
    target_idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    out = load_gdelt_tone(target_idx, cache_path=cache)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["tone"]
    assert out.index.equals(target_idx)


def test_publication_lag_no_same_day_leak(tmp_path):
    """Tone stamped at date t is visible at date t+1, NOT t."""
    cache = _write_synth_cache(
        tmp_path,
        ["2024-01-01", "2024-01-02", "2024-01-03"],
        [-3.0, +2.0, -1.0],
    )
    target_idx = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    out = load_gdelt_tone(target_idx, cache_path=cache)
    # 2024-01-01 bar: no prior stamp -> NaN
    assert pd.isna(out.loc[pd.Timestamp("2024-01-01", tz="UTC"), "tone"])
    # 2024-01-02 bar: sees Jan-01 (-3.0)
    assert out.loc[pd.Timestamp("2024-01-02", tz="UTC"), "tone"] == -3.0
    # 2024-01-03 bar: sees Jan-02 (+2.0)
    assert out.loc[pd.Timestamp("2024-01-03", tz="UTC"), "tone"] == 2.0


def test_ffill_holds_across_gap(tmp_path):
    """When cache has gaps, target bars between fill from the prior shifted stamp."""
    cache = _write_synth_cache(
        tmp_path,
        ["2024-01-05", "2024-01-08"],
        [-2.0, -1.0],
    )
    target_idx = pd.date_range("2024-01-06", periods=4, freq="D", tz="UTC")
    out = load_gdelt_tone(target_idx, cache_path=cache)
    # 2024-01-06 sees Jan-05 (-2.0)
    assert out.loc[pd.Timestamp("2024-01-06", tz="UTC"), "tone"] == -2.0
    # 2024-01-07 sees Jan-05 still
    assert out.loc[pd.Timestamp("2024-01-07", tz="UTC"), "tone"] == -2.0
    # 2024-01-08 still sees Jan-05 (Jan-08's value not visible till Jan-09)
    assert out.loc[pd.Timestamp("2024-01-08", tz="UTC"), "tone"] == -2.0
    # 2024-01-09 sees Jan-08 (-1.0)
    assert out.loc[pd.Timestamp("2024-01-09", tz="UTC"), "tone"] == -1.0


def test_pre_coverage_returns_nan(tmp_path):
    """Bars before the first cache row return NaN."""
    cache = _write_synth_cache(
        tmp_path, ["2024-01-15"], [-1.0],
    )
    target_idx = pd.date_range("2024-01-10", periods=3, freq="D", tz="UTC")
    out = load_gdelt_tone(target_idx, cache_path=cache)
    assert out["tone"].isna().all()


def test_missing_cache_raises_helpful(tmp_path):
    target_idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    with pytest.raises(GdeltToneCacheMissing, match="ingest_gdelt_tone"):
        load_gdelt_tone(target_idx, cache_path=tmp_path / "missing.parquet")


def test_empty_target_index_returns_empty(tmp_path):
    cache = _write_synth_cache(tmp_path, ["2024-01-01"], [-1.0])
    out = load_gdelt_tone(pd.DatetimeIndex([]), cache_path=cache)
    assert len(out) == 0
    assert list(out.columns) == ["tone"]
