"""Tests for sentiment_features module (Tier 1 Phase 2).

Invariants:
  1. load_fear_greed_index returns DatetimeIndex'd DF with fgi_value col
  2. build_sentiment_features shifts FGI by 1 bar (look-ahead invariant)
  3. build_sentiment_features handles missing cache gracefully (NaN fill)
  4. Non-crypto assets get NaN columns (FGI not applicable)
  5. Output index aligns exactly with target primary_index
  6. Multiple calls are idempotent (no state leakage)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pipeline.sentiment_features import build_sentiment_features, load_fear_greed_index


def _h4_idx(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="4h", tz="UTC")


def _daily_idx(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D", tz="UTC")


def _write_fgi_parquet(tmp_path: Path, values: list[int], start: str = "2024-01-01") -> Path:
    cache_dir = tmp_path / "sentiment"
    cache_dir.mkdir(parents=True, exist_ok=True)
    idx = _daily_idx(start, len(values))
    df = pd.DataFrame({"fgi_value": values}, index=idx)
    df.index.name = "time"
    df.to_parquet(cache_dir / "fear_greed.parquet")
    return cache_dir


def test_load_fear_greed_index_schema(tmp_path):
    cache_dir = _write_fgi_parquet(tmp_path, [25, 50, 75])
    df = load_fear_greed_index(cache_dir)
    assert "fgi_value" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None


def test_load_fear_greed_index_raises_on_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fear_greed_index(tmp_path / "nonexistent")


def test_build_sentiment_features_shifts_fgi_by_one_bar(tmp_path):
    """FGI value stamped at day t must NOT appear in H4 bars on day t.
    Only day-(t+1) bars can use day-t's FGI value (shift(1) discipline)."""
    cache_dir = _write_fgi_parquet(
        tmp_path,
        # Strong contrast: day 0 = 99 (greed), day 1 = 1 (fear)
        values=[99, 1, 50],
        start="2024-06-01",
    )
    # Target H4 bars on day 0 (2024-06-01) must NOT see the value 99 — must
    # see the SHIFTED value (NaN, since there's no day -1).
    target_idx = _h4_idx("2024-06-01", 6)   # 6 H4 bars = 1 day
    feats = build_sentiment_features(target_idx, cache_dir, "crypto")
    # All bars on day 0 should be NaN (shift(1) → can't see day 0's own value)
    assert feats["sent__fgi_value"].isna().all(), (
        f"Day-0 bars must not see day-0 FGI; got {feats['sent__fgi_value'].tolist()}"
    )

    # Now bars on day 1 (2024-06-02) should see day 0's value (99)
    target_idx_day1 = _h4_idx("2024-06-02", 6)
    feats_d1 = build_sentiment_features(target_idx_day1, cache_dir, "crypto")
    assert (feats_d1["sent__fgi_value"] == 99).all(), (
        f"Day-1 bars must see day-0 FGI=99; got {feats_d1['sent__fgi_value'].tolist()}"
    )


def test_build_sentiment_features_handles_missing_cache(tmp_path):
    """If FGI parquet is missing, emit NaN columns (no exception)."""
    target_idx = _h4_idx("2024-01-01", 20)
    feats = build_sentiment_features(target_idx, tmp_path / "empty", "crypto")
    assert feats.index.equals(target_idx)
    assert feats["sent__fgi_value"].isna().all()
    assert feats["sent__fgi_pct_chg_5d"].isna().all()


def test_build_sentiment_features_nan_for_non_crypto(tmp_path):
    """FX and metal assets get NaN FGI columns (only crypto is meaningful)."""
    cache_dir = _write_fgi_parquet(tmp_path, [25, 50, 75, 50, 25])
    target_idx = _h4_idx("2024-01-03", 6)
    for asset_class in ("fx", "metal"):
        feats = build_sentiment_features(target_idx, cache_dir, asset_class)
        assert feats["sent__fgi_value"].isna().all(), (
            f"{asset_class} should have NaN FGI"
        )


def test_build_sentiment_features_index_alignment(tmp_path):
    cache_dir = _write_fgi_parquet(tmp_path, list(range(50)))
    target_idx = _h4_idx("2024-01-15", 30)
    feats = build_sentiment_features(target_idx, cache_dir, "crypto")
    assert feats.index.equals(target_idx)
    assert len(feats) == len(target_idx)


def test_build_sentiment_features_idempotent(tmp_path):
    """Multiple calls produce identical results."""
    cache_dir = _write_fgi_parquet(tmp_path, [30, 60, 90, 40, 20])
    target_idx = _h4_idx("2024-01-03", 12)
    feats_a = build_sentiment_features(target_idx, cache_dir, "crypto")
    feats_b = build_sentiment_features(target_idx, cache_dir, "crypto")
    pd.testing.assert_frame_equal(feats_a, feats_b)
