"""Tests for pipeline.cot_features (Tier 1 Phase 3 — CFTC COT positioning).

Invariants:
  1. fetch_cot_for_asset caches results to parquet under cache_dir.
  2. Non-COT assets (BTC/ETH/SOL) return an empty DataFrame from build_cot_features.
  3. Publication lag: a Tuesday-stamped COT report published Friday must NOT
     be visible to any market bar at time t < that Friday's publication.
     Mondays following must see it.
  4. cot_extreme_long / cot_extreme_short flags trigger when |z| > 2.
  5. 52-week rolling z-score at row N uses rows < N (no look-ahead).
  6. The feature schema columns are stable for COT assets.
"""
from __future__ import annotations
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pipeline.cot_features import (
    ASSET_TO_CFTC_CONTRACT,
    build_cot_features,
    fetch_cot_for_asset,
    COT_FEATURE_COLUMNS,
)


def _synth_weekly_cot(
    start: str = "2022-01-04",
    n_weeks: int = 80,
    net_noncomm_pattern: list[float] | None = None,
    total_oi: float = 500_000.0,
) -> pd.DataFrame:
    """Return a synthetic weekly COT DataFrame with the schema the loader expects.

    `report_date` is Tuesday UTC midnight, weekly cadence.
    """
    tuesdays = pd.date_range(start=start, periods=n_weeks, freq="7D", tz="UTC")
    if net_noncomm_pattern is None:
        # Mild oscillation around zero — non-extreme by default.
        net_noncomm_pattern = list(np.sin(np.linspace(0, 4 * np.pi, n_weeks)) * 10_000.0)
    return pd.DataFrame(
        {
            "report_date": tuesdays,
            "net_noncomm": net_noncomm_pattern,
            "total_oi": [total_oi] * n_weeks,
        }
    )


def test_fetch_cot_caches_to_parquet(tmp_path, monkeypatch):
    """fetch_cot_for_asset must write parquet to cache_dir and read from it on re-fetch."""
    synth = _synth_weekly_cot(n_weeks=20)

    # Monkeypatch the HTTP-bound fetch with a synthesizer.
    call_count = {"n": 0}

    def fake_download(asset, start, end):
        call_count["n"] += 1
        return synth

    monkeypatch.setattr("pipeline.cot_features._download_cot_for_asset", fake_download)

    cache_dir = tmp_path / "cot"
    out = fetch_cot_for_asset(
        "XAUUSD",
        start=pd.Timestamp("2022-01-01", tz="UTC"),
        end=pd.Timestamp("2023-01-01", tz="UTC"),
        cache_dir=str(cache_dir),
    )
    assert "report_date" in out.columns
    assert "net_noncomm" in out.columns
    assert "total_oi" in out.columns
    parquet_files = list(cache_dir.glob("*.parquet"))
    assert len(parquet_files) >= 1, f"expected parquet in {cache_dir}; got {parquet_files}"
    assert call_count["n"] == 1

    # Second call covering the same window — should hit cache, not call downloader.
    out2 = fetch_cot_for_asset(
        "XAUUSD",
        start=pd.Timestamp("2022-01-01", tz="UTC"),
        end=pd.Timestamp("2023-01-01", tz="UTC"),
        cache_dir=str(cache_dir),
    )
    assert call_count["n"] == 1, "downloader called twice despite cache hit"
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), out2.reset_index(drop=True), check_dtype=False
    )


def test_build_cot_features_for_non_cot_asset_returns_empty(tmp_path):
    """BTC/ETH/SOL have no CFTC report under our mapping → empty DataFrame."""
    target_idx = pd.date_range("2023-01-01", periods=50, freq="D", tz="UTC")
    for asset in ("BTCUSD", "ETHUSD", "SOLUSD"):
        feats = build_cot_features(asset, target_idx, cache_dir=str(tmp_path / "cot"))
        assert isinstance(feats, pd.DataFrame)
        assert feats.shape[1] == 0, f"{asset} should produce 0 COT columns; got {feats.columns.tolist()}"
        # Index alignment still upheld so caller can pd.concat safely.
        assert feats.index.equals(target_idx)


def test_publication_lag_no_leak(tmp_path, monkeypatch):
    """A COT report stamped Tue 2024-01-09 is published Fri 2024-01-12.

    - Market bar at Thu 2024-01-11 must NOT see that COT value.
    - Market bar at Mon 2024-01-15 MUST see it.
    """
    # Two reports: Tue 2024-01-02 (published Fri 2024-01-05) and Tue 2024-01-09
    # (published Fri 2024-01-12). Net positions deliberately different so we can
    # detect leakage by value.
    report_dates = pd.to_datetime(["2024-01-02", "2024-01-09"], utc=True)
    synth = pd.DataFrame(
        {
            "report_date": report_dates,
            "net_noncomm": [10_000.0, 50_000.0],
            "total_oi": [500_000.0, 500_000.0],
        }
    )
    monkeypatch.setattr(
        "pipeline.cot_features._download_cot_for_asset",
        lambda asset, start, end: synth,
    )

    # Daily market index over Jan 2024.
    target_idx = pd.date_range("2024-01-01", "2024-01-20", freq="D", tz="UTC")
    feats = build_cot_features(
        "XAUUSD", target_idx, cache_dir=str(tmp_path / "cot")
    )

    # The Thu 2024-01-11 bar should be in the pre-publication window of report #2,
    # so it should still see the prior report (net_noncomm = 10000) or NaN if
    # there's not enough history. The KEY assertion: it must NOT see the
    # 2024-01-09 report's value (50000).
    thu = pd.Timestamp("2024-01-11", tz="UTC")
    mon = pd.Timestamp("2024-01-15", tz="UTC")

    thu_val = feats.loc[thu, "cot_net_noncomm_pct"]
    mon_val = feats.loc[mon, "cot_net_noncomm_pct"]

    # The 2024-01-09 report would give net/oi = 50000/500000 = 0.10.
    # The 2024-01-02 report would give 10000/500000 = 0.02.
    assert thu_val != pytest.approx(0.10), (
        f"Thu 2024-01-11 leaked from Fri-published 2024-01-09 report; "
        f"got {thu_val}, must not equal 0.10"
    )
    # Monday after the Friday publication must see the new report.
    assert mon_val == pytest.approx(0.10), (
        f"Mon 2024-01-15 must see Tue 2024-01-09 / Fri 2024-01-12 report "
        f"(net/oi=0.10); got {mon_val}"
    )


def test_cot_extreme_flags(tmp_path, monkeypatch):
    """cot_extreme_long fires when z > +2; cot_extreme_short fires when z < -2."""
    # Build a 60-week series where the last 3 weeks are extreme.
    n_weeks = 60
    base = np.zeros(n_weeks - 6)
    # Mild noise to seed the rolling z denominator.
    rng = np.random.default_rng(7)
    base = rng.normal(0, 1_000.0, size=n_weeks - 6)
    # Plant strong positive then strong negative extremes at the end.
    tail = [60_000.0, 65_000.0, 70_000.0, -60_000.0, -65_000.0, -70_000.0]
    pattern = list(base) + tail
    synth = _synth_weekly_cot(
        start="2022-01-04", n_weeks=n_weeks, net_noncomm_pattern=pattern,
        total_oi=500_000.0,
    )
    monkeypatch.setattr(
        "pipeline.cot_features._download_cot_for_asset",
        lambda asset, start, end: synth,
    )

    # Daily market index spanning the whole window plus a tail for publication.
    target_idx = pd.date_range(
        synth["report_date"].min() - pd.Timedelta(days=2),
        synth["report_date"].max() + pd.Timedelta(days=14),
        freq="D",
        tz="UTC",
    )
    feats = build_cot_features(
        "XAUUSD", target_idx, cache_dir=str(tmp_path / "cot")
    )

    # At the tail of the index, after the extreme negative weeks have been
    # published, cot_extreme_short should equal 1 for at least one bar.
    tail_window = feats.iloc[-5:]
    assert tail_window["cot_extreme_short"].max() == 1.0, (
        "expected cot_extreme_short to fire (z<-2) at series tail; "
        f"got {tail_window['cot_extreme_short'].tolist()}"
    )

    # Earlier, after the extreme positive weeks (positions 54-56 of synth)
    # are published, cot_extreme_long should fire for some bars.
    # The extreme positives are at synth rows -6..-4 (positions 54..56).
    # They are published roughly 3-4 days after Tue → check bars between
    # synth.report_date.iloc[54] + 4d and synth.report_date.iloc[56] + 12d.
    extreme_long_pub_start = synth["report_date"].iloc[-6] + pd.Timedelta(days=4)
    extreme_long_pub_end = synth["report_date"].iloc[-4] + pd.Timedelta(days=12)
    window_mask = (feats.index >= extreme_long_pub_start) & (
        feats.index <= extreme_long_pub_end
    )
    long_window = feats.loc[window_mask]
    assert long_window["cot_extreme_long"].max() == 1.0, (
        "expected cot_extreme_long to fire (z>+2) after positive extreme reports; "
        f"got max={long_window['cot_extreme_long'].max()}"
    )


def test_no_lookahead_in_rolling_zscore(tmp_path, monkeypatch):
    """The 52-week rolling z at row N must use only data with publication time < bar time N.

    Concretely: if we shift the LAST weekly value to an arbitrary spike, no bar
    BEFORE that report's publication should change. Compare two scenarios:
    a baseline series, and an alternate where only the last weekly value differs.
    """
    rng = np.random.default_rng(11)
    n_weeks = 80
    base_pattern = list(rng.normal(0, 5_000.0, size=n_weeks))
    spiked_pattern = list(base_pattern)
    spiked_pattern[-1] = 9e6  # absurd spike on the last week

    base_synth = _synth_weekly_cot(
        start="2022-01-04", n_weeks=n_weeks, net_noncomm_pattern=base_pattern
    )
    spiked_synth = _synth_weekly_cot(
        start="2022-01-04", n_weeks=n_weeks, net_noncomm_pattern=spiked_pattern
    )

    target_idx = pd.date_range(
        base_synth["report_date"].min() - pd.Timedelta(days=2),
        base_synth["report_date"].max() + pd.Timedelta(days=14),
        freq="D",
        tz="UTC",
    )

    cache_a = tmp_path / "a"
    cache_b = tmp_path / "b"

    monkeypatch.setattr(
        "pipeline.cot_features._download_cot_for_asset",
        lambda asset, start, end: base_synth,
    )
    feats_base = build_cot_features("XAUUSD", target_idx, cache_dir=str(cache_a))

    monkeypatch.setattr(
        "pipeline.cot_features._download_cot_for_asset",
        lambda asset, start, end: spiked_synth,
    )
    feats_spike = build_cot_features("XAUUSD", target_idx, cache_dir=str(cache_b))

    # Anywhere BEFORE the last report's publication (Friday after the last Tuesday)
    # — features must be identical, regardless of the spike.
    last_tue = base_synth["report_date"].iloc[-1]
    last_friday_pub = last_tue + pd.Timedelta(days=3)

    pre_pub = feats_base.index < last_friday_pub
    pd.testing.assert_frame_equal(
        feats_base.loc[pre_pub].fillna(-1.0),
        feats_spike.loc[pre_pub].fillna(-1.0),
        check_dtype=False,
    )


def test_cot_feature_columns_schema(tmp_path, monkeypatch):
    """For COT-supported assets the output has exactly the documented columns."""
    synth = _synth_weekly_cot(n_weeks=80)
    monkeypatch.setattr(
        "pipeline.cot_features._download_cot_for_asset",
        lambda asset, start, end: synth,
    )
    target_idx = pd.date_range("2022-02-01", periods=120, freq="D", tz="UTC")
    feats = build_cot_features(
        "EURUSD", target_idx, cache_dir=str(tmp_path / "cot")
    )
    assert list(feats.columns) == list(COT_FEATURE_COLUMNS)
    assert feats.index.equals(target_idx)
