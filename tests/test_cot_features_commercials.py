"""Tests for pipeline.cot_features.build_cot_commercials_raw (B0015b).

The commercials-side extraction is parallel to the existing non-commercials
features pipeline but returns RAW values (no z-score, no extremes) at the
target_index. Computation of derived features (z, change, extremes) lives in
the phase5_cot_extremes primary, by design — keeps commercials information
off the meta's view by separating where the columns are computed.

Invariants:
  1. For disagg-mapped assets (XAUUSD, XAGUSD), function returns a 2-col
     DataFrame with columns {net_long, total_oi}.
  2. For non-disagg-mapped assets (BTCUSD, EURUSD), function returns an empty
     2-col DataFrame (zero rows but the columns).
  3. Publication-lag discipline: a Tuesday report stamped 2024-01-09 is
     published Friday 2024-01-12; a Friday-midnight-UTC market bar must NOT
     see that report's value.
  4. Output is indexed to target_index (not the weekly cadence).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.cot_features import (
    ASSET_TO_CFTC_CONTRACT,
    build_cot_commercials_raw,
)


def _synth_weekly_commercials(
    start: str = "2022-01-04",
    n_weeks: int = 60,
    commercials_net_pattern: list[float] | None = None,
    total_oi: float = 500_000.0,
) -> pd.DataFrame:
    """Return a synthetic weekly long-frame mimicking what _download_cot_for_asset
    would return after the Task 1 extension (with commercials_net column added).
    """
    tuesdays = pd.date_range(start=start, periods=n_weeks, freq="7D", tz="UTC")
    if commercials_net_pattern is None:
        commercials_net_pattern = list(
            -np.sin(np.linspace(0, 4 * np.pi, n_weeks)) * 50_000.0
        )
    return pd.DataFrame(
        {
            "report_date": tuesdays,
            # The existing schema; commercials_net is the new column.
            "net_noncomm": [0.0] * n_weeks,
            "commercials_net": commercials_net_pattern,
            "total_oi": [total_oi] * n_weeks,
        }
    )


def test_returns_correct_columns_for_disagg_asset(tmp_path, monkeypatch):
    """For XAUUSD the function returns a DataFrame with net_long, total_oi columns."""
    synth = _synth_weekly_commercials(n_weeks=60)
    monkeypatch.setattr(
        "pipeline.cot_features._download_cot_for_asset",
        lambda asset, start, end: synth,
    )
    target_idx = pd.date_range("2022-02-01", periods=200, freq="D", tz="UTC")
    out = build_cot_commercials_raw(
        "XAUUSD", target_idx, cache_dir=str(tmp_path / "cot")
    )
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) == {"net_long", "total_oi"}
    assert out.index.equals(target_idx)


def test_returns_empty_for_non_disagg_asset(tmp_path):
    """For non-CFTC assets, returns empty 2-col DataFrame indexed to target_idx."""
    target_idx = pd.date_range("2022-01-01", periods=50, freq="D", tz="UTC")
    for asset in ("BTCUSD", "ETHUSD", "SOLUSD"):
        out = build_cot_commercials_raw(
            asset, target_idx, cache_dir=str(tmp_path / "cot")
        )
        assert isinstance(out, pd.DataFrame)
        assert out.shape[0] == 0 or out["net_long"].isna().all(), (
            f"{asset} should produce no data; got {out}"
        )


def test_returns_empty_for_tff_only_asset(tmp_path):
    """TFF-mapped assets (EUR/GBP/JPY) do NOT have commercials in the same
    Producer-Merchant+Swap-Dealer sense as disaggregated metals. For v1 of
    B0015b we only ship disagg-asset support; TFF returns empty."""
    target_idx = pd.date_range("2022-01-01", periods=50, freq="D", tz="UTC")
    for asset in ("EURUSD", "GBPUSD", "USDJPY"):
        out = build_cot_commercials_raw(
            asset, target_idx, cache_dir=str(tmp_path / "cot")
        )
        assert isinstance(out, pd.DataFrame)
        assert out.shape[0] == 0 or out["net_long"].isna().all(), (
            f"{asset} (TFF-only for v1) should produce empty data; got {out}"
        )


def test_publication_lag_no_leak(tmp_path, monkeypatch):
    """A COT report stamped Tue 2024-01-09 (published Fri 2024-01-12) must NOT
    be visible to a Friday-midnight-UTC market bar at 2024-01-12 00:00 UTC.

    Mondays AFTER the publication MUST see it.
    """
    # Two reports with deliberately different commercials_net so we can detect
    # leakage by value.
    report_dates = pd.to_datetime(["2024-01-02", "2024-01-09"], utc=True)
    synth = pd.DataFrame(
        {
            "report_date": report_dates,
            "net_noncomm": [0.0, 0.0],
            "commercials_net": [10_000.0, 80_000.0],
            "total_oi": [500_000.0, 500_000.0],
        }
    )
    monkeypatch.setattr(
        "pipeline.cot_features._download_cot_for_asset",
        lambda asset, start, end: synth,
    )
    target_idx = pd.date_range("2024-01-01", "2024-01-20", freq="D", tz="UTC")
    out = build_cot_commercials_raw(
        "XAUUSD", target_idx, cache_dir=str(tmp_path / "cot")
    )

    # Friday-midnight-UTC bar 2024-01-12 00:00 UTC. The 2024-01-09 report has
    # NOT yet been published at that time (publication is Friday 15:30 ET ≈
    # 20:30 UTC). The defensive .shift(1) on the weekly frame in
    # build_cot_commercials_raw ensures this bar sees the prior week's value.
    fri_midnight = pd.Timestamp("2024-01-12", tz="UTC")
    fri_val = out.loc[fri_midnight, "net_long"]

    # Monday 2024-01-15 — comfortably after publication; must see the new report.
    mon = pd.Timestamp("2024-01-15", tz="UTC")
    mon_val = out.loc[mon, "net_long"]

    # The 2024-01-09 report would be net_long=80000.
    # The 2024-01-02 report (or NaN before warm-up) would be 10000 or NaN.
    assert fri_val != pytest.approx(80_000.0), (
        f"Fri 2024-01-12 00:00 UTC leaked from Fri-published 2024-01-09 report; "
        f"got {fri_val}, must not equal 80000"
    )
    assert mon_val == pytest.approx(80_000.0), (
        f"Mon 2024-01-15 must see the Tue 2024-01-09 / Fri 2024-01-12 report; "
        f"got {mon_val}, expected 80000"
    )


def test_returns_total_oi_alongside_net_long(tmp_path, monkeypatch):
    """The function returns total_oi so the primary can compute pct = net/oi."""
    synth = _synth_weekly_commercials(n_weeks=60, total_oi=750_000.0)
    monkeypatch.setattr(
        "pipeline.cot_features._download_cot_for_asset",
        lambda asset, start, end: synth,
    )
    target_idx = pd.date_range("2022-02-01", periods=200, freq="D", tz="UTC")
    out = build_cot_commercials_raw(
        "XAUUSD", target_idx, cache_dir=str(tmp_path / "cot")
    )
    # After warm-up the total_oi should be 750_000 (constant in the fixture).
    last = out.iloc[-1]
    assert last["total_oi"] == pytest.approx(750_000.0)


def test_empty_dataframe_when_download_returns_nothing(tmp_path, monkeypatch):
    """If the downloader returns an empty frame (no rows matching contract), the
    function still returns a 2-col DataFrame at target_index with all NaN."""
    empty_synth = pd.DataFrame(
        {"report_date": [], "net_noncomm": [], "commercials_net": [], "total_oi": []}
    )
    monkeypatch.setattr(
        "pipeline.cot_features._download_cot_for_asset",
        lambda asset, start, end: empty_synth,
    )
    target_idx = pd.date_range("2022-01-01", periods=50, freq="D", tz="UTC")
    out = build_cot_commercials_raw(
        "XAUUSD", target_idx, cache_dir=str(tmp_path / "cot")
    )
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) >= {"net_long", "total_oi"}
    assert out.index.equals(target_idx)
