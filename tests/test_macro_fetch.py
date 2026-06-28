"""Tests for pipeline.macro_fetch."""
from __future__ import annotations
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest

from pipeline.macro_fetch import fetch_series, build_macro_frame, MacroFetchError, FRED_SERIES


def _mock_fred(series_values):
    """Return a mock fredapi.Fred whose get_series returns the given pd.Series."""
    fred = MagicMock()
    fred.get_series = lambda code, observation_start=None, observation_end=None: series_values[code]
    return fred


def test_fetch_series_uses_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]), name="DGS5")
    with patch("pipeline.macro_fetch._make_fred_client", return_value=_mock_fred({"DGS5": s})):
        out = fetch_series("DGS5", "2020-01-01", "2020-01-03", cache_dir=cache_dir)
        assert (cache_dir / "DGS5.parquet").exists()
        assert list(out.values) == [1.0, 2.0, 3.0]

    # Second call: API client should not be called again — cache hit.
    sentinel = MagicMock(side_effect=AssertionError("API was called despite cache hit"))
    with patch("pipeline.macro_fetch._make_fred_client", return_value=sentinel):
        out2 = fetch_series("DGS5", "2020-01-01", "2020-01-03", cache_dir=cache_dir)
        assert list(out2.values) == [1.0, 2.0, 3.0]


def test_fetch_series_trusts_existing_cache_fully(tmp_path):
    """If the cache file exists, fetch_series MUST use it without calling the
    API — even when the request window extends past cache.max() (UMCSENT-
    style staleness) OR before cache.min() (weekend/holiday gap at the
    start). Freshness/coverage are managed explicitly by scripts/ingest_*.
    """
    cache_dir = tmp_path / "cache"
    cached = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-08"]),
        name="DTWEXBGS",
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.to_frame().to_parquet(cache_dir / "DTWEXBGS.parquet")

    sentinel = MagicMock(side_effect=AssertionError("API called despite existing cache"))
    with patch("pipeline.macro_fetch._make_fred_client", return_value=sentinel):
        # start (2020-01-01) is BEFORE cache.min (2020-01-06): used to refetch.
        out_start = fetch_series("DTWEXBGS", "2020-01-01", "2020-01-08", cache_dir=cache_dir)
        # end (2020-08-31) is AFTER cache.max (2020-01-08): used to refetch.
        out_end = fetch_series("DTWEXBGS", "2020-01-06", "2020-08-31", cache_dir=cache_dir)

    # Both calls return whatever the cache holds in the requested range.
    assert list(out_start.values) == [1.0, 2.0, 3.0]
    assert list(out_end.values) == [1.0, 2.0, 3.0]


def test_build_macro_frame_applies_publication_lag(tmp_path):
    """build_macro_frame must shift macro values by 1 day so features[t] uses <= t-1 info."""
    cache_dir = tmp_path / "cache"
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"], utc=True)
    raw = {
        "DTWEXBGS": pd.Series([100.0, 101.0, 102.0, 103.0], index=idx),
        "DFII5": pd.Series([0.5, 0.6, 0.7, 0.8], index=idx),
        "DGS5": pd.Series([1.5, 1.6, 1.7, 1.8], index=idx),
        "DGS2": pd.Series([0.8, 0.9, 1.0, 1.1], index=idx),
        "T5YIE": pd.Series([2.0, 2.1, 2.2, 2.3], index=idx),
        "VIXCLS": pd.Series([15.0, 16.0, 17.0, 18.0], index=idx),
        "UMCSENT": pd.Series([80.0, 81.0, 82.0, 83.0], index=idx),
    }
    with patch("pipeline.macro_fetch._make_fred_client", return_value=_mock_fred(raw)):
        frame = build_macro_frame("2020-01-01", "2020-01-04", cache_dir=cache_dir)

    # After shift(1), the value at 2020-01-02 should be the raw value at 2020-01-01.
    assert frame.loc["2020-01-02", "DTWEXBGS"] == 100.0
    assert frame.loc["2020-01-02", "VIXCLS"] == 15.0
    # The first row must be NaN (no previous day's data).
    assert pd.isna(frame.loc["2020-01-01", "DTWEXBGS"])


def test_build_macro_frame_raises_without_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(MacroFetchError, match="FRED_API_KEY"):
        build_macro_frame("2020-01-01", "2020-01-04", cache_dir=tmp_path / "cache")


def test_umcsent_in_fred_series():
    """UMCSENT (University of Michigan Consumer Sentiment) is part of the FRED bundle."""
    assert "UMCSENT" in FRED_SERIES


def test_build_macro_frame_emits_umcsent_and_precomputed_3m_change(tmp_path):
    """UMCSENT is monthly. We precompute the 3-month change on the daily-aligned
    series BEFORE reindexing to market_index so that downstream D1 and H4
    builders see the same calendar-anchored delta (instead of a bar-count
    diff that means 3 months on D1 but ~10 days on H4).
    """
    cache_dir = tmp_path / "cache"
    n = 100
    idx = pd.to_datetime(pd.date_range("2020-01-01", periods=n, freq="D"))
    raw = {
        "DTWEXBGS": pd.Series(np.linspace(100, 110, n), index=idx),
        "DFII5": pd.Series(np.linspace(0.5, 1.5, n), index=idx),
        "DGS5": pd.Series(np.linspace(1.5, 2.5, n), index=idx),
        "DGS2": pd.Series(np.linspace(0.8, 1.8, n), index=idx),
        "T5YIE": pd.Series(np.linspace(2.0, 3.0, n), index=idx),
        "VIXCLS": pd.Series(np.linspace(15, 25, n), index=idx),
        # UMCSENT: monthly cadence — only 4 observations in 100 days.
        "UMCSENT": pd.Series(
            [80.0, 82.0, 85.0, 88.0],
            index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]),
        ),
    }
    with patch("pipeline.macro_fetch._make_fred_client", return_value=_mock_fred(raw)):
        frame = build_macro_frame("2020-01-01", "2020-04-09", cache_dir=cache_dir)

    assert "UMCSENT" in frame.columns
    assert "UMCSENT_chg_3m" in frame.columns

    # Daily ffill + .shift(1): UMCSENT at "2020-01-02" should be the raw value
    # at "2020-01-01" = 80.0.
    assert frame.loc["2020-01-02", "UMCSENT"] == 80.0
    # By "2020-02-02" the Feb release has propagated (shifted by 1 day from Feb 1).
    assert frame.loc["2020-02-02", "UMCSENT"] == 82.0

    # First row NaN (publication lag).
    assert pd.isna(frame.loc["2020-01-01", "UMCSENT"])

    # 3-month change is computed on the daily-aligned series with a 63-trading-
    # day window — should be NaN for the first ~63 rows then turn finite.
    chg = frame["UMCSENT_chg_3m"]
    assert chg.iloc[:63].isna().all()
    assert not pd.isna(chg.iloc[70])
