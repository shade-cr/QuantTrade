"""Tests for crypto perpetual-swap funding-rate features (Tier B2).

Network-free: HTTP calls to Binance are monkeypatched via the
`_fetch_funding_page` test seam to return synthetic JSON.

Invariants under test:
  - Strict-less-than alignment: a funding stamp at exactly H4 bar time t
    must NOT be visible to the bar at t (it publishes AT t — see
    macro_fetch_intraday.get_macro_value_at_bar pattern).
  - Forward-fill: between two funding settlements (8h apart), the
    intermediate H4 bars see the most recent prior funding rate.
  - No look-ahead in rolling z-score: z[t] is built from rows < t (the
    funding value at t is itself eligible, since by the strict-less-than
    rule it is already lagged from t).
  - Pagination: Binance caps each call at 1000 rows; long ranges fan out
    across multiple calls.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pipeline.funding_features as ff
from pipeline.funding_features import (
    ASSET_TO_PERP_SYMBOL,
    build_funding_features,
    fetch_funding_for_asset,
)


# ---------------------------------------------------------------------------
# Fake Binance HTTP
# ---------------------------------------------------------------------------


def _make_funding_payload(
    symbol: str,
    start_ms: int,
    end_ms: int,
    step_hours: int = 8,
    base_rate: float = 1e-4,
) -> list[dict]:
    """Build a synthetic Binance fundingRate JSON payload.

    Funding stamps every `step_hours` hours between [start_ms, end_ms],
    with rates that vary smoothly so rolling stats are non-degenerate.
    """
    step_ms = step_hours * 3600 * 1000
    rows = []
    t = start_ms
    i = 0
    while t < end_ms:
        # Deterministic sinusoidal funding around base_rate.
        rate = base_rate + 5e-5 * np.sin(i / 4.0)
        rows.append(
            {
                "symbol": symbol,
                "fundingTime": t,
                "fundingRate": f"{rate:.10f}",
                "markPrice": f"{40000.0 + 100.0 * np.sin(i / 8.0):.4f}",
            }
        )
        t += step_ms
        i += 1
    return rows


# ---------------------------------------------------------------------------
# fetch_funding_for_asset: caching + pagination
# ---------------------------------------------------------------------------


def test_fetch_funding_caches_to_parquet(monkeypatch, tmp_path):
    """Verify a parquet appears in cache_dir after a successful fetch."""
    call_log = []

    def fake_page(symbol, start_ms, end_ms, limit):
        call_log.append((symbol, start_ms, end_ms, limit))
        return _make_funding_payload(symbol, start_ms, min(end_ms, start_ms + 30 * 24 * 3600 * 1000))

    monkeypatch.setattr(ff, "_fetch_funding_page", fake_page)

    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-20", tz="UTC")
    df = fetch_funding_for_asset("BTCUSD", start, end, cache_dir=tmp_path)

    assert isinstance(df, pd.DataFrame)
    assert {"funding_time", "funding_rate", "mark_price"} <= set(df.columns)
    assert len(df) > 0
    # All funding times within range.
    assert df["funding_time"].min() >= start
    assert df["funding_time"].max() <= end + pd.Timedelta(hours=8)
    # Cache parquet written.
    assert any(Path(tmp_path).glob("*.parquet")), "expected parquet cache file"


def test_fetch_funding_uses_cache_on_second_call(monkeypatch, tmp_path):
    """Second fetch within the cached range must not hit the HTTP layer."""
    call_count = {"n": 0}

    def fake_page(symbol, start_ms, end_ms, limit):
        call_count["n"] += 1
        return _make_funding_payload(symbol, start_ms, min(end_ms, start_ms + 30 * 24 * 3600 * 1000))

    monkeypatch.setattr(ff, "_fetch_funding_page", fake_page)

    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-20", tz="UTC")
    fetch_funding_for_asset("BTCUSD", start, end, cache_dir=tmp_path)
    calls_after_first = call_count["n"]

    # Fetch a sub-range — should hit cache.
    fetch_funding_for_asset(
        "BTCUSD",
        pd.Timestamp("2024-01-05", tz="UTC"),
        pd.Timestamp("2024-01-15", tz="UTC"),
        cache_dir=tmp_path,
    )
    assert call_count["n"] == calls_after_first, "expected cache hit; HTTP was called again"


def test_binance_pagination(monkeypatch, tmp_path):
    """When the time range exceeds Binance's 1000-row-per-call limit, the
    fetcher must fan out across multiple HTTP calls."""
    call_log = []

    def fake_page(symbol, start_ms, end_ms, limit):
        # Mimic Binance behaviour: return at most `limit` rows starting
        # from start_ms, regardless of how far end_ms is.
        call_log.append((start_ms, end_ms, limit))
        step_ms = 8 * 3600 * 1000
        rows = []
        t = start_ms
        for i in range(limit):
            if t >= end_ms:
                break
            rows.append(
                {
                    "symbol": symbol,
                    "fundingTime": t,
                    "fundingRate": f"{1e-4 + 1e-6 * i:.10f}",
                    "markPrice": "40000.0",
                }
            )
            t += step_ms
        return rows

    monkeypatch.setattr(ff, "_fetch_funding_page", fake_page)

    # 5 years × 365 days × 3 funding stamps/day ≈ 5475 rows → requires ≥6 calls
    # at limit=1000.
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2025-01-01", tz="UTC")
    df = fetch_funding_for_asset("BTCUSD", start, end, cache_dir=tmp_path)

    assert len(call_log) >= 2, f"expected pagination across multiple calls; got {len(call_log)}"
    # All funding times strictly monotone (no duplicates from pagination overlap).
    assert df["funding_time"].is_monotonic_increasing
    assert df["funding_time"].duplicated().sum() == 0


# ---------------------------------------------------------------------------
# build_funding_features: gating + alignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("asset", ["XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "USDJPY"])
def test_build_funding_features_for_non_crypto_asset_returns_empty(asset, tmp_path):
    """Non-crypto assets must get an empty DataFrame (no funding info)."""
    target_index = pd.date_range("2024-01-01", periods=24, freq="4h", tz="UTC")
    df = build_funding_features(asset, target_index, cache_dir=tmp_path)
    assert df.empty or len(df.columns) == 0, f"expected empty frame for {asset}, got {df.columns.tolist()}"
    # Index should still match target_index even when empty (consistent shape).
    if not df.empty:
        assert df.index.equals(target_index)


def test_strict_less_than_alignment_no_leak(monkeypatch, tmp_path):
    """Funding stamped at 2024-03-15 08:00:00 UTC must NOT appear in the
    H4 bar stamped 2024-03-15 08:00:00 UTC (only in the bar at 12:00:00).

    Mirrors the strict-less-than convention from macro_fetch_intraday."""
    # Build a synthetic funding parquet directly (bypass HTTP).
    funding_idx = pd.DatetimeIndex([
        pd.Timestamp("2024-03-15 00:00:00", tz="UTC"),
        pd.Timestamp("2024-03-15 08:00:00", tz="UTC"),
        pd.Timestamp("2024-03-15 16:00:00", tz="UTC"),
    ])
    funding_rates = [0.0001, 0.0005, 0.0002]  # distinct values per stamp

    def fake_page(symbol, start_ms, end_ms, limit):
        rows = []
        for ts, rate in zip(funding_idx, funding_rates):
            ms = int(ts.value // 1_000_000)
            if start_ms <= ms < end_ms:
                rows.append({
                    "symbol": symbol,
                    "fundingTime": ms,
                    "fundingRate": f"{rate:.10f}",
                    "markPrice": "40000.0",
                })
        return rows

    monkeypatch.setattr(ff, "_fetch_funding_page", fake_page)

    target_index = pd.DatetimeIndex([
        pd.Timestamp("2024-03-15 04:00:00", tz="UTC"),  # after 00:00 stamp → sees 0.0001
        pd.Timestamp("2024-03-15 08:00:00", tz="UTC"),  # AT 08:00 stamp → must use 00:00 stamp (0.0001), NOT 08:00 (0.0005)
        pd.Timestamp("2024-03-15 12:00:00", tz="UTC"),  # after 08:00 stamp → sees 0.0005
        pd.Timestamp("2024-03-15 16:00:00", tz="UTC"),  # AT 16:00 stamp → must use 08:00 stamp (0.0005), NOT 16:00 (0.0002)
        pd.Timestamp("2024-03-15 20:00:00", tz="UTC"),  # after 16:00 stamp → sees 0.0002
    ])
    df = build_funding_features("BTCUSD", target_index, cache_dir=tmp_path)

    assert df.loc[target_index[0], "funding_h4"] == pytest.approx(0.0001)
    assert df.loc[target_index[1], "funding_h4"] == pytest.approx(0.0001), \
        "strict-less-than violated: bar AT 08:00 saw the 08:00 stamp (lookahead)"
    assert df.loc[target_index[2], "funding_h4"] == pytest.approx(0.0005)
    assert df.loc[target_index[3], "funding_h4"] == pytest.approx(0.0005), \
        "strict-less-than violated: bar AT 16:00 saw the 16:00 stamp (lookahead)"
    assert df.loc[target_index[4], "funding_h4"] == pytest.approx(0.0002)


def test_forward_fill_within_8h_window(monkeypatch, tmp_path):
    """H4 bars at 08, 12, 16 all see the same funding (the one stamped
    before 08); the next funding at 16:00 takes effect for the H4 bar at 20:00."""
    funding_stamps = [
        (pd.Timestamp("2024-03-15 00:00:00", tz="UTC"), 0.0001),
        (pd.Timestamp("2024-03-15 08:00:00", tz="UTC"), 0.0005),
        (pd.Timestamp("2024-03-15 16:00:00", tz="UTC"), 0.0002),
    ]

    def fake_page(symbol, start_ms, end_ms, limit):
        rows = []
        for ts, rate in funding_stamps:
            ms = int(ts.value // 1_000_000)
            if start_ms <= ms < end_ms:
                rows.append({
                    "symbol": symbol,
                    "fundingTime": ms,
                    "fundingRate": f"{rate:.10f}",
                    "markPrice": "40000.0",
                })
        return rows

    monkeypatch.setattr(ff, "_fetch_funding_page", fake_page)

    # H4 bars covering one full day.
    target_index = pd.date_range("2024-03-15 04:00:00", periods=6, freq="4h", tz="UTC")
    df = build_funding_features("BTCUSD", target_index, cache_dir=tmp_path)

    # 04:00 → uses 00:00 stamp = 0.0001
    # 08:00 → strict-less-than → 00:00 stamp = 0.0001
    # 12:00 → uses 08:00 stamp = 0.0005
    # 16:00 → strict-less-than → 08:00 stamp = 0.0005
    # 20:00 → uses 16:00 stamp = 0.0002
    # 00:00 next day → uses 16:00 stamp = 0.0002
    expected = [0.0001, 0.0001, 0.0005, 0.0005, 0.0002, 0.0002]
    assert df["funding_h4"].tolist() == pytest.approx(expected)


def test_rolling_zscore_no_lookahead(monkeypatch, tmp_path):
    """funding_h4_z252[t] must depend only on funding values at index positions ≤ t.

    Since funding_h4 is already strict-less-than aligned, the rolling stat is
    safe as long as it does NOT center or look ahead. We assert the standard
    pandas behaviour explicitly: at position k the rolling window uses the
    most recent N values up to and including position k.
    """
    # Build a long funding history so the 252-window kicks in.
    n_stamps = 400
    funding_stamps = [
        (
            pd.Timestamp("2024-01-01 00:00:00", tz="UTC") + pd.Timedelta(hours=8 * i),
            1e-4 + 1e-5 * np.sin(i / 5.0),
        )
        for i in range(n_stamps)
    ]

    def fake_page(symbol, start_ms, end_ms, limit):
        rows = []
        for ts, rate in funding_stamps:
            ms = int(ts.value // 1_000_000)
            if start_ms <= ms < end_ms:
                rows.append({
                    "symbol": symbol,
                    "fundingTime": ms,
                    "fundingRate": f"{rate:.10f}",
                    "markPrice": "40000.0",
                })
        return rows

    monkeypatch.setattr(ff, "_fetch_funding_page", fake_page)

    target_index = pd.date_range("2024-01-01 04:00:00", periods=500, freq="4h", tz="UTC")
    df = build_funding_features("BTCUSD", target_index, cache_dir=tmp_path)

    # Recompute the z-score manually from funding_h4 using only past data
    # (rolling window ending at t). Must match the module's output.
    fh = df["funding_h4"]
    mean = fh.rolling(252).mean()
    std = fh.rolling(252).std()
    expected_z = (fh - mean) / std

    # Compare on the non-NaN tail.
    common = df["funding_h4_z252"].dropna().index.intersection(expected_z.dropna().index)
    assert len(common) > 0, "z-score never computed — check window size"
    np.testing.assert_allclose(
        df.loc[common, "funding_h4_z252"].values,
        expected_z.loc[common].values,
        rtol=1e-9,
        atol=1e-12,
    )

    # Also assert: at any t where z is non-NaN, removing all rows > t from
    # funding_h4 and recomputing yields the same z[t] (no peek into future).
    sample_t = common[len(common) // 2]
    past_only = fh.loc[:sample_t]
    expected_z_past_only = (
        (past_only.iloc[-1] - past_only.iloc[-252:].mean()) / past_only.iloc[-252:].std()
    )
    assert df.loc[sample_t, "funding_h4_z252"] == pytest.approx(expected_z_past_only, rel=1e-9)


def test_extreme_flags(monkeypatch, tmp_path):
    """funding_extreme_positive fires when z > 1.5; funding_extreme_negative
    when z < -1.5. Verified by constructing a series with a known spike."""
    # 252 baseline rates at 0.0001, then a giant spike, then a giant negative.
    rates = [0.0001] * 260 + [0.01] + [0.0001] * 5 + [-0.01] + [0.0001] * 5
    funding_stamps = [
        (
            pd.Timestamp("2024-01-01 00:00:00", tz="UTC") + pd.Timedelta(hours=8 * i),
            r,
        )
        for i, r in enumerate(rates)
    ]

    def fake_page(symbol, start_ms, end_ms, limit):
        rows = []
        for ts, rate in funding_stamps:
            ms = int(ts.value // 1_000_000)
            if start_ms <= ms < end_ms:
                rows.append({
                    "symbol": symbol,
                    "fundingTime": ms,
                    "fundingRate": f"{rate:.10f}",
                    "markPrice": "40000.0",
                })
        return rows

    monkeypatch.setattr(ff, "_fetch_funding_page", fake_page)

    # Long target index spanning the funding history; 3 H4 bars between settlements.
    target_index = pd.date_range(
        "2024-01-01 04:00:00",
        periods=len(rates) * 2,
        freq="4h",
        tz="UTC",
    )
    df = build_funding_features("BTCUSD", target_index, cache_dir=tmp_path)

    # At least one bar must trip funding_extreme_positive (around the spike) and
    # at least one must trip funding_extreme_negative (around the negative spike).
    assert (df["funding_extreme_positive"] == 1.0).any(), \
        "expected at least one extreme-positive bar after a +0.01 spike"
    assert (df["funding_extreme_negative"] == 1.0).any(), \
        "expected at least one extreme-negative bar after a -0.01 spike"

    # Negative flag must never coincide with positive flag.
    both = (df["funding_extreme_positive"] == 1.0) & (df["funding_extreme_negative"] == 1.0)
    assert not both.any(), "extreme flags fired simultaneously — sign error"


def test_feature_columns_present(monkeypatch, tmp_path):
    """Schema check: build_funding_features for a crypto asset emits all the
    documented columns."""
    funding_stamps = [
        (
            pd.Timestamp("2024-01-01 00:00:00", tz="UTC") + pd.Timedelta(hours=8 * i),
            1e-4 + 1e-5 * np.sin(i / 5.0),
        )
        for i in range(200)
    ]

    def fake_page(symbol, start_ms, end_ms, limit):
        rows = []
        for ts, rate in funding_stamps:
            ms = int(ts.value // 1_000_000)
            if start_ms <= ms < end_ms:
                rows.append({
                    "symbol": symbol,
                    "fundingTime": ms,
                    "fundingRate": f"{rate:.10f}",
                    "markPrice": "40000.0",
                })
        return rows

    monkeypatch.setattr(ff, "_fetch_funding_page", fake_page)

    target_index = pd.date_range("2024-01-02", periods=24, freq="4h", tz="UTC")
    df = build_funding_features("BTCUSD", target_index, cache_dir=tmp_path)

    expected_cols = {
        "funding_h4",
        "funding_h4_z252",
        "funding_h4_chg_5bars",
        "funding_cumulative_7d",
        "funding_extreme_positive",
        "funding_extreme_negative",
    }
    assert expected_cols <= set(df.columns), (
        f"missing columns: {expected_cols - set(df.columns)}"
    )
    assert df.index.equals(target_index)


def test_asset_to_perp_symbol_map_contains_btc_eth():
    """Symbol map sanity check (no typos in the constant)."""
    assert ASSET_TO_PERP_SYMBOL["BTCUSD"] == "BTCUSDT"
    assert ASSET_TO_PERP_SYMBOL["ETHUSD"] == "ETHUSDT"
    assert ASSET_TO_PERP_SYMBOL["SOLUSD"] == "SOLUSDT"
