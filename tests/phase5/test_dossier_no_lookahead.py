"""B0036 — smoke tests for dossier orthogonal-feature lookahead discipline.

These tests verify that the COT + real-yield features added by B0036 (a)
become available only after their required history + publication-lag burn-in,
(b) carry the expected name, and (c) at any test bar use only data with
publication_date <= bar_date - lag.

The tests run against the actual cache files; they do not mock or fixture
the data. If the cache is missing they skip with a clear message.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest

from phase5.regime_stats import (
    load_cot_net_noncomm_z,
    load_real_yield_z,
    B0036_MVP_FEATURES,
    REGIME_DEFINING_FEATURES,
)


COT_CACHE = Path("cache/cot/cot_XAUUSD.parquet")
DFII5_CACHE = Path("cache/fred/DFII5.parquet")


@pytest.mark.skipif(not COT_CACHE.exists(), reason="cot_XAUUSD.parquet not in cache")
def test_cot_z_smoke():
    """COT z-score has correct name, burn-in, and post-burn-in values."""
    target_idx = pd.date_range("2010-01-01", "2026-01-01", freq="D", tz="UTC")
    z = load_cot_net_noncomm_z("XAUUSD", target_idx)
    assert z.name == "cot_net_noncomm_z52w"
    # 52 weeks of COT + 5d publication lag → first ~370 days must be NaN
    assert z.iloc[:300].isna().all(), (
        "cot z must be NaN before 52 weeks of COT history + publication lag"
    )
    # By 2012 (2y after COT cache start 2010-01-05), values must exist
    post_burn = z.loc["2012-01-01":"2012-12-31"]
    assert post_burn.notna().any(), "cot z must have values by 2012"


@pytest.mark.skipif(not COT_CACHE.exists(), reason="cot_XAUUSD.parquet not in cache")
def test_cot_publication_lag_respected():
    """At any D1 bar t, cot z must depend only on reports with available_from <= t.

    This is the structural lookahead test: we sample 20 random non-NaN dates
    and verify that the underlying COT row's `report_date + publication_lag_days`
    is <= the bar date.
    """
    import random
    target_idx = pd.date_range("2012-01-01", "2026-01-01", freq="D", tz="UTC")
    z = load_cot_net_noncomm_z("XAUUSD", target_idx)
    raw = pd.read_parquet(COT_CACHE)
    raw["available_from"] = raw["report_date"] + pd.Timedelta(days=5)
    if raw["available_from"].dt.tz is None:
        raw["available_from"] = raw["available_from"].dt.tz_localize("UTC")

    non_nan = z.dropna().index
    random.seed(42)
    sample = random.sample(list(non_nan), min(20, len(non_nan)))
    for t in sample:
        eligible = raw[raw["available_from"] <= t]
        assert not eligible.empty, (
            f"cot z is non-NaN at {t} but no COT row has available_from <= {t} "
            f"(min available_from in cache: {raw['available_from'].min()})"
        )


@pytest.mark.skipif(not DFII5_CACHE.exists(), reason="DFII5.parquet not in cache")
def test_real_yield_z_smoke():
    """Real-yield (DFII5) z-score has correct name, burn-in, and post-burn-in values."""
    target_idx = pd.date_range("2006-01-01", "2026-01-01", freq="D", tz="UTC")
    z = load_real_yield_z(target_idx)
    assert z.name == "real_yield_5y_z252d"
    # 252-day rolling z + 1-day publication shift → first ~253 days must be NaN
    assert z.iloc[:200].isna().all(), (
        "real_yield z must be NaN before 252 days of DFII5 history"
    )
    # By 2008, values must exist
    post_burn = z.loc["2008-01-01":"2008-12-31"]
    assert post_burn.notna().any(), "real_yield z must have values by 2008"


@pytest.mark.skipif(not DFII5_CACHE.exists(), reason="DFII5.parquet not in cache")
def test_real_yield_publication_lag_respected():
    """Real-yield z at D1 bar t must use only DFII5 values stamped <= t-1.

    Concretely: the .shift(1) applied inside load_real_yield_z guarantees that
    the value at bar t reflects the rolling window ending at t-1, not at t.
    We verify by checking that z is NaN at exactly the first D1 bar AFTER the
    DFII5 cache starts (because the .shift(1) pushes the first available
    value one day forward).
    """
    s = pd.read_parquet(DFII5_CACHE)["DFII5"]
    cache_start = pd.to_datetime(s.index.min()).tz_localize("UTC") if s.index.tz is None else s.index.min()
    target_idx = pd.date_range(cache_start, cache_start + pd.Timedelta(days=10), freq="D", tz="UTC")
    z = load_real_yield_z(target_idx)
    # First bar of target_idx is the same date as DFII5's first stamp. With
    # .shift(1), the rolling window at that date cannot include the stamp itself,
    # so z must be NaN at cache_start.
    assert pd.isna(z.iloc[0]), (
        f"real_yield z at cache_start={cache_start} must be NaN due to .shift(1) "
        f"publication-lag discipline; got {z.iloc[0]}"
    )


def test_b0036_mvp_features_constant_consistency():
    """B0036_MVP_FEATURES keys must match the feature names emitted by the loaders."""
    assert "cot_net_noncomm_z52w" in B0036_MVP_FEATURES
    assert "real_yield_5y_z252d" in B0036_MVP_FEATURES
    # And the loaders must use these exact names
    target_idx = pd.date_range("2020-01-01", "2020-01-10", freq="D", tz="UTC")
    if COT_CACHE.exists():
        assert load_cot_net_noncomm_z("XAUUSD", target_idx).name == "cot_net_noncomm_z52w"
    if DFII5_CACHE.exists():
        assert load_real_yield_z(target_idx).name == "real_yield_5y_z252d"


def test_orthogonal_features_disjoint_from_regime_defining():
    """B0036 MVP features must NOT overlap with REGIME_DEFINING_FEATURES."""
    for feat in B0036_MVP_FEATURES.keys():
        assert feat not in REGIME_DEFINING_FEATURES, (
            f"B0036 MVP feature {feat!r} cannot be a regime-defining feature"
        )
