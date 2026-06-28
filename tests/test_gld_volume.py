"""B0147 — GLD real-volume features loader (pipeline/alt_data/gld_volume.py).

The load-bearing property is the PIT calendar shift: GLD's value stamped at
trading date t must be visible to market bars at date >= t+1 only, with
Friday's stamp covering the weekend via ffill (same convention as
gld_holdings.py, pinned here independently).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.alt_data.gld_volume import (
    GLD_VOLUME_FEATURES,
    GldVolumeCacheMissing,
    load_gld_volume_features,
)


@pytest.fixture
def gld_cache(tmp_path):
    """Synthetic GLD daily cache: weekdays only, deterministic volume ramp."""
    idx = pd.bdate_range("2020-01-01", periods=400, tz="UTC")
    rng = np.random.default_rng(0)
    close = 150.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    vol = np.linspace(5e6, 2e7, len(idx)) * (1 + 0.1 * rng.normal(size=len(idx))).clip(0.5)
    df = pd.DataFrame({"gld_close": close, "gld_volume": vol}, index=idx)
    path = tmp_path / "gld_volume.parquet"
    df.to_parquet(path)
    return path, df


def test_missing_cache_raises(tmp_path):
    with pytest.raises(GldVolumeCacheMissing):
        load_gld_volume_features(pd.DatetimeIndex([]), cache_path=tmp_path / "nope.parquet")


def test_empty_index_returns_empty_frame(gld_cache):
    path, _ = gld_cache
    out = load_gld_volume_features(pd.DatetimeIndex([], tz="UTC"), cache_path=path)
    assert list(out.columns) == list(GLD_VOLUME_FEATURES)
    assert out.empty


def test_columns_and_warmup(gld_cache):
    path, df = gld_cache
    target = pd.date_range(df.index[0], df.index[-1] + pd.Timedelta(days=1),
                           freq="D", tz="UTC")
    out = load_gld_volume_features(target, cache_path=path)
    assert list(out.columns) == list(GLD_VOLUME_FEATURES)
    # 252d amihud z warmup: early region NaN, tail populated.
    assert out["gld_dvol_z42"].iloc[-5:].notna().all()
    assert out["gld_amihud_z252"].iloc[-5:].notna().all()
    assert out["gld_dvol_z42"].iloc[:30].isna().all()


def test_pit_calendar_shift_no_same_day_visibility(gld_cache):
    """The feature value visible at bar date t must equal the feature computed
    from GLD data stamped <= t-1 — never t's own stamp."""
    path, df = gld_cache
    target = pd.date_range(df.index[0], df.index[-1] + pd.Timedelta(days=1),
                           freq="D", tz="UTC")
    out = load_gld_volume_features(target, cache_path=path)

    # Recompute the native-series feature independently.
    dollar = df["gld_close"] * df["gld_volume"]
    z42 = (np.log(dollar) - np.log(dollar).rolling(42).mean()) / np.log(dollar).rolling(42).std()

    t = df.index[300]                      # a GLD trading day past warmup
    next_day = t + pd.Timedelta(days=1)
    # Visible at t+1 == value stamped at t...
    assert out.loc[next_day, "gld_dvol_z42"] == pytest.approx(z42.loc[t])
    # ...and at t itself, only the PREVIOUS stamp (t-1) is visible.
    prev_stamp = df.index[299]
    assert out.loc[t, "gld_dvol_z42"] == pytest.approx(z42.loc[prev_stamp])


def test_weekend_bars_carry_friday_stamp(gld_cache):
    path, df = gld_cache
    target = pd.date_range(df.index[0], df.index[-1] + pd.Timedelta(days=3),
                           freq="D", tz="UTC")
    out = load_gld_volume_features(target, cache_path=path)
    fridays = [ts for ts in df.index[260:] if ts.weekday() == 4]
    f = fridays[0]
    sat, sun = f + pd.Timedelta(days=1), f + pd.Timedelta(days=2)
    # Saturday sees Friday's stamp; Sunday still does (ffill), since Monday's
    # own stamp only becomes visible Tuesday.
    assert out.loc[sat, "gld_dvol_z42"] == out.loc[sun, "gld_dvol_z42"]
    assert np.isfinite(out.loc[sat, "gld_dvol_z42"])


def test_future_mutation_does_not_change_past(gld_cache, tmp_path):
    path, df = gld_cache
    target = pd.date_range(df.index[0], df.index[-1], freq="D", tz="UTC")
    a = load_gld_volume_features(target, cache_path=path)
    df2 = df.copy()
    df2.iloc[350, df2.columns.get_loc("gld_volume")] *= 10.0
    path2 = tmp_path / "mut.parquet"
    df2.to_parquet(path2)
    b = load_gld_volume_features(target, cache_path=path2)
    cut = df.index[349] + pd.Timedelta(days=1)
    pd.testing.assert_frame_equal(a.loc[:cut], b.loc[:cut])
