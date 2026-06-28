"""Tests for the HAR volatility-forecast sizing seam (B0101).

Two pure building blocks (daily_realized_variance, har_vol_forecast) and the
optional `vol_forecast` parameter on compute_survival_target. The load-bearing
invariant: vol_forecast=None reproduces the CURRENT behavior EXACTLY (backtest/live
parity must not move), and a provided forecast rescales the per-asset vols used for
both the inverse-vol weights and the covariance diagonal (keeping correlations).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.survival_book import (
    compute_survival_target,
    daily_realized_variance,
    har_vol_forecast,
)


# --- daily_realized_variance ------------------------------------------------ #
def test_daily_realized_variance_sums_squared_intraday_logrets():
    # two UTC days of hourly closes; RV_day = sum of squared hourly log-returns.
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")  # all day 1
    idx2 = pd.date_range("2024-01-02", periods=4, freq="h", tz="UTC")  # day 2
    close = pd.Series([100, 101, 100, 102, 101, 103], index=idx)
    close2 = pd.Series([103, 104, 103, 105], index=idx2)
    intraday = pd.concat([close, close2])

    rv = daily_realized_variance(intraday)

    r = np.log(intraday / intraday.shift(1)).dropna()
    exp_d1 = float((r[r.index.normalize() == pd.Timestamp("2024-01-01", tz="UTC")] ** 2).sum())
    exp_d2 = float((r[r.index.normalize() == pd.Timestamp("2024-01-02", tz="UTC")] ** 2).sum())
    assert rv.loc[pd.Timestamp("2024-01-01", tz="UTC")] == pytest.approx(exp_d1)
    assert rv.loc[pd.Timestamp("2024-01-02", tz="UTC")] == pytest.approx(exp_d2)


# --- har_vol_forecast ------------------------------------------------------- #
def test_har_forecast_on_constant_rv_returns_that_vol():
    # constant daily variance c -> HAR forecasts ~c -> annualized vol = sqrt(c*252)
    c = (0.01) ** 2  # 1% daily vol -> daily variance
    rv = pd.Series(c, index=pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC"))
    f = har_vol_forecast(rv, burn_in=252)
    assert f == pytest.approx(np.sqrt(c * 252), rel=1e-3)


def test_har_forecast_insufficient_history_is_nan():
    rv = pd.Series([1e-4] * 30, index=pd.date_range("2022-01-01", periods=30, freq="D", tz="UTC"))
    assert np.isnan(har_vol_forecast(rv, burn_in=252))


def test_har_forecast_reacts_up_when_recent_rv_rises():
    # calm then a sustained vol spike -> forecast should exceed the calm-regime vol
    calm = [(0.005) ** 2] * 300
    hot = [(0.03) ** 2] * 60
    rv = pd.Series(calm + hot,
                   index=pd.date_range("2022-01-01", periods=360, freq="D", tz="UTC"))
    f_hot = har_vol_forecast(rv, burn_in=200)
    calm_vol = np.sqrt((0.005) ** 2 * 252)
    assert f_hot > calm_vol * 1.5  # forecast clearly elevated by the recent spike


# --- compute_survival_target vol_forecast param ----------------------------- #
UNIVERSE = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD", "SOLUSD"]
CRYPTO = ["BTCUSD", "ETHUSD", "SOLUSD"]


def _rets(n=320, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    out = {}
    for a in UNIVERSE:
        scale = 0.02 if a in CRYPTO else 0.006
        out[a] = pd.Series(rng.normal(0, scale, n), index=idx)
    return pd.DataFrame(out)


def _call(rets, vol_forecast=None):
    return compute_survival_target(
        rets, universe=UNIVERSE, crypto=CRYPTO, vol_window=60, cov_window=120,
        target_vol=0.06, max_leverage=0.5, crypto_var_cap=0.30,
        per_asset_weight_cap=0.40, asset_weight_caps={"SOLUSD": 0.10},
        crypto_corr_stress_floor=0.95, vol_forecast=vol_forecast,
    )


def test_vol_forecast_none_is_identical_to_omitting_it():
    """PARITY GUARD: the default vol_forecast=None must not change any weight."""
    rets = _rets()
    a = compute_survival_target(
        rets, universe=UNIVERSE, crypto=CRYPTO, vol_window=60, cov_window=120,
        target_vol=0.06, max_leverage=0.5, crypto_var_cap=0.30,
        per_asset_weight_cap=0.40, asset_weight_caps={"SOLUSD": 0.10},
        crypto_corr_stress_floor=0.95,
    )
    b = _call(rets, vol_forecast=None)
    assert np.allclose(a.weights.values, b.weights.values, atol=0.0)


def test_higher_forecast_vol_reduces_that_assets_weight_and_scales_book_down():
    """A forecast that an asset will be MUCH more volatile than its sample vol must
    (a) reduce that asset's inverse-vol weight and (b) lower the book's leverage
    scale (higher ex-ante vol -> smaller k)."""
    rets = _rets()
    base = _call(rets, vol_forecast=None)
    # Forecast 5x vol for XAUUSD, sample vol elsewhere (NaN -> fall back to sample).
    vf = pd.Series({a: np.nan for a in UNIVERSE})
    # use the base sample vol proxy: derive from the returns std (annualized)
    samp = rets["XAUUSD"].std() * np.sqrt(252)
    vf["XAUUSD"] = samp * 5.0
    bumped = _call(rets, vol_forecast=vf)
    assert bumped.weights["XAUUSD"] < base.weights["XAUUSD"] - 1e-6
    assert bumped.leverage_scale <= base.leverage_scale + 1e-9
