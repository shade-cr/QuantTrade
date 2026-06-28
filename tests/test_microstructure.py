"""Tests for pipeline.microstructure â€” Kyle's lambda price-impact features (AFML Â§19.4.1).

The canonical model (AFML Â§19.4.1, eq. around book line 6895) is

    Î”p_t = Î» Â· (b_t Â· V_t) + Îµ

where b_t âˆˆ {-1, +1} is the trade aggressor sign (proxied by the tick rule on bar
closes) and b_t Â· V_t is signed volume / net order flow. LdP (Â§19.4, line 6866)
prefers the *t-value* of Î» over the mean estimate, because the t-value is re-scaled
by the standard deviation of the estimation error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.microstructure import (
    amihud_lambda,
    kyle_lambda,
    kyle_lambda_tvalue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_known_impact(n: int, lam_true: float, noise: float, seed: int):
    """Construct a (close, volume) pair where dp_t = lam_true*(b_t*V_t) + noise.

    We pick the aggressor signs and volumes first, build the signed-volume
    driver, then integrate dp into a close series. Because b_t is recovered
    downstream as sign(close.diff()), we choose dp so that sign(dp_t) == b_t
    (lam_true>0 and dominating noise makes this hold by construction).
    """
    rng = np.random.default_rng(seed)
    b = rng.choice([-1.0, 1.0], size=n)
    vol = rng.uniform(1.0, 5.0, size=n)
    signed_vol = b * vol
    eps = rng.normal(0.0, noise, size=n)
    dp = lam_true * signed_vol + eps
    close = pd.Series(100.0 + np.cumsum(dp))
    volume = pd.Series(vol)
    return close, volume, b


# ---------------------------------------------------------------------------
# Known-impact recovery
# ---------------------------------------------------------------------------
def test_kyle_lambda_recovers_positive_slope():
    close, volume, _ = _build_known_impact(n=400, lam_true=0.5, noise=0.02, seed=1)
    lam = kyle_lambda(close, volume, window=40)
    mature = lam.iloc[60:]
    assert mature.notna().all()
    # Slope should be positive and near lam_true.
    assert mature.median() > 0
    assert abs(mature.median() - 0.5) < 0.25


def test_kyle_lambda_tvalue_large_and_positive_under_real_impact():
    close, volume, _ = _build_known_impact(n=400, lam_true=0.5, noise=0.02, seed=2)
    tval = kyle_lambda_tvalue(close, volume, window=40)
    mature = tval.iloc[60:]
    assert mature.notna().all()
    # Strong, consistent impact => large positive t-values.
    assert mature.median() > 2.0
    assert (mature > 2.0).mean() > 0.8


def test_kyle_lambda_tvalue_smaller_when_no_volume_impact_relationship():
    # NULL regime: the *magnitude* of dp is unrelated to volume. Because the
    # tick-rule proxy ties b_t = sign(dp_t) to the response, the signed-volume
    # regressor always co-moves in sign with dp (documented proxy bias), so the
    # t-value is not centered at zero. The honest, falsifiable claim is that a
    # genuine impact relationship (|dp| scaling with V) yields a SUBSTANTIALLY
    # larger t-value than the null where |dp| is i.i.d. and volume carries no
    # magnitude information.
    rng = np.random.default_rng(3)
    n = 400
    window = 40

    # Null: |dp| i.i.d., volume independent.
    b_null = rng.choice([-1.0, 1.0], size=n)
    dp_null = b_null * np.abs(rng.normal(0.0, 1.0, size=n))
    close_null = pd.Series(100.0 + np.cumsum(dp_null))
    vol_null = pd.Series(rng.uniform(1.0, 5.0, size=n))
    t_null = kyle_lambda_tvalue(close_null, vol_null, window=window).iloc[60:].dropna()

    # Impact: |dp| scales with volume (true Kyle relationship), low noise.
    close_imp, vol_imp, _ = _build_known_impact(n=n, lam_true=0.5, noise=0.02, seed=33)
    t_imp = kyle_lambda_tvalue(close_imp, vol_imp, window=window).iloc[60:].dropna()

    assert t_imp.median() > t_null.median()
    # And the genuine-impact t-values clear the conventional significance bar.
    assert t_imp.median() > 2.0


# ---------------------------------------------------------------------------
# Causality (no look-ahead)
# ---------------------------------------------------------------------------
def test_kyle_lambda_tvalue_is_causal(synth_ohlcv):
    close = synth_ohlcv["close"].copy()
    volume = synth_ohlcv["volume"].copy()
    base = kyle_lambda_tvalue(close, volume, window=20)

    # Mutate a FUTURE close bar at position 300.
    mutate_at = 300
    close2 = close.copy()
    close2.iloc[mutate_at] = close2.iloc[mutate_at] * 1.10
    mutated = kyle_lambda_tvalue(close2, volume, window=20)

    # All values strictly BEFORE the mutated bar must be unchanged.
    before_base = base.iloc[:mutate_at]
    before_mut = mutated.iloc[:mutate_at]
    pd.testing.assert_series_equal(before_base, before_mut)


def test_kyle_lambda_is_causal(synth_ohlcv):
    close = synth_ohlcv["close"].copy()
    volume = synth_ohlcv["volume"].copy()
    base = kyle_lambda(close, volume, window=20)
    mutate_at = 250
    close2 = close.copy()
    close2.iloc[mutate_at] = close2.iloc[mutate_at] * 0.9
    mutated = kyle_lambda(close2, volume, window=20)
    pd.testing.assert_series_equal(base.iloc[:mutate_at], mutated.iloc[:mutate_at])


# ---------------------------------------------------------------------------
# NaN warm-up, index/length preservation
# ---------------------------------------------------------------------------
def test_warmup_nan_and_index_preserved(synth_ohlcv):
    close = synth_ohlcv["close"]
    volume = synth_ohlcv["volume"]
    window = 20
    out = kyle_lambda_tvalue(close, volume, window=window)
    assert len(out) == len(close)
    assert out.index.equals(close.index)
    # A full trailing window of length `window` first closes at index window-1,
    # so the first `window-1` rows lack sufficient history => NaN.
    assert out.iloc[: window - 1].isna().all()


def test_kyle_lambda_index_preserved_with_datetime_index(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    out = kyle_lambda(df["close"], df["volume"], window=20)
    assert out.index.equals(df.index)
    assert isinstance(out.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# Degenerate guards
# ---------------------------------------------------------------------------
def test_constant_close_gives_nan_no_exception():
    # Constant close => dp == 0 => zero-variance regressor => NaN, no crash.
    close = pd.Series(np.full(100, 50.0))
    volume = pd.Series(np.random.default_rng(0).uniform(1, 5, size=100))
    lam = kyle_lambda(close, volume, window=20)
    tval = kyle_lambda_tvalue(close, volume, window=20)
    assert lam.isna().all()
    assert tval.isna().all()


def test_zero_variance_signed_volume_gives_nan():
    # Monotone increasing close => b_t == +1 everywhere; constant volume =>
    # signed volume is constant => zero regressor variance => NaN.
    close = pd.Series(np.arange(100, dtype=float))  # strictly increasing
    volume = pd.Series(np.full(100, 3.0))
    tval = kyle_lambda_tvalue(close, volume, window=20)
    mature = tval.iloc[30:]
    assert mature.isna().all()


def test_zero_volume_bars_no_exception():
    rng = np.random.default_rng(5)
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, size=100)))
    volume = pd.Series(rng.uniform(0, 5, size=100))
    volume.iloc[10:15] = 0.0  # some zero-volume bars
    # Should not raise.
    lam = kyle_lambda(close, volume, window=20)
    tval = kyle_lambda_tvalue(close, volume, window=20)
    assert len(lam) == len(close)
    assert len(tval) == len(close)


def test_short_series_all_nan():
    close = pd.Series([1.0, 2.0])
    volume = pd.Series([1.0, 1.0])
    out = kyle_lambda_tvalue(close, volume, window=20)
    assert out.isna().all()
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Amihud (optional diagnostic)
# ---------------------------------------------------------------------------
def test_amihud_lambda_nonnegative_and_causal(synth_ohlcv):
    close = synth_ohlcv["close"]
    volume = synth_ohlcv["volume"]
    out = amihud_lambda(close, volume, window=20)
    assert len(out) == len(close)
    mature = out.iloc[40:].dropna()
    assert (mature >= 0).all()  # |log_ret| / (close*volume) is non-negative

    # Causality.
    mutate_at = 200
    close2 = close.copy()
    close2.iloc[mutate_at] = close2.iloc[mutate_at] * 1.05
    mutated = amihud_lambda(close2, volume, window=20)
    pd.testing.assert_series_equal(out.iloc[:mutate_at], mutated.iloc[:mutate_at])


# ---------------------------------------------------------------------------
# B0135 â€” Corwin-Schultz high-low spread
# ---------------------------------------------------------------------------
from pipeline.microstructure import corwin_schultz_spread  # noqa: E402


def _cs_ohlc(n=300, seed=11, spread_frac=0.01):
    """Random-walk close with a synthetic bid-ask bounce of known size:
    high prints at ask (mid*(1+s/2)), low at bid (mid*(1-s/2))."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    mid = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    high = pd.Series(mid * (1 + spread_frac / 2), index=idx)
    low = pd.Series(mid * (1 - spread_frac / 2), index=idx)
    return high, low


def test_cs_spread_nonnegative_and_bounded():
    high, low = _cs_ohlc()
    s = corwin_schultz_spread(high, low)
    valid = s.dropna()
    assert (valid >= 0).all()
    assert (valid < 1.0).all()


def test_cs_spread_recovers_synthetic_spread_order_of_magnitude():
    """With a pure bounce (tiny true vol), the estimator should land in the
    neighbourhood of the planted 1% spread â€” order of magnitude, not exact."""
    rng = np.random.default_rng(5)
    n = 400
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    mid = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 1e-5, n))), index=idx)
    high = mid * 1.005
    low = mid * 0.995
    s = corwin_schultz_spread(high, low).dropna()
    assert 0.002 < s.median() < 0.05


def test_cs_spread_zero_when_trend_dominates():
    """Strong drift with narrow intra-bar ranges: the joint 2-bar range
    (gamma) far exceeds what the single-bar ranges (beta) imply, driving
    alpha negative -> clamped to 0 (the authors clamp)."""
    n = 300
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    mid = pd.Series(100.0 * np.exp(np.arange(n) * 0.01), index=idx)  # +1%/bar drift
    high = mid * 1.0005   # 10bp intra-bar range << 1% bar-to-bar displacement
    low = mid * 0.9995
    s = corwin_schultz_spread(high, low).dropna()
    assert (s == 0).mean() > 0.9


def test_cs_spread_causal_prefix_stability():
    high, low = _cs_ohlc()
    full = corwin_schultz_spread(high, low)
    k = 200
    prefix = corwin_schultz_spread(high.iloc[:k], low.iloc[:k])
    pd.testing.assert_series_equal(full.iloc[:k], prefix)


def test_cs_spread_future_mutation_does_not_change_past():
    high, low = _cs_ohlc()
    a = corwin_schultz_spread(high, low)
    high2, low2 = high.copy(), low.copy()
    high2.iloc[250] *= 2.0
    b = corwin_schultz_spread(high2, low2)
    pd.testing.assert_series_equal(a.iloc[:250], b.iloc[:250])
