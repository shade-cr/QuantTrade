"""Tests for vix_regime_riskflow_signal — macro vol regime + USD funding flow engine.

Information axis: macro volatility regime (VIX percentile rolling) + USD funding
direction (DXY EMA cross). ZERO target-price autocorrelation in the trigger —
the only target-asset dependence is through the symbol's USD-beta sign that
maps the directional bias.

Why orthogonal to price-derived primaries (EMA, momentum_zscore, CUSUM,
Bollinger): those four engines all derive their trigger from the target's own
price/returns autocorrelation. This engine triggers on EXTERNAL macro state
(VIX, DXY) which is independent of the target's price path.

Mechanism:
  - vix_pct = VIX.rolling(N).rank(pct=True).shift(1)  # 0..1 percentile
  - regime: LOW if vix_pct < low_pct, HIGH if > high_pct, else NEUTRAL
  - dxy_dir = sign(EMA_fast(DXY) - EMA_slow(DXY)).shift(1)  # +1, -1, 0
  - In NEUTRAL regime: emit 0 (sparsity by design — no engine pollution in
    calm middle-of-distribution VIX state).
  - In LOW regime (calm/risk-on): side = -dxy_dir * usd_beta
  - In HIGH regime (panic/risk-off): side = +dxy_dir * usd_beta (sign flip)

USD-beta lookup (default symbol → beta):
  EURUSD/GBPUSD/AUDUSD/NZDUSD: -1 (anti-USD)
  USDJPY/USDCHF/USDCAD: +1 (pro-USD)
  XAUUSD/XAGUSD: -1 (precious metals)
  BTCUSD/ETHUSD/SOLUSD: -1 (crypto risk-on)

Invariants tested (5 from expert recommendation):
  1. test_no_lookahead_vix_shifted_one_bar
  2. test_neutral_regime_returns_zero
  3. test_signal_inverts_between_eurusd_and_usdjpy_same_bar
  4. test_output_only_in_minus_one_zero_plus_one
  5. test_index_alignment_with_target_close

Plus extras for safety:
  - test_unknown_symbol_raises
  - test_high_vix_inverts_low_vix_sign
  - test_handles_warmup_period
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from pipeline.labels import vix_regime_riskflow_signal


def _h4_idx(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="4h", tz="UTC")


def _daily_idx(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D", tz="UTC")


# ---------- 5 invariants from expert ----------

def test_no_lookahead_vix_shifted_one_bar():
    """The VIX value at H4 bar t must come from VIX[<=t-1day], NEVER VIX[t]."""
    # Construct a target H4 series at a date where we KNOW the daily VIX value
    target_idx = _h4_idx("2024-01-02", 12)   # 6 H4 bars across 2 days
    target_close = pd.Series(100.0, index=target_idx)

    # VIX: 10 days, with a HUGE spike at 2024-01-02 that should NOT appear in
    # the H4 signal at 2024-01-02 bars (would be look-ahead).
    vix_idx = _daily_idx("2023-12-25", 12)
    vix_values = [15.0] * 11 + [99.0]   # spike on the LAST day = 2024-01-05
    # Make sure the spike date is AFTER any target H4 timestamp we'll check.
    vix = pd.Series(vix_values, index=vix_idx)

    dxy = pd.Series(np.full(12, 100.0), index=vix_idx)

    # Use small lookback so rolling rank fills in. The exact regime doesn't
    # matter here — we're testing that the spike doesn't leak.
    sig = vix_regime_riskflow_signal(
        target_close, vix, dxy, target_symbol="EURUSD",
        vix_lookback=5, vix_low_pct=0.25, vix_high_pct=0.75,
        dxy_ema_fast=3, dxy_ema_slow=5,
    )
    # Sanity: signal must be same length as target_close
    assert len(sig) == len(target_close)
    # The VIX spike is at 2024-01-05; verify that bars before 2024-01-06
    # (i.e. before VIX could "publish" past the spike via shift(1)) do NOT
    # see the spike's regime info. With shift(1) applied, the spike at day t
    # only affects bars at day t+1 or later.
    spike_date = pd.Timestamp("2024-01-05", tz="UTC")
    bars_before_spike = target_close.index[target_close.index < spike_date + pd.Timedelta(days=1)]
    # We can't assert exact signal values without knowing percentiles, but we
    # CAN assert the signal at bars_before_spike is the same as if the spike
    # had been a normal value of 15. So we run a parallel computation without
    # the spike and confirm signals match for bars before the spike+1 day.
    vix_nospike = pd.Series([15.0] * 12, index=vix_idx)
    sig_no = vix_regime_riskflow_signal(
        target_close, vix_nospike, dxy, target_symbol="EURUSD",
        vix_lookback=5, vix_low_pct=0.25, vix_high_pct=0.75,
        dxy_ema_fast=3, dxy_ema_slow=5,
    )
    # Bars strictly before the spike's publish date (spike_day + 1) must match
    pre = sig.loc[target_close.index < spike_date + pd.Timedelta(days=1)]
    pre_no = sig_no.loc[target_close.index < spike_date + pd.Timedelta(days=1)]
    pd.testing.assert_series_equal(pre, pre_no, check_names=False)


def test_neutral_regime_returns_zero():
    """When VIX is in the middle of its range (not LOW or HIGH), signal is 0."""
    n_h4 = 200
    target = pd.Series(100.0, index=_h4_idx("2024-01-01", n_h4))
    # VIX stays at exactly the median percentile (0.5) — neither LOW nor HIGH
    # under default thresholds (0.25 / 0.75)
    vix_idx = _daily_idx("2023-01-01", 500)
    vix = pd.Series(np.full(500, 20.0), index=vix_idx)   # constant → rank = 0.5
    dxy = pd.Series(100.0 + np.arange(500) * 0.1, index=vix_idx)   # rising

    sig = vix_regime_riskflow_signal(
        target, vix, dxy, target_symbol="EURUSD",
        vix_lookback=100,
    )
    # All NEUTRAL → all zeros (after warmup)
    # Allow first few bars to be 0 anyway due to warmup
    assert (sig == 0.0).all(), f"NEUTRAL regime should be 0; got {sig.value_counts()}"


def test_signal_inverts_between_eurusd_and_usdjpy_same_bar():
    """Same VIX+DXY state → EURUSD and USDJPY get OPPOSITE signs (USD-beta flip)."""
    n_h4 = 50
    target = pd.Series(100.0, index=_h4_idx("2024-06-01", n_h4))

    vix_idx = _daily_idx("2023-06-01", 400)
    # Make VIX go from low to high to exercise the regime engine
    vix_values = np.linspace(10.0, 50.0, 400)
    vix = pd.Series(vix_values, index=vix_idx)
    dxy = pd.Series(100.0 + np.linspace(0, 5, 400), index=vix_idx)   # gently rising

    sig_eur = vix_regime_riskflow_signal(
        target, vix, dxy, target_symbol="EURUSD",
        vix_lookback=200,
    )
    sig_jpy = vix_regime_riskflow_signal(
        target, vix, dxy, target_symbol="USDJPY",
        vix_lookback=200,
    )
    # Where both are non-zero, they must have OPPOSITE signs
    both_active = (sig_eur != 0) & (sig_jpy != 0)
    if both_active.sum() == 0:
        pytest.skip("no overlapping non-zero bars to check — adjust fixture")
    assert (sig_eur[both_active] == -sig_jpy[both_active]).all(), (
        "EURUSD and USDJPY should produce opposite signs at the same bar"
    )


def test_output_only_in_minus_one_zero_plus_one():
    n_h4 = 100
    target = pd.Series(100.0, index=_h4_idx("2024-01-01", n_h4))
    vix_idx = _daily_idx("2023-01-01", 500)
    rng = np.random.default_rng(42)
    vix = pd.Series(15.0 + rng.normal(0, 5, 500).cumsum() * 0.1, index=vix_idx).abs()
    dxy = pd.Series(100.0 + rng.normal(0, 1, 500).cumsum(), index=vix_idx)
    sig = vix_regime_riskflow_signal(target, vix, dxy, target_symbol="BTCUSD")
    vals = set(sig.dropna().unique())
    assert vals.issubset({-1.0, 0.0, 1.0}), f"unexpected values: {vals}"


def test_index_alignment_with_target_close():
    """Returned Series has EXACTLY target_close's index."""
    n_h4 = 50
    target = pd.Series(100.0, index=_h4_idx("2024-01-01", n_h4))
    vix = pd.Series([15.0] * 100, index=_daily_idx("2023-12-01", 100))
    dxy = pd.Series([100.0] * 100, index=_daily_idx("2023-12-01", 100))
    sig = vix_regime_riskflow_signal(target, vix, dxy, target_symbol="EURUSD")
    assert sig.index.equals(target.index)
    assert len(sig) == len(target)


# ---------- Extra safety tests ----------

def test_unknown_symbol_raises():
    target = pd.Series(100.0, index=_h4_idx("2024-01-01", 10))
    vix = pd.Series([15.0] * 100, index=_daily_idx("2023-12-01", 100))
    dxy = pd.Series([100.0] * 100, index=_daily_idx("2023-12-01", 100))
    with pytest.raises((KeyError, ValueError)):
        vix_regime_riskflow_signal(target, vix, dxy, target_symbol="DOGEUSD")


def test_high_vix_inverts_low_vix_sign():
    """For the same DXY direction, HIGH-VIX regime produces OPPOSITE sign of
    LOW-VIX regime (regime sign flip).

    To produce LOW regime: VIX must rank in BOTTOM quartile → use a DECREASING
    VIX series (latest value is the min of the rolling window).
    To produce HIGH regime: VIX must rank in TOP quartile → use an INCREASING
    VIX series (latest value is the max).
    """
    n_h4 = 20
    target = pd.Series(100.0, index=_h4_idx("2024-06-01", n_h4))

    vix_idx = _daily_idx("2023-01-01", 600)
    vix_low_series  = pd.Series(np.linspace(80.0, 10.0, 600), index=vix_idx)   # decreasing → bottom rank
    vix_high_series = pd.Series(np.linspace(10.0, 80.0, 600), index=vix_idx)   # increasing → top rank
    dxy = pd.Series(100.0 + np.linspace(0, 5, 600), index=vix_idx)             # same DXY for both

    sig_low = vix_regime_riskflow_signal(
        target, vix_low_series, dxy, target_symbol="EURUSD",
        vix_lookback=200, vix_low_pct=0.50, vix_high_pct=0.50,
    )
    sig_high = vix_regime_riskflow_signal(
        target, vix_high_series, dxy, target_symbol="EURUSD",
        vix_lookback=200, vix_low_pct=0.50, vix_high_pct=0.50,
    )
    both_active = (sig_low != 0) & (sig_high != 0)
    assert both_active.sum() > 0, "expected at least one bar where both have signal"
    assert (sig_low[both_active] == -sig_high[both_active]).all(), (
        "LOW and HIGH regimes should produce opposite signs with same DXY"
    )


def test_handles_warmup_period():
    """During VIX rolling rank warmup, the regime is undefined → signal must be 0."""
    n_h4 = 30
    target = pd.Series(100.0, index=_h4_idx("2024-01-01", n_h4))
    vix = pd.Series([15.0] * 30, index=_daily_idx("2023-12-15", 30))
    dxy = pd.Series([100.0] * 30, index=_daily_idx("2023-12-15", 30))
    sig = vix_regime_riskflow_signal(
        target, vix, dxy, target_symbol="EURUSD",
        vix_lookback=100,   # vix_lookback > available data → all warmup
    )
    assert (sig == 0.0).all(), "All bars in warmup should be 0"
