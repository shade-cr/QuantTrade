"""Tests for the load-bearing causal primitives of the market-neutral pairs spike.

The three pieces that MUST be causal/correct or the whole spike lies:
  1. half_life            — OU mean-reversion half-life of a spread.
  2. rolling_beta_zscore  — hedge ratio (beta) and z-score at bar t use ONLY
                            data <= t (no look-ahead). Rolling + shifted.
  3. dollar_neutral_legs  — leg notionals are dollar-neutral given beta.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from pipeline.pairs_spread import (
    half_life,
    rolling_beta_zscore,
    dollar_neutral_legs,
    simulate_spread_meanrev,
)


# ---------------------------------------------------------------------------
# 1. half_life — OU process
# ---------------------------------------------------------------------------
def test_half_life_recovers_known_ar1_decay():
    """A synthetic AR(1) spread with known phi has half-life = -ln(2)/ln(phi).

    spread_t - mu = phi * (spread_{t-1} - mu) + eps
    The OU half-life from the discrete AR(1) coefficient phi is ln(2)/(-ln(phi)).
    """
    rng = np.random.default_rng(0)
    n = 20000
    phi = 0.95
    mu = 1.0
    s = np.empty(n)
    s[0] = mu
    for t in range(1, n):
        s[t] = mu + phi * (s[t - 1] - mu) + rng.normal(0, 0.01)
    expected = np.log(2) / (-np.log(phi))  # ~13.51 bars
    hl = half_life(pd.Series(s))
    assert hl == pytest.approx(expected, rel=0.10)


def test_half_life_positive_for_mean_reverting():
    rng = np.random.default_rng(1)
    n = 5000
    s = np.empty(n)
    for t in range(1, n):
        s[t] = 0.9 * s[t - 1] + rng.normal(0, 0.01)
    assert half_life(pd.Series(s)) > 0


def test_half_life_nan_or_inf_for_random_walk():
    """A pure random walk is NOT mean reverting: lambda ~ 0, half-life huge/non-finite."""
    rng = np.random.default_rng(2)
    s = np.cumsum(rng.normal(0, 1, size=5000))
    hl = half_life(pd.Series(s))
    # Either non-finite or implausibly large (well outside the sane 5-500 bar
    # trading band) — a random walk has no usable mean-reversion timescale.
    assert (not np.isfinite(hl)) or hl > 500


# ---------------------------------------------------------------------------
# 2. rolling_beta_zscore — CAUSALITY is the whole point
# ---------------------------------------------------------------------------
def test_rolling_beta_zscore_no_lookahead():
    """Truncating the series AFTER bar t must not change beta[t] or z[t].

    This is the look-ahead test: the value at bar t depends only on data <= t.
    Append arbitrary future bars; values up to t must be byte-identical.
    """
    rng = np.random.default_rng(3)
    n = 800
    logp = pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + 5.0)
    logo = pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + 4.0)

    beta_full, z_full = rolling_beta_zscore(logp, logo, beta_lookback=120, z_lookback=60)

    cut = 600
    beta_cut, z_cut = rolling_beta_zscore(
        logp.iloc[:cut], logo.iloc[:cut], beta_lookback=120, z_lookback=60
    )

    # Compare the overlapping region [0, cut). Must be identical (no future leak).
    pd.testing.assert_series_equal(
        beta_full.iloc[:cut], beta_cut, check_names=False
    )
    pd.testing.assert_series_equal(
        z_full.iloc[:cut], z_cut, check_names=False
    )


def test_rolling_beta_zscore_warmup_is_nan():
    """Before enough bars for both beta and z windows, outputs are NaN."""
    rng = np.random.default_rng(4)
    n = 300
    logp = pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + 5.0)
    logo = pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + 4.0)
    beta, z = rolling_beta_zscore(logp, logo, beta_lookback=120, z_lookback=60)
    # First bar can't have a beta estimate.
    assert np.isnan(beta.iloc[0])
    assert np.isnan(z.iloc[0])
    # Deep into the series both are finite.
    assert np.isfinite(beta.iloc[-1])
    assert np.isfinite(z.iloc[-1])


def test_rolling_beta_recovers_constant_hedge_ratio():
    """If logp = 2.0 * logo + noise, rolling beta should hover near 2.0."""
    rng = np.random.default_rng(5)
    n = 1500
    logo = pd.Series(np.cumsum(rng.normal(0, 0.01, n)) + 4.0)
    logp = 2.0 * logo + rng.normal(0, 0.001, n)
    beta, _ = rolling_beta_zscore(logp, logo, beta_lookback=250, z_lookback=60)
    assert beta.iloc[-1] == pytest.approx(2.0, abs=0.1)


def test_zscore_uses_shifted_spread_stats():
    """z[t] standardizes spread[t] using mean/std of the window ending at t-1
    (or t), NOT a window that peeks past t. We assert the z-score at the last
    bar is unchanged when future data is appended (subset of the no-lookahead
    test, but isolates the z path with a deterministic spread)."""
    n = 400
    logo = pd.Series(np.linspace(4.0, 5.0, n))
    logp = pd.Series(np.linspace(8.0, 10.0, n))  # beta ~2 constant
    _, z_full = rolling_beta_zscore(logp, logo, beta_lookback=100, z_lookback=50)
    _, z_cut = rolling_beta_zscore(logp.iloc[:300], logo.iloc[:300],
                                   beta_lookback=100, z_lookback=50)
    assert z_full.iloc[299] == pytest.approx(z_cut.iloc[299], nan_ok=True)


# ---------------------------------------------------------------------------
# 3. dollar_neutral_legs
# ---------------------------------------------------------------------------
def test_dollar_neutral_legs_balanced_notional():
    """For a LONG-spread (+1) position of gross notional N on the primary,
    the other leg is SHORT beta * N (dollar-neutral hedge ratio in notional)."""
    legs = dollar_neutral_legs(direction=1, gross_notional=1000.0, beta=2.0)
    assert legs["primary_notional"] == pytest.approx(1000.0)
    assert legs["other_notional"] == pytest.approx(-2000.0)


def test_dollar_neutral_legs_short_spread_flips_both():
    legs = dollar_neutral_legs(direction=-1, gross_notional=1000.0, beta=2.0)
    assert legs["primary_notional"] == pytest.approx(-1000.0)
    assert legs["other_notional"] == pytest.approx(2000.0)


def test_dollar_neutral_legs_flat_is_zero():
    legs = dollar_neutral_legs(direction=0, gross_notional=1000.0, beta=2.0)
    assert legs["primary_notional"] == 0.0
    assert legs["other_notional"] == 0.0


# ---------------------------------------------------------------------------
# 4. simulate_spread_meanrev — round-trip accounting + dual-leg pnl
# ---------------------------------------------------------------------------
def _toy_meanrev_inputs(n=400):
    """Deterministic z that crosses -entry then reverts to 0, so we get exactly
    one clean LONG-spread round trip. Spread per-bar return crafted so the
    spread RISES while in the position (profitable for a long-spread)."""
    z = np.zeros(n)
    # bars 100..109: z deep negative (entry long spread at first cross)
    z[100:110] = -3.0
    # bars 110..119: z recovers toward 0 (exit when |z| < exit band)
    z[110:120] = 0.1
    z = pd.Series(z)
    # log legs: make primary drift up vs other so long-spread (long primary,
    # short other) earns a positive spread return over the holding window.
    logp = pd.Series(np.linspace(0.0, 0.05, n))      # primary up 5%
    logo = pd.Series(np.zeros(n))                     # other flat
    beta = pd.Series(np.ones(n))                      # beta=1 for clean accounting
    return z, logp, logo, beta


def test_simulate_one_long_round_trip_counts_and_sign():
    z, logp, logo, beta = _toy_meanrev_inputs()
    res = simulate_spread_meanrev(
        z=z, log_primary=logp, log_other=logo, beta=beta,
        entry=2.0, exit_band=0.5, stop=4.0, cost_bps_per_leg=0.0,
    )
    # exactly one round trip
    assert res["n_round_trips"] == 1
    # long spread on a rising primary vs flat other -> positive gross pnl
    assert res["trade_pnls"][0] > 0


def test_simulate_cost_reduces_pnl():
    z, logp, logo, beta = _toy_meanrev_inputs()
    gross = simulate_spread_meanrev(
        z=z, log_primary=logp, log_other=logo, beta=beta,
        entry=2.0, exit_band=0.5, stop=4.0, cost_bps_per_leg=0.0,
    )
    net = simulate_spread_meanrev(
        z=z, log_primary=logp, log_other=logo, beta=beta,
        entry=2.0, exit_band=0.5, stop=4.0, cost_bps_per_leg=5.0,
    )
    assert net["trade_pnls"][0] < gross["trade_pnls"][0]
    # both legs * (entry+exit) = 4 leg-transactions; cost = 4 * 5bps applied
    assert gross["trade_pnls"][0] - net["trade_pnls"][0] == pytest.approx(4 * 5e-4, rel=1e-6)


def test_simulate_no_entry_when_z_never_breaches():
    n = 300
    z = pd.Series(np.full(n, 0.5))  # never exceeds entry=2.0
    logp = pd.Series(np.zeros(n))
    logo = pd.Series(np.zeros(n))
    beta = pd.Series(np.ones(n))
    res = simulate_spread_meanrev(
        z=z, log_primary=logp, log_other=logo, beta=beta,
        entry=2.0, exit_band=0.5, stop=4.0, cost_bps_per_leg=2.0,
    )
    assert res["n_round_trips"] == 0
    assert len(res["trade_pnls"]) == 0


def test_simulate_no_lookahead_position_uses_prior_bar_signal():
    """Entry/exit decisions act on z at the PRIOR bar (signal on close, fill next
    bar). Truncating the series after the exit must not change the realized trade."""
    z, logp, logo, beta = _toy_meanrev_inputs()
    full = simulate_spread_meanrev(
        z=z, log_primary=logp, log_other=logo, beta=beta,
        entry=2.0, exit_band=0.5, stop=4.0, cost_bps_per_leg=1.0,
    )
    cut = 200
    part = simulate_spread_meanrev(
        z=z.iloc[:cut], log_primary=logp.iloc[:cut], log_other=logo.iloc[:cut],
        beta=beta.iloc[:cut], entry=2.0, exit_band=0.5, stop=4.0, cost_bps_per_leg=1.0,
    )
    # The single round trip closes well before bar 200, so it must be identical.
    assert full["n_round_trips"] == part["n_round_trips"] == 1
    assert full["trade_pnls"][0] == pytest.approx(part["trade_pnls"][0], rel=1e-9)
