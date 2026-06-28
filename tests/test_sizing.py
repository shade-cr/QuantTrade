"""B0001 — position-sizing diagnostics (Gaussian vs empirical Kelly)."""
from __future__ import annotations

import numpy as np

from pipeline.sizing import gaussian_kelly, empirical_kelly, kelly_sizing


def test_gaussian_kelly_basic_formula():
    rng = np.random.default_rng(0)
    r = rng.normal(0.01, 0.05, size=2000)
    f = gaussian_kelly(r)
    assert np.isclose(f, r.mean() / r.var(ddof=1), rtol=1e-6)
    assert f > 0


def test_gaussian_kelly_zero_when_mean_nonpositive():
    r = np.array([-0.02, 0.01, -0.03, 0.005, -0.01])  # negative mean
    assert gaussian_kelly(r) == 0.0


def test_empirical_kelly_zero_when_mean_nonpositive():
    r = np.array([-0.02, 0.01, -0.03, 0.005, -0.01])
    assert empirical_kelly(r) == 0.0


def test_empirical_kelly_respects_no_bankruptcy_bound():
    # Worst loss is -0.25 -> f must stay below 1/0.25 = 4 so 1 + f*r > 0.
    r = np.array([0.30, 0.20, -0.25, 0.10, 0.15, -0.05])
    f = empirical_kelly(r)
    assert 0.0 < f < 4.0
    # All 1 + f*r strictly positive (no bankruptcy).
    assert np.all(1.0 + f * r > 0)


def test_empirical_more_conservative_under_negative_skew():
    """Core motivation: with a heavy left tail, empirical Kelly < Gaussian Kelly."""
    rng = np.random.default_rng(7)
    # Mostly small wins, occasional large losses -> negative skew, positive mean.
    base = rng.normal(0.02, 0.01, size=1000)
    base[::50] = -0.20  # rare big losses
    assert base.mean() > 0
    g = gaussian_kelly(base)
    e = empirical_kelly(base)
    assert g > 0 and e > 0
    assert e < g


def test_no_losses_caps_at_max_fraction():
    r = np.array([0.01, 0.02, 0.015, 0.03])  # all positive -> growth monotone in f
    # Bounded optimizer lands at the cap (within solver tolerance).
    assert empirical_kelly(r, max_fraction=10.0) >= 10.0 * (1 - 1e-4)


def test_kelly_sizing_block_shape():
    rng = np.random.default_rng(1)
    r = rng.normal(0.01, 0.04, size=500)
    block = kelly_sizing(r)
    assert block["n_trades"] == 500
    assert np.isclose(block["half_gaussian_kelly"], block["gaussian_kelly"] / 2)
    assert np.isclose(block["half_empirical_kelly"], block["empirical_kelly"] / 2)
    assert np.isfinite(block["skew_pnl"])
    assert block["empirical_to_gaussian_ratio"] > 0


def test_empty_and_tiny_series_are_safe():
    assert gaussian_kelly(np.array([])) == 0.0
    assert empirical_kelly(np.array([])) == 0.0
    assert gaussian_kelly(np.array([0.01])) == 0.0  # n<2
    block = kelly_sizing(np.array([]))
    assert block["n_trades"] == 0
    assert block["gaussian_kelly"] == 0.0
    assert block["empirical_kelly"] == 0.0


def test_nan_values_filtered():
    r = np.array([0.01, np.nan, 0.02, -0.01, np.nan, 0.015])
    block = kelly_sizing(r)
    assert block["n_trades"] == 4  # NaNs dropped


# ---------------------------------------------------------------------------
# B0120 — AFML ch.10 probability-aware bet sizing
# ---------------------------------------------------------------------------
from pipeline.sizing import (  # noqa: E402
    bet_size_from_prob,
    average_active_bets,
    discretize_bet,
    bet_sizing_diagnostics,
)


def test_bet_size_zero_at_coin_flip_and_monotone():
    p = np.array([0.5, 0.55, 0.65, 0.80, 0.95])
    m = bet_size_from_prob(p)
    assert abs(m[0]) < 1e-12
    assert np.all(np.diff(m) > 0), "size must be monotone in p"
    assert np.all(m <= 1.0)


def test_bet_size_antisymmetric_and_saturating():
    m_lo = bet_size_from_prob(np.array([0.2]))[0]
    m_hi = bet_size_from_prob(np.array([0.8]))[0]
    assert np.isclose(m_lo, -m_hi, atol=1e-12)
    # extreme confidence saturates near +/-1 without inf/nan (clip_eps)
    m_ext = bet_size_from_prob(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(m_ext))
    assert m_ext[0] < -0.999 and m_ext[1] > 0.999


def test_bet_size_known_value():
    # p=0.75: z = 0.25/sqrt(0.1875) = 0.57735, m = 2*Phi(z)-1 ~= 0.43621
    m = bet_size_from_prob(np.array([0.75]))[0]
    assert np.isclose(m, 0.4362, atol=2e-4)


def test_average_active_bets_overlap_window():
    # Event 1 (bars 0-10, size 1.0) still active when event 2 starts (bar 5,
    # size 0.0) -> event 2's averaged size is mean(1.0, 0.0) = 0.5.
    bets = np.array([1.0, 0.0, 0.6])
    start = np.array([0, 5, 20])
    end = np.array([10, 15, 30])
    avg = average_active_bets(bets, start, end)
    assert np.isclose(avg[0], 1.0)     # only itself active at bar 0
    assert np.isclose(avg[1], 0.5)     # events 1+2 active at bar 5
    assert np.isclose(avg[2], 0.6)     # disjoint -> only itself


def test_average_active_bets_excludes_nan():
    bets = np.array([np.nan, 0.4])
    start = np.array([0, 5])
    end = np.array([10, 15])
    avg = average_active_bets(bets, start, end)
    assert np.isnan(avg[0])            # only a NaN bet active -> no measurement
    assert np.isclose(avg[1], 0.4)     # NaN neighbour excluded from the mean


def test_discretize_bet_rounds_and_clips():
    m = np.array([0.07, -0.12, 1.4, -1.4, 0.024])
    d = discretize_bet(m, step=0.05)
    assert np.allclose(d, [0.05, -0.10, 1.0, -1.0, 0.0])


def test_discretize_rejects_bad_step():
    import pytest
    with pytest.raises(ValueError):
        discretize_bet(np.array([0.5]), step=0.0)


def test_bet_sizing_diagnostics_contract():
    rng = np.random.default_rng(3)
    n = 200
    probs = rng.uniform(0.3, 0.8, n)
    take = probs >= 0.55
    start = np.arange(n)
    end = start + 5
    block = bet_sizing_diagnostics(probs, take, start, end)
    assert block["n_events"] == n
    assert block["n_taken"] == int(take.sum())
    assert 0.0 <= block["mean_size_taken"] <= 1.0
    assert block["max_size_taken"] <= 1.0
    assert block["mean_abs_size_change"] >= 0.0


def test_bet_sizing_diagnostics_no_trade_is_floored_not_flipped():
    """A below-coin-flip prob on an untaken event must contribute size 0,
    never a negative (counter-trade) size."""
    probs = np.array([0.2, 0.2, 0.2])
    take = np.array([False, False, False])
    block = bet_sizing_diagnostics(probs, take, np.arange(3), np.arange(3) + 2)
    assert block["n_taken"] == 0
    assert np.isnan(block["mean_size_taken"])  # nothing taken -> no level stat


def test_average_active_bets_oom_guard_degrades_to_nan():
    """Risk-officer caveat: pooled-scale event counts must not OOM the n^2
    broadcast — the guard degrades to NaN with a warning."""
    import pytest
    n = 20_001
    with pytest.warns(RuntimeWarning, match="broadcast"):
        out = average_active_bets(np.zeros(n), np.arange(n), np.arange(n) + 1)
    assert np.isnan(out).all()
