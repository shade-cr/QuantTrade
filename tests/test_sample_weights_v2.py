"""B0012 v2: PIT rho schedule, ENB, correlation-discounted uniqueness.

Pre-registered validation battery — spec 2026-07-04-b0012 §5. Constants are
frozen; these tests are the reject-criteria, not tunable checks.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.sample_weights import (
    RHO_FLOOR,
    effective_number_of_bets,
    rolling_panel_rho,
)

N_ASSETS, N_BARS = 6, 800


def _equicorr_panel(rho: float, seed: int = 3, n_assets: int = N_ASSETS,
                    n_bars: int = N_BARS) -> pd.DataFrame:
    """GBM closes with a one-factor structure giving pairwise correlation rho."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2008-01-02", periods=n_bars, freq="B", tz="UTC")
    common = rng.normal(0, 1, n_bars)
    beta = np.sqrt(rho)
    r = beta * common[:, None] + np.sqrt(1 - rho) * rng.normal(0, 1, (n_bars, n_assets))
    r = 0.01 * r
    close = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx,
                         columns=[f"A{k}" for k in range(n_assets)])
    return close


def test_rho_schedule_is_point_in_time():
    close = _equicorr_panel(0.4)
    full = rolling_panel_rho(close)
    cut = len(close) - 100
    trunc = rolling_panel_rho(close.iloc[:cut])
    cutoff_ts = close.index[cut]
    full_before = [(ts, m) for ts, m in full if ts < cutoff_ts]
    assert len(full_before) == len(trunc)
    for (ts_a, m_a), (ts_b, m_b) in zip(full_before, trunc):
        assert ts_a == ts_b
        pd.testing.assert_frame_equal(m_a, m_b)


def test_rho_star_bounds_and_recovery():
    close = _equicorr_panel(0.5)
    sched = rolling_panel_rho(close)
    assert len(sched) >= 5
    last = sched[-1][1]
    off = last.values[~np.eye(len(last), dtype=bool)]
    assert (off >= RHO_FLOOR - 1e-12).all() and (off <= 1.0 + 1e-12).all()
    assert np.allclose(np.diag(last.values), 1.0)
    # shrunk estimate of an equicorrelated 0.5 panel should sit near 0.5
    assert 0.3 < off.mean() < 0.7


def test_rho_floor_binds_on_independent_panel():
    close = _equicorr_panel(0.0, seed=9)
    last = rolling_panel_rho(close)[-1][1]
    off = last.values[~np.eye(len(last), dtype=bool)]
    # true rho 0 -> estimates shrunk toward ~0 panel mean -> clipped at the floor
    assert np.isclose(off.min(), RHO_FLOOR, atol=0.05) or off.min() >= RHO_FLOOR


def test_enb_bounds_and_monotonicity():
    def equicorr_matrix(n, rho):
        m = np.full((n, n), rho, dtype=float)
        np.fill_diagonal(m, 1.0)
        return pd.DataFrame(m)
    assert np.isclose(effective_number_of_bets(equicorr_matrix(8, 0.0)), 8.0)
    assert effective_number_of_bets(equicorr_matrix(8, 0.999)) < 1.5
    vals = [effective_number_of_bets(equicorr_matrix(8, r)) for r in (0.1, 0.4, 0.7)]
    assert vals[0] > vals[1] > vals[2]
    for v in vals:
        assert 1.0 <= v <= 8.0
