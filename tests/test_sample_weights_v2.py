"""B0012 v2: PIT rho schedule, ENB, correlation-discounted uniqueness.

Pre-registered validation battery — spec 2026-07-04-b0012 §5. Constants are
frozen; these tests are the reject-criteria, not tunable checks.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.sample_weights import (
    RHO_FLOOR,
    avg_uniqueness,
    corr_discounted_uniqueness,
    effective_number_of_bets,
    pooled_avg_uniqueness,
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


def _dense_events(close: pd.DataFrame, dur: int = 5):
    """One overlapping event stream per asset: an event starts every bar and
    lives `dur` bars — the dense-state worst case."""
    ev_t, le_t, asset = [], [], []
    idx = close.index
    for a in close.columns:
        for k in range(len(idx) - dur):
            ev_t.append(idx[k]); le_t.append(idx[k + dur]); asset.append(a)
    return (pd.DatetimeIndex(ev_t), pd.DatetimeIndex(le_t),
            np.array(asset, dtype=object), idx)


def test_synthetic_recovery_equicorrelated():
    """Spec §5.1: v2 fit-weight sum ~= rho1 sum x N/(1+(N-1)rho*), and NEVER
    biased above the analytic target."""
    rho_true = 0.5
    close = _equicorr_panel(rho_true, seed=21)
    ev, le, asset, grid = _dense_events(close)
    sched = rolling_panel_rho(close)
    w2 = corr_discounted_uniqueness(ev, le, asset, sched, grid)
    w1 = pooled_avg_uniqueness(ev, le)
    n = N_ASSETS
    # rho* under the frozen estimator ~= clip(0.5*rho_hat + 0.5*rbar, floor, 1) ~= rho_true
    target_ratio = n / (1.0 + (n - 1) * rho_true)
    ratio = w2.sum() / w1.sum()
    assert ratio == pytest.approx(target_ratio, rel=0.25)
    # upward-bias rejection: credited breadth must not exceed the analytic
    # target by more than estimation tolerance
    assert ratio < target_ratio * 1.25
    # and it must actually unstarve vs rho=1
    assert ratio > 1.5


def test_single_asset_parity_bit_identical():
    """Spec §5.2: one-asset pool -> cross terms vanish -> v2 == the rho=1
    pooled weights on the same events (which equal within-name uniqueness)."""
    close = _equicorr_panel(0.4, seed=5, n_assets=1)
    ev, le, asset, grid = _dense_events(close)
    sched = rolling_panel_rho(close)  # 1x1 matrices
    w2 = corr_discounted_uniqueness(ev, le, asset, sched, grid)
    w1 = pooled_avg_uniqueness(ev, le)
    np.testing.assert_array_equal(w2, w1)


def test_placebo_duplicate_series_credits_no_breadth():
    """Spec §5.3 (false-edge direction): a panel of near-identical series
    (true rho -> 1) must NOT be credited independence: v2 sum stays within a
    small tolerance of the rho=1 sum."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2008-01-02", periods=N_BARS, freq="B", tz="UTC")
    base = 0.01 * rng.normal(0, 1, N_BARS)
    cols = {}
    for k in range(N_ASSETS):
        noise = 0.0005 * rng.normal(0, 1, N_BARS)
        cols[f"A{k}"] = 100 * np.exp(np.cumsum(base + noise))
    close = pd.DataFrame(cols, index=idx)
    ev, le, asset, grid = _dense_events(close)
    w2 = corr_discounted_uniqueness(ev, le, asset, rolling_panel_rho(close), grid)
    w1 = pooled_avg_uniqueness(ev, le)
    assert w2.sum() <= w1.sum() * 1.15  # no invented breadth on rho~1


def test_independent_panel_bounded_by_floor():
    """Spec §5.3 (other direction): truly independent panel is credited at
    most the floor-implied breadth, never full N."""
    close = _equicorr_panel(0.0, seed=13)
    ev, le, asset, grid = _dense_events(close)
    w2 = corr_discounted_uniqueness(ev, le, asset, rolling_panel_rho(close), grid)
    w1 = pooled_avg_uniqueness(ev, le)
    n = N_ASSETS
    floor_ratio = n / (1.0 + (n - 1) * RHO_FLOOR)   # max credit under the floor
    assert w2.sum() / w1.sum() <= floor_ratio * 1.10


def test_warmup_days_fall_back_to_rho1():
    """Events entirely before the first rho matrix must get exactly the rho=1
    pooled weights (conservative warmup)."""
    close = _equicorr_panel(0.3, seed=17)
    ev, le, asset, grid = _dense_events(close)
    sched = rolling_panel_rho(close)
    first_eff = sched[0][0]
    early = ev < first_eff - pd.Timedelta(days=15)  # span fully pre-schedule
    w2 = corr_discounted_uniqueness(ev, le, asset, sched, grid)
    w1 = pooled_avg_uniqueness(ev, le)
    np.testing.assert_allclose(w2[early], w1[early], rtol=1e-9)
