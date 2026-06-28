"""Tests for pipeline/otr.py — LdP Optimal Trading Rule exits (B0164).

Port of D:\\PROJECTS\\QuantTradingDocs\\code\\OTR.py.txt (Lopez de Prado 2014,
"Determining optimal trading rules without backtesting"): fit a discrete O-U
process on TRAINING-day prices, Monte-Carlo a (TP, SL) mesh, rank by the
Sharpe of the rule's exit pnl, return the best pair as triple-barrier
multipliers. The 1-month tick sample is too small for an exit grid search on
real data — this derives exits from the FITTED process instead.

Leakage discipline is structural: the public entry point takes a
keyword-only ``train_prices`` argument and nothing else data-shaped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.otr import OUCoeffs, derive_otr_exit, fit_ou, otr_mesh


def simulate_ou(n=20000, hl=10.0, sigma=1.0, mean=5.0, p0=0.0, seed=0):
    """Discrete O-U exactly as OTR.py.txt: p_t = (1-phi)*forecast + phi*p_{t-1} + sigma*eps."""
    rng = np.random.default_rng(seed)
    phi = 2.0 ** (-1.0 / hl)
    p = np.empty(n)
    prev = p0
    for i in range(n):
        prev = (1 - phi) * mean + phi * prev + sigma * rng.normal()
        p[i] = prev
    return pd.Series(p)


class TestFitOU:
    def test_recovers_known_parameters(self):
        hl, sigma, mean = 10.0, 1.0, 5.0
        prices = simulate_ou(n=50000, hl=hl, sigma=sigma, mean=mean, seed=1)
        c = fit_ou(prices)
        assert isinstance(c, OUCoeffs)
        assert c.phi == pytest.approx(2.0 ** (-1.0 / hl), abs=0.02)
        assert c.sigma == pytest.approx(sigma, rel=0.05)
        assert c.forecast == pytest.approx(mean, abs=0.3)
        assert c.half_life == pytest.approx(hl, rel=0.25)

    def test_nonstationary_series_raises(self):
        # Long random walk: OLS phi_hat sits just below 1 (Dickey-Fuller
        # bias is O(1/n)), well above the 0.999 near-unit-root guard.
        rng = np.random.default_rng(2)
        rw = pd.Series(np.cumsum(rng.normal(size=200_000)))
        with pytest.raises(ValueError, match="stationar"):
            fit_ou(rw)

    def test_too_short_series_raises(self):
        with pytest.raises(ValueError, match="short"):
            fit_ou(pd.Series([1.0, 2.0, 3.0]))


class TestMesh:
    COEFFS = OUCoeffs(phi=2.0 ** (-1.0 / 10.0), forecast=0.0, sigma=1.0,
                      half_life=10.0)

    def test_mesh_covers_full_grid(self):
        rpt = np.array([1.0, 2.0])
        rsl = np.array([1.0, 2.0, 3.0])
        mesh = otr_mesh(self.COEFFS, rPT=rpt, rSLm=rsl, n_iter=500,
                        max_hp=50, seed=0)
        assert len(mesh) == 6
        assert set(mesh.columns) >= {"tp", "sl", "mean", "std", "sharpe"}
        assert set(np.unique(mesh["tp"])) == {1.0, 2.0}

    def test_deterministic_under_seed(self):
        rpt = rsl = np.array([1.0, 2.0])
        m1 = otr_mesh(self.COEFFS, rPT=rpt, rSLm=rsl, n_iter=400, max_hp=40, seed=7)
        m2 = otr_mesh(self.COEFFS, rPT=rpt, rSLm=rsl, n_iter=400, max_hp=40, seed=7)
        pd.testing.assert_frame_equal(m1, m2)

    def test_zero_drift_symmetric_rules_sharpe_near_zero(self):
        """forecast == entry: by symmetry of the process, SYMMETRIC rules
        (tp == sl) have ~0 Sharpe. (Asymmetric rules legitimately do not:
        mean reversion truncates one side faster — the OTR paper's point.)"""
        rpt = rsl = np.array([2.0, 4.0])
        mesh = otr_mesh(self.COEFFS, rPT=rpt, rSLm=rsl, n_iter=4000,
                        max_hp=100, seed=3)
        sym = mesh[mesh["tp"] == mesh["sl"]]
        assert len(sym) == 2
        assert np.abs(sym["sharpe"]).max() < 0.1
        # and the asymmetric mirror cells have opposite-signed Sharpes
        a = mesh[(mesh["tp"] == 2.0) & (mesh["sl"] == 4.0)]["sharpe"].iloc[0]
        b = mesh[(mesh["tp"] == 4.0) & (mesh["sl"] == 2.0)]["sharpe"].iloc[0]
        assert a == pytest.approx(-b, abs=0.05)

    def test_positive_drift_yields_positive_sharpe(self):
        coeffs = OUCoeffs(phi=2.0 ** (-1.0 / 10.0), forecast=2.0, sigma=1.0,
                          half_life=10.0)
        mesh = otr_mesh(coeffs, rPT=np.array([2.0]), rSLm=np.array([4.0]),
                        n_iter=3000, max_hp=100, seed=4)
        assert mesh["sharpe"].iloc[0] > 0.5


class TestDeriveExit:
    def test_returns_best_mesh_pair_in_sigma_units(self):
        # Mean-reverting series; entry 2 sigmas below the mean (overshoot
        # fade) -> positive drift toward TP -> a confident best rule exists.
        prices = simulate_ou(n=30000, hl=10.0, sigma=1.0, mean=0.0, seed=5)
        res = derive_otr_exit(
            train_prices=prices,
            entry_offset_sigmas=2.0,
            rPT=np.linspace(0.5, 3.0, 4),
            rSLm=np.linspace(0.5, 3.0, 4),
            n_iter=2000, max_hp=60, seed=6,
        )
        assert (res.tp, res.sl) in {(t, s) for t in np.linspace(0.5, 3.0, 4)
                                    for s in np.linspace(0.5, 3.0, 4)}
        best = res.mesh.loc[res.mesh["sharpe"].idxmax()]
        assert res.tp == best["tp"] and res.sl == best["sl"]
        assert res.sharpe == pytest.approx(best["sharpe"])
        assert res.sharpe > 0
        assert isinstance(res.coeffs, OUCoeffs)

    def test_train_prices_is_keyword_only(self):
        """Leakage made structurally awkward: positional data is rejected."""
        prices = simulate_ou(n=5000, seed=8)
        with pytest.raises(TypeError):
            derive_otr_exit(prices)  # noqa — intentional misuse

    def test_tp_sl_usable_as_triple_barrier_multipliers(self):
        """Returned pair is in sigma units of the fitted process — positive
        finite floats, directly usable as tp_mult/sl_mult against a per-bar
        vol estimate in pipeline.labels.triple_barrier_labels."""
        prices = simulate_ou(n=20000, hl=8.0, sigma=0.5, mean=0.0, seed=9)
        res = derive_otr_exit(train_prices=prices, entry_offset_sigmas=1.0,
                              rPT=np.array([1.0, 2.0]), rSLm=np.array([1.0, 2.0]),
                              n_iter=1000, max_hp=50, seed=10)
        assert np.isfinite(res.tp) and res.tp > 0
        assert np.isfinite(res.sl) and res.sl > 0
