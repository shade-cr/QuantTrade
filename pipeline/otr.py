"""Optimal Trading Rule (OTR) exits from an O-U fit — no exit grid search (B0164).

Port of Lopez de Prado (2014), "Determining optimal trading rules without
backtesting" (reference implementation:
``D:\\PROJECTS\\QuantTradingDocs\\code\\OTR.py.txt``). The 18 usable days of
the Jun-15 tick sample cannot support an honest TP/SL grid search (every grid
point is a trial against the same tiny sample); instead we:

1. fit a discrete Ornstein-Uhlenbeck process
   ``p_t = (1 - phi) * forecast + phi * p_{t-1} + sigma * eps_t``
   on TRAINING-day prices only (:func:`fit_ou`);
2. Monte-Carlo the FITTED process over a (TP, SL) mesh and rank each rule by
   the Sharpe of its exit pnl (:func:`otr_mesh` — same loop as OTR.py.txt's
   ``batch``, vectorized with common random paths across mesh cells);
3. return the argmax pair (:func:`derive_otr_exit`) in SIGMA UNITS of the
   fitted process, directly usable as ``tp_mult`` / ``sl_mult`` against a
   per-bar vol estimate in :func:`pipeline.labels.triple_barrier_labels`.

Leakage discipline: the only public entry point that touches data,
:func:`derive_otr_exit`, takes a KEYWORD-ONLY ``train_prices`` argument and
nothing else data-shaped — feeding it validation/holdout prices requires
typing ``train_prices=<not training data>``, which is deliberately awkward
and trivially greppable in review.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OUCoeffs:
    """Discrete O-U coefficients in OTR.py.txt's parametrization."""

    phi: float        #: AR(1) persistence, phi = 2 ** (-1 / half_life)
    forecast: float   #: long-run mean the process reverts to
    sigma: float      #: innovation std per step
    half_life: float  #: -1 / log2(phi), in steps


@dataclass(frozen=True)
class OTRResult:
    tp: float            #: optimal profit-take, in sigma units
    sl: float            #: optimal stop-loss, in sigma units (positive)
    sharpe: float        #: per-exit Sharpe of the best rule (NOT annualized)
    coeffs: OUCoeffs
    mesh: pd.DataFrame   #: full (tp, sl, mean, std, sharpe) diagnostic grid


def fit_ou(prices: pd.Series, *, min_obs: int = 50,
           max_phi: float = 0.999) -> OUCoeffs:
    """Fit the discrete O-U by OLS of ``p_t`` on ``p_{t-1}`` (AR(1)).

    slope = phi; intercept = (1 - phi) * forecast; sigma = residual std.
    Raises if the series is too short or not mean-reverting
    (``phi >= max_phi``; the default 0.999 ~ half-life 693 steps, far beyond
    any intraday holding period — and a finite-sample random walk fits phi
    just BELOW 1, so testing ``phi >= 1`` alone would wave unit roots
    through). A near-unit-root fit on a price LEVEL series usually means the
    caller should hand in a de-trended / spread / z-scored series instead.
    """
    p = pd.Series(prices).astype("float64").dropna().to_numpy()
    if len(p) < min_obs:
        raise ValueError(
            f"fit_ou: series too short ({len(p)} < {min_obs} observations)."
        )
    x, y = p[:-1], p[1:]
    x_var = x.var()
    if x_var <= 0:
        raise ValueError("fit_ou: constant series.")
    phi = float(np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1))
    if phi >= max_phi:
        raise ValueError(
            f"fit_ou: phi={phi:.5f} >= {max_phi} — series is not stationary/"
            f"mean-reverting at any tradable horizon; OTR exits are "
            f"undefined. De-trend or z-score first."
        )
    intercept = float(y.mean() - phi * x.mean())
    forecast = intercept / (1.0 - phi)
    resid = y - (intercept + phi * x)
    sigma = float(resid.std(ddof=2))
    half_life = float(-1.0 / np.log2(phi)) if phi > 0 else 0.0
    return OUCoeffs(phi=phi, forecast=forecast, sigma=sigma,
                    half_life=half_life)


def _simulate_cp_paths(
    coeffs: OUCoeffs, seed_price: float, n_iter: int, max_hp: int, seed: int
) -> np.ndarray:
    """(n_iter, max_hp) matrix of cumulated pnl ``cP_t = p_t - seed_price``
    under the fitted process, starting at ``p_0 = seed_price`` — the inner
    ``while`` loop of OTR.py.txt's ``batch``, vectorized across iterations.
    Common random paths are reused across mesh cells (variance reduction:
    cells are ranked on identical draws)."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((n_iter, max_hp))
    p = np.empty((n_iter, max_hp))
    prev = np.full(n_iter, float(seed_price))
    drift = (1.0 - coeffs.phi) * coeffs.forecast
    for t in range(max_hp):
        prev = drift + coeffs.phi * prev + coeffs.sigma * eps[:, t]
        p[:, t] = prev
    return p - float(seed_price)


def otr_mesh(
    coeffs: OUCoeffs,
    *,
    rPT: np.ndarray,
    rSLm: np.ndarray,
    n_iter: int = 10_000,
    max_hp: int = 100,
    seed: int = 0,
    seed_price: float = 0.0,
) -> pd.DataFrame:
    """Sharpe of every (TP, SL) rule on the fitted process (OTR ``batch``).

    ``rPT`` / ``rSLm`` are in SIGMA UNITS of the fitted process (the original
    uses absolute units with sigma=1; scaling by ``coeffs.sigma`` makes the
    mesh vol-relative and the output reusable as barrier multipliers). For
    each cell, a position opened at ``seed_price`` exits at the first step
    where ``cP > tp*sigma`` or ``cP < -sl*sigma``, else at ``max_hp``
    (timeout); the cell's score is mean/std of the exit pnl across paths —
    exactly the ranking OTR.py.txt prints.
    """
    cp = _simulate_cp_paths(coeffs, seed_price, int(n_iter), int(max_hp), seed)
    timeout_pnl = cp[:, -1]
    rows = []
    idx = np.arange(cp.shape[0])
    for tp in np.asarray(rPT, dtype=float):
        for sl in np.asarray(rSLm, dtype=float):
            hit = (cp > tp * coeffs.sigma) | (cp < -sl * coeffs.sigma)
            any_hit = hit.any(axis=1)
            first = hit.argmax(axis=1)
            pnl = np.where(any_hit, cp[idx, first], timeout_pnl)
            mean, std = float(pnl.mean()), float(pnl.std(ddof=1))
            rows.append({
                "tp": float(tp), "sl": float(sl), "mean": mean, "std": std,
                "sharpe": mean / std if std > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


def derive_otr_exit(
    *,
    train_prices: pd.Series,
    entry_offset_sigmas: float = 0.0,
    rPT: np.ndarray | None = None,
    rSLm: np.ndarray | None = None,
    n_iter: int = 10_000,
    max_hp: int = 100,
    seed: int = 0,
) -> OTRResult:
    """Derive the optimal (TP, SL) pair for triple-barrier exits.

    Args:
      train_prices: TRAINING-day price series ONLY (keyword-only by design —
        see module docstring). For the overshoot-reversion primary this is
        the same vol-bar close/z-score series the entry signal is built on.
      entry_offset_sigmas: entry displacement from the O-U mean, in sigma
        units, in the ADVERSE direction the strategy fades (an overshoot
        fade enters ~k sigmas away from the mean and profits from the pull
        back; 0 = enter at the mean, i.e. zero expected drift).
      rPT / rSLm: TP / SL mesh in sigma units (default: OTR.py.txt's
        ``linspace(0, 10, 21)`` without the degenerate 0 point).

    Returns the mesh-argmax (tp, sl) in sigma units, its per-exit Sharpe,
    the fitted coefficients, and the full mesh for the report.
    """
    if rPT is None:
        rPT = np.linspace(0.5, 10.0, 20)
    if rSLm is None:
        rSLm = np.linspace(0.5, 10.0, 20)
    coeffs = fit_ou(train_prices)
    # Enter `entry_offset_sigmas` BELOW the long-run mean (long-side
    # convention; the process is symmetric, shorts mirror).
    seed_price = coeffs.forecast - entry_offset_sigmas * coeffs.sigma
    mesh = otr_mesh(coeffs, rPT=rPT, rSLm=rSLm, n_iter=n_iter,
                    max_hp=max_hp, seed=seed, seed_price=seed_price)
    best = mesh.loc[mesh["sharpe"].idxmax()]
    return OTRResult(
        tp=float(best["tp"]),
        sl=float(best["sl"]),
        sharpe=float(best["sharpe"]),
        coeffs=coeffs,
        mesh=mesh,
    )
