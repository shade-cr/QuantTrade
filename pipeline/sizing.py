"""Position-sizing diagnostics (B0001) + AFML ch.10 probability-aware sizing (B0120).

Growth-optimal leverage estimates from a realized per-trade PnL series, and the
2N(z)-1 calibrated-probability bet sizer with concurrency averaging and
discretization. These are INFORMATIONAL ONLY — reported in summary.json, never
wired into live order sizing here. The point of computing both Gaussian and empirical Kelly is the
gap between them: meta-filtered PnL usually has a heavy left tail (the meta
filters winners more aggressively than losers in some regimes), and the
Gaussian formula f* = mu/sigma^2 overstates leverage exactly when skew is
negative. Empirical Kelly maximizes realized log-growth directly, so it is
automatically more conservative under negative skew.

maxdama.pdf §6.4.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def gaussian_kelly(pnl) -> float:
    """Gaussian Kelly fraction f* = mu / sigma^2 on per-trade PnL.

    Returns 0.0 when mean <= 0 (don't allocate) or variance is non-positive.
    Clamped at 0 — this is long-the-strategy sizing, never a short of it.
    """
    r = np.asarray(pnl, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    mu = float(r.mean())
    var = float(r.var(ddof=1))
    if var <= 0.0 or mu <= 0.0:
        return 0.0
    return mu / var


def empirical_kelly(pnl, *, max_fraction: float = 10.0) -> float:
    """Growth-optimal fraction maximizing mean(log(1 + f * r)) on realized PnL.

    Bounded by the no-bankruptcy constraint 1 + f * r_i > 0 for every trade,
    i.e. f < 1 / |worst_loss|. Returns 0.0 when mean <= 0 (don't allocate).
    When there are no losing trades the growth is monotone-increasing in f and
    the constraint is vacuous, so the result is capped at `max_fraction`.
    Clamped at 0 — long-the-strategy only.
    """
    r = np.asarray(pnl, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or float(r.mean()) <= 0.0:
        return 0.0
    losses = r[r < 0]
    if losses.size == 0:
        upper = max_fraction
    else:
        worst = float(-losses.min())  # magnitude of the most negative trade
        upper = min(max_fraction, (1.0 / worst) * (1.0 - 1e-6))
    if upper <= 0.0:
        return 0.0

    def neg_growth(f: float) -> float:
        return -float(np.mean(np.log1p(f * r)))

    res = minimize_scalar(neg_growth, bounds=(0.0, upper), method="bounded")
    f_star = float(res.x) if res.success else 0.0
    return max(0.0, f_star)


# ---------------------------------------------------------------------------
# B0120 — AFML ch.10 probability-aware bet sizing (2N(z) - 1)
# ---------------------------------------------------------------------------
def bet_size_from_prob(p, *, clip_eps: float = 1e-6) -> np.ndarray:
    """AFML §10.3: map calibrated P(label=1) to a bet size in [-1, 1].

    z = (p - 1/2) / sqrt(p (1 - p));  m = 2 * Phi(z) - 1.

    Monotone in p, ~0 at p=0.5, saturating toward +/-1 as confidence grows.
    In the meta-labeling design the SIGN of the trade comes from the primary
    side; this m scales that side by meta confidence — a negative m on a
    below-coin-flip p means "the meta bets against taking this trade" and is
    floored to 0 by the caller's take-mask, never flipped into a counter-trade.

    Depends on CALIBRATION quality (the z-standardization is meaningless on
    uncalibrated scores) — this pipeline's sigmoid/Platt frozen-tail
    calibration is the prerequisite, see pipeline/train.py.

    WIRING-TIME GATES (risk-officer review 2026-06-11, PROCEED_WITH_CAVEAT —
    hard conditions on any future live wiring, recorded on B0120):
      1. Live code must NEVER consume this raw signed output directly — only
         through a take-mask-aware wrapper that hard-floors negatives to 0.
      2. Tail saturation: p→1 maps to m→1.0 (full size) exactly where Platt
         calibration is least trustworthy. Live wiring must apply a fractional
         multiplier and/or a max-size cap < 1.0 (half-Kelly convention), and
         summary.json's max_size_taken must be cross-read against the
         calibration plots before trusting tail sizes.
    """
    from scipy.stats import norm
    p = np.clip(np.asarray(p, dtype=float), clip_eps, 1.0 - clip_eps)
    z = (p - 0.5) / np.sqrt(p * (1.0 - p))
    return 2.0 * norm.cdf(z) - 1.0


def average_active_bets(bet, start_bar, end_bar) -> np.ndarray:
    """AFML §10.4 averaging: at each event start, average the sizes of all
    bets still ACTIVE (start_j <= start_i <= end_j). Smooths the size path so
    it does not churn on every probability wiggle of a single new event —
    without this, discretized live sizing whipsaws and friction eats the edge.

    NaN bets (events with no measurement) are excluded from every average.
    O(n^2) broadcast — fine for event counts in the thousands; guarded above
    `max_events` (multi-asset pooled runs can reach tens of thousands of
    events, where the n^2 float64 intermediate is a multi-GB OOM): degrades
    to all-NaN with a RuntimeWarning instead of crashing the report.
    """
    b = np.asarray(bet, dtype=float)
    s = np.asarray(start_bar, dtype=float)
    e = np.asarray(end_bar, dtype=float)
    if b.size == 0:
        return b.copy()
    _MAX_EVENTS_BROADCAST = 20_000
    if b.size > _MAX_EVENTS_BROADCAST:
        import warnings
        warnings.warn(
            f"average_active_bets: {b.size} events exceeds the O(n^2) broadcast "
            f"guard ({_MAX_EVENTS_BROADCAST}); returning NaN diagnostics. "
            f"Replace with a sorted-interval sweep before pooled-scale use.",
            RuntimeWarning, stacklevel=2,
        )
        return np.full(b.shape, np.nan)
    active = (s[None, :] <= s[:, None]) & (s[:, None] <= e[None, :])  # [i, j]
    active &= np.isfinite(b)[None, :]
    counts = active.sum(axis=1)
    sums = np.where(active, np.nan_to_num(b, nan=0.0)[None, :], 0.0).sum(axis=1)
    out = np.divide(sums, counts, out=np.full(b.shape, np.nan), where=counts > 0)
    return out


def discretize_bet(m, *, step: float = 0.05) -> np.ndarray:
    """AFML §10.4 discretization: round to a step grid, clip to [-1, 1].
    Prevents order churn from sub-step size updates."""
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    m = np.asarray(m, dtype=float)
    return np.clip(np.round(m / step) * step, -1.0, 1.0)


def bet_sizing_diagnostics(
    probs, take, start_bar, end_bar, *, step: float = 0.05,
) -> dict:
    """B0120 pre-check block for summary.json — INFORMATIONAL ONLY, never
    wired into live order sizing (same contract as kelly_sizing above).

    Computes the full ch.10 chain (raw size -> concurrency-averaged ->
    discretized) on the OOF calibrated probs, sizes floored to 0 where the
    take-mask is False (meta said no-trade), and reports level + churn stats
    so a human can verify the sizes do not whipsaw fold-to-fold before any
    future live wiring (the B0120 'cheap pre-check').

    INTERPRETATION (risk-officer caveat): untaken events enter the chain as
    genuine 0-size active bets, so (a) the churn stats include 0<->x
    transitions at take-mask boundaries — this is the EVENT-sequence size
    path, not an active book's; (b) the concurrency average drags
    mean_size_taken below the raw 2N(z)-1 level when pct_signals_kept is low.
    Both biases are conservative (smaller, smoother sizes).
    """
    p = np.asarray(probs, dtype=float)
    t = np.asarray(take, dtype=bool)
    raw = bet_size_from_prob(p)
    sized = np.where(t, np.maximum(raw, 0.0), 0.0)   # no-trade -> 0; never flip
    sized = np.where(np.isfinite(p), sized, np.nan)  # no measurement -> NaN
    avg = average_active_bets(sized, start_bar, end_bar)
    disc = discretize_bet(np.nan_to_num(avg, nan=0.0), step=step)
    disc = np.where(np.isfinite(avg), disc, np.nan)

    taken = disc[t & np.isfinite(disc)]
    valid = disc[np.isfinite(disc)]
    churn = np.abs(np.diff(valid)) if valid.size > 1 else np.array([])
    return {
        "step": step,
        "n_events": int(p.size),
        "n_taken": int(t.sum()),
        "mean_size_taken": float(taken.mean()) if taken.size else float("nan"),
        "max_size_taken": float(taken.max()) if taken.size else float("nan"),
        "mean_abs_size_change": float(churn.mean()) if churn.size else float("nan"),
        "p90_abs_size_change": float(np.quantile(churn, 0.9)) if churn.size else float("nan"),
        "pct_steps_with_change": float((churn > 0).mean()) if churn.size else float("nan"),
    }


def kelly_sizing(pnl, *, max_fraction: float = 10.0) -> dict:
    """Both Kelly estimates + half-Kelly + distribution diagnostics.

    `pnl` is the realized per-trade net PnL of TAKEN trades (not the full
    sparse event series). half_kelly_* are the conventional fractional-Kelly
    safety margin practitioners actually size on.
    """
    r = np.asarray(pnl, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    g = gaussian_kelly(r)
    e = empirical_kelly(r, max_fraction=max_fraction)
    if n >= 3 and r.std(ddof=1) > 0:
        m = r.mean()
        s = r.std(ddof=1)
        skew = float(np.mean(((r - m) / s) ** 3))
    else:
        skew = float("nan")
    return {
        "n_trades": n,
        "mean_pnl": float(r.mean()) if n else float("nan"),
        "std_pnl": float(r.std(ddof=1)) if n >= 2 else float("nan"),
        "skew_pnl": skew,
        "gaussian_kelly": g,
        "empirical_kelly": e,
        "half_gaussian_kelly": g / 2.0,
        "half_empirical_kelly": e / 2.0,
        # When this is well below 1.0 the Gaussian formula is over-levering
        # relative to the realized (negatively-skewed) distribution.
        "empirical_to_gaussian_ratio": (e / g) if g > 0 else float("nan"),
    }
