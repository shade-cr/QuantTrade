"""Dual-leg market-neutral pairs / spread mean-reversion primitives.

This is the REAL dual-leg spread engine (contrast with pipeline/cointegration.py,
whose v1 is single-leg DIRECTIONAL — it informs entry on the primary's own price
barrier and does NOT enforce spread mean-reversion on exit). Here a position is a
DOLLAR-NEUTRAL pair: long/short `gross_notional` on the primary and the opposite
`beta * gross_notional` on the other leg, and pnl is the SPREAD return.

Causality contract (no look-ahead — backtest-only purge/embargo is irrelevant here,
the spread/z-score are inherently causal by construction):
  * Rolling hedge ratio beta[t] is OLS of log(primary) on log(other) over the
    window of bars ending at t-1, then SHIFTED so beta[t] uses only data <= t-1.
  * Spread[t] = log(primary)[t] - beta[t] * log(other)[t].
  * Z-score[t] standardizes spread[t] using rolling mean/std of the spread over a
    window ending at t-1 (shifted). No statistic at bar t sees bar t's own
    realization in its mean/std, and never any future bar.

Load-bearing pieces are TDD'd in tests/test_pairs_spread.py:
  - half_life          (OU half-life of mean reversion)
  - rolling_beta_zscore (causal rolling hedge ratio + spread z-score)
  - dollar_neutral_legs (notional split given beta)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck half-life of mean reversion
# ---------------------------------------------------------------------------
def half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion of an OU/AR(1) process, in bars.

    Regress delta_spread[t] = lambda * spread[t-1] + c + eps.
    For a mean-reverting series lambda < 0 and the continuous-time half-life is
    -ln(2) / lambda. Equivalent to ln(2) / (-ln(phi)) where phi = 1 + lambda is
    the AR(1) coefficient.

    Returns +inf when the series is not mean reverting (lambda >= 0), which is the
    correct "no reversion" signal (a random walk has lambda ~ 0 -> half-life ~ inf).
    """
    s = pd.Series(np.asarray(spread, dtype=float)).dropna()
    if len(s) < 10:
        return float("inf")
    s_lag = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    # align
    s_lag = s_lag.loc[delta.index]
    x = np.column_stack([s_lag.values, np.ones(len(s_lag))])
    y = delta.values
    # OLS via lstsq
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    lam = coef[0]
    if lam >= 0:
        return float("inf")
    hl = -np.log(2) / lam
    return float(hl)


# ---------------------------------------------------------------------------
# Causal rolling hedge ratio + spread z-score
# ---------------------------------------------------------------------------
def rolling_beta_zscore(
    log_primary: pd.Series,
    log_other: pd.Series,
    *,
    beta_lookback: int,
    z_lookback: int,
) -> tuple[pd.Series, pd.Series]:
    """Causal rolling hedge ratio beta and spread z-score.

    beta[t]   = slope of OLS(log_primary ~ log_other) over the window ending at
                t-1 (shifted by 1 so bar t never sees its own value in the fit).
    spread[t] = log_primary[t] - beta[t] * log_other[t]
    z[t]      = (spread[t] - mean(spread, window ending t-1)) / std(...)

    Returns (beta, z) both indexed like log_primary.

    No-look-ahead guarantee: every rolling stat is shifted by 1 bar, so the value
    at index i is a deterministic function of indices <= i only. Truncating the
    inputs after bar i leaves beta[i]/z[i] unchanged (tested).
    """
    if not log_primary.index.equals(log_other.index):
        log_other = log_other.reindex(log_primary.index)

    x = log_other
    y = log_primary

    # Rolling OLS slope of y on x:
    #   beta = Cov(x, y) / Var(x)  over the window.
    # Computed via rolling moments, then SHIFTED by 1 so beta[t] excludes bar t.
    roll_x = x.rolling(beta_lookback)
    roll_y = y.rolling(beta_lookback)
    mean_x = roll_x.mean()
    mean_y = roll_y.mean()
    # E[xy] over window
    mean_xy = (x * y).rolling(beta_lookback).mean()
    mean_xx = (x * x).rolling(beta_lookback).mean()
    cov_xy = mean_xy - mean_x * mean_y
    var_x = mean_xx - mean_x * mean_x
    beta_raw = cov_xy / var_x.replace(0, np.nan)
    beta = beta_raw.shift(1)

    spread = y - beta * x

    roll_s = spread.rolling(z_lookback)
    s_mean = roll_s.mean().shift(1)
    s_std = roll_s.std().shift(1)
    z = (spread - s_mean) / s_std.replace(0, np.nan)

    beta.name = "beta"
    z.name = "z"
    return beta, z


# ---------------------------------------------------------------------------
# Dollar-neutral leg sizing
# ---------------------------------------------------------------------------
def dollar_neutral_legs(direction: int, gross_notional: float, beta: float) -> dict:
    """Dollar-neutral leg notionals for a spread position.

    direction:
      +1  LONG spread  -> LONG primary, SHORT beta*notional of other
      -1  SHORT spread -> SHORT primary, LONG beta*notional of other
       0  flat
    """
    if direction == 0:
        return {"primary_notional": 0.0, "other_notional": 0.0}
    sign = float(np.sign(direction))
    return {
        "primary_notional": sign * gross_notional,
        "other_notional": -sign * beta * gross_notional,
    }


# ---------------------------------------------------------------------------
# Spread mean-reversion backtest simulator (dual-leg, dollar-neutral)
# ---------------------------------------------------------------------------
def simulate_spread_meanrev(
    z: pd.Series,
    log_primary: pd.Series,
    log_other: pd.Series,
    beta: pd.Series,
    *,
    entry: float,
    exit_band: float,
    stop: float,
    cost_bps_per_leg: float,
) -> dict:
    """Event-driven dual-leg mean-reversion backtest on the spread z-score.

    Decision timing (no look-ahead): at bar t we observe z[t-1] (signal on close)
    and act on bar t. Position pnl accrues from bar t's leg log-returns onward.
    A trade's return is the notional-weighted spread log-return over the holding
    window:

        long-spread per-bar return  =  d(log_primary) - beta_entry * d(log_other)
        short-spread per-bar return = -(d(log_primary) - beta_entry * d(log_other))

    `beta_entry` is frozen at entry (the hedge ratio you actually put on).

    Costs: each round trip transacts BOTH legs on entry AND exit = 4 leg
    transactions; total cost = 4 * cost_bps_per_leg (in fractional terms).

    Rules:
      enter LONG  spread when prior z < -entry  (spread cheap)
      enter SHORT spread when prior z > +entry  (spread rich)
      exit when |prior z| < exit_band  (mean reverted) OR |prior z| > stop (divergence stop)

    Returns dict with:
      n_round_trips, trade_pnls (np.ndarray net fractional pnl per trade),
      holding_bars (list), per_bar_pnl (np.ndarray, net, aligned to index for DD/equity),
      pct_time_in_market.
    """
    zv = np.asarray(z, dtype=float)
    lp = np.asarray(log_primary, dtype=float)
    lo = np.asarray(log_other, dtype=float)
    bv = np.asarray(beta, dtype=float)
    n = len(zv)

    dlp = np.diff(lp, prepend=lp[0])   # d(log_primary)[t] = lp[t]-lp[t-1]
    dlo = np.diff(lo, prepend=lo[0])

    cost_round_trip = 4.0 * cost_bps_per_leg / 1e4

    per_bar_pnl = np.zeros(n)
    trade_pnls: list[float] = []
    holding_bars: list[int] = []
    bars_in_market = 0

    position = 0          # -1 short spread, +1 long spread, 0 flat
    beta_entry = np.nan
    entry_bar = -1
    cur_gross = 0.0       # accumulated gross (pre-cost) pnl for the open trade

    for t in range(1, n):
        prior_z = zv[t - 1]

        if position != 0:
            # accrue this bar's spread return on the open position
            spread_ret = dlp[t] - beta_entry * dlo[t]
            bar_pnl = position * spread_ret
            per_bar_pnl[t] += bar_pnl
            cur_gross += bar_pnl
            bars_in_market += 1

            # exit decision on prior z
            if np.isnan(prior_z):
                pass
            elif abs(prior_z) < exit_band or abs(prior_z) > stop:
                net = cur_gross - cost_round_trip
                # book the cost on the exit bar so equity curve nets out
                per_bar_pnl[t] -= cost_round_trip
                trade_pnls.append(net)
                holding_bars.append(t - entry_bar)
                position = 0
                beta_entry = np.nan
                cur_gross = 0.0
                entry_bar = -1
            continue

        # flat: look for entry on prior z
        if np.isnan(prior_z) or np.isnan(bv[t]):
            continue
        if prior_z < -entry:
            position = 1
            beta_entry = bv[t]
            entry_bar = t
            cur_gross = 0.0
        elif prior_z > entry:
            position = -1
            beta_entry = bv[t]
            entry_bar = t
            cur_gross = 0.0

    return {
        "n_round_trips": len(trade_pnls),
        "trade_pnls": np.asarray(trade_pnls, dtype=float),
        "holding_bars": holding_bars,
        "per_bar_pnl": per_bar_pnl,
        "pct_time_in_market": float(bars_in_market / n) if n else 0.0,
    }
