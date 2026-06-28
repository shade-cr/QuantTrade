"""Load-bearing primitives for the consensus-ensemble feasibility spike.

Question being answered (CEO): can several individually-coin-flip weak signals be
COMBINED into a tradeable edge by a SIMPLE, NON-FITTED consensus (no learned
weights), which trades less often so cost is amortized?

The fitted combiner (ML meta-labeler) was already run 442x -> 0 stable (it overfits
the in-sample mix; see CLAUDE.md "squeeze the orange twice" failure). This module
provides ONLY non-fitted operations:

  * standardize_on_train        — TRAIN-only z-score (no holdout leakage into scale)
  * consensus_zsum_sign         — sign(sum of equal-weight component z); no weights
  * consensus_kofn              — k-of-N agreement gate (selective, lower turnover)
  * annualized_sharpe           — replicates pipeline/metrics.strategy_metrics L80-91
  * forward_log_return          — strictly-forward return (no look-ahead)
  * information_coefficient      — corr(signal[t], forward_return[t])

All combiners are pure functions of already-causal component series — they contain
no .fit(), no parameter learned from data. walk_forward purge/embargo is a BACKTEST
construct and is intentionally NOT used here: this is a single chronological 70/30
split, inherently causal given the .shift / strict-alignment discipline in the
component constructions and forward_log_return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_TRADES_FOR_SHARPE = 30


# --------------------------------------------------------------------------- #
# TRAIN-only standardization
# --------------------------------------------------------------------------- #
def standardize_on_train(x: pd.Series, train_mask: np.ndarray) -> pd.Series:
    """z-score `x` using the TRAIN slice's mean and std ONLY.

    z[t] = (x[t] - mean(x[train])) / std(x[train], ddof=0)

    The holdout is standardized with the TRAIN moments — no holdout statistic
    enters the scale (that would be a subtle leak). A component that is constant
    on TRAIN (std == 0) is degenerate; it returns all zeros (it abstains from the
    consensus) rather than NaN/inf, which would poison a sum/sign.

    NaNs in `x` (warm-up bars) stay NaN; downstream combiners treat NaN as abstain.
    """
    train_mask = np.asarray(train_mask, dtype=bool)
    tr = x[train_mask]
    mu = tr.mean()
    sd = tr.std(ddof=0)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(0.0, index=x.index, name=x.name)
    return ((x - mu) / sd).rename(x.name)


# --------------------------------------------------------------------------- #
# No-fit consensus combiners
# --------------------------------------------------------------------------- #
def consensus_zsum_sign(components: pd.DataFrame) -> pd.Series:
    """Equal-weight z-sum consensus: combined[t] = sign(sum_i z_i[t]).

    No learned weights — every component enters with weight 1. NaN components
    (warm-up / unavailable) are treated as 0 (abstain). sign(0) -> 0 (flat).
    """
    z = components.fillna(0.0)
    total = z.sum(axis=1)
    return np.sign(total).rename("consensus_zsum")


def consensus_kofn(components: pd.DataFrame, k: int) -> pd.Series:
    """k-of-N agreement gate (selective, low-turnover, NON-fitted).

    Take a position ONLY when at least `k` components agree on the SAME non-zero
    direction (by sign). The position is that agreed direction (+1/-1); otherwise
    flat (0). If both directions exist but neither reaches k, flat.

    Thresholds on the SIGN of each (possibly continuous) component z, so it works
    on standardized z-scores as well as on pre-signed {-1,0,+1} inputs.
    """
    s = np.sign(components.fillna(0.0))
    n_long = (s > 0).sum(axis=1)
    n_short = (s < 0).sum(axis=1)
    out = pd.Series(0.0, index=components.index, name="consensus_kofn")
    out[n_long >= k] = 1.0
    out[n_short >= k] = -1.0
    # If somehow both reach k (only possible when k <= N/2), prefer the larger;
    # a strict majority k > N/2 makes this branch unreachable.
    both = (n_long >= k) & (n_short >= k)
    if both.any():
        out[both] = np.sign(n_long[both] - n_short[both])
    return out


# --------------------------------------------------------------------------- #
# Sharpe invariant (replicates pipeline/metrics.strategy_metrics L80-91)
# --------------------------------------------------------------------------- #
def annualized_sharpe(
    pnl: np.ndarray,
    *,
    trades_per_year: float,
    min_trades: int = MIN_TRADES_FOR_SHARPE,
) -> float:
    """Annualized Sharpe via sqrt(trades_per_year), NaN-safe.

    Replicates the load-bearing rule from pipeline/metrics.py strategy_metrics
    (lines 80-91):
      * NaN when n_trades < min_trades (default 30) — too few to estimate.
      * NaN when std(ddof=1) <= 1e-12 — all-equal pnl.
      * else (mean / std(ddof=1)) * sqrt(trades_per_year).

    `pnl` is the per-trade net return series. The DAILY-bars sqrt(252) convention
    is WRONG for sparse, selective pnl (it inflates Sharpe ~3x); trades_per_year
    is the correct annualization base. See CLAUDE.md "Sharpe annualization".
    """
    pnl = np.asarray(pnl, dtype=float)
    n = pnl.size
    if n < min_trades or trades_per_year <= 0:
        return float("nan")
    sd = pnl.std(ddof=1)
    if sd <= 1e-12:
        return float("nan")
    return float((pnl.mean() / sd) * np.sqrt(trades_per_year))


# --------------------------------------------------------------------------- #
# Forward return + IC (the decisive diagnostic)
# --------------------------------------------------------------------------- #
def forward_log_return(close: pd.Series, h_bars: int) -> pd.Series:
    """Strictly-forward h-bar log return: f[t] = log(close[t+h] / close[t]).

    The last h entries are NaN (no future bar). NEVER uses a past bar — this is
    the future the per-bar signal is graded against (no look-ahead in the signal,
    by construction; the return is forward by construction).
    """
    return np.log(close.shift(-h_bars) / close).rename(f"fwd_logret_{h_bars}")


def information_coefficient(signal: pd.Series, forward_return: pd.Series) -> float:
    """IC = Pearson corr(signal[t], forward_return[t]) on jointly-valid rows.

    A direct read of "does this signal predict the next move?". Positive OOS IC
    is the precondition for the Fundamental Law (IR ~ IC*sqrt(breadth)) to help;
    if every component's HOLDOUT IC is ~0/negative, no combination can manufacture
    edge (garbage x breadth = garbage).
    """
    df = pd.concat(
        [pd.Series(signal).rename("s"), pd.Series(forward_return).rename("f")], axis=1
    ).dropna()
    if len(df) < 3 or df["s"].std(ddof=0) <= 1e-12 or df["f"].std(ddof=0) <= 1e-12:
        return float("nan")
    return float(df["s"].corr(df["f"]))
