"""Pure evaluation functions for the 101-alpha screen (no I/O)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pipeline.metrics import probabilistic_sharpe_ratio, deflated_sharpe_ratio

PERIODS_PER_YEAR_D1 = 252


def forward_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Return realized over t → t+1, aligned to bar t whose signal predicts it.

    Args:
        close: DataFrame of close prices (time × symbol).

    Returns:
        DataFrame of forward returns, same shape as close.
    """
    return close.pct_change().shift(-1)


def long_short_returns(alpha: pd.DataFrame, fwd: pd.DataFrame, k: int = 2) -> tuple[pd.Series, pd.Series]:
    """Equal-weight top-k / bottom-k long-short portfolio returns and turnover.

    Args:
        alpha: DataFrame of alpha values (time × symbol).
        fwd: DataFrame of forward returns (time × symbol).
        k: Number of assets in long and short legs.

    Returns:
        (gross_ret, turnover) as pd.Series indexed by time.
    """
    a = alpha.reindex_like(fwd)
    gross, turnover = [], []
    prev_w = pd.Series(0.0, index=fwd.columns)
    for t in fwd.index:
        row = a.loc[t].dropna()
        w = pd.Series(0.0, index=fwd.columns)
        if len(row) >= 2 * k:
            ordered = row.sort_values()
            shorts, longs = ordered.index[:k], ordered.index[-k:]
            w[longs] = 1.0 / k
            w[shorts] = -1.0 / k
        r = fwd.loc[t].reindex(w.index).fillna(0.0)
        gross.append(float((w * r).sum()))
        turnover.append(float((w - prev_w).abs().sum()))
        prev_w = w
    return (pd.Series(gross, index=fwd.index), pd.Series(turnover, index=fwd.index))


def per_asset_returns(alpha: pd.DataFrame, fwd: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-asset gross and turnover using sign(alpha) as position.

    Args:
        alpha: DataFrame of alpha values (time × symbol).
        fwd: DataFrame of forward returns (time × symbol).

    Returns:
        (gross_ret_per_sym, turnover_per_sym) as DataFrames.
    """
    pos = np.sign(alpha.reindex_like(fwd))
    gross = pos * fwd
    prev = pos.shift(1).fillna(0.0)  # enter from flat on the first bar
    turnover = (pos - prev).abs()
    return gross, turnover


def net_returns(gross, turnover, cost_bps: float):
    """Deduct turnover costs from gross returns.

    Args:
        gross: Series or DataFrame of gross returns.
        turnover: Series or DataFrame of turnover (fraction).
        cost_bps: Cost per unit turnover in basis points.

    Returns:
        Same shape as gross, with costs subtracted.
    """
    return gross - turnover * cost_bps * 1e-4


def information_coefficient(alpha: pd.DataFrame, fwd: pd.DataFrame) -> tuple[float, float]:
    """Cross-sectional Spearman correlation of alpha vs forward returns.

    Args:
        alpha: DataFrame of alpha values (time × symbol).
        fwd: DataFrame of forward returns (time × symbol).

    Returns:
        (mean_IC, IC_IR) where IC_IR is mean / std(ddof=1).
    """
    a = alpha.reindex_like(fwd)
    ics = []
    for t in fwd.index:
        x, y = a.loc[t], fwd.loc[t]
        mask = x.notna() & y.notna()
        if mask.sum() >= 3:
            ics.append(x[mask].corr(y[mask], method="spearman"))
    ics = pd.Series(ics, dtype=float).dropna()
    if len(ics) < 2 or ics.std(ddof=1) == 0:
        return (float(ics.mean()) if len(ics) else float("nan"), float("nan"))
    return (float(ics.mean()), float(ics.mean() / ics.std(ddof=1)))


def per_bar_stats(ret: pd.Series) -> dict:
    """Compute per-bar Sharpe, annualized Sharpe, skew, and raw kurtosis of a return series.

    Args:
        ret: Series of per-bar returns.

    Returns:
        Dictionary with keys: n, sr_per_bar, ann_sharpe, skew, kurt.
        Sharpe is per-bar (mean/std); ann_sharpe = sr_per_bar * sqrt(252).
        kurt is raw kurtosis (normal = 3.0).
    """
    r = pd.Series(ret).dropna()
    n = int(len(r))
    sd = r.std(ddof=1) if n >= 2 else float("nan")
    sr = float(r.mean() / sd) if (pd.notna(sd) and sd > 1e-12) else float("nan")
    _sk = r.skew()
    sk = float(_sk) if (n >= 3 and pd.notna(_sk)) else 0.0
    ku = r.kurt()
    kurt = float(ku + 3.0) if n >= 4 and not pd.isna(ku) else 3.0
    return {
        "n": n,
        "sr_per_bar": sr,
        "ann_sharpe": sr * np.sqrt(PERIODS_PER_YEAR_D1) if not pd.isna(sr) else float("nan"),
        "skew": sk,
        "kurt": kurt,
    }


def survival_verdict(sr_observed: float, sr_trials, n: int, skew: float, kurt: float, n_trials: int, min_obs: int = 30) -> dict:
    """Determine whether an observed Sharpe survives against DSR threshold.

    A Sharpe "survives" if: (1) n >= min_obs AND (2) sr_observed > 0 AND (3) dsr >= 0.95.

    Args:
        sr_observed: Observed Sharpe ratio (per-bar or annualized, must match sr_trials).
        sr_trials: Array of trial Sharpes (e.g., from competing models/primaries).
        n: Sample size (number of observations used to compute sr_observed).
        skew: Skewness of the return series.
        kurt: Raw kurtosis (normal = 3.0).
        n_trials: Total family-wise trial count (for DSR deflation).
        min_obs: Minimum sample size required to trust the Sharpe ratio (default 30).
            A high DSR on <min_obs bars is a small-sample artifact; mirrors the CLAUDE.md
            n_trades >= 30 invariant for strategy_metrics.

    Returns:
        Dictionary with keys: psr, dsr, survives.
        psr = probabilistic Sharpe ratio (vs benchmark 0).
        dsr = deflated Sharpe ratio (vs E[max trial] under H0).
        survives = bool(n >= min_obs AND sr_observed > 0 AND dsr >= 0.95).
    """
    trials = np.asarray(sr_trials, dtype=float)
    trials = trials[~np.isnan(trials)]
    psr = probabilistic_sharpe_ratio(sr_observed, 0.0, n, skew, kurt)
    if len(trials) < 2 or np.isnan(sr_observed):
        dsr = float("nan")
    else:
        dsr = deflated_sharpe_ratio(sr_observed, trials, n, skew, kurt, n_trials=n_trials)
    survives = bool(
        (n >= min_obs)
        and (sr_observed > 0)
        and (not np.isnan(dsr))
        and (dsr >= 0.95)
    )
    return {"psr": float(psr), "dsr": float(dsr), "survives": survives}
