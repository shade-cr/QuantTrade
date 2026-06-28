"""Classification + strategy metrics for the meta-labeling pipeline."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, average_precision_score, brier_score_loss
from sklearn.metrics import precision_recall_curve, roc_auc_score


def precision_at_recall(y_true: np.ndarray, y_prob: np.ndarray, recall: float) -> float:
    p, r, _ = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns arrays sorted by descending threshold; recall increases.
    # Find the first index where r >= recall, take its precision.
    eligible = np.where(r >= recall)[0]
    if eligible.size == 0:
        return 0.0
    # precision_recall_curve sorts by descending recall (ascending threshold).
    # eligible[-1] is the minimum recall that still meets the target, giving the
    # highest-precision operating point at that recall level.
    return float(p[eligible[-1]])


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    single_class = len(np.unique(y_true)) < 2
    return {
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if not single_class else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if not single_class else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "precision_at_recall_0.3": precision_at_recall(y_true, y_prob, 0.3),
        "precision_at_recall_0.5": precision_at_recall(y_true, y_prob, 0.5),
    }


def strategy_metrics(
    side: pd.Series,
    prediction: pd.Series,        # 0/1: 1 = take the trade
    fwd_return: pd.Series,        # signed log-return realized between entry and exit
    cost_bps: float,
    threshold: float,
    *,
    years_in_window: float,
    min_trades_for_sharpe: int = 30,
) -> dict[str, float]:
    """Compute net Sharpe, max drawdown, hit ratio, and pct_kept on a per-trade pnl series.

    Sharpe annualization
    --------------------
    Standard daily-returns Sharpe uses sqrt(bars_per_year). That is wrong here:
    `pnl[t]` is non-zero only when a trade is taken, and `mu/sd` is computed on
    the filtered per-trade pnl. The correct annualization factor is
    sqrt(trades_per_year) where trades_per_year = n_trades / years_in_window.

    With ~25 trades/year on D1, the right factor is sqrt(25) ≈ 5, not sqrt(252) ≈ 16.
    Using sqrt(252) here produced Sharpes of 11–62 in the first Phase 1 run —
    physically impossible numbers, all coming from this bug.

    NaN semantics
    -------------
    Sharpe is returned as NaN (not 0) when:
      * n_trades < min_trades_for_sharpe — the sample is too small to estimate
        Sharpe reliably; 0 would conflate "no skill" with "no measurement".
      * pnl std is 0 — all-winner or all-loser tiny samples.
      * years_in_window <= 0 — caller bug.
    Downstream code must use nan-safe aggregations (nanmedian, nanmean) and the
    stack-decision logic must skip NaN Sharpes when counting competing models.
    """
    take = prediction.values >= threshold
    n_trades = int(take.sum())
    if n_trades == 0:
        return {"sharpe_net": 0.0, "max_drawdown": 0.0, "hit_ratio": float("nan"),
                "pct_signals_kept": 0.0, "n_trades": 0,
                "per_trade_pnl": np.empty(0, dtype=float)}
    pnl_take = side.values[take] * fwd_return.values[take] - (cost_bps / 1e4)
    pnl_full = side.values * fwd_return.values * take - (cost_bps / 1e4) * take
    equity = pnl_full.cumsum()
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max

    if n_trades < min_trades_for_sharpe or years_in_window <= 0:
        sharpe = float("nan")
    else:
        sd = pnl_take.std(ddof=1)
        # 1e-12 absolute threshold catches both true zero and float-noise zero
        # (e.g. all-winners with identical returns) without false positives on
        # realistic financial pnl, where per-trade stdev is O(1e-3) to O(1e-1).
        if sd <= 1e-12:
            sharpe = float("nan")
        else:
            trades_per_year = n_trades / years_in_window
            sharpe = float((pnl_take.mean() / sd) * np.sqrt(trades_per_year))

    hit_ratio = float(((side.values * fwd_return.values > 0) & take).sum() / take.sum())
    return {
        "sharpe_net": sharpe,
        "max_drawdown": float(drawdown.min()),
        "hit_ratio": hit_ratio,
        "pct_signals_kept": float(take.mean()),
        "n_trades": n_trades,
        # Per-trade pnl (net of cost) for downstream PSR/DSR computation on
        # the REAL pnl distribution moments — not the prior Sharpe-distribution
        # proxy. Strip this key before JSON-serializing the metrics dict.
        "per_trade_pnl": pnl_take,
    }


def aggregate_per_trade_pnl_metrics(
    pnl_per_model_per_fold: dict[str, list[np.ndarray]],
) -> dict[str, dict]:
    """Concatenate per-trade pnl across folds per model and compute moments.

    Output per model:
      - n_trades: total across folds
      - sr_per_trade: per-trade Sharpe (mean/std, NOT annualized) — the
        quantity to pass to `probabilistic_sharpe_ratio` along with `n_trades`,
        `skew`, and `kurt` (formula is timescale-invariant under linear
        scaling of returns, but mixing annualized SR with per-trade γ would
        be inconsistent — keep everything in per-trade units).
      - skew: skewness of the aggregated per-trade pnl distribution.
      - kurt: raw kurtosis (normal = 3.0).
      - per_fold_sr_per_trade: per-trade Sharpe per fold (folds with <2
        trades or std=0 are dropped). Used to build the DSR trial pool.

    The previous PSR/DSR path used moments of the PER-FOLD ANNUALIZED Sharpe
    distribution as a proxy — that collapsed when n_folds was small or many
    folds had 0 trades. The pnl-based moments here are the AFML §14 standard.
    """
    out: dict[str, dict] = {}
    for m, fold_pnls in pnl_per_model_per_fold.items():
        non_empty = [p for p in fold_pnls if len(p) > 0]
        all_pnl = np.concatenate(non_empty) if non_empty else np.empty(0, dtype=float)
        per_fold_sr: list[float] = []
        for pnl_f in non_empty:
            if len(pnl_f) < 2:
                continue
            sd_f = pnl_f.std(ddof=1)
            if sd_f > 1e-12:
                per_fold_sr.append(float(pnl_f.mean() / sd_f))
        if len(all_pnl) < 2:
            out[m] = {
                "n_trades": int(len(all_pnl)),
                "sr_per_trade": float("nan"),
                "skew": float("nan"),
                "kurt": float("nan"),
                "per_fold_sr_per_trade": per_fold_sr,
            }
            continue
        sd = all_pnl.std(ddof=1)
        if sd <= 1e-12:
            sr = float("nan")
        else:
            sr = float(all_pnl.mean() / sd)
        sk_raw = pd.Series(all_pnl).skew()
        ku_raw = pd.Series(all_pnl).kurt()
        out[m] = {
            "n_trades": int(len(all_pnl)),
            "sr_per_trade": sr,
            "skew": float(sk_raw) if not pd.isna(sk_raw) else 0.0,
            # Raw kurtosis (pd.Series.kurt() returns excess; add 3 for raw).
            "kurt": float(ku_raw + 3.0) if not pd.isna(ku_raw) else 3.0,
            "per_fold_sr_per_trade": per_fold_sr,
        }
    return out


def probabilistic_sharpe_ratio(
    sr_observed: float,
    sr_benchmark: float,
    n: int,
    skew: float,
    kurt: float,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012, AFML §14).

    Returns Pr(true SR > sr_benchmark | sample). All quantities in the SAME time units:
    if sr_observed/sr_benchmark are per-bar Sharpes, n is the number of bars; if they are
    annualized Sharpes, the formula still holds when applied consistently.

    Args:
      sr_observed: Sample Sharpe ratio.
      sr_benchmark: Benchmark Sharpe (0 = "any positive skill", or set to E[max trial]).
      n: Sample size (number of returns used to compute sr_observed).
      skew: Skewness of the return series (gamma_3).
      kurt: Kurtosis (gamma_4, raw moment — 3.0 for normal, not excess).
    """
    from scipy.stats import norm
    denom_sq = 1.0 - skew * sr_observed + (kurt - 1.0) / 4.0 * sr_observed ** 2
    if denom_sq <= 0:
        return float("nan")
    z = (sr_observed - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom_sq)
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    sr_observed: float,
    sr_trials: np.ndarray,
    n: int,
    skew: float,
    kurt: float,
    n_trials: int | None = None,
) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Equivalent to PSR with sr_benchmark = E[max trial Sharpe under H0 of zero skill].
    `sr_trials` is the array of candidate Sharpes considered (e.g., one per model-primary
    combo). Returns Pr(true SR > best-by-chance | sample).

    B0132 (AFML §11/§13 — deflate by the FULL number of trials carried out):
    `n_trials` decouples the trial COUNT used in the E[max] threshold from the
    SIZE of `sr_trials`. The variance V[{SR_n}] is always estimated empirically
    from `sr_trials` (the observed dispersion of attempts), but the
    expected-maximum order statistic E[max_N] grows with the family-wise number
    of trials N, which is typically LARGER than the handful of Sharpes we happen
    to have collected (e.g. the threshold-grid axis and other primaries are real
    search dimensions that inflate N but are not present in `sr_trials`).
    Passing the larger family-wise N raises the rejection threshold and lowers
    DSR, preventing the per-primary undercount that overstates significance.
    When `n_trials is None`, falls back to `len(sr_trials)` (legacy behaviour).
    """
    from scipy.stats import norm
    sr_trials = np.asarray(sr_trials, dtype=float)
    n_sample = len(sr_trials)
    if n_sample < 2:
        raise ValueError("DSR requires at least 2 trials")
    # Trial count for the E[max] order statistic: family-wise N if supplied,
    # else the number of collected trial Sharpes. Never below the sample size.
    n_eff = max(int(n_trials), n_sample) if n_trials is not None else n_sample
    sr_var = float(sr_trials.var(ddof=1))
    if sr_var <= 0:
        return probabilistic_sharpe_ratio(sr_observed, 0.0, n, skew, kurt)
    gamma_euler = 0.5772156649
    sr_max_expected = np.sqrt(sr_var) * (
        (1 - gamma_euler) * norm.ppf(1.0 - 1.0 / n_eff)
        + gamma_euler * norm.ppf(1.0 - 1.0 / (n_eff * np.e))
    )
    return probabilistic_sharpe_ratio(sr_observed, sr_max_expected, n, skew, kurt)
