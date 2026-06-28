"""Tests for pipeline.metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from pipeline.metrics import (
    classification_metrics,
    strategy_metrics,
    precision_at_recall,
)


def test_precision_at_recall_extreme():
    """Perfect classifier: precision @ any recall == 1.0."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.9, 0.95])
    assert precision_at_recall(y, p, recall=0.5) == 1.0


def test_classification_metrics_returns_required_keys():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.6, 0.3, 0.8, 0.7, 0.2, 0.55, 0.45])
    m = classification_metrics(y, p)
    for k in ("mcc", "pr_auc", "brier", "precision_at_recall_0.3", "precision_at_recall_0.5"):
        assert k in m


def test_strategy_metrics_zero_cost_recovers_dir_acc():
    """With cost=0 and confidence=1, dir_acc must equal hit_ratio + the equity is monotone with returns."""
    rng = np.random.default_rng(0)
    side = pd.Series(rng.choice([-1, 1], size=200))
    fwd_ret = pd.Series(rng.normal(0, 0.01, size=200))
    pred = pd.Series(np.ones(200))  # take every trade
    m = strategy_metrics(side=side, prediction=pred, fwd_return=fwd_ret,
                         cost_bps=0.0, threshold=0.5, years_in_window=1.0)
    assert "sharpe_net" in m and "max_drawdown" in m and "hit_ratio" in m
    assert 0.0 <= m["hit_ratio"] <= 1.0


def test_sharpe_annualized_by_trades_per_year_not_bars_per_year():
    """Sharpe must annualize by sqrt(n_trades / years_in_window), not sqrt(252).

    With 100 trades over 1 year, annualization factor = sqrt(100) = 10.
    The old `(mu/sd) * sqrt(252)` would give a different (inflated) answer
    whenever trades_per_year != bars_per_year.
    """
    rng = np.random.default_rng(42)
    n = 100
    side = pd.Series(np.ones(n))
    fwd_ret = pd.Series(rng.normal(0.001, 0.01, size=n))
    pred = pd.Series(np.ones(n))  # take every trade
    m = strategy_metrics(side, pred, fwd_ret, cost_bps=0.0, threshold=0.5,
                         years_in_window=1.0)
    pnl_take = side.values * fwd_ret.values
    raw = pnl_take.mean() / pnl_take.std(ddof=1)
    expected = raw * np.sqrt(100 / 1.0)
    assert m["sharpe_net"] == pytest.approx(expected, rel=1e-9)
    # Sanity: the buggy formula would give raw * sqrt(252), which differs by ~sqrt(2.52).
    assert not np.isclose(m["sharpe_net"], raw * np.sqrt(252))


def test_sharpe_with_sparse_trades_correctly_annualized():
    """Filtered subset of trades: trades_per_year = n_trades / years_in_window."""
    rng = np.random.default_rng(7)
    n = 200
    side = pd.Series(np.ones(n))
    fwd_ret = pd.Series(rng.normal(0.001, 0.01, size=n))
    # Take every 4th → 50 trades over 2 years → 25 trades/year.
    take = np.zeros(n, dtype=float)
    take[::4] = 1.0
    pred = pd.Series(take)
    m = strategy_metrics(side, pred, fwd_ret, cost_bps=0.0, threshold=0.5,
                         years_in_window=2.0)
    assert m["n_trades"] == 50
    pnl_take = (side.values * fwd_ret.values)[take.astype(bool)]
    expected = (pnl_take.mean() / pnl_take.std(ddof=1)) * np.sqrt(50 / 2.0)
    assert m["sharpe_net"] == pytest.approx(expected, rel=1e-9)


def test_sharpe_is_nan_when_too_few_trades():
    """Below min_trades_for_sharpe (default 30), Sharpe must be NaN — not 0.

    Zero conflates 'no edge' with 'insufficient sample to measure edge', which
    was a source of bogus stack-decisions in the first XAU D1 run.
    """
    n = 100
    side = pd.Series(np.ones(n))
    fwd_ret = pd.Series(np.full(n, 0.01))  # all winners → sd=0 too
    take = np.zeros(n, dtype=float)
    take[:5] = 1.0  # only 5 trades
    pred = pd.Series(take)
    m = strategy_metrics(side, pred, fwd_ret, cost_bps=0.0, threshold=0.5,
                         years_in_window=0.2)
    assert np.isnan(m["sharpe_net"])
    assert m["n_trades"] == 5


def test_sharpe_is_nan_when_pnl_std_is_zero():
    """All-winner small sample: std=0 must yield NaN, not the inf/exception path."""
    n = 100
    side = pd.Series(np.ones(n))
    fwd_ret = pd.Series(np.full(n, 0.01))  # constant returns → sd=0
    pred = pd.Series(np.ones(n))  # all trades, 100 of them
    m = strategy_metrics(side, pred, fwd_ret, cost_bps=0.0, threshold=0.5,
                         years_in_window=1.0)
    assert np.isnan(m["sharpe_net"])
    assert m["n_trades"] == 100


def test_strategy_metrics_returns_per_trade_pnl_array():
    """The dict must include a `per_trade_pnl` ndarray of length n_trades.

    Needed for PSR/DSR to consume realized per-trade skewness/kurtosis instead
    of the prior Sharpe-distribution proxy (which collapsed when n_folds was
    small or many folds had 0 trades).
    """
    rng = np.random.default_rng(0)
    n = 200
    side = pd.Series(np.ones(n))
    fwd_ret = pd.Series(rng.normal(0.001, 0.01, n))
    # Take every 3rd → ~67 trades
    take = np.zeros(n, dtype=float); take[::3] = 1.0
    pred = pd.Series(take)
    m = strategy_metrics(side, pred, fwd_ret, cost_bps=0.0, threshold=0.5,
                         years_in_window=1.0)
    assert "per_trade_pnl" in m
    pnl = m["per_trade_pnl"]
    assert isinstance(pnl, np.ndarray)
    assert len(pnl) == m["n_trades"]
    # Manually compute expected pnl for taken trades, no cost.
    expected = side.values[take.astype(bool)] * fwd_ret.values[take.astype(bool)]
    np.testing.assert_allclose(pnl, expected)


def test_per_trade_pnl_empty_array_when_zero_trades():
    """When n_trades=0 the per_trade_pnl array must be empty, not None / missing."""
    side = pd.Series(np.ones(50))
    fwd_ret = pd.Series(np.zeros(50))
    pred = pd.Series(np.zeros(50))  # no trades
    m = strategy_metrics(side, pred, fwd_ret, cost_bps=0.0, threshold=0.5,
                         years_in_window=1.0)
    assert "per_trade_pnl" in m
    assert isinstance(m["per_trade_pnl"], np.ndarray)
    assert len(m["per_trade_pnl"]) == 0


def test_aggregate_per_trade_pnl_metrics_concatenates_folds_and_computes_moments():
    """Aggregator concatenates per-fold per-trade pnl arrays, returns mean/std
    Sharpe (per-trade, NOT annualized) plus skewness and kurtosis of the raw
    per-trade pnl distribution. Per-fold per-trade Sharpes are also returned
    so the trial pool for DSR can be assembled from them."""
    from pipeline.metrics import aggregate_per_trade_pnl_metrics
    rng = np.random.default_rng(7)
    fold_a = rng.normal(0.005, 0.01, size=80)
    fold_b = rng.normal(0.002, 0.012, size=60)
    fold_c = rng.normal(-0.001, 0.011, size=40)
    pnl_per_fold = {"xgb": [fold_a, fold_b, fold_c]}
    agg = aggregate_per_trade_pnl_metrics(pnl_per_fold)
    assert agg["xgb"]["n_trades"] == 180
    combined = np.concatenate([fold_a, fold_b, fold_c])
    assert agg["xgb"]["sr_per_trade"] == pytest.approx(
        combined.mean() / combined.std(ddof=1), rel=1e-9
    )
    assert agg["xgb"]["skew"] == pytest.approx(pd.Series(combined).skew(), rel=1e-9)
    assert agg["xgb"]["kurt"] == pytest.approx(pd.Series(combined).kurt() + 3.0, rel=1e-9)
    assert len(agg["xgb"]["per_fold_sr_per_trade"]) == 3


def test_aggregator_skips_folds_with_no_trades_in_per_fold_sr():
    """A fold with 0 trades contributes nothing to either the aggregate or the
    per-fold Sharpe list — it carries no information."""
    from pipeline.metrics import aggregate_per_trade_pnl_metrics
    rng = np.random.default_rng(0)
    pnl_per_fold = {
        "xgb": [rng.normal(0.001, 0.01, size=50), np.array([]), rng.normal(0, 0.01, size=40)],
    }
    agg = aggregate_per_trade_pnl_metrics(pnl_per_fold)
    assert agg["xgb"]["n_trades"] == 90
    assert len(agg["xgb"]["per_fold_sr_per_trade"]) == 2  # the empty fold is skipped


def test_aggregator_returns_nan_for_model_with_fewer_than_two_trades():
    """Below 2 trades total, mean/std can't be estimated — return NaN moments."""
    from pipeline.metrics import aggregate_per_trade_pnl_metrics
    pnl_per_fold = {"xgb": [np.array([0.01]), np.array([])]}
    agg = aggregate_per_trade_pnl_metrics(pnl_per_fold)
    assert agg["xgb"]["n_trades"] == 1
    assert np.isnan(agg["xgb"]["sr_per_trade"])
    assert np.isnan(agg["xgb"]["skew"])


def test_psr_returns_high_prob_for_clearly_positive_sharpe():
    """A clearly positive, non-fat-tailed series should yield PSR ≈ 1 vs benchmark 0."""
    from pipeline.metrics import probabilistic_sharpe_ratio
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=2520)  # 10y of daily, strong drift
    sr = r.mean() / r.std()
    psr = probabilistic_sharpe_ratio(sr_observed=sr, sr_benchmark=0.0, n=len(r),
                                     skew=float(pd.Series(r).skew()),
                                     kurt=float(pd.Series(r).kurt() + 3))  # full kurtosis
    assert 0.99 <= psr <= 1.0


def test_psr_returns_low_prob_when_observed_equals_benchmark():
    """PSR(sr=k, benchmark=k) ≈ 0.5 by symmetry of the normal."""
    from pipeline.metrics import probabilistic_sharpe_ratio
    psr = probabilistic_sharpe_ratio(sr_observed=0.1, sr_benchmark=0.1, n=500, skew=0.0, kurt=3.0)
    assert abs(psr - 0.5) < 1e-6


def test_dsr_lower_than_psr_when_many_trials():
    """DSR(N=100) ≤ PSR(vs 0) by construction (deflation by max-trial expectation)."""
    from pipeline.metrics import probabilistic_sharpe_ratio, deflated_sharpe_ratio
    rng = np.random.default_rng(1)
    # Single observed SR vs 100 fake trials drawn around 0 with realistic variance.
    sr_observed = 0.10
    sr_trials = rng.normal(0.0, 0.05, size=100)
    psr = probabilistic_sharpe_ratio(sr_observed, 0.0, n=500, skew=0.0, kurt=3.0)
    dsr = deflated_sharpe_ratio(sr_observed, sr_trials, n=500, skew=0.0, kurt=3.0)
    assert dsr < psr


def test_dsr_raises_with_single_trial():
    from pipeline.metrics import deflated_sharpe_ratio
    with pytest.raises(ValueError, match="at least 2 trials"):
        deflated_sharpe_ratio(0.1, np.array([0.05]), n=500, skew=0.0, kurt=3.0)
