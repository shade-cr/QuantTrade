import numpy as np
import pandas as pd
import pytest
from pipeline.alpha_eval import (
    forward_returns, long_short_returns, per_asset_returns, net_returns,
    information_coefficient, per_bar_stats, survival_verdict,
)


def _frame(vals):
    # vals is a dict: get the length from the first value's length
    first_val = next(iter(vals.values()))
    n_periods = len(first_val) if isinstance(first_val, (list, tuple)) else 1
    idx = pd.date_range("2020-01-01", periods=n_periods, freq="D", tz="UTC")
    return pd.DataFrame(vals, index=idx)


def test_forward_returns_shifts_back_one():
    close = _frame({"A": [10.0, 11, 12, 13]})
    fwd = forward_returns(close)
    # fwd at t = close[t+1]/close[t]-1
    assert fwd["A"].iloc[0] == pytest.approx(0.1)
    assert np.isnan(fwd["A"].iloc[-1])


def test_long_short_picks_top_and_bottom():
    # 4 symbols; alpha ranks A<B<C<D every bar; long {C,D} short {A,B}
    alpha = _frame({"A": [1, 1], "B": [2, 2], "C": [3, 3], "D": [4, 4]})
    fwd = _frame({"A": [-0.01, -0.01], "B": [-0.02, -0.02],
                  "C": [0.03, 0.03], "D": [0.04, 0.04]})
    gross, turnover = long_short_returns(alpha, fwd, k=2)
    # long mean(0.03,0.04)=0.035 ; short mean(-0.01,-0.02)=-0.015
    # L/S = 0.035 - (-0.015) = 0.05
    assert gross.iloc[0] == pytest.approx(0.05)


def test_net_returns_subtracts_cost():
    gross = pd.Series([0.05])
    turnover = pd.Series([1.0])
    out = net_returns(gross, turnover, cost_bps=2.0)
    assert out.iloc[0] == pytest.approx(0.05 - 2.0 * 1e-4)


def test_per_asset_returns_charges_first_bar_entry_and_signs():
    idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    alpha = pd.DataFrame({"A": [1.0, -1.0, 1.0]}, index=idx)   # long, short, long
    fwd = pd.DataFrame({"A": [0.02, 0.03, -0.01]}, index=idx)
    gross, turnover = per_asset_returns(alpha, fwd)
    # position = sign(alpha) = [1,-1,1]; gross = pos*fwd
    assert gross["A"].tolist() == pytest.approx([0.02, -0.03, -0.01])
    # turnover: enter from flat |1-0|=1 ; flip 1->-1 =2 ; -1->1 =2 (first bar NOT NaN)
    assert turnover["A"].tolist() == pytest.approx([1.0, 2.0, 2.0])


def test_information_coefficient_perfect_alignment_zero_std_ir():
    idx = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
    alpha = pd.DataFrame({"A": [1]*4, "B": [2]*4, "C": [3]*4, "D": [4]*4}, index=idx).astype(float)
    fwd = pd.DataFrame({"A": [0.01]*4, "B": [0.02]*4, "C": [0.03]*4, "D": [0.04]*4}, index=idx)
    mean_ic, ic_ir = information_coefficient(alpha, fwd)
    assert mean_ic == pytest.approx(1.0)        # alpha rank == fwd rank every bar
    assert np.isnan(ic_ir)                       # all ICs identical -> std 0 -> IR NaN (guard)


def test_per_bar_stats_basic():
    ret = pd.Series(np.r_[np.full(50, 0.001), np.full(50, -0.0005)])
    st = per_bar_stats(ret)
    assert st["n"] == 100
    assert st["ann_sharpe"] == pytest.approx(st["sr_per_bar"] * np.sqrt(252))
    assert st["kurt"] > 0  # raw kurtosis reported


def test_survival_requires_positive_and_high_dsr():
    # Strong, lone signal among weak trials -> high DSR; should survive.
    rng = np.random.default_rng(1)
    trials = np.r_[0.02, rng.normal(0, 0.01, size=40)]
    v = survival_verdict(sr_observed=0.30, sr_trials=trials, n=2000,
                         skew=0.0, kurt=3.0, n_trials=200)
    assert 0.0 <= v["dsr"] <= 1.0
    assert v["dsr"] >= 0.95
    assert v["survives"] is True


def test_negative_sharpe_never_survives():
    trials = np.array([0.1, 0.2, -0.3, 0.05])
    v = survival_verdict(sr_observed=-0.1, sr_trials=trials, n=500,
                         skew=0.0, kurt=3.0, n_trials=100)
    assert v["survives"] is False


def test_survival_blocked_by_min_obs_floor():
    """A strong lone signal with DSR >= 0.95 but n < 30 must be rejected.

    Small-sample artifacts can produce high DSR by chance. The min_obs=30 floor
    mirrors the CLAUDE.md n_trades >= 30 invariant for strategy_metrics, ensuring
    that tiny-sample flukes do not survive (e.g., a 24-bar run with luck).
    """
    rng = np.random.default_rng(5)
    trials = np.r_[0.40, rng.normal(0, 0.01, size=40)]

    # Same signal, n=24 < 30 floor -> must be rejected
    v_small = survival_verdict(sr_observed=0.40, sr_trials=trials, n=24,
                               skew=0.0, kurt=3.0, n_trials=160)
    assert v_small["survives"] is False

    # Same signal, n=2000 >= 30 floor -> can survive if DSR high enough
    v_ok = survival_verdict(sr_observed=0.40, sr_trials=trials, n=2000,
                            skew=0.0, kurt=3.0, n_trials=160)
    assert v_ok["survives"] is True
