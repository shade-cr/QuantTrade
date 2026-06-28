"""Tests for the load-bearing pieces of scripts/spike_ensemble_consensus.py.

The spike asks the decisive question "can weak signals be COMBINED into a tradeable
edge via a NON-fitted consensus?". The load-bearing, must-be-correct pieces are:

  1. causal alignment of each component into a common index (closed-bar; no look-ahead),
  2. TRAIN-only standardization (the z-score uses ONLY the train mean/std),
  3. the no-fit consensus combiners (equal-weight z-sum sign; k-of-N agreement gate),
  4. the Sharpe invariant (replicates pipeline/metrics.strategy_metrics L80-91:
     sqrt(trades_per_year), NaN when n_trades < 30 or std == 0),
  5. the OOS IC (correlation of a signal to a FORWARD return) is forward-only.

These live in pipeline/consensus_ensemble.py and are imported by the spike script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.consensus_ensemble import (  # noqa: E402
    annualized_sharpe,
    consensus_kofn,
    consensus_zsum_sign,
    forward_log_return,
    information_coefficient,
    standardize_on_train,
)


# --------------------------------------------------------------------------- #
# 1. TRAIN-only standardization
# --------------------------------------------------------------------------- #
def test_standardize_uses_only_train_mean_and_std():
    """z = (x - train_mean) / train_std, with train_std from the TRAIN slice only.

    The holdout must be standardized with the TRAIN moments (no holdout leakage
    into the scale). After standardizing, the TRAIN slice has ~0 mean / unit std,
    and the holdout is offset by the train moments (NOT re-centered on itself).
    """
    idx = pd.date_range("2020-01-01", periods=10, freq="h", tz="UTC")
    x = pd.Series(np.arange(10, dtype=float), index=idx)
    train_mask = np.array([True] * 6 + [False] * 4)

    z = standardize_on_train(x, train_mask)

    tr = x[train_mask]
    expected_mean = tr.mean()
    expected_std = tr.std(ddof=0)
    # Train slice standardized by its own moments:
    assert z[train_mask].mean() == pytest.approx(0.0, abs=1e-12)
    # Holdout standardized by TRAIN moments, NOT holdout moments:
    ho_expected = (x[~train_mask] - expected_mean) / expected_std
    np.testing.assert_allclose(z[~train_mask].to_numpy(), ho_expected.to_numpy(), rtol=1e-12)


def test_standardize_constant_train_returns_zeros_not_nan():
    """A component that is constant on TRAIN (std==0) must not blow up to NaN/inf;
    a degenerate component contributes 0 to the consensus, not a poison value."""
    idx = pd.date_range("2020-01-01", periods=6, freq="h", tz="UTC")
    x = pd.Series([5.0, 5.0, 5.0, 9.0, 1.0, 7.0], index=idx)
    train_mask = np.array([True, True, True, False, False, False])
    z = standardize_on_train(x, train_mask)
    assert np.isfinite(z.to_numpy()).all()
    assert (z[train_mask] == 0.0).all()


# --------------------------------------------------------------------------- #
# 2. No-fit consensus combiners
# --------------------------------------------------------------------------- #
def test_zsum_sign_is_unweighted_majority_direction():
    """combined = sign(sum of component z-scores). No learned weights — each
    component enters with weight 1. NaN components are treated as 0 (abstain)."""
    comps = pd.DataFrame(
        {
            "a": [1.0, -1.0, 0.5, np.nan],
            "b": [1.0, 2.0, -2.0, 1.0],
            "c": [-0.5, -1.0, 1.0, 1.0],
        }
    )
    out = consensus_zsum_sign(comps)
    # row0: 1+1-0.5 = +1.5 -> +1 ; row1: -1+2-1 = 0 -> 0 ; row2: 0.5-2+1 = -0.5 -> -1
    # row3: nan(0)+1+1 = 2 -> +1
    np.testing.assert_array_equal(out.to_numpy(), np.array([1.0, 0.0, -1.0, 1.0]))


def test_kofn_gate_requires_k_agreeing_components():
    """k-of-N: take a position ONLY when >= k components agree on a NON-zero
    direction; the position is that agreed direction; otherwise flat (0).

    Crucially this is a SELECTIVE (lower-turnover) combiner: disagreement -> flat.
    """
    comps = pd.DataFrame(
        {
            "a": [1.0, 1.0, 1.0, -1.0, 0.0],
            "b": [1.0, 1.0, -1.0, -1.0, 0.0],
            "c": [1.0, -1.0, -1.0, 1.0, 0.0],
        }
    )
    # k = majority of 3 = 2
    out = consensus_kofn(comps, k=2)
    # row0: 3 long -> +1 ; row1: 2 long 1 short -> +1 ; row2: 1 long 2 short -> -1
    # row3: 2 short 1 long -> -1 ; row4: all flat -> 0
    np.testing.assert_array_equal(out.to_numpy(), np.array([1.0, 1.0, -1.0, -1.0, 0.0]))


def test_kofn_no_majority_is_flat():
    """If neither direction reaches k, the gate is flat even if components are
    non-zero (e.g. exact tie with an even count below threshold)."""
    comps = pd.DataFrame({"a": [1.0], "b": [-1.0], "c": [0.0]})
    out = consensus_kofn(comps, k=2)  # 1 long, 1 short, neither reaches 2
    assert out.iloc[0] == 0.0


def test_kofn_uses_sign_of_continuous_z():
    """k-of-N must threshold on the SIGN of each (continuous) component z, so it
    works on standardized z-scores, not only on pre-signed {-1,0,+1} inputs."""
    comps = pd.DataFrame({"a": [0.3], "b": [2.1], "c": [-0.01]})
    out = consensus_kofn(comps, k=2)  # signs: +,+,- -> 2 long -> +1
    assert out.iloc[0] == 1.0


# --------------------------------------------------------------------------- #
# 3. Sharpe invariant (replicates pipeline/metrics.py L80-91)
# --------------------------------------------------------------------------- #
def test_sharpe_uses_sqrt_trades_per_year():
    """sharpe = (mean/std(ddof=1)) * sqrt(trades_per_year)."""
    rng = np.random.default_rng(0)
    pnl = rng.normal(0.001, 0.01, size=500)
    tpy = 252.0
    expected = (pnl.mean() / pnl.std(ddof=1)) * np.sqrt(tpy)
    got = annualized_sharpe(pnl, trades_per_year=tpy)
    assert got == pytest.approx(expected, rel=1e-12)


def test_sharpe_nan_when_fewer_than_30_trades():
    pnl = np.full(29, 0.001)
    assert np.isnan(annualized_sharpe(pnl, trades_per_year=252.0))


def test_sharpe_nan_when_zero_std():
    pnl = np.full(50, 0.001)  # std == 0
    assert np.isnan(annualized_sharpe(pnl, trades_per_year=252.0))


# --------------------------------------------------------------------------- #
# 4. Forward return causality + IC
# --------------------------------------------------------------------------- #
def test_forward_log_return_is_strictly_forward():
    """forward_log_return(close, h)[t] = log(close[t+h]/close[t]); the last h
    entries are NaN (no future bar). It must NEVER use a past bar."""
    close = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0])
    f1 = forward_log_return(close, 1)
    assert f1.iloc[0] == pytest.approx(np.log(2.0))
    assert f1.iloc[1] == pytest.approx(np.log(2.0))
    assert np.isnan(f1.iloc[-1])  # no bar ahead


def test_information_coefficient_is_signal_to_forward_corr():
    """IC = Pearson corr(signal[t], forward_return[t]); aligned, NaN-dropped.
    A signal that perfectly predicts the forward return sign has IC ~ +1 when the
    magnitudes co-move."""
    sig = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    fwd = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert information_coefficient(sig, fwd) == pytest.approx(1.0, abs=1e-12)
    # anti-correlated:
    assert information_coefficient(sig, -fwd) == pytest.approx(-1.0, abs=1e-12)


def test_information_coefficient_drops_misaligned_nans():
    sig = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
    fwd = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])
    # Jointly-valid rows: 0,2,3 -> corr of [1,3,4] vs [1,3,4] = 1.0
    assert information_coefficient(sig, fwd) == pytest.approx(1.0, abs=1e-12)


def test_information_coefficient_nan_with_fewer_than_3_points():
    """Two points trivially correlate +/-1 — meaningless. Require >= 3 jointly
    valid rows or return NaN (no measurement, not a spurious +1)."""
    sig = pd.Series([1.0, np.nan, np.nan, 4.0])
    fwd = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert np.isnan(information_coefficient(sig, fwd))
