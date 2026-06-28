"""Tests for pipeline.stack.should_stack decision logic."""
from __future__ import annotations
import numpy as np
import pytest

from pipeline.stack import should_stack, StackDecision


def test_no_stack_when_under_2_of_3_beat_baseline():
    sharpe_per_fold_per_model = {
        "xgb":      [0.9, 0.7, 0.8, 0.6],   # beats all 4 folds (assume baseline=0.5)
        "catboost": [0.4, 0.45, 0.3, 0.4],  # never beats
        "rf":       [0.3, 0.4, 0.2, 0.45],  # never beats
    }
    sharpe_baseline_per_fold = [0.5, 0.5, 0.5, 0.5]
    oof_corr = np.array([[1.0, 0.4, 0.4], [0.4, 1.0, 0.4], [0.4, 0.4, 1.0]])
    decision = should_stack(
        sharpe_per_fold_per_model, sharpe_baseline_per_fold, oof_corr,
        min_models=2, min_folds=3, max_corr=0.7,
    )
    assert decision.stack is False
    assert "competence" in decision.reason


def test_no_stack_when_oof_correlation_too_high():
    sharpe_per_fold_per_model = {
        "xgb":      [0.9, 0.7, 0.8, 0.6],
        "catboost": [0.85, 0.75, 0.8, 0.62],
        "rf":       [0.88, 0.72, 0.79, 0.61],
    }
    sharpe_baseline_per_fold = [0.5, 0.5, 0.5, 0.5]
    oof_corr = np.array([[1.0, 0.95, 0.93], [0.95, 1.0, 0.92], [0.93, 0.92, 1.0]])
    decision = should_stack(
        sharpe_per_fold_per_model, sharpe_baseline_per_fold, oof_corr,
        min_models=2, min_folds=3, max_corr=0.7,
    )
    assert decision.stack is False
    assert "correlation" in decision.reason


def test_stack_when_both_criteria_pass():
    sharpe_per_fold_per_model = {
        "xgb":      [0.9, 0.7, 0.8, 0.6],
        "catboost": [0.85, 0.75, 0.8, 0.62],
        "rf":       [0.55, 0.4, 0.78, 0.59],  # beats in 3/4
    }
    sharpe_baseline_per_fold = [0.5, 0.5, 0.5, 0.5]
    oof_corr = np.array([[1.0, 0.5, 0.4], [0.5, 1.0, 0.45], [0.4, 0.45, 1.0]])
    decision = should_stack(
        sharpe_per_fold_per_model, sharpe_baseline_per_fold, oof_corr,
        min_models=2, min_folds=3, max_corr=0.7,
    )
    assert decision.stack is True


def test_stack_rejects_folds_with_too_few_trades():
    """A model whose Sharpe beats baseline but only on folds with <min_trades_per_fold
    trades cannot count as 'competing' — the Sharpe is statistical noise.

    Regression from Phase 1 XAU D1: catboost had Sharpe 62.7 from 3 trades,
    xgb had Sharpe 0 from 1 trade. Both were counted toward stack approval
    even though the sample sizes were microscopic.
    """
    sharpe_per_fold_per_model = {
        "xgb":      [5.0, 4.0, 6.0, 5.5],     # spectacular but on tiny samples
        "catboost": [0.85, 0.75, 0.8, 0.62],   # genuine signal
        "rf":       [0.78, 0.85, 0.82, 0.79],  # genuine signal
    }
    n_trades_per_fold_per_model = {
        "xgb":      [3, 2, 3, 1],                  # microscopic — must NOT count
        "catboost": [100, 110, 95, 105],
        "rf":       [120, 115, 130, 125],
    }
    sharpe_baseline_per_fold = [0.5, 0.5, 0.5, 0.5]
    oof_corr = np.array([[1.0, 0.4, 0.4], [0.4, 1.0, 0.4], [0.4, 0.4, 1.0]])
    decision = should_stack(
        sharpe_per_fold_per_model, sharpe_baseline_per_fold, oof_corr,
        n_trades_per_fold_per_model=n_trades_per_fold_per_model,
        min_models=2, min_folds=3, max_corr=0.7, min_trades_per_fold=30,
    )
    # catboost + rf still compete (both with >30 trades and beating baseline).
    assert decision.stack is True
    assert decision.n_models_passing == 2


def test_stack_rejected_when_only_competitor_has_too_few_trades():
    """If the only model beating baseline does so with tiny samples, stack must be rejected."""
    sharpe_per_fold_per_model = {
        "xgb":      [5.0, 4.0, 6.0, 5.5],     # beats but tiny n_trades
        "catboost": [0.4, 0.4, 0.45, 0.3],
        "rf":       [0.3, 0.4, 0.2, 0.45],
    }
    n_trades_per_fold_per_model = {
        "xgb":      [3, 2, 3, 1],
        "catboost": [100, 110, 95, 105],
        "rf":       [120, 115, 130, 125],
    }
    sharpe_baseline_per_fold = [0.5, 0.5, 0.5, 0.5]
    oof_corr = np.array([[1.0, 0.4, 0.4], [0.4, 1.0, 0.4], [0.4, 0.4, 1.0]])
    decision = should_stack(
        sharpe_per_fold_per_model, sharpe_baseline_per_fold, oof_corr,
        n_trades_per_fold_per_model=n_trades_per_fold_per_model,
        min_trades_per_fold=30,
    )
    assert decision.stack is False
    assert "competence" in decision.reason


def test_stack_backward_compat_without_n_trades_arg():
    """If n_trades_per_fold_per_model is None, behave like before — no trade-size gate."""
    sharpe_per_fold_per_model = {
        "xgb":      [0.9, 0.7, 0.8, 0.6],
        "catboost": [0.85, 0.75, 0.8, 0.62],
        "rf":       [0.55, 0.4, 0.78, 0.59],
    }
    sharpe_baseline_per_fold = [0.5, 0.5, 0.5, 0.5]
    oof_corr = np.array([[1.0, 0.5, 0.4], [0.5, 1.0, 0.45], [0.4, 0.45, 1.0]])
    decision = should_stack(
        sharpe_per_fold_per_model, sharpe_baseline_per_fold, oof_corr,
        n_trades_per_fold_per_model=None,
        min_models=2, min_folds=3, max_corr=0.7,
    )
    assert decision.stack is True
