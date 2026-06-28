"""Tests for the DSR-aware best-model selector (Phase 2 T13).

The selector decides which of (xgb, catboost, rf) earns the "best" label
in deployment config. Two failure modes the old median-Sharpe selector
allowed and the DSR-aware one must catch:

1. A model with high median Sharpe but low DSR (catboost on XAU D1 ema_cross:
   Sharpe +0.54, DSR 0.001) — picked it as best, but Phase 1 v4 showed
   rf was the only model with materially positive deflated significance.
2. A model with high DSR but no fold-level evidence (rf has DSR 0.257 on
   XAU D1 but its trade counts per fold are [25, 0, 21, 0] — zero folds
   meet the n_trades_per_fold=30 floor, so DSR is statistically suspect
   despite the headline number).

The selector handles both: gates qualification on n_folds with trades,
then ranks by DSR among the qualified set. Falls back to median Sharpe
when no model qualifies.
"""
from __future__ import annotations
import pytest

from pipeline.best_model import select_best_model


def test_phase1_v4_data_falls_back_to_median_sharpe():
    """Calibrated to actual Phase 1 v4 ema_cross results.

    n_trades per fold:
      xgb:      [48, 0, 23, 0]  → 1 fold ≥30 trades
      catboost: [92, 0,  7, 1]  → 1 fold ≥30 trades
      rf:       [25, 0, 21, 0]  → 0 folds ≥30 trades

    With default `min_folds_with_trades=2`, no model qualifies → fallback
    to median Sharpe → catboost (median +0.54). This is the result XAU
    D1 single-asset should produce; H4 multi-asset is expected to clear
    the gate.
    """
    psr_dsr = {
        "xgb":      {"psr": 0.952, "dsr": 0.004},
        "catboost": {"psr": 0.981, "dsr": 0.001},
        "rf":       {"psr": 0.995, "dsr": 0.257},
    }
    sharpe_median = {"xgb": 0.0, "catboost": 0.54, "rf": 0.0}
    n_trades = {
        "xgb":      [48, 0, 23, 0],
        "catboost": [92, 0,  7, 1],
        "rf":       [25, 0, 21, 0],
    }

    best, reason = select_best_model(
        psr_dsr, sharpe_median, n_trades,
        min_trades_per_fold=30, min_folds_with_trades=2,
    )

    assert best == "catboost", f"fallback should pick catboost (max median sharpe), got {best!r}"
    assert reason["criterion"] == "median_sharpe_fallback"
    assert reason["qualified_models"] == []
    assert "warning" in reason


def test_relaxed_gate_picks_dsr_ranking_when_2_qualify():
    """Same Phase 1 v4 data but with `min_folds_with_trades=1`.

    Now xgb and catboost qualify (each has 1 fold with ≥30 trades). rf
    still does NOT qualify (0 folds reach the 30-trade floor) — this is
    the gate doing its job: even though rf has DSR=0.257, the
    distributional support is too thin to trust.

    Among xgb (DSR 0.004) and catboost (DSR 0.001), DSR-ranking picks xgb.
    This contradicts median Sharpe (which would pick catboost at +0.54) —
    that's the value of the DSR-aware criterion.
    """
    psr_dsr = {
        "xgb":      {"psr": 0.952, "dsr": 0.004},
        "catboost": {"psr": 0.981, "dsr": 0.001},
        "rf":       {"psr": 0.995, "dsr": 0.257},
    }
    sharpe_median = {"xgb": 0.0, "catboost": 0.54, "rf": 0.0}
    n_trades = {
        "xgb":      [48, 0, 23, 0],
        "catboost": [92, 0,  7, 1],
        "rf":       [25, 0, 21, 0],
    }

    best, reason = select_best_model(
        psr_dsr, sharpe_median, n_trades,
        min_trades_per_fold=30, min_folds_with_trades=1,
    )

    assert best == "xgb", f"DSR ranking should pick xgb (0.004 > 0.001), got {best!r}"
    assert reason["criterion"] == "dsr_aware"
    assert set(reason["qualified_models"]) == {"xgb", "catboost"}
    assert "rf" not in reason["qualified_models"]


def test_h4_multi_asset_scenario_dsr_aware_overrides_median_sharpe():
    """Expected H4 multi-asset case: ~600 events per fold across 4 folds
    means all 3 models clear the n_trades=30 floor in ≥2 folds.

    Median Sharpe: catboost wins (1.25 > xgb 1.10 > rf 0.85)
    DSR:           rf wins      (0.55 > xgb 0.42 > catboost 0.31)

    DSR-aware selector picks rf — this is the deployment-relevant choice.
    """
    psr_dsr = {
        "xgb":      {"psr": 0.98, "dsr": 0.42},
        "catboost": {"psr": 0.97, "dsr": 0.31},
        "rf":       {"psr": 0.96, "dsr": 0.55},
    }
    sharpe_median = {"xgb": 1.10, "catboost": 1.25, "rf": 0.85}
    n_trades = {
        "xgb":      [620, 580, 560, 540],
        "catboost": [610, 590, 570, 550],
        "rf":       [600, 570, 540, 510],
    }

    best, reason = select_best_model(
        psr_dsr, sharpe_median, n_trades,
        min_trades_per_fold=30, min_folds_with_trades=2,
    )

    assert best == "rf", (
        f"DSR ranking should override median Sharpe: rf has DSR=0.55 (top), "
        f"catboost has median Sharpe=1.25 (top). Got {best!r}."
    )
    assert reason["criterion"] == "dsr_aware"
    assert set(reason["qualified_models"]) == {"xgb", "catboost", "rf"}
    assert reason["best_dsr"] == pytest.approx(0.55)


def test_zero_qualified_warning_in_reason():
    """When no model has enough fold-level support, the fallback path is
    used and the diagnostic includes a `warning` so downstream code
    (deployment config writer) can surface it."""
    psr_dsr = {
        "xgb":      {"psr": 0.6, "dsr": 0.01},
        "catboost": {"psr": 0.7, "dsr": 0.02},
        "rf":       {"psr": 0.8, "dsr": 0.05},
    }
    sharpe_median = {"xgb": 0.1, "catboost": 0.2, "rf": 0.3}
    # All models have ZERO folds ≥30 trades.
    n_trades = {
        "xgb":      [5, 10, 8, 12],
        "catboost": [4, 9, 7, 11],
        "rf":       [6, 8, 5, 14],
    }

    best, reason = select_best_model(
        psr_dsr, sharpe_median, n_trades,
        min_trades_per_fold=30, min_folds_with_trades=2,
    )

    assert reason["criterion"] == "median_sharpe_fallback"
    assert reason["qualified_models"] == []
    assert "warning" in reason
    # Fallback picks max median sharpe.
    assert best == "rf"
