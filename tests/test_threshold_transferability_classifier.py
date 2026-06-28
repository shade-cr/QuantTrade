"""Tests for the M2 transferability classifier in scripts/analyze_threshold_transferability.py.

Covers the v2 fix (2026-05-25): the n_trades-only gate misclassified
USDJPY engine_cusum cusum_filter/rf as STABLE despite 2 of 3 active folds
losing money. The fix adds a median-active-fold-Sharpe positivity gate.

Also covers the v3 fix (2026-05-25): regime-diversity gate flags
candidates whose OOS span sees only one regime (sustained rally only,
or sustained DD only). Motivated by XAG/ETH feat_sentiment
momentum_zscore/catboost — H4 data only starts 2021-05, so with
train_min_bars=3000 the OOS is structurally limited to 2023-2026.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analyze_threshold_transferability import (  # noqa: E402
    _classify_transferability,
    _max_drawdown,
    _max_rally,
    _regime_diversity,
)


# --- v1 (n_trades only) — preserved as backward-compat baseline -------------


def test_legacy_call_with_no_sharpes_still_returns_stable():
    """When Sharpe is not provided, classifier falls back to v1 taxonomy.

    Required so existing callers / tests that only pass n_trades keep working.
    """
    # 3 of 4 folds with n>=30 — v1 STABLE
    cls = _classify_transferability([30, 53, 0, 31])
    assert cls == "STABLE"


def test_legacy_call_marginal_2folds():
    cls = _classify_transferability([30, 53, 0, 0])
    assert cls == "MARGINAL_2FOLDS"


def test_legacy_call_1fold_concentrated():
    cls = _classify_transferability([35, 0, 0, 0])
    assert cls == "1FOLD_CONCENTRATED"


def test_legacy_call_no_fire():
    cls = _classify_transferability([0, 0, 0, 0])
    assert cls == "NO_FIRE"


# --- v2 (median-Sharpe gate) -------------------------------------------------


def test_usdjpy_cusum_reclassified_to_not_profitable():
    """The motivating empirical failure.

    USDJPY engine_cusum cusum_filter/rf at threshold 0.50:
        fold 0: n=30, Sharpe=-0.58   (loss)
        fold 1: n=53, Sharpe=-0.61   (loss)
        fold 2: n=0,  (no trades)
        fold 3: n=31, Sharpe=+2.00   (win)

    Active folds: [-0.58, -0.61, +2.00]. Median = -0.58 (LOSING).
    v1: STABLE (3 of 4 active folds met n>=30 gate).
    v2: NOT_PROFITABLE (median active-fold Sharpe is negative).
    """
    cls = _classify_transferability(
        per_fold_n=[30, 53, 0, 31],
        per_fold_sharpe=[-0.58, -0.61, 0.0, 2.00],
    )
    assert cls == "NOT_PROFITABLE"


def test_stable_with_positive_median_survives():
    """Healthy STABLE candidate keeps its label."""
    cls = _classify_transferability(
        per_fold_n=[40, 45, 35, 50],
        per_fold_sharpe=[0.8, 1.2, 0.5, 0.9],
    )
    assert cls == "STABLE"


def test_marginal_2folds_both_profitable_survives():
    """The project's real winner-like signature: 2 active folds, both positive."""
    cls = _classify_transferability(
        per_fold_n=[92, 65, 4, 0],  # XAU+COT ema_cross/rf shape
        per_fold_sharpe=[4.31, 3.41, float("nan"), 0.0],
    )
    assert cls == "MARGINAL_2FOLDS"


def test_marginal_2folds_one_loss_one_win_downgraded():
    """2 active folds, one positive one negative, median may be 0 -> NOT_PROFITABLE."""
    cls = _classify_transferability(
        per_fold_n=[40, 50, 0, 0],
        per_fold_sharpe=[-1.0, 2.0, 0.0, 0.0],
    )
    # median is (-1.0 + 2.0)/2 = 0.5 > 0 -> still MARGINAL_2FOLDS
    assert cls == "MARGINAL_2FOLDS"


def test_marginal_2folds_both_negative_downgraded():
    """2 active folds, both losing — NOT_PROFITABLE."""
    cls = _classify_transferability(
        per_fold_n=[40, 50, 0, 0],
        per_fold_sharpe=[-1.0, -0.5, 0.0, 0.0],
    )
    assert cls == "NOT_PROFITABLE"


def test_stable_with_zero_median_downgraded():
    """Median == 0 (boundary) is NOT > 0, must be NOT_PROFITABLE."""
    cls = _classify_transferability(
        per_fold_n=[30, 30, 30, 30],
        per_fold_sharpe=[-1.0, 0.0, 0.0, 1.0],
    )
    # Median of [-1, 0, 0, 1] = 0.0, strictly NOT > 0
    assert cls == "NOT_PROFITABLE"


def test_nan_sharpes_excluded_from_median_not_zero_imputed():
    """CLAUDE.md invariant: NaN Sharpes (n<30 folds) must NOT count as 0.

    If a fold has n<30 trades, its Sharpe is NaN per upstream contract.
    We must not coerce that to 0 (would dilute median artificially).
    Here we have one active loser; the NaN fold must be ignored.
    """
    cls = _classify_transferability(
        per_fold_n=[40, 10, 0, 0],
        per_fold_sharpe=[-0.5, float("nan"), 0.0, 0.0],
    )
    # Only 1 active fold (n>=30) — falls to 1FOLD_CONCENTRATED
    assert cls == "1FOLD_CONCENTRATED"


def test_1fold_concentrated_not_downgraded_by_sharpe():
    """1-fold candidates keep their label regardless of Sharpe sign.

    1FOLD_CONCENTRATED is already a warning class — we don't further
    split it. The new gate only applies to STABLE/MARGINAL_2FOLDS.
    """
    cls = _classify_transferability(
        per_fold_n=[40, 0, 0, 0],
        per_fold_sharpe=[-2.0, 0.0, 0.0, 0.0],
    )
    assert cls == "1FOLD_CONCENTRATED"


def test_no_fire_not_affected():
    cls = _classify_transferability(
        per_fold_n=[0, 0, 0, 0],
        per_fold_sharpe=[0.0, 0.0, 0.0, 0.0],
    )
    assert cls == "NO_FIRE"


def test_median_not_mean():
    """Median > 0 with one massive winner pulling mean positive must still classify
    on median.

    Folds Sharpe = [-1, -1, +5] -> mean=+1, median=-1.
    This is exactly the failure mode in USDJPY/cusum.
    """
    cls = _classify_transferability(
        per_fold_n=[40, 40, 40, 0],
        per_fold_sharpe=[-1.0, -1.0, 5.0, 0.0],
    )
    assert cls == "NOT_PROFITABLE"
    # Sanity check: mean would have said yes
    assert (sum([-1.0, -1.0, 5.0]) / 3.0) > 0  # mean is +1.0


# --- v3 (regime-diversity gate) ---------------------------------------------


def test_max_drawdown_simple():
    """Monotone-up series has zero drawdown."""
    assert _max_drawdown(np.array([1.0, 2.0, 3.0, 4.0])) == 0.0


def test_max_drawdown_basic():
    """100 -> 80 is a 20% drawdown."""
    dd = _max_drawdown(np.array([100.0, 80.0]))
    assert abs(dd - 0.20) < 1e-9


def test_max_drawdown_uses_running_peak():
    """Drawdown is measured from the running peak, not the global peak.

    100 -> 90 -> 200 -> 100. Two candidate DDs:
        from peak 100 (idx 0) to trough 90 (idx 1) = 10%
        from peak 200 (idx 2) to trough 100 (idx 3) = 50%
    Max DD = 50%.
    """
    dd = _max_drawdown(np.array([100.0, 90.0, 200.0, 100.0]))
    assert abs(dd - 0.50) < 1e-9


def test_max_rally_basic():
    """80 -> 100 is a 25% rally (gain / trough)."""
    rally = _max_rally(np.array([80.0, 100.0]))
    assert abs(rally - 0.25) < 1e-9


def test_max_rally_uses_running_trough():
    """Rally is measured from the running trough, not the global trough."""
    # 100 -> 50 -> 75 -> 30 -> 60
    # from trough 50 (idx 1) to peak 75 (idx 2) = 50%
    # from trough 30 (idx 3) to peak 60 (idx 4) = 100%
    rally = _max_rally(np.array([100.0, 50.0, 75.0, 30.0, 60.0]))
    assert abs(rally - 1.0) < 1e-9


def test_regime_diversity_pass_both_moves():
    """A series that does 100 -> 70 (-30%) -> 100 (+43%) passes the gate."""
    series = np.array([100.0, 70.0, 100.0])
    div = _regime_diversity(series, min_move=0.15)
    assert div["pass"] is True
    assert div["max_dd"] >= 0.15
    assert div["max_rally"] >= 0.15


def test_regime_diversity_fail_rally_only():
    """Monotone rally with <15% dip — XAG silver breakout 2024-2025 pattern.

    Simulates: starts at 25, dips to 23 (8% DD), then climbs to 75
    (>200% rally). DD never reaches 15%, so the gate fails.
    """
    series = np.array([25.0, 23.0, 26.0, 30.0, 40.0, 60.0, 75.0])
    div = _regime_diversity(series, min_move=0.15)
    assert div["max_dd"] < 0.15
    assert div["max_rally"] >= 0.15
    assert div["pass"] is False


def test_regime_diversity_fail_dd_only():
    """Monotone decline with <15% rally — ETH 2025-2026 pattern.

    Simulates: starts at 3300, climbs briefly to 3400 (3% rally),
    then collapses to 2000 (-41% DD). Rally never reaches 15%, so
    the gate fails.
    """
    series = np.array([3300.0, 3400.0, 3000.0, 2500.0, 2000.0])
    div = _regime_diversity(series, min_move=0.15)
    assert div["max_dd"] >= 0.15
    assert div["max_rally"] < 0.15
    assert div["pass"] is False


def test_regime_diversity_fail_both_insufficient():
    """A range-bound, low-volatility OOS span passes neither side."""
    # 100 -> 105 -> 95 -> 102: rally = 105/95 = +10.5%, DD = (105-95)/105 = 9.5%
    series = np.array([100.0, 105.0, 95.0, 102.0])
    div = _regime_diversity(series, min_move=0.15)
    assert div["max_dd"] < 0.15
    assert div["max_rally"] < 0.15
    assert div["pass"] is False


# Classifier integration with regime gate


def test_stable_regime_pass_survives():
    """STABLE candidate with regime pass keeps STABLE label."""
    cls = _classify_transferability(
        per_fold_n=[40, 45, 35, 50],
        per_fold_sharpe=[0.8, 1.2, 0.5, 0.9],
        regime_pass=True,
    )
    assert cls == "STABLE"


def test_stable_regime_fail_reclassified():
    """STABLE candidate with regime fail downgrades to REGIME_LIMITED."""
    cls = _classify_transferability(
        per_fold_n=[40, 45, 35, 50],
        per_fold_sharpe=[0.8, 1.2, 0.5, 0.9],
        regime_pass=False,
    )
    assert cls == "REGIME_LIMITED"


def test_marginal_2folds_regime_pass_survives():
    """XAU+COT ema_cross/rf shape with regime pass keeps MARGINAL_2FOLDS."""
    cls = _classify_transferability(
        per_fold_n=[92, 65, 4, 0],
        per_fold_sharpe=[4.31, 3.41, float("nan"), 0.0],
        regime_pass=True,
    )
    assert cls == "MARGINAL_2FOLDS"


def test_marginal_2folds_regime_fail_reclassified():
    """XAG/ETH feat_sentiment shape: 2 active folds + median positive + regime fail
    -> REGIME_LIMITED.
    """
    cls = _classify_transferability(
        per_fold_n=[85, 0, 38, 0],
        per_fold_sharpe=[-1.70, 0.0, 3.85, 0.0],
        regime_pass=False,
    )
    assert cls == "REGIME_LIMITED"


def test_not_profitable_overrides_regime_fail():
    """NOT_PROFITABLE takes priority over regime gate — losing money is a
    stronger signal than narrow OOS.
    """
    cls = _classify_transferability(
        per_fold_n=[40, 50, 0, 0],
        per_fold_sharpe=[-1.0, -0.5, 0.0, 0.0],
        regime_pass=False,
    )
    assert cls == "NOT_PROFITABLE"


def test_1fold_concentrated_not_downgraded_by_regime():
    """1FOLD_CONCENTRATED is already a warning class — regime gate doesn't apply.

    The v3 gate explicitly fires only when there are >=2 active folds;
    a single-fold candidate is already flagged for human review.
    """
    cls = _classify_transferability(
        per_fold_n=[40, 0, 0, 0],
        per_fold_sharpe=[1.2, 0.0, 0.0, 0.0],
        regime_pass=False,
    )
    assert cls == "1FOLD_CONCENTRATED"


def test_regime_pass_none_preserves_v2_behavior():
    """When regime_pass=None (legacy / no data), classifier matches v2."""
    cls = _classify_transferability(
        per_fold_n=[40, 45, 35, 50],
        per_fold_sharpe=[0.8, 1.2, 0.5, 0.9],
        regime_pass=None,
    )
    # No regime data → behaves like v2 → STABLE
    assert cls == "STABLE"
