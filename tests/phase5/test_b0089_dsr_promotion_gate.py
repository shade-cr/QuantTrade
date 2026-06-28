"""B0089 — Deflated Sharpe Ratio (DSR) as a HARD GATE in the M3 promotion decision.

DSR is computed per-model by scripts/run_backtest.py (deflated_sharpe_ratio ->
dsr_per_model, persisted to psr_dsr.json) but was previously used ONLY for
deployment-tier Kelly sizing. The promotion gate (evaluate_against_falsification)
keyed on audit_class_in + median_active_fold_sharpe_min + n_trades_total_min only,
so a candidate with strong raw fold Sharpe but trial-DEFLATED DSR ~0 could still
class STABLE/MARGINAL and promote.

B0089 threads the audited DSR into the gate. When falsification_criterion.dsr_min
is set and the candidate's DSR < dsr_min, the verdict is 'falsified' and the
audit_class is downgraded to NOT_PROFITABLE EVEN IF raw fold Sharpe + n_trades
pass. dsr_min defaults to None (backward-compatible: old gate unchanged). This
subsumes the external "OOS can't beat IS by >30%" heuristic — DSR deflates the
observed Sharpe by the expected max-of-N-trials Sharpe, directly penalizing
selection over many configurations.
"""
from __future__ import annotations

import numpy as np
import pytest

from phase5.run_proposal import evaluate_against_falsification
from phase5.proposal import FalsificationCriterion, ProposalValidationError


# Three folds each with adequate trades and strongly positive Sharpe ->
# classifies STABLE and clears median_active_fold_sharpe_min + n_trades_total_min.
PASSING_FOLDS_N = [40, 45, 50]
PASSING_FOLDS_SHARPE = [1.2, 1.5, 1.1]


def _criterion(dsr_min=None):
    return {
        "audit_class_in": ["STABLE", "MARGINAL_2FOLDS"],
        "median_active_fold_sharpe_min": 0.5,
        "n_trades_total_min": 50,
        "dsr_min": dsr_min,
    }


# ---------------- schema (FalsificationCriterion.dsr_min) ----------------

def test_dsr_min_defaults_to_none():
    fc = FalsificationCriterion()
    assert fc.dsr_min is None
    fc.validate()  # no raise


def test_dsr_min_in_unit_interval_valid():
    FalsificationCriterion(dsr_min=0.90).validate()
    FalsificationCriterion(dsr_min=0.0).validate()
    FalsificationCriterion(dsr_min=1.0).validate()


@pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0])
def test_dsr_min_out_of_range_rejected(bad):
    with pytest.raises(ProposalValidationError, match="dsr_min"):
        FalsificationCriterion(dsr_min=bad).validate()


# ---------------- gate behavior ----------------

def test_backward_compat_dsr_min_none_ignores_dsr():
    """dsr_min=None -> old behavior: a passing-folds candidate survives even
    when DSR is supplied and is terrible (DSR=0.0)."""
    cls, verdict = evaluate_against_falsification(
        PASSING_FOLDS_N, PASSING_FOLDS_SHARPE, True, _criterion(dsr_min=None), dsr=0.0
    )
    assert cls == "STABLE"
    assert verdict == "survives"


def test_strong_folds_but_dsr_below_gate_is_falsified():
    """Strong raw fold Sharpe + adequate trades, but DSR < dsr_min ->
    falsified and audit_class downgraded to NOT_PROFITABLE."""
    cls, verdict = evaluate_against_falsification(
        PASSING_FOLDS_N, PASSING_FOLDS_SHARPE, True, _criterion(dsr_min=0.90), dsr=0.10
    )
    assert verdict == "falsified"
    assert cls == "NOT_PROFITABLE"


def test_strong_folds_and_dsr_at_or_above_gate_survives():
    """Same passing folds, DSR >= dsr_min -> promotes (survives), class unchanged."""
    cls, verdict = evaluate_against_falsification(
        PASSING_FOLDS_N, PASSING_FOLDS_SHARPE, True, _criterion(dsr_min=0.90), dsr=0.97
    )
    assert cls == "STABLE"
    assert verdict == "survives"


def test_dsr_exactly_at_gate_survives():
    """Boundary: DSR == dsr_min is NOT below the gate -> survives."""
    cls, verdict = evaluate_against_falsification(
        PASSING_FOLDS_N, PASSING_FOLDS_SHARPE, True, _criterion(dsr_min=0.90), dsr=0.90
    )
    assert cls == "STABLE"
    assert verdict == "survives"


def test_nan_dsr_with_gate_on_is_falsified():
    """A NaN DSR (n_trades<2 / degenerate trial pool) cannot clear a set gate ->
    falsified. NaN means 'no DSR measurement', which must not promote."""
    cls, verdict = evaluate_against_falsification(
        PASSING_FOLDS_N, PASSING_FOLDS_SHARPE, True, _criterion(dsr_min=0.90),
        dsr=float("nan"),
    )
    assert verdict == "falsified"
    assert cls == "NOT_PROFITABLE"


def test_dsr_gate_does_not_rescue_already_falsified():
    """If the candidate already fails the fold criterion (low Sharpe), a good
    DSR does NOT make it survive — the DSR gate only tightens, never loosens."""
    cls, verdict = evaluate_against_falsification(
        [40, 45, 50], [-0.3, -0.2, 0.1], True, _criterion(dsr_min=0.90), dsr=0.99
    )
    assert verdict == "falsified"


def test_dsr_none_argument_with_gate_on_is_falsified():
    """dsr_min set but the audited DSR is unavailable (dsr=None) -> cannot
    clear the gate -> falsified. Absence of measurement must not promote."""
    cls, verdict = evaluate_against_falsification(
        PASSING_FOLDS_N, PASSING_FOLDS_SHARPE, True, _criterion(dsr_min=0.90), dsr=None
    )
    assert verdict == "falsified"
    assert cls == "NOT_PROFITABLE"
