"""B0149 — system-level DSR floor: the gate is ALWAYS on at result-reading time.

B0089 made the Deflated Sharpe Ratio a hard gate, but only when the proposal
itself set `falsification_criterion.dsr_min` — and none of the Loop-A corpus
proposals did, so DSR enforcement was a manual reviewer-discretion step.

B0149 injects a system floor (SYSTEM_DSR_MIN = 0.95, the Bailey–López de Prado
significance bar: DSR is the probability that the deflated true Sharpe exceeds
0) into the criterion dict at audit-evaluation time:

  - proposal omits dsr_min          → floor applied, source = "system_floor"
  - proposal sets dsr_min >= floor  → proposal value kept, source = "proposal"
  - proposal sets dsr_min <  floor  → RAISED to the floor (tightening-only,
                                      consistent with the methodology's
                                      "criteria may never be loosened")

The locked-proposal invariant is respected: frozen proposal JSONs are not
edited; the floor only ever TIGHTENS the promotion decision.
"""
from __future__ import annotations

from types import SimpleNamespace

from phase5.run_proposal import (
    SYSTEM_DSR_MIN,
    _proposal_criterion_as_dict,
    evaluate_against_falsification,
)
from phase5.proposal import FalsificationCriterion


def _p(fc: FalsificationCriterion) -> SimpleNamespace:
    """_proposal_criterion_as_dict only touches p.falsification_criterion."""
    return SimpleNamespace(falsification_criterion=fc)


def test_floor_value_is_ldp_significance_bar():
    assert SYSTEM_DSR_MIN == 0.95


def test_system_floor_injected_when_proposal_omits_dsr_min():
    crit = _proposal_criterion_as_dict(_p(FalsificationCriterion()))
    assert crit["dsr_min"] == SYSTEM_DSR_MIN
    assert crit["dsr_gate_source"] == "system_floor"


def test_proposal_dsr_min_above_floor_is_kept():
    crit = _proposal_criterion_as_dict(_p(FalsificationCriterion(dsr_min=0.99)))
    assert crit["dsr_min"] == 0.99
    assert crit["dsr_gate_source"] == "proposal"


def test_proposal_dsr_min_at_floor_is_kept_as_proposal():
    crit = _proposal_criterion_as_dict(_p(FalsificationCriterion(dsr_min=0.95)))
    assert crit["dsr_min"] == 0.95
    assert crit["dsr_gate_source"] == "proposal"


def test_proposal_dsr_min_below_floor_is_raised():
    """Tightening-only: a proposal cannot set a laxer bar than the system."""
    crit = _proposal_criterion_as_dict(_p(FalsificationCriterion(dsr_min=0.50)))
    assert crit["dsr_min"] == SYSTEM_DSR_MIN
    assert crit["dsr_gate_source"] == "proposal_raised_to_system_floor"


def test_gate_active_end_to_end_for_dsr_less_proposal():
    """A dsr-less proposal's criterion now FALSIFIES a strong-folds candidate
    whose deflated edge is poor — previously it survived (gate off)."""
    crit = _proposal_criterion_as_dict(_p(FalsificationCriterion()))
    cls, verdict = evaluate_against_falsification(
        [40, 45, 50], [1.2, 1.5, 1.1], True, crit, dsr=0.10
    )
    assert verdict == "falsified"
    assert cls == "NOT_PROFITABLE"


def test_gate_active_end_to_end_missing_dsr_cannot_promote():
    """No DSR measurement at all (None) under the always-on gate → falsified."""
    crit = _proposal_criterion_as_dict(_p(FalsificationCriterion()))
    cls, verdict = evaluate_against_falsification(
        [40, 45, 50], [1.2, 1.5, 1.1], True, crit, dsr=None
    )
    assert verdict == "falsified"
    assert cls == "NOT_PROFITABLE"


def test_gate_passes_good_dsr_end_to_end():
    crit = _proposal_criterion_as_dict(_p(FalsificationCriterion()))
    cls, verdict = evaluate_against_falsification(
        [40, 45, 50], [1.2, 1.5, 1.1], True, crit, dsr=0.97
    )
    assert verdict == "survives"
    assert cls == "STABLE"
