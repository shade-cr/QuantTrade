"""B0041 — proposal length-validation regression test.

Loop A tick 12's T012D2 PROCEED_WITH_CAVEAT survivor pre-flight-failed on
`causal_story length 1275 not in [30, 800]` after a full DA round-trip.
stage_review now calls Proposal.validate() before DA dispatch so the failure
surfaces in milliseconds instead of after a multi-minute DA subprocess.

This test pins the validator behavior so the early-fail path keeps catching
the same shape of violation that motivated B0041.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from phase5.proposal import load_proposal, ProposalValidationError


def _minimal_valid_proposal() -> dict:
    return {
        "id": "TEST-B0041",
        "asset": "XAUUSD",
        "asset_class": "metal",
        "regime_scope": ["BULL_QUIET"],
        "hypothesis": "Hypothesis text long enough to clear the 30-char lower bound for sure.",
        "causal_story": "Causal story text, well above the 30-char floor for the validator.",
        "primary": "ema_cross",
        "primary_params": {"fast": 10, "slow": 30},
        "feature_overrides": {"add": [], "drop": []},
        "regime_gate": {"mode": "filter_events", "feature_added": True},
        "falsification_criterion": {
            "audit_class_in": ["STABLE", "MARGINAL_2FOLDS"],
            "median_active_fold_sharpe_min": 0.5,
            "n_trades_total_min": 50,
        },
        "lookahead_attestation": {"checklist_version": "v1", "linter_passed": None},
        "lookahead_shape_attestation": {
            "target_regime_episode_ordinals": [0, 2],
            "cross_asset_falsifiable_in": ["fx"],
        },
        "barrier_geometry_attestation": {"tp_atr_mult": 3.0, "sl_atr_mult": 1.0},
        "parent_proposal": None,
        "git_sha_at_propose": None,
        "diagnostic_only": False,
    }


def _write(payload: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(payload, tmp)
    tmp.close()
    return Path(tmp.name)


def test_causal_story_over_800_chars_rejected():
    p = _minimal_valid_proposal()
    p["causal_story"] = "x" * 1275
    path = _write(p)
    try:
        with pytest.raises(ProposalValidationError, match=r"causal_story length 1275 not in \[30, 800\]"):
            load_proposal(path).validate()
    finally:
        path.unlink()


def test_hypothesis_over_800_chars_rejected():
    p = _minimal_valid_proposal()
    p["hypothesis"] = "y" * 801
    path = _write(p)
    try:
        with pytest.raises(ProposalValidationError, match=r"hypothesis length 801 not in \[30, 800\]"):
            load_proposal(path).validate()
    finally:
        path.unlink()


def test_hypothesis_under_30_chars_rejected():
    p = _minimal_valid_proposal()
    p["hypothesis"] = "too short"
    path = _write(p)
    try:
        with pytest.raises(ProposalValidationError, match=r"hypothesis length 9 not in \[30, 800\]"):
            load_proposal(path).validate()
    finally:
        path.unlink()


def test_700_char_causal_story_passes():
    """Confirms the doc-suggested ~600-700 target stays under the cap."""
    p = _minimal_valid_proposal()
    p["causal_story"] = "z" * 700
    path = _write(p)
    try:
        load_proposal(path).validate()  # raises on failure
    finally:
        path.unlink()
