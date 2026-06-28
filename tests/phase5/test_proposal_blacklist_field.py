"""Tests for primary_feature_blacklist field in Proposal schema (B0015b Task 6).

Per docs/superpowers/plans/2026-05-26-cot-extremes-primary.md Task 6: the
Proposal dataclass gains a `primary_feature_blacklist: list[str]` field with
default empty list. Validation is enforced at orchestration time
(scripts/run_xau_d1.py via assert_primary_inputs_disjoint), NOT in
Proposal.validate() — the dataclass only ensures the field shape.
"""
from __future__ import annotations

import pytest

from phase5.proposal import (
    LookaheadShapeAttestation,
    Proposal,
    _build_dataclass,
)


def _minimal_proposal(**overrides) -> Proposal:
    """Build a valid Proposal for use as the baseline in tests."""
    defaults = dict(
        id="test",
        asset="XAUUSD",
        asset_class="metal",
        regime_scope=["BULL_QUIET"],
        hypothesis="x" * 50,
        causal_story="y" * 50,
        primary="ema_cross",
        lookahead_shape_attestation=LookaheadShapeAttestation(
            target_regime_episode_ordinals=[0, 2],
            cross_asset_falsifiable_in=["XAGUSD"],
        ),
    )
    defaults.update(overrides)
    return Proposal(**defaults)


def test_proposal_primary_feature_blacklist_optional_defaults_to_empty():
    """primary_feature_blacklist defaults to empty list (backward compat)."""
    p = _minimal_proposal()
    p.validate()  # no raise
    assert p.primary_feature_blacklist == []


def test_proposal_blacklist_field_round_trip_through_dict():
    """A blacklist with values survives to_dict/_build_dataclass round-trip."""
    payload = {
        "id": "20260526-XAU-BULLQ-COT-001",
        "asset": "XAUUSD",
        "asset_class": "metal",
        "regime_scope": ["BULL_QUIET"],
        "hypothesis": "x" * 50,
        "causal_story": "y" * 50,
        "primary": "phase5_cot_extremes",
        "custom_primary_pseudocode": "see spec docs",
        "primary_feature_blacklist": [
            "cot_extreme_long", "cot_net_noncomm_z52", "dxy_*", "dtwexbgs_*"
        ],
        "lookahead_shape_attestation": {
            "target_regime_episode_ordinals": [0, 2, 4],
            "cross_asset_falsifiable_in": ["XAGUSD", "USDJPY"],
        },
        "barrier_geometry_attestation": {
            "tp_atr_mult": 3.0,
            "sl_atr_mult": 1.0,
            "rationale": "Trend-style R:R=3 per spec.",
        },
    }
    p = _build_dataclass(Proposal, payload)
    assert p.primary_feature_blacklist == [
        "cot_extreme_long", "cot_net_noncomm_z52", "dxy_*", "dtwexbgs_*"
    ]


def test_proposal_with_blacklist_validates_when_phase5_primary():
    """A phase5_* primary with a non-empty blacklist passes validate() (the
    actual disjointness check happens at orchestration time, not here)."""
    p = _minimal_proposal(
        primary="phase5_cot_extremes",
        custom_primary_pseudocode="see spec docs",
        primary_feature_blacklist=["cot_extreme_long", "dxy_*"],
    )
    p.validate()  # no raise


def test_proposal_validate_does_not_check_blacklist_completeness():
    """Per design: Proposal.validate() does NOT verify the blacklist is complete
    against build_tier2_features outputs. That check lives in
    test_primary_feature_blacklist.py and at orchestration time."""
    # An empty blacklist on a phase5_* primary is allowed at the schema level.
    p = _minimal_proposal(
        primary="phase5_cot_extremes",
        custom_primary_pseudocode="see spec docs",
        primary_feature_blacklist=[],
    )
    p.validate()  # no raise — the schema is permissive; orchestration enforces.
