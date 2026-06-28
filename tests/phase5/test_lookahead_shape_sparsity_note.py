"""Tests for the optional `sparsity_note` field on LookaheadShapeAttestation (B0038).

Background: Loop A tick 10 (2026-05-26, BULL_STRESSED) crashed pre-flight with
TypeError because the hypothesizer added a narrative `sparsity_note` inside
lookahead_shape_attestation in response to DA's must_have_mod asking for
sparsity acknowledgment. The dataclass previously rejected unknown keys.
B0038 makes the field a first-class optional, schema-permissive (no validation).
"""
from __future__ import annotations

from phase5.proposal import (
    LookaheadShapeAttestation,
    Proposal,
    _build_dataclass,
)


def test_sparsity_note_defaults_to_none():
    a = LookaheadShapeAttestation(
        target_regime_episode_ordinals=[0, 2],
        cross_asset_falsifiable_in=["XAGUSD"],
    )
    assert a.sparsity_note is None
    a.validate()  # no raise


def test_sparsity_note_round_trips_through_build_dataclass():
    """A proposal payload with sparsity_note inside lookahead_shape_attestation
    loads cleanly (the bug B0038 was a TypeError on this exact path)."""
    note = (
        "Only 5 BULL_STRESSED episodes exist. Under 5-fold WF with "
        "regime_gate=filter_events, the expected count of folds with "
        ">=30 trades is 2-3, not all 5; MARGINAL_2FOLDS is the binding "
        "audit class."
    )
    payload = {
        "id": "test-b0038",
        "asset": "XAUUSD",
        "asset_class": "metal",
        "regime_scope": ["BULL_STRESSED"],
        "hypothesis": "x" * 50,
        "causal_story": "y" * 50,
        "primary": "cusum_filter",
        "lookahead_shape_attestation": {
            "target_regime_episode_ordinals": [1, 13, 15, 21, 23],
            "cross_asset_falsifiable_in": ["XAGUSD", "fx_DXY_short_proxy"],
            "sparsity_note": note,
        },
    }
    p = _build_dataclass(Proposal, payload)
    assert p.lookahead_shape_attestation.sparsity_note == note
    assert p.lookahead_shape_attestation.target_regime_episode_ordinals == [
        1, 13, 15, 21, 23,
    ]
    p.validate()  # ordinals >=2, cross_asset non-empty → no raise


def test_sparsity_note_survives_to_dict():
    a = LookaheadShapeAttestation(
        target_regime_episode_ordinals=[0, 2, 4],
        cross_asset_falsifiable_in=["XAGUSD"],
        sparsity_note="2 active folds expected",
    )
    from dataclasses import asdict

    d = asdict(a)
    assert d["sparsity_note"] == "2 active folds expected"
