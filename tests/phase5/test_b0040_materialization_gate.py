"""B0040 (Option B) — custom-primary materialization gate.

A phase5_* custom primary whose backing signal module does not yet exist on
disk must park at `pending_materialization` instead of dispatching an audit
subprocess that would crash on ImportError. Re-running the audit after a human
writes pipeline/primaries_phase5/<primary>.py IS the promotion step.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from phase5 import run_proposal
from phase5.proposal import load_proposal


def _base_proposal_dict(primary: str) -> dict:
    """A lint-clean, pre-flight-passing proposal shaped like a real Loop A
    custom survivor (modeled on T011D2M), with the primary swappable."""
    return {
        "id": "TEST-B0040-GATE",
        "asset": "XAUUSD",
        "asset_class": "metal",
        "regime_scope": ["BULL_QUIET"],
        "hypothesis": (
            "In quiet bullish conditions, large speculator net positioning conviction "
            "measured as a rolling z-score carries information about durable price "
            "continuation orthogonal to recent price trend, so upper-tier positioning "
            "should convert directional moves to favorable barrier outcomes."
        ),
        "causal_story": (
            "Net non-commercial positioning aggregates the conviction of large "
            "speculative traders and updates weekly with a publication lag, so it "
            "cannot leak from future price. Its modest correlation with the trend "
            "feature means conditioning on its upper tier admits a different event "
            "set than the regime gate alone, which is where the residual edge lives."
        ),
        "primary": primary,
        "primary_params": {"cot_quantile_floor": 0.75, "quantile_lookback_bars": 504},
        "custom_primary_pseudocode": (
            "Long when cot_net_noncomm_z52w rolling-quantile rank over the trailing "
            "504 bars >= 0.75 AND a CUSUM filter emits an upward event. Short disabled."
        ),
        "feature_overrides": {"add": ["cot_net_noncomm_z52w", "volume"], "drop": []},
        "regime_gate": {"mode": "filter_events", "feature_added": True},
        "falsification_criterion": {
            "audit_class_in": ["STABLE"],
            "median_active_fold_sharpe_min": 0.7,
            "n_trades_total_min": 75,
        },
        "lookahead_attestation": {"checklist_version": "v1", "linter_passed": None},
        "lookahead_shape_attestation": {
            "target_regime_episode_ordinals": [2, 6, 10],
            "cross_asset_falsifiable_in": ["XAGUSD", "fx"],
        },
        "barrier_geometry_attestation": {"tp_atr_mult": 3.0, "sl_atr_mult": 1.0},
        "parent_proposal": None,
        "git_sha_at_propose": None,
        "diagnostic_only": False,
    }


# ---------------- helper unit tests ----------------

def test_helper_false_for_builtin_primary():
    p = load_proposal_from_dict(_base_proposal_dict("ema_cross"))
    assert run_proposal._is_unmaterialized_custom_primary(p) is False


def test_helper_false_for_existing_custom_module():
    # phase5_t011d2 was materialized (pipeline/primaries_phase5/phase5_t011d2.py).
    p = load_proposal_from_dict(_base_proposal_dict("phase5_t011d2"))
    assert run_proposal._is_unmaterialized_custom_primary(p) is False


def test_helper_true_for_missing_custom_module():
    p = load_proposal_from_dict(_base_proposal_dict("phase5_does_not_exist_zzz"))
    assert run_proposal._is_unmaterialized_custom_primary(p) is True


# ---------------- integration test: run() parks at pending_materialization ----------------

def test_run_parks_unmaterialized_custom_at_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(run_proposal, "AUDIT_RESULTS_DIR", tmp_path / "audit_results")

    prop = _base_proposal_dict("phase5_custom")  # phase5_custom has no module
    prop_path = tmp_path / "TEST-B0040-GATE.json"
    prop_path.write_text(json.dumps(prop), encoding="utf-8")

    record = run_proposal.run(prop_path)

    assert record["status"] == "pending_materialization"
    mat = record["materialization"]
    assert mat["expected_module"] == "pipeline/primaries_phase5/phase5_custom.py"
    assert "signal(ohlcv, features, cfg)" in mat["entry_point"]
    assert mat["custom_primary_pseudocode"] == prop["custom_primary_pseudocode"]
    assert "run_proposal" in mat["promote_command"]
    # Pre-flight result is carried through so the human sees it passed.
    assert record["preflight"]["passed"] is True


def test_run_does_not_park_builtin_primary_at_pending(tmp_path, monkeypatch):
    """A built-in primary must NOT be intercepted by the gate (it would proceed
    to mask/config build; we stop there via skip_subprocess to stay light)."""
    monkeypatch.setattr(run_proposal, "AUDIT_RESULTS_DIR", tmp_path / "audit_results")

    prop = _base_proposal_dict("bollinger_meanrev")
    # bollinger_meanrev is trend-exempt mean-rev: keep R:R within [0.5, 2.0].
    prop["barrier_geometry_attestation"] = {"tp_atr_mult": 1.5, "sl_atr_mult": 1.0}
    prop_path = tmp_path / "TEST-B0040-BUILTIN.json"
    prop_path.write_text(json.dumps(prop), encoding="utf-8")

    record = run_proposal.run(prop_path, skip_subprocess=True)
    assert record["status"] != "pending_materialization"


# ---------------- local loader (avoids writing a temp file just to build a Proposal) ----

def load_proposal_from_dict(d: dict):
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(d, f)
        path = f.name
    try:
        return load_proposal(path)
    finally:
        Path(path).unlink()
