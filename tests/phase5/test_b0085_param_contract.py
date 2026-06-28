"""B0085 — built-in-primary param contract enforcement in the audit path.

build_transient_config must normalize a built-in proposal's primary_params to
canonical keys before writing the transient YAML, so _select_primary never dies
on an opaque KeyError inside the audit subprocess. Divergent-unit synonyms must
fail fast at build time instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from phase5 import run_proposal
from phase5.proposal import load_proposal
from pipeline.primary_contracts import PrimaryParamError


def _proposal_dict(primary: str, primary_params: dict) -> dict:
    return {
        "id": "TEST-B0085",
        "asset": "XAUUSD",
        "asset_class": "metal",
        "regime_scope": ["BEAR_QUIET"],
        "hypothesis": (
            "In quiet bearish conditions a volatility-adaptive CUSUM event filter "
            "isolates the sparse directional impulses where a meta-labeler can add "
            "value, because most bars in the regime are noise that should not trade."
        ),
        "causal_story": (
            "The CUSUM accumulator only fires after a run of same-sign returns "
            "exceeds an ATR-scaled threshold, so events are endogenous to realized "
            "price action and cannot reference future bars, while the regime gate "
            "restricts them to the quiet-bear episodes the hypothesis is about."
        ),
        "primary": primary,
        "primary_params": primary_params,
        "feature_overrides": {"add": [], "drop": []},
        "regime_gate": {"mode": "filter_events", "feature_added": False},
        "falsification_criterion": {
            "audit_class_in": ["STABLE", "MARGINAL_2FOLDS"],
            "median_active_fold_sharpe_min": 0.7,
            "n_trades_total_min": 75,
        },
        "lookahead_attestation": {"checklist_version": "v1", "linter_passed": None},
        "lookahead_shape_attestation": {
            "target_regime_episode_ordinals": [1, 3, 5],
            "cross_asset_falsifiable_in": ["XAGUSD"],
        },
        "barrier_geometry_attestation": {"tp_atr_mult": 3.0, "sl_atr_mult": 1.0},
        "parent_proposal": None,
        "git_sha_at_propose": None,
        "diagnostic_only": False,
    }


def _load(primary: str, primary_params: dict):
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(_proposal_dict(primary, primary_params), f)
        path = f.name
    try:
        return load_proposal(path)
    finally:
        Path(path).unlink()


def test_build_transient_config_normalizes_alias_to_canonical(tmp_path, monkeypatch):
    """T015D1-shaped: cusum_filter with threshold_atr_mult (no threshold_atr) must
    yield a config carrying the canonical threshold_atr — the direct regression
    for the KeyError crash."""
    monkeypatch.setattr(run_proposal, "RUNTIME_DIR", tmp_path)
    p = _load("cusum_filter", {"threshold_atr_mult": 1.0, "side": "short_only"})

    cfg_path = run_proposal.build_transient_config(p, tmp_path / "mask.parquet")

    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    assert cfg["primary"]["cusum_filter"]["threshold_atr"] == 1.0
    assert "threshold_atr_mult" not in cfg["primary"]["cusum_filter"]
    # harmless extra keys are preserved
    assert cfg["primary"]["cusum_filter"]["side"] == "short_only"


def test_build_transient_config_fails_fast_on_divergent_synonym(tmp_path, monkeypatch):
    """threshold_sigma is vol-sigma units, not ATR — must raise at build time,
    not silently coerce and not crash later inside the subprocess."""
    monkeypatch.setattr(run_proposal, "RUNTIME_DIR", tmp_path)
    p = _load("cusum_filter", {"threshold_sigma": 1.0})

    with pytest.raises(PrimaryParamError):
        run_proposal.build_transient_config(p, tmp_path / "mask.parquet")
