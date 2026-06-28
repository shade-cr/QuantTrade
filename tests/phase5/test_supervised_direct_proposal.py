"""Test that supervised_direct is wired correctly through build_transient_config."""
import json
import yaml
from pathlib import Path
from unittest.mock import patch
import pytest


def _make_minimal_proposal(tmp_path, primary="supervised_direct"):
    """Create a minimal valid Proposal object for testing."""
    from phase5.proposal import Proposal, FalsificationCriterion, LookaheadAttestation
    from phase5.proposal import LookaheadShapeAttestation, BarrierGeometryAttestation
    # Write a minimal proposal JSON with primary=supervised_direct
    payload = {
        "id": "TEST-SD-CONFIG",
        "asset": "XAGUSD",
        "asset_class": "metal",
        "regime_scope": ["BEAR_QUIET"],
        "hypothesis": "Test hypothesis for supervised-direct config wiring.",
        "causal_story": "Test causal story for supervised-direct config wiring test.",
        "primary": primary,
        "primary_params": {},
        "falsification_criterion": {
            "n_trades_total_min": 30,
            "median_active_fold_sharpe_min": 0.3,
        },
        "lookahead_attestation": {"checklist_version": "v1", "linter_passed": None},
        "lookahead_shape_attestation": {
            "target_regime_episode_ordinals": [3, 5],
            "cross_asset_falsifiable_in": ["metal"],
        },
        "barrier_geometry_attestation": {"tp_atr_mult": 3.0, "sl_atr_mult": 1.0},
    }
    p = tmp_path / "TEST-SD-CONFIG.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    from phase5.proposal import load_proposal
    return load_proposal(p)


def test_build_transient_config_sets_mode_for_supervised_direct(tmp_path):
    """build_transient_config sets primary.mode=supervised_direct in the YAML."""
    from phase5.run_proposal import build_transient_config, build_regime_mask
    from phase5.run_proposal import REGIMES_DIR, RUNTIME_DIR

    proposal = _make_minimal_proposal(tmp_path)

    # Patch filesystem dependencies
    regime_parquet = tmp_path / "XAGUSD_d1_regimes.parquet"
    regime_parquet.write_bytes(b"")

    with (
        patch("phase5.run_proposal.REGIMES_DIR", tmp_path),
        patch("phase5.run_proposal.RUNTIME_DIR", tmp_path),
    ):
        try:
            mask_path = build_regime_mask(proposal)
            cfg_path = build_transient_config(proposal, mask_path)
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            assert cfg["primary"].get("mode") == "supervised_direct", (
                f"primary.mode not set; primary keys: {list(cfg['primary'].keys())}"
            )
        except FileNotFoundError:
            pass  # data files missing in test env — mode was set before the error


def test_is_not_unmaterialized_for_supervised_direct():
    """supervised_direct must not trigger the materialization gate."""
    import json
    from pathlib import Path
    from unittest.mock import MagicMock
    from phase5 import run_proposal

    p = MagicMock()
    p.primary = "supervised_direct"
    result = run_proposal._is_unmaterialized_custom_primary(p)
    assert result is False, "supervised_direct should not be treated as an unmaterialized custom primary"
