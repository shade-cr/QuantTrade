"""B0043 — devil's-advocate dossier-threading regression test.

Loop A tick 13's BULL_STRESSED T013->D1->D2 lineage was BLOCKED three times in
a row by the devil's advocate, every time on the SAME #1 high-severity
objection: the DA could not verify the dossier statistics cited in the
proposal's causal_story because `stage_review` built the DA decision payload
via `make_decision_payload(...)` WITHOUT passing `regime_stats_dossier`, so the
DA always received `{}`. The retry budget burned out against a payload-assembly
defect no proposal revision could clear.

This test pins the fix: stage_review must thread the current tick's dossier
(the same regime-aggregate, point-in-time dossier the hypothesizer saw) into
the DA payload, so the DA can verify the cited figures.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _dossier(regime: str = "BULL_STRESSED") -> dict:
    return {
        "asset_class": "metal",
        "regime_id": regime,
        "n_bars": 383,
        "n_episodes": 5,
        "sample_sufficient": True,
        "features_quantile_summary": {
            "close": {"median_quantile": 0.83, "vs_other_regimes_rank": "higher"},
            "real_yield_5y_z252d": {"median_quantile": 0.31, "vs_other_regimes_rank": "lower"},
        },
        "regime_defining_features": ["roc_63", "ma_50", "ma_200", "rv_20"],
        "regime_episode_ordinals": [1, 13, 15, 21, 23],
    }


def _proposal() -> dict:
    # ema_cross(10, 30) with no feature-overrides clears the anti-circularity
    # lint and the length/schema validator, so stage_review reaches DA staging.
    return {
        "id": "TEST-B0043",
        "asset": "XAUUSD",
        "asset_class": "metal",
        "regime_scope": ["BULL_STRESSED"],
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
            "target_regime_episode_ordinals": [1, 13],
            "cross_asset_falsifiable_in": ["fx"],
        },
        "barrier_geometry_attestation": {"tp_atr_mult": 3.0, "sl_atr_mult": 1.0},
        "parent_proposal": None,
        "git_sha_at_propose": None,
        "diagnostic_only": False,
    }


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """Redirect loop_a_tick + dispatch module paths into tmp_path and stage a tick."""
    import loop_a_tick as lat
    from phase5 import devils_advocate_dispatch as dad

    signals = tmp_path / "signals"
    regime_stats = signals / "regime_stats"
    runtime = tmp_path / "phase5" / "runtime"
    asset, regime = "XAUUSD", "BULL_STRESSED"

    (regime_stats / f"{asset}_d1").mkdir(parents=True, exist_ok=True)
    (regime_stats / f"{asset}_d1" / f"{regime}.json").write_text(
        json.dumps(_dossier(regime)), encoding="utf-8"
    )

    monkeypatch.setattr(lat, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(lat, "SIGNALS_DIR", signals)
    monkeypatch.setattr(lat, "STATE_PATH", signals / "loop_a_state.json")
    monkeypatch.setattr(lat, "REGIME_STATS_DIR", regime_stats)
    monkeypatch.setattr(lat, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(dad, "RUNTIME_DIR", runtime)

    state = {
        "version": 1,
        "tick_count": 12,
        "asset_scope": [{"asset": asset, "frequency": "D1"}],
        "regime_scope": [regime],
        "regime_history": [],
        "current_tick": {
            "tick_number": 13,
            "stage": "awaiting_hypothesizer",
            "asset": asset,
            "frequency": "D1",
            "regime": regime,
            "proposal_id_hint": "TEST-B0043",
        },
    }
    signals.mkdir(parents=True, exist_ok=True)
    (signals / "loop_a_state.json").write_text(json.dumps(state), encoding="utf-8")

    prop_path = tmp_path / "TEST-B0043.json"
    prop_path.write_text(json.dumps(_proposal()), encoding="utf-8")

    return lat, prop_path, _dossier(regime)


def test_stage_review_threads_dossier_into_da_payload(staged):
    lat, prop_path, dossier = staged
    out = lat.stage_review(str(prop_path))

    assert out["stage"] == "awaiting_devils_advocate"
    threaded = out["payload"]["input"]["regime_stats_dossier"]
    # The regression: this used to be {} regardless of the dossier on disk.
    assert threaded, "DA payload must carry the regime_stats_dossier, not an empty dict"
    assert threaded["regime_id"] == dossier["regime_id"]
    assert threaded["features_quantile_summary"] == dossier["features_quantile_summary"]
