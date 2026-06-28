"""B0070: Loop A frequency-awareness — state migration + freq-aware addressing."""
from __future__ import annotations
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import importlib.util


def _load_lat():
    spec = importlib.util.spec_from_file_location(
        "loop_a_tick", REPO_ROOT / "scripts" / "loop_a_tick.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migrate_state_wraps_bare_asset_scope_and_backfills_frequency():
    lat = _load_lat()
    raw = {
        "asset_scope": ["XAUUSD"],
        "regime_scope": ["BULL_QUIET"],
        "regime_history": [{"asset": "XAUUSD", "regime": "BULL_QUIET", "last_ticked": "x"}],
        "current_tick": {"asset": "XAUUSD", "regime": "BULL_QUIET", "stage": "awaiting_hypothesizer"},
    }
    migrated = lat._migrate_state(raw)
    assert migrated["asset_scope"] == [{"asset": "XAUUSD", "frequency": "D1"}]
    assert migrated["regime_history"][0]["frequency"] == "D1"
    assert migrated["current_tick"]["frequency"] == "D1"
    # idempotent
    assert lat._migrate_state(migrated) == migrated


def test_migrate_state_leaves_already_migrated_untouched():
    lat = _load_lat()
    raw = {
        "asset_scope": [{"asset": "BTCUSD", "frequency": "H4"}],
        "regime_scope": ["BEAR_QUIET"],
        "regime_history": [{"asset": "BTCUSD", "regime": "BEAR_QUIET", "frequency": "H4", "last_ticked": "x"}],
        "current_tick": None,
    }
    assert lat._migrate_state(raw) == raw


def test_load_dossier_uses_freq_dir(tmp_path):
    lat = _load_lat()
    import json
    rs = tmp_path / "regime_stats"
    (rs / "BTCUSD_h4").mkdir(parents=True)
    (rs / "BTCUSD_h4" / "BEAR_QUIET.json").write_text(json.dumps({"regime_id": "BEAR_QUIET"}))
    lat.REGIME_STATS_DIR = rs
    assert lat._load_dossier("BTCUSD", "H4", "BEAR_QUIET")["regime_id"] == "BEAR_QUIET"
    assert lat._load_dossier("BTCUSD", "D1", "BEAR_QUIET") is None  # no d1 dir


def test_lru_pick_distinguishes_frequency(tmp_path, monkeypatch):
    lat = _load_lat()
    import json
    rs = tmp_path / "regime_stats"
    for cell in ("XAUUSD_d1", "BTCUSD_h4"):
        (rs / cell).mkdir(parents=True)
        (rs / cell / "BULL_QUIET.json").write_text(
            json.dumps({"regime_id": "BULL_QUIET", "sample_sufficient": True})
        )
    monkeypatch.setattr(lat, "REGIME_STATS_DIR", rs)
    state = {
        "asset_scope": [{"asset": "XAUUSD", "frequency": "D1"}, {"asset": "BTCUSD", "frequency": "H4"}],
        "regime_scope": ["BULL_QUIET"],
        "regime_history": [
            {"asset": "XAUUSD", "frequency": "D1", "regime": "BULL_QUIET", "last_ticked": "2026-01-02"},
            {"asset": "BTCUSD", "frequency": "H4", "regime": "BULL_QUIET", "last_ticked": "2026-01-01"},
        ],
    }
    asset, frequency, regime = lat._regime_lru_pick(state)
    assert (asset, frequency, regime) == ("BTCUSD", "H4", "BULL_QUIET")


def test_proposal_id_hint_embeds_frequency_and_infers_back():
    lat = _load_lat()
    from phase5.run_proposal import _infer_frequency
    from phase5.proposal import _build_dataclass, Proposal

    def _mk(pid: str) -> Proposal:
        return _build_dataclass(Proposal, {
            "id": pid,
            "asset": "XAUUSD",
            "asset_class": "metal",
            "regime_scope": ["BEAR_STRESSED"],
            "hypothesis": "Hypothesis text long enough to clear the 30-char lower bound for sure.",
            "causal_story": "Causal story text, well above the 30-char floor for the validator.",
            "primary": "ema_cross",
        })

    h4_id = lat._proposal_id_hint("XAUUSD", "H4", "BEAR_STRESSED", 7)
    d1_id = lat._proposal_id_hint("XAUUSD", "D1", "BEAR_STRESSED", 7)
    assert "-H4-" in h4_id and "-D1-" in d1_id
    assert _infer_frequency(_mk(h4_id)) == "H4"
    assert _infer_frequency(_mk(d1_id)) == "D1"
