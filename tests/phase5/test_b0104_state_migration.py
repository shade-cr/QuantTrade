"""B0104 — Component 3 behavioral tests: record_tick bounded growth + migration
idempotency / parity / backup. (Spec test plan §4 and §5.)
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def lat(tmp_path, monkeypatch):
    """Fresh loop_a_tick module with STATE_PATH / RESULTS_DIR redirected to tmp."""
    import scripts.loop_a_tick as mod
    importlib.reload(mod)
    state_path = tmp_path / "loop_a_state.json"
    results = tmp_path / "results_loop_a"
    monkeypatch.setattr(mod, "STATE_PATH", state_path)
    monkeypatch.setattr(mod, "RESULTS_DIR", results)
    monkeypatch.setattr(mod, "SIGNALS_DIR", tmp_path)
    # every cell eligible
    monkeypatch.setattr(mod, "_load_dossier", lambda a, f, r: {"sample_sufficient": True})
    return mod


def _seed_v2_state(mod, current):
    state = {
        "version": 2,
        "tick_count": current["tick_number"] - 1,
        "asset_scope": [{"asset": "XAUUSD", "frequency": "D1"}],
        "regime_scope": ["BULL_QUIET"],
        "current_tick": current,
        "counters": {"total": 0},
        "lru_by_cell": {},
        "recent": [],
        "survivors": [],
    }
    mod.STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _completed_tick(n):
    return {
        "tick_number": n,
        "stage": "completed",
        "asset": "XAUUSD",
        "frequency": "D1",
        "regime": "BULL_QUIET",
        "started_at": f"2026-05-{n + 1:02d}T00:00:00+00:00",
        "completed_at": f"2026-05-{n + 1:02d}T01:00:00+00:00",
        "result": {
            "proposal_id": f"prop-{n}",
            "da_verdict": "BLOCK",
            "preflight_status": "skipped_due_to_da_block",
            "da_objections_high": 1,
            "da_objections_total": 2,
        },
    }


def test_record_tick_archives_row_and_bumps_counter(lat):
    _seed_v2_state(lat, _completed_tick(1))
    out = lat.record_tick()
    assert out["outcome"] == "da_blocked"

    state = json.loads(lat.STATE_PATH.read_text(encoding="utf-8"))
    assert state["counters"]["total"] == 1
    assert state["counters"]["da_blocked"] == 1
    assert "regime_history" not in state  # full rows no longer in the index
    assert state["lru_by_cell"]["XAUUSD|D1|BULL_QUIET"] == "2026-05-02T00:00:00+00:00"
    assert len(state["recent"]) == 1 and state["recent"][0]["proposal_id"] == "prop-1"

    # full row landed in the JSONL archive
    archive = list(lat.RESULTS_DIR.glob("*.jsonl"))
    assert len(archive) == 1
    rows = [json.loads(l) for l in archive[0].read_text().splitlines()]
    assert rows[0]["proposal_id"] == "prop-1" and rows[0]["da_verdict"] == "BLOCK"


def test_record_tick_index_size_constant_across_many_ticks(lat):
    """Spec §4: index byte size does not grow with tick count (recent capped at K)."""
    sizes = []
    for n in range(1, 101):
        _seed_v2_state(lat, _completed_tick(n))
        # _seed resets state each loop; replay accumulated index by carrying it forward
        if n > 1:
            prev = json.loads((lat.STATE_PATH.parent / "_carry.json").read_text())
            cur = json.loads(lat.STATE_PATH.read_text())
            cur["counters"] = prev["counters"]
            cur["lru_by_cell"] = prev["lru_by_cell"]
            cur["recent"] = prev["recent"]
            lat.STATE_PATH.write_text(json.dumps(cur, indent=2))
        lat.record_tick()
        (lat.STATE_PATH.parent / "_carry.json").write_text(lat.STATE_PATH.read_text())
        if n >= 40:  # after `recent` saturates at K=20
            sizes.append(lat.STATE_PATH.stat().st_size)

    state = json.loads(lat.STATE_PATH.read_text())
    assert len(state["recent"]) == lat.RECENT_K == 20
    assert state["counters"]["total"] == 100
    # bounded: size variance after saturation is tiny (only counter integers grow)
    assert max(sizes) - min(sizes) < 200, f"index grew unboundedly: {min(sizes)}..{max(sizes)}"


# ---- migration script ----

def _v1_state(n_rows):
    rows = []
    for i in range(1, n_rows + 1):
        rows.append({
            "asset": "XAUUSD", "frequency": "D1", "regime": "BULL_QUIET",
            "tick_number": i, "proposal_id": f"p{i}",
            "last_ticked": f"2026-05-{(i % 28) + 1:02d}T00:00:00+00:00",
            "outcome": "da_blocked" if i % 2 else "preflight_failed",
            "da_verdict": "BLOCK", "da_objections_high": 1,
        })
    return {
        "version": 1, "tick_count": n_rows,
        "asset_scope": [{"asset": "XAUUSD", "frequency": "D1"},
                        {"asset": "BTCUSD", "frequency": "H4"}],
        "regime_scope": ["BULL_QUIET", "BEAR_QUIET"],
        "regime_history": rows, "survivors": [],
        "da_blocked_count": 7, "preflight_failed_count": 3,
        "current_tick": None, "notes": "x", "lru_strategy": "lru",
    }


@pytest.fixture()
def mig(tmp_path, monkeypatch):
    import scripts.loop_a_tick as lat_mod
    importlib.reload(lat_mod)
    import scripts.migrate_loop_a_state_v2 as m
    importlib.reload(m)
    state_path = tmp_path / "loop_a_state.json"
    results = tmp_path / "results_loop_a"
    backup = tmp_path / "loop_a_state.v1.bak.json"
    for mod in (lat_mod, m):
        monkeypatch.setattr(mod, "STATE_PATH", state_path, raising=False)
        monkeypatch.setattr(mod, "RESULTS_DIR", results, raising=False)
    monkeypatch.setattr(m, "BACKUP_PATH", backup)
    return m, state_path, results, backup


def test_migration_idempotent_with_backup_and_parity(mig):
    m, state_path, results, backup = mig
    state_path.write_text(json.dumps(_v1_state(40), indent=2), encoding="utf-8")

    out1 = m.migrate()
    assert out1["status"] == "migrated"
    assert out1["rows_archived"] == 40
    assert backup.exists()  # v1 preserved

    v2 = json.loads(state_path.read_text())
    assert v2["version"] == 2
    assert "regime_history" not in v2
    assert v2["counters"]["total"] == 40
    assert len(v2["recent"]) == 20
    archive = (results / "_migrated_v1.jsonl")
    assert archive.exists()
    assert len(archive.read_text().splitlines()) == 40

    # idempotent: second run is a no-op, archive not duplicated
    out2 = m.migrate()
    assert out2["status"] == "already_v2"
    assert len(archive.read_text().splitlines()) == 40


def test_migration_parity_assertion_picks_match(mig):
    m, state_path, results, backup = mig
    v1 = _v1_state(30)
    state_path.write_text(json.dumps(v1, indent=2), encoding="utf-8")
    out = m.migrate()
    # the gate's recorded pick is the v1 scan pick — BTCUSD/H4 cells never ticked
    # ("" sorts first), so the parity pick must be an untouched cell.
    assert tuple(out["parity_pick"]) == m._oracle_v1_pick(v1)
