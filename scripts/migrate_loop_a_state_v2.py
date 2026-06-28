"""B0104 — one-shot, idempotent migration of signals/loop_a_state.json v1 -> v2.

The v1 state carries the full per-tick `regime_history` (unbounded, ~77% of the
file's tokens). v2 replaces it with a bounded index:

  * `counters`     — integer tally by outcome (was scattered `*_count` fields)
  * `lru_by_cell`  — max(last_ticked) per "ASSET|FREQ|REGIME" cell (<= 24)
  * `recent`       — the last K=20 tick summaries

The full rows are NOT discarded — they are replayed into the day-partitioned
machine-readable archive `results/loop_a/_migrated_v1.jsonl` (the JSONL archive
is the source of truth; the index is a cache).

Safety:
  * backs up the original to signals/loop_a_state.v1.bak.json
  * ASSERTS post-migration that the new _regime_lru_pick returns the SAME pick as
    a v1 full-scan oracle BEFORE finalizing the swap (fails loudly otherwise)
  * idempotent: re-running on an already-v2 file is a no-op (no double-archiving,
    no counter drift) — the v1 backup is the trigger that gates the replay.

    uv run python scripts/migrate_loop_a_state_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.util.state_store import append_archive_jsonl, atomic_write_text  # noqa: E402
from scripts.loop_a_tick import (  # noqa: E402
    STATE_PATH, RESULTS_DIR, RECENT_K, _regime_lru_pick, _cell_key, _load_dossier,
    _rel,
)

BACKUP_PATH = STATE_PATH.with_name("loop_a_state.v1.bak.json")
MIGRATED_ARCHIVE_DATE = "_migrated_v1"  # results/loop_a/_migrated_v1.jsonl

# Carried-over top-level keys (everything that is NOT regime_history / *_count).
_PRESERVE = (
    "tick_count", "last_tick_at", "created_at", "current_tick", "survivors",
    "asset_scope", "regime_scope", "lru_strategy", "notes",
)
# v1 *_count field -> v2 counters bucket name.
_COUNT_FIELDS = {
    "discarded_count": "discarded",
    "preflight_failed_count": "preflight_failed",
    "da_blocked_count": "da_blocked",
    "circularity_violation_count": "circularity_violation",
}


def _oracle_v1_pick(v1_state: dict) -> tuple[str, str, str]:
    """The pre-B0104 full-history scan, run against a v1 state with a stubbed
    dossier loader (every cell eligible) so the parity check isolates the
    history->cell collapse, not dossier availability on disk."""
    history_by_key: dict[tuple[str, str, str], str] = {}
    for e in v1_state.get("regime_history", []):
        key = (e["asset"], e.get("frequency", "D1"), e["regime"])
        if key not in history_by_key or e["last_ticked"] > history_by_key[key]:
            history_by_key[key] = e["last_ticked"]
    eligible = []
    for cell in v1_state["asset_scope"]:
        a, f = cell["asset"], cell["frequency"]
        for r in v1_state["regime_scope"]:
            eligible.append((history_by_key.get((a, f, r), ""), a, f, r))
    eligible.sort()
    _, a, f, r = eligible[0]
    return a, f, r


def _build_v2(v1: dict) -> tuple[dict, list[dict]]:
    """Compute the v2 shape + the list of full rows to archive."""
    history = v1.get("regime_history", [])

    counters = {"total": 0}
    lru_by_cell: dict[str, str] = {}
    for e in history:
        counters["total"] += 1
        bucket = e.get("outcome")
        if bucket:
            counters[bucket] = counters.get(bucket, 0) + 1
        key = _cell_key(e["asset"], e.get("frequency", "D1"), e["regime"])
        if key not in lru_by_cell or e["last_ticked"] > lru_by_cell[key]:
            lru_by_cell[key] = e["last_ticked"]

    recent = []
    for e in history[-RECENT_K:]:
        recent.append({
            "tick_number": e.get("tick_number"),
            "asset": e.get("asset"),
            "frequency": e.get("frequency", "D1"),
            "regime": e.get("regime"),
            "proposal_id": e.get("proposal_id"),
            "outcome": e.get("outcome"),
            "da_verdict": e.get("da_verdict"),
            "da_objections_high": e.get("da_objections_high"),
            "last_ticked": e.get("last_ticked"),
        })

    v2: dict = {"version": 2}
    for k in _PRESERVE:
        if k in v1:
            v2[k] = v1[k]
    v2["counters"] = counters
    v2["lru_by_cell"] = lru_by_cell
    v2["recent"] = recent
    return v2, history


def migrate() -> dict:
    raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))

    if raw.get("version") == 2 and "lru_by_cell" in raw:
        return {"status": "already_v2", "note": "no-op (idempotent)",
                "tick_count": raw.get("tick_count"),
                "lru_cells": len(raw.get("lru_by_cell", {}))}

    if "regime_history" not in raw:
        raise RuntimeError(
            "v1 state has no regime_history to migrate and is not v2 — refusing "
            "to guess. Inspect signals/loop_a_state.json manually."
        )

    v2, history = _build_v2(raw)

    # --- PARITY GATE: new lru_by_cell pick must equal the v1 full-scan pick ---
    # Stub the dossier loader so eligibility is identical in both branches; the
    # gate isolates the history->cell collapse + sort, which is what changes.
    import scripts.loop_a_tick as lat
    orig_loader = lat._load_dossier
    lat._load_dossier = lambda a, f, r: {"sample_sufficient": True}
    try:
        v1_pick = _oracle_v1_pick(raw)
        v2_state_for_pick = {
            "asset_scope": raw["asset_scope"],
            "regime_scope": raw["regime_scope"],
            "lru_by_cell": v2["lru_by_cell"],
        }
        v2_pick = lat._regime_lru_pick(v2_state_for_pick)
    finally:
        lat._load_dossier = orig_loader

    if v1_pick != v2_pick:
        raise AssertionError(
            f"MIGRATION ABORTED — LRU parity FAILED. v1 scan picks {v1_pick} but "
            f"v2 lru_by_cell picks {v2_pick}. State NOT modified."
        )

    # --- replay full rows to the machine-readable archive (idempotent) ---
    archive_path = RESULTS_DIR / f"{MIGRATED_ARCHIVE_DATE}.jsonl"
    if archive_path.exists():
        archive_path.unlink()  # idempotent: rebuild cleanly on re-run
    for row in history:
        append_archive_jsonl(RESULTS_DIR, MIGRATED_ARCHIVE_DATE, row)

    # --- back up v1, then swap in v2 atomically ---
    atomic_write_text(BACKUP_PATH, json.dumps(raw, indent=2))
    atomic_write_text(STATE_PATH, json.dumps(v2, indent=2, default=str))

    return {
        "status": "migrated",
        "parity_pick": list(v1_pick),
        "rows_archived": len(history),
        "archive": _rel(archive_path),
        "backup": _rel(BACKUP_PATH),
        "counters": v2["counters"],
        "lru_cells": len(v2["lru_by_cell"]),
        "recent_kept": len(v2["recent"]),
    }


if __name__ == "__main__":
    print(json.dumps(migrate(), indent=2, default=str))
