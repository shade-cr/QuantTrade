"""Backlog DB — thin specialization of pipeline.registry.Registry.

Agents call this; nothing else writes JSON under backlog/.

CLI:
    python -m backlog.db validate_all
    python -m backlog.db rebuild_index
    python -m backlog.db list_active
"""
from __future__ import annotations

import sys
from typing import Any, Optional

from pipeline.registry import (
    EntityNotFoundError,
    Registry,
    RegistryError,
    ValidationError,
)
from backlog.schema import (
    ACTIVE_STATUSES,
    ARCHIVE_STATUSES,
    BacklogEntry,
    BacklogValidationError,
    INDEX_PATH,
    ROOT,
    STATUSES,
    make_created_event,
)

_registry = Registry(
    root=ROOT,
    id_prefix="B",
    id_width=4,
    statuses=STATUSES,
    active_statuses=ACTIVE_STATUSES,
    archive_statuses=ARCHIVE_STATUSES,
    from_dict=BacklogEntry.from_dict,
    index_path=INDEX_PATH,
    default_new_status="proposed",
)


def ensure_layout() -> None:
    _registry.ensure_layout()


# ---------- reads ----------

def get_entry(b_id: str) -> BacklogEntry:
    return _registry.get(b_id)


def exists(b_id: str) -> bool:
    return _registry.exists(b_id)


def status_of(b_id: str) -> str:
    return _registry.status_of(b_id)


def list_active(value: Optional[str] = None, tag: Optional[str] = None) -> list[str]:
    ids = _registry.list_active()
    if value is None and tag is None:
        return ids
    out = []
    for eid in ids:
        e = _registry.get(eid)
        if value is not None and e.value != value:
            continue
        if tag is not None and tag not in (e.tags or []):
            continue
        out.append(eid)
    return out


def list_by_status(status: str) -> list[str]:
    return _registry.list_by_status(status)


def list_archive(status: Optional[str] = None) -> list[str]:
    return _registry.list_archive(status)


def list_all() -> list[str]:
    return _registry.list_all()


def load_active_entries() -> list[BacklogEntry]:
    return _registry.load_active_entries()


def list_blocking(b_id: str, transitive: bool = False) -> list[str]:
    return _registry.list_blocking(b_id, transitive=transitive)


def list_blocked_by(b_id: str, transitive: bool = False) -> list[str]:
    return _registry.list_blocked_by(b_id, transitive=transitive)


def load_index() -> dict:
    return _registry.load_index()


# ---------- writes ----------

def add_entry(entry: BacklogEntry, status: Optional[str] = None) -> None:
    if not entry.history:
        entry.history = [make_created_event()]
    _registry.add(entry, status=status)


def change_status(b_id: str, new_status: str, reason: str) -> None:
    _registry.change_status(b_id, new_status, reason)


def update_field(b_id: str, field: str, value: Any, reason: str) -> None:
    _registry.update_field(b_id, field, value, reason)


def append_history(b_id: str, event: str, reason: str, extra: Optional[dict] = None) -> None:
    _registry.append_history(b_id, event, reason, extra=extra)


def add_link(b_id_a: str, b_id_b: str, relation: str) -> None:
    """Add a symmetric link between two backlog entries.

    Supported relations and their inverses:
        blocks <-> blocked_by
        supersedes <-> superseded_by
        spawned <-> spawned_by
        related <-> related   (self-inverse)
    """
    inverse = {
        "blocks": "blocked_by",
        "blocked_by": "blocks",
        "supersedes": "superseded_by",
        "superseded_by": "supersedes",
        "spawned": "spawned_by",
        "spawned_by": "spawned",
        "related": "related",
    }.get(relation)
    if inverse is None:
        raise RegistryError(f"unsupported relation {relation!r}")
    for src, dst, rel in ((b_id_a, b_id_b, relation), (b_id_b, b_id_a, inverse)):
        entry = _registry.get(src)
        targets = list(entry.links.get(rel, []))
        if dst not in targets:
            targets.append(dst)
            entry.links[rel] = targets
            _registry.update_field(src, "links", entry.links, reason=f"add_link {rel}->{dst}")


def add_da_review(b_id: str, review_path: str, verdict: str, severity_counts: Optional[dict] = None, review_date: Optional[str] = None) -> None:
    entry = _registry.get(b_id)
    entry.da_reviews.append(
        {
            "path": review_path,
            "verdict": verdict,
            "severity_counts": severity_counts or {},
            "date": review_date or "",
        }
    )
    _registry.update_field(b_id, "da_reviews", entry.da_reviews, reason=f"DA review {verdict}")


def tick_falsification_clock(b_id: str, delta: int = 1, reason: str = "") -> dict:
    """Advance the falsification_clock.current counter on a backlog entry."""
    entry = _registry.get(b_id)
    if entry.falsification_clock is None:
        raise RegistryError(f"{b_id}: no falsification_clock set")
    clock = dict(entry.falsification_clock)
    clock["current"] = int(clock.get("current", 0)) + delta
    _registry.update_field(b_id, "falsification_clock", clock, reason=reason or f"tick {delta}")
    return clock


# ---------- maintenance ----------

def rebuild_index(extra: Optional[dict] = None) -> None:
    _registry.rebuild_index(extra=extra)


def validate_all() -> list[ValidationError]:
    return _registry.validate_all()


def _cli() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m backlog.db <validate_all|rebuild_index|list_active|list_all>")
        return 2
    cmd = sys.argv[1]
    if cmd == "validate_all":
        errors = validate_all()
        if errors:
            for e in errors:
                print(f"  FAIL  {e}")
            print(f"\n{len(errors)} validation error(s)")
            return 1
        print("OK: all entries valid")
        return 0
    if cmd == "rebuild_index":
        rebuild_index()
        print(f"OK: rebuilt {INDEX_PATH}")
        return 0
    if cmd == "list_active":
        for eid in list_active():
            print(eid)
        return 0
    if cmd == "list_all":
        for eid in list_all():
            print(eid)
        return 0
    print(f"unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(_cli())
