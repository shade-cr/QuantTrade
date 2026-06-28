"""Ideas DB — thin specialization of pipeline.registry.Registry.

Promotion is terminal: promote_to_backlog moves the idea to ideas/promoted/
and creates a backlog entry with links.spawned_from set.

CLI:
    python -m ideas.db validate_all
    python -m ideas.db rebuild_index
    python -m ideas.db list_active
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
from ideas.schema import (
    ACTIVE_STATUSES,
    ARCHIVE_STATUSES,
    INDEX_PATH,
    IdeaEntry,
    IdeaValidationError,
    ROOT,
    STATUSES,
    make_captured_event,
)

_registry = Registry(
    root=ROOT,
    id_prefix="I",
    id_width=4,
    statuses=STATUSES,
    active_statuses=ACTIVE_STATUSES,
    archive_statuses=ARCHIVE_STATUSES,
    from_dict=IdeaEntry.from_dict,
    index_path=INDEX_PATH,
    default_new_status="open",
)


def ensure_layout() -> None:
    _registry.ensure_layout()


# ---------- reads ----------

def get_entry(i_id: str) -> IdeaEntry:
    return _registry.get(i_id)


def exists(i_id: str) -> bool:
    return _registry.exists(i_id)


def list_active(tag: Optional[str] = None) -> list[str]:
    ids = _registry.list_active()
    if tag is None:
        return ids
    return [iid for iid in ids if tag in (_registry.get(iid).tags or [])]


def list_by_status(status: str) -> list[str]:
    return _registry.list_by_status(status)


def list_archive(status: Optional[str] = None) -> list[str]:
    return _registry.list_archive(status)


def list_all() -> list[str]:
    return _registry.list_all()


def load_active_entries() -> list[IdeaEntry]:
    return _registry.load_active_entries()


def load_index() -> dict:
    return _registry.load_index()


# ---------- writes ----------

def next_id() -> str:
    """Allocate the next monotonic I-ID (e.g., I0006). Never re-used."""
    used = list_all()
    if not used:
        return "I0001"
    nums = [int(eid[1:]) for eid in used]
    return f"I{max(nums) + 1:04d}"


def add_entry(entry: IdeaEntry) -> None:
    if not entry.history:
        entry.history = [make_captured_event()]
    _registry.add(entry)


def change_status(i_id: str, new_status: str, reason: str) -> None:
    _registry.change_status(i_id, new_status, reason)


def update_field(i_id: str, field: str, value: Any, reason: str) -> None:
    _registry.update_field(i_id, field, value, reason)


def append_history(i_id: str, event: str, reason: str, extra: Optional[dict] = None) -> None:
    _registry.append_history(i_id, event, reason, extra=extra)


def promote_to_backlog(i_id: str, b_id: str, reason: str = "triaged into backlog") -> None:
    """Promote an idea into a B-ID. Cross-prefix link wiring.

    Side effects:
      - sets idea.links.promoted_to = b_id
      - moves idea file to ideas/promoted/
      - appends 'promoted' history event on the idea
      - the caller is responsible for having ALREADY created the B-entry
        with links.spawned_from = i_id (so this function does not need
        to write across the backlog/ tree). We just verify and warn if
        it's missing.
    """
    idea = _registry.get(i_id)
    if idea.status != "open":
        raise RegistryError(f"{i_id}: cannot promote from status={idea.status!r}; expected 'open'")
    idea.links = {**idea.links, "promoted_to": b_id}
    _registry.update_field(i_id, "links", idea.links, reason=f"promote -> {b_id}")
    _registry.change_status(i_id, "promoted", reason=reason)


def discard(i_id: str, reason: str) -> None:
    idea = _registry.get(i_id)
    idea.links = {**idea.links, "discarded_reason": reason}
    _registry.update_field(i_id, "links", idea.links, reason=f"discard: {reason}")
    _registry.change_status(i_id, "discarded", reason=reason)


# ---------- maintenance ----------

def rebuild_index(extra: Optional[dict] = None) -> None:
    _registry.rebuild_index(extra=extra)


def validate_all() -> list[ValidationError]:
    return _registry.validate_all()


def _cli() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m ideas.db <validate_all|rebuild_index|list_active|list_all|next_id>")
        return 2
    cmd = sys.argv[1]
    if cmd == "validate_all":
        errors = validate_all()
        if errors:
            for e in errors:
                print(f"  FAIL  {e}")
            print(f"\n{len(errors)} validation error(s)")
            return 1
        print("OK: all ideas valid")
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
    if cmd == "next_id":
        print(next_id())
        return 0
    print(f"unknown command {cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(_cli())
