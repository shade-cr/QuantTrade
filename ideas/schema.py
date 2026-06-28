"""Ideas entry schema — lighter than backlog (raw captures, no effort/value).

Promotion to a B-ID is terminal: the idea moves to ideas/promoted/ and the
B-entry takes over. ideas.db.promote_to_backlog wires both sides.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

ROOT = Path("ideas")
INDEX_PATH = ROOT / "INDEX.json"

STATUSES = ("open", "promoted", "discarded")
ACTIVE_STATUSES = ("open",)
ARCHIVE_STATUSES = ("promoted", "discarded")

I_ID_REGEX = re.compile(r"^I\d{4}$")
B_ID_REGEX = re.compile(r"^B\d{4}$")
DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class IdeaValidationError(ValueError):
    pass


@dataclass
class IdeaEntry:
    id: str = ""
    title: str = ""
    status: str = "open"
    tags: list[str] = field(default_factory=list)

    created_at: str = ""
    updated_at: str = ""

    description: str = ""
    source: str = ""
    context: str = ""

    links: dict = field(
        default_factory=lambda: {
            "related_backlog": [],
            "related_ideas": [],
            "promoted_to": None,
            "discarded_reason": None,
        }
    )
    history: list[dict] = field(default_factory=list)

    def validate(self) -> None:
        if not I_ID_REGEX.match(self.id):
            raise IdeaValidationError(f"id={self.id!r} must match ^I\\d{{4}}$")
        if not self.title:
            raise IdeaValidationError(f"{self.id}: title required")
        if self.status not in STATUSES:
            raise IdeaValidationError(f"{self.id}: status={self.status!r} not in {STATUSES}")
        if not isinstance(self.tags, list) or not all(isinstance(t, str) for t in self.tags):
            raise IdeaValidationError(f"{self.id}: tags must be list[str]")
        for fname in ("created_at", "updated_at"):
            v = getattr(self, fname)
            if not v or not DATE_REGEX.match(v):
                raise IdeaValidationError(f"{self.id}: {fname} must be YYYY-MM-DD; got {v!r}")
        if not isinstance(self.description, str) or not self.description.strip():
            raise IdeaValidationError(f"{self.id}: description required (non-empty)")
        self._validate_links()
        self._validate_history()
        # invariants by status
        if self.status == "promoted" and not self.links.get("promoted_to"):
            raise IdeaValidationError(f"{self.id}: status=promoted requires links.promoted_to")
        if self.status == "discarded" and not self.links.get("discarded_reason"):
            raise IdeaValidationError(f"{self.id}: status=discarded requires links.discarded_reason")

    def _validate_links(self) -> None:
        if not isinstance(self.links, dict):
            raise IdeaValidationError(f"{self.id}: links must be dict")
        rb = self.links.get("related_backlog") or []
        ri = self.links.get("related_ideas") or []
        if not isinstance(rb, list) or not all(B_ID_REGEX.match(x) for x in rb if isinstance(x, str)):
            raise IdeaValidationError(f"{self.id}: links.related_backlog must be list of B-IDs")
        if not isinstance(ri, list) or not all(I_ID_REGEX.match(x) for x in ri if isinstance(x, str)):
            raise IdeaValidationError(f"{self.id}: links.related_ideas must be list of I-IDs")
        pt = self.links.get("promoted_to")
        if pt is not None and not (isinstance(pt, str) and B_ID_REGEX.match(pt)):
            raise IdeaValidationError(f"{self.id}: links.promoted_to must be null or a B-ID; got {pt!r}")

    def _validate_history(self) -> None:
        if not isinstance(self.history, list) or not self.history:
            raise IdeaValidationError(f"{self.id}: history must be non-empty list")
        if not any(ev.get("event") == "captured" for ev in self.history):
            raise IdeaValidationError(f"{self.id}: history must contain a 'captured' event")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "IdeaEntry":
        default_links = {
            "related_backlog": [],
            "related_ideas": [],
            "promoted_to": None,
            "discarded_reason": None,
        }
        links = payload.get("links") or {}
        merged_links = {**default_links, **links}
        # Coerce list fields
        for key in ("related_backlog", "related_ideas"):
            merged_links[key] = list(merged_links.get(key) or [])
        return cls(
            id=payload.get("id", ""),
            title=payload.get("title", ""),
            status=payload.get("status", "open"),
            tags=list(payload.get("tags", [])),
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
            description=payload.get("description", ""),
            source=payload.get("source", ""),
            context=payload.get("context", ""),
            links=merged_links,
            history=list(payload.get("history", [])),
        )


def make_captured_event(via: str = "/capture") -> dict:
    return {"date": date.today().isoformat(), "event": "captured", "via": via}
