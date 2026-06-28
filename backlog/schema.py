"""Backlog entry schema — stdlib dataclasses + .validate().

Mirrors `phase5/proposal.py` style. Used by `backlog/db.py` (via
`pipeline.registry.Registry`) and `backlog/lint.py`.

Field structure documented in `backlog/SCHEMA.md`. Pure JSON — no embedded
markdown body. Prose lives in `why` / `value_rationale` / `source` as plain
strings (agent-readable; humans read the legacy/ archive if curious).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

ROOT = Path("backlog")
INDEX_PATH = ROOT / "INDEX.json"

STATUSES = ("proposed", "in_progress", "blocked", "done", "discarded")
ACTIVE_STATUSES = ("proposed", "in_progress", "blocked")
ARCHIVE_STATUSES = ("done", "discarded")

EFFORTS = ("XS", "S", "M", "L")
VALUES = ("L", "M", "H")

LINK_RELATIONS = (
    "blocks",
    "blocked_by",
    "related",
    "supersedes",
    "superseded_by",
    "spawned",
    "spawned_by",
    "spawned_from",  # I-ID this B-entry came from on promotion (cross-prefix)
)

ID_REGEX = re.compile(r"^B\d{4}$")
B_ID_REGEX = re.compile(r"^B\d{4}$")
I_ID_REGEX = re.compile(r"^I\d{4}$")
DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class BacklogValidationError(ValueError):
    """Raised when a backlog entry violates schema or commit-time rules."""


@dataclass
class BacklogEntry:
    """One B-ID ticket. Serialized as JSON; loaded via from_dict."""

    id: str = ""
    title: str = ""
    status: str = "proposed"
    effort: Any = "M"  # str enum OR dict for multi-option entries
    value: str = "M"
    tags: list[str] = field(default_factory=list)

    created_at: str = ""
    updated_at: str = ""

    source: str = ""
    why: str = ""
    value_rationale: str = ""
    scope: Any = field(default_factory=list)  # list[str] OR list[dict] for multi-option
    recommendation: str = ""
    out_of_scope: str = ""

    links: dict = field(default_factory=lambda: {rel: [] for rel in LINK_RELATIONS})
    external_refs: list[dict] = field(default_factory=list)
    da_reviews: list[dict] = field(default_factory=list)
    falsification_clock: Optional[dict] = None
    history: list[dict] = field(default_factory=list)

    # ---------------------------------------------------------------- validate

    def validate(self) -> None:
        if not ID_REGEX.match(self.id):
            raise BacklogValidationError(f"id={self.id!r} must match ^B\\d{{4}}$")
        if not self.title or not isinstance(self.title, str):
            raise BacklogValidationError(f"{self.id}: title must be non-empty string")
        if self.status not in STATUSES:
            raise BacklogValidationError(
                f"{self.id}: status={self.status!r} not in {STATUSES}"
            )
        self._validate_effort()
        if self.value not in VALUES:
            raise BacklogValidationError(f"{self.id}: value={self.value!r} not in {VALUES}")
        if not isinstance(self.tags, list) or not all(isinstance(t, str) for t in self.tags):
            raise BacklogValidationError(f"{self.id}: tags must be list[str]")
        for fname in ("created_at", "updated_at"):
            v = getattr(self, fname)
            if not v or not DATE_REGEX.match(v):
                raise BacklogValidationError(f"{self.id}: {fname} must be YYYY-MM-DD; got {v!r}")
        self._validate_links()
        self._validate_external_refs()
        self._validate_da_reviews()
        self._validate_falsification_clock()
        self._validate_history()

    def _validate_effort(self) -> None:
        if isinstance(self.effort, str):
            if self.effort not in EFFORTS:
                raise BacklogValidationError(f"{self.id}: effort={self.effort!r} not in {EFFORTS}")
        elif isinstance(self.effort, dict):
            if not self.effort:
                raise BacklogValidationError(f"{self.id}: effort dict must be non-empty")
            for k, v in self.effort.items():
                if not isinstance(k, str):
                    raise BacklogValidationError(f"{self.id}: effort dict key {k!r} must be str")
                if not isinstance(v, str):
                    raise BacklogValidationError(f"{self.id}: effort.{k} must be str")
        else:
            raise BacklogValidationError(
                f"{self.id}: effort must be str enum or dict; got {type(self.effort).__name__}"
            )

    def _validate_links(self) -> None:
        if not isinstance(self.links, dict):
            raise BacklogValidationError(f"{self.id}: links must be dict")
        for relation, targets in self.links.items():
            if relation not in LINK_RELATIONS:
                raise BacklogValidationError(
                    f"{self.id}: links.{relation} not in {LINK_RELATIONS}"
                )
            if not isinstance(targets, list):
                raise BacklogValidationError(f"{self.id}: links.{relation} must be list")
            for t in targets:
                if not isinstance(t, str):
                    raise BacklogValidationError(f"{self.id}: links.{relation} contains non-string {t!r}")
                if not (B_ID_REGEX.match(t) or I_ID_REGEX.match(t)):
                    raise BacklogValidationError(
                        f"{self.id}: links.{relation} entry {t!r} must match B\\d{{4}} or I\\d{{4}}"
                    )

    def _validate_external_refs(self) -> None:
        if not isinstance(self.external_refs, list):
            raise BacklogValidationError(f"{self.id}: external_refs must be list")
        for ref in self.external_refs:
            if not isinstance(ref, dict):
                raise BacklogValidationError(f"{self.id}: external_refs entry must be dict; got {ref!r}")
            if "path" not in ref:
                raise BacklogValidationError(f"{self.id}: external_refs entry missing 'path'")

    def _validate_da_reviews(self) -> None:
        if not isinstance(self.da_reviews, list):
            raise BacklogValidationError(f"{self.id}: da_reviews must be list")
        for r in self.da_reviews:
            if not isinstance(r, dict):
                raise BacklogValidationError(f"{self.id}: da_reviews entry must be dict")
            if "path" not in r:
                raise BacklogValidationError(f"{self.id}: da_reviews entry missing 'path'")

    def _validate_falsification_clock(self) -> None:
        if self.falsification_clock is None:
            return
        if not isinstance(self.falsification_clock, dict):
            raise BacklogValidationError(f"{self.id}: falsification_clock must be dict or null")
        required = {"metric", "current", "limit", "trigger"}
        missing = required - set(self.falsification_clock.keys())
        if missing:
            raise BacklogValidationError(
                f"{self.id}: falsification_clock missing fields {sorted(missing)}"
            )

    def _validate_history(self) -> None:
        if not isinstance(self.history, list) or not self.history:
            raise BacklogValidationError(
                f"{self.id}: history must be non-empty list (at least the 'created' event)"
            )
        for ev in self.history:
            if not isinstance(ev, dict):
                raise BacklogValidationError(f"{self.id}: history entry must be dict; got {ev!r}")
            if "date" not in ev or "event" not in ev:
                raise BacklogValidationError(f"{self.id}: history entry missing date/event")
        # at least one 'created' event
        if not any(ev.get("event") == "created" for ev in self.history):
            raise BacklogValidationError(f"{self.id}: history must contain a 'created' event")

    # --------------------------------------------------------------- to/from

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "BacklogEntry":
        """Reconstruct from a JSON payload, filling defaults for missing fields."""
        # Ensure all link relations present even if some are absent in older entries
        links = payload.get("links") or {}
        full_links = {rel: list(links.get(rel, [])) for rel in LINK_RELATIONS}
        return cls(
            id=payload.get("id", ""),
            title=payload.get("title", ""),
            status=payload.get("status", "proposed"),
            effort=payload.get("effort", "M"),
            value=payload.get("value", "M"),
            tags=list(payload.get("tags", [])),
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
            source=payload.get("source", ""),
            why=payload.get("why", ""),
            value_rationale=payload.get("value_rationale", ""),
            scope=payload.get("scope", []),
            recommendation=payload.get("recommendation", ""),
            out_of_scope=payload.get("out_of_scope", ""),
            links=full_links,
            external_refs=list(payload.get("external_refs", [])),
            da_reviews=list(payload.get("da_reviews", [])),
            falsification_clock=payload.get("falsification_clock"),
            history=list(payload.get("history", [])),
        )


def make_created_event(reason: str = "initial entry") -> dict:
    """Convenience factory for the mandatory first history event."""
    return {"date": date.today().isoformat(), "event": "created", "reason": reason}
