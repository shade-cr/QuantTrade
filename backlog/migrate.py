"""One-shot migration: BACKLOG.md -> backlog/<status>/B*.json + INDEX.json.

Parses the legacy markdown using regex on `### B<NNNN>` headers + bulleted
metadata lines. Imperfect by design — a manual review pass is expected after
this runs. Idempotent: re-running overwrites existing JSON entries.

Usage:
    uv run python -m backlog.migrate           # full run (writes files)
    uv run python -m backlog.migrate --dry-run # parse + report; no writes
    uv run python -m backlog.migrate --source BACKLOG.md  # override source path
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from backlog import db
from backlog.schema import (
    BacklogEntry,
    EFFORTS,
    LINK_RELATIONS,
    STATUSES,
    VALUES,
)


HEADER_RE = re.compile(r"^### (B\d{4})\s+—\s+(.+?)$", re.MULTILINE)
SECTION_HEADER_RE = re.compile(r"^## (.+?)$", re.MULTILINE)
BULLET_RE = re.compile(r"^- \*\*([A-Za-z_][A-Za-z _0-9-]*?)\*\*:\s*(.+?)$", re.MULTILINE)
B_ID_RE = re.compile(r"\b(B\d{4})\b")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
DA_REVIEW_RE = re.compile(r"backlog_reviews/(\d{8}[A-Za-z0-9_\-]+\.json)")


def parse_status_line(status_text: str) -> str:
    """Normalize a free-form 'Status:' field into one of STATUSES."""
    t = status_text.lower()
    # Aggressive normalization
    if "in-progress" in t or "in progress" in t or "in_progress" in t:
        return "in_progress"
    if any(k in t for k in ("archived", "closed", "done")):
        return "done"
    if "discarded" in t:
        return "discarded"
    if "blocked" in t or "deferred" in t:
        return "blocked"
    if "proposed" in t:
        return "proposed"
    return "proposed"  # default fallback


def parse_effort_field(effort_text: str):
    """Normalize the 'Effort:' field. Returns str (single enum) or dict (multi-option)."""
    # Multi-option pattern: "Option A = S, Option B = XS"
    if "option" in effort_text.lower() and "=" in effort_text:
        out: dict[str, str] = {}
        for chunk in re.split(r"[;,]", effort_text):
            m = re.search(r"option\s*([a-z])\s*=\s*([A-Z\-]+)", chunk, re.IGNORECASE)
            if m:
                opt = "option_" + m.group(1).lower()
                eff = m.group(2).upper().split("-")[0]  # "M-L" -> "M"
                if eff in EFFORTS:
                    out[opt] = eff
        if out:
            return out
    # Single value: take the first uppercase token that matches
    for tok in re.split(r"\W+", effort_text):
        tok_up = tok.upper().strip()
        if tok_up in EFFORTS:
            return tok_up
        # collapse compound like M-L -> M
        first = tok_up.split("-")[0] if "-" in tok_up else tok_up
        if first in EFFORTS:
            return first
    return "M"  # fallback


def parse_value_field(value_text: str) -> str:
    """Normalize 'Value:' field to one of {L, M, H}. Aggressive."""
    for tok in re.split(r"\W+", value_text):
        tok_up = tok.upper().strip()
        if tok_up in VALUES:
            return tok_up
        # "M-H" -> first letter that's valid
        if "-" in tok_up:
            for part in tok_up.split("-"):
                if part in VALUES:
                    return part
    return "M"


def extract_dates(text: str) -> tuple[str, str]:
    """Extract earliest + latest YYYY-MM-DD dates from a block of text."""
    dates = DATE_RE.findall(text)
    if not dates:
        return "2026-05-26", "2026-05-26"
    return min(dates), max(dates)


def parse_entry_block(entry_id: str, title: str, body: str, default_status: str) -> BacklogEntry:
    """Parse one ### entry block into a BacklogEntry."""
    bullets = {m.group(1).strip().lower().replace(" ", "_"): m.group(2).strip()
               for m in BULLET_RE.finditer(body)}

    status_text = bullets.get("status", "")
    status = parse_status_line(status_text) if status_text else default_status
    effort = parse_effort_field(bullets.get("effort", "M"))
    value = parse_value_field(bullets.get("value", "M"))

    why = bullets.get("why", "").strip()
    source = bullets.get("source", "").strip()
    value_rationale = bullets.get("value", "").strip()
    out_of_scope = bullets.get("out_of_scope", "").strip()
    recommendation = bullets.get("recommendation", "").strip()

    scope_text = bullets.get("scope", "").strip()
    scope: list = []
    if scope_text:
        scope = [scope_text]
    # Capture multi-line scope sub-items (lines starting with "  -" under Scope)
    scope_sub_re = re.compile(r"^\s+-\s+(.+?)$", re.MULTILINE)
    # Only grab bullets directly after "- **Scope" or "- **Scope(...)**:" line
    scope_match = re.search(r"- \*\*Scope[^*]*\*\*:.*?(\n(?:\s{2,}-.+\n?)+)", body)
    if scope_match:
        sub_items = scope_sub_re.findall(scope_match.group(1))
        if sub_items:
            scope = sub_items

    # Cross-references
    related = sorted({m for m in B_ID_RE.findall(body) if m != entry_id})

    # External refs
    external_refs: list[dict] = []
    for label, path in MD_LINK_RE.findall(body):
        # filter out the obvious non-file URLs and same-doc anchors
        if path.startswith(("http", "#", "mailto:")):
            continue
        external_refs.append({"path": path, "relation": "reference", "label": label})

    # DA reviews
    da_reviews: list[dict] = []
    for fname in DA_REVIEW_RE.findall(body):
        da_reviews.append({
            "path": f"backlog_reviews/{fname}",
            "verdict": "BLOCK" if "block" in body.lower() else "UNKNOWN",
            "severity_counts": {},
            "date": fname[:8],
        })

    # Dates
    created_at, updated_at = extract_dates(body)

    # Tags — derive from header keywords (simple heuristic)
    tags: list[str] = []
    body_lower = body.lower()
    for tag, keywords in {
        "phase5": ["phase 5", "phase5"],
        "loop-a": ["loop a", "loop-a"],
        "h4": ["h4 ", " h4", "h4-"],
        "audit": ["audit"],
        "alt-data": ["alt-data", "alt data"],
        "cross-asset": ["cross-asset", "cross asset"],
        "infrastructure": ["infrastructure", "calibration"],
        "methodology": ["methodology"],
        "portfolio": ["portfolio"],
        "lint": ["lint"],
    }.items():
        if any(k in body_lower for k in keywords):
            tags.append(tag)

    entry = BacklogEntry(
        id=entry_id,
        title=title.strip(),
        status=status,
        effort=effort,
        value=value,
        tags=tags,
        created_at=created_at,
        updated_at=updated_at,
        source=source,
        why=why,
        value_rationale=value_rationale,
        scope=scope,
        recommendation=recommendation,
        out_of_scope=out_of_scope,
        links={
            **{rel: [] for rel in LINK_RELATIONS},
            "related": related,
        },
        external_refs=external_refs,
        da_reviews=da_reviews,
        falsification_clock=None,
        history=[
            {
                "date": created_at,
                "event": "created",
                "reason": "migrated from BACKLOG.md (auto-parsed)",
            }
        ],
    )
    return entry


def split_into_entries(md_text: str) -> list[tuple[str, str, str, str]]:
    """Split BACKLOG.md into (entry_id, title, body, default_status) tuples."""
    # Identify section ranges so we can carry default_status forward.
    section_starts: list[tuple[int, str]] = []
    for m in SECTION_HEADER_RE.finditer(md_text):
        section_starts.append((m.start(), m.group(1).strip()))

    def section_for(pos: int) -> str:
        current = "Open"
        for start, name in section_starts:
            if start <= pos:
                current = name
            else:
                break
        return current

    section_to_default_status = {
        "Open": "proposed",
        "Done": "done",
        "Discarded": "discarded",
    }

    headers = list(HEADER_RE.finditer(md_text))
    out: list[tuple[str, str, str, str]] = []
    seen_ids: set[str] = set()
    for i, h in enumerate(headers):
        entry_id = h.group(1)
        if entry_id in seen_ids:
            # duplicate header (e.g., short stub in Loop A v1 cluster + full Done entry).
            # Take the LATER one — it's the canonical entry.
            for j in range(len(out)):
                if out[j][0] == entry_id:
                    out.pop(j)
                    break
        seen_ids.add(entry_id)
        title = h.group(2)
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(md_text)
        body = md_text[start:end]
        section = section_for(h.start())
        # Best-effort match for sub-section headers like "5 palancas" or "Loop A v1"
        # — they're still under "Open" / "Done" by content.
        default_status = section_to_default_status.get(section, "proposed")
        out.append((entry_id, title, body, default_status))
    return out


def normalize_status_with_overrides(entry: BacklogEntry, body: str) -> None:
    """Apply known status-override patterns from the prose body."""
    # archived/closed/etc. land in done/ regardless of containing section
    if entry.status == "proposed" and any(k in body.lower() for k in ("**archived", "closed 2026", "archived 2026")):
        entry.status = "done"


def run(source: Path, dry_run: bool = False) -> int:
    md_text = source.read_text(encoding="utf-8")
    entries_raw = split_into_entries(md_text)
    if not entries_raw:
        print(f"FAIL: no entries parsed from {source}")
        return 1

    db.ensure_layout()
    counters: dict[str, int] = {s: 0 for s in STATUSES}
    parsed: list[BacklogEntry] = []

    for entry_id, title, body, default_status in entries_raw:
        entry = parse_entry_block(entry_id, title, body, default_status)
        normalize_status_with_overrides(entry, body)
        parsed.append(entry)
        counters[entry.status] += 1

    print(f"\nParsed {len(parsed)} entries:")
    for s in STATUSES:
        print(f"  {s:<13} {counters[s]}")

    if dry_run:
        print("\n(dry run — no files written)")
        return 0

    # write each entry
    written = 0
    errors: list[str] = []
    for entry in parsed:
        target = db._registry._entity_path(entry.id, entry.status)
        try:
            entry.validate()
        except Exception as e:
            errors.append(f"{entry.id}: validation failed -> {e}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(entry.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written += 1

    print(f"\nWrote {written} JSON entries to {db._registry.root}/")
    if errors:
        print(f"\n{len(errors)} validation errors:")
        for e in errors:
            print(f"  {e}")

    # rebuild INDEX from per-file truth
    db.rebuild_index()
    print(f"Rebuilt {db._registry.index_path}")

    # link integrity check
    validation_errors = db.validate_all()
    if validation_errors:
        print(f"\n{len(validation_errors)} integrity errors (expected — links need manual review):")
        for v in validation_errors[:20]:
            print(f"  {v}")
        if len(validation_errors) > 20:
            print(f"  ... ({len(validation_errors) - 20} more)")
    else:
        print("Integrity check clean.")

    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="BACKLOG.md", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return run(args.source, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
