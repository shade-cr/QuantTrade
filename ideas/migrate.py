"""One-shot migration: IDEAS.md -> ideas/<status>/I*.json + INDEX.json.

IDEAS.md uses ## date headers + bulleted captures. Each bullet that starts
with **bold title** is one idea.

Usage:
    uv run python -m ideas.migrate
    uv run python -m ideas.migrate --dry-run
    uv run python -m ideas.migrate --source IDEAS.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ideas import db
from ideas.schema import IdeaEntry, STATUSES


DATE_SECTION_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2})", re.MULTILINE)
BULLET_RE = re.compile(r"^- \*\*([^*]+?)\*\*\s*[—\-]\s*(.+?)(?=\n- \*\*|\n##|\Z)", re.MULTILINE | re.DOTALL)
SOURCE_TAG_RE = re.compile(r"\(src:\s*([^)]+)\)", re.IGNORECASE)
B_ID_RE = re.compile(r"\b(B\d{4})\b")


def parse_idea_capture(capture_date: str, title: str, body: str, idea_id: str) -> IdeaEntry:
    """Build an IdeaEntry from one bulleted capture."""
    src = ""
    m = SOURCE_TAG_RE.search(body)
    if m:
        src = m.group(1).strip()
        body = SOURCE_TAG_RE.sub("", body).strip()
    description = body.strip().rstrip(".") + "."

    related_backlog = sorted(set(B_ID_RE.findall(description)))

    # Tag heuristics
    tags: list[str] = []
    body_lower = description.lower()
    if any(k in body_lower for k in ("cot", "gld", "gdelt", "alt-data", "alt data")):
        tags.append("alt-data")
    if "ratio" in body_lower or "spread" in body_lower:
        tags.append("cross-asset")
    if "regime" in body_lower:
        tags.append("regime")
    if "feature" in body_lower:
        tags.append("feature-engineering")
    if "primary" in body_lower:
        tags.append("primary")

    entry = IdeaEntry(
        id=idea_id,
        title=title.strip(),
        status="open",
        tags=tags,
        created_at=capture_date,
        updated_at=capture_date,
        description=description,
        source=src,
        context="",
        links={
            "related_backlog": related_backlog,
            "related_ideas": [],
            "promoted_to": None,
            "discarded_reason": None,
        },
        history=[
            {
                "date": capture_date,
                "event": "captured",
                "via": "/capture",
                "note": "migrated from IDEAS.md",
            }
        ],
    )
    return entry


def split_into_captures(md_text: str) -> list[tuple[str, str, str]]:
    """Split IDEAS.md into (capture_date, title, body) tuples."""
    out: list[tuple[str, str, str]] = []
    # Find each date section
    section_matches = list(DATE_SECTION_RE.finditer(md_text))
    for i, sm in enumerate(section_matches):
        capture_date = sm.group(1)
        section_start = sm.end()
        section_end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(md_text)
        section_body = md_text[section_start:section_end]
        for bm in BULLET_RE.finditer(section_body):
            title = bm.group(1).strip()
            body = bm.group(2).strip()
            out.append((capture_date, title, body))
    return out


def run(source: Path, dry_run: bool = False, start_id: int = 1) -> int:
    md_text = source.read_text(encoding="utf-8")
    captures = split_into_captures(md_text)
    if not captures:
        print(f"FAIL: no captures parsed from {source}")
        return 1

    db.ensure_layout()
    parsed: list[IdeaEntry] = []
    next_n = start_id
    for capture_date, title, body in captures:
        idea_id = f"I{next_n:04d}"
        next_n += 1
        parsed.append(parse_idea_capture(capture_date, title, body, idea_id))

    print(f"\nParsed {len(parsed)} ideas (IDs I{start_id:04d}..I{next_n - 1:04d}):")
    for entry in parsed:
        print(f"  {entry.id}  {entry.title[:70]}")

    if dry_run:
        print("\n(dry run — no files written)")
        return 0

    errors: list[str] = []
    for entry in parsed:
        target = db._registry._entity_path(entry.id, entry.status)
        try:
            entry.validate()
        except Exception as e:
            errors.append(f"{entry.id}: {e}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(entry.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    db.rebuild_index()
    print(f"\nWrote {len(parsed) - len(errors)} entries; rebuilt {db._registry.index_path}")

    if errors:
        print(f"\n{len(errors)} validation errors:")
        for e in errors:
            print(f"  {e}")

    validation_errors = db.validate_all()
    if validation_errors:
        print(f"\n{len(validation_errors)} integrity errors:")
        for v in validation_errors[:20]:
            print(f"  {v}")
    else:
        print("Integrity check clean.")

    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="IDEAS.md", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-id", type=int, default=1)
    args = parser.parse_args(argv)
    return run(args.source, dry_run=args.dry_run, start_id=args.start_id)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
