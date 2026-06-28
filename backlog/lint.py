"""Backlog repo-level lint.

Walks backlog/ tree; verifies JSON parse, schema, folder/status match,
link integrity, symmetric backlinks, cycle freedom, INDEX consistency,
external_refs existence.

CLI:
    uv run python -m backlog.lint               # human; exit 1 on failure
    uv run python -m backlog.lint --json        # machine output
    uv run python -m backlog.lint --fix         # safe-only fixes: rebuild INDEX

Exit codes:
    0  clean
    1  validation errors
    2  invocation error
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backlog import db
from backlog.schema import ROOT


REPO_ROOT = Path(".")


def check_external_refs() -> list[dict]:
    """Verify external_refs[].path entries point at existing files."""
    out: list[dict] = []
    for eid in db.list_all():
        entry = db.get_entry(eid)
        for ref in entry.external_refs:
            path = ref.get("path", "")
            if not path or path.startswith(("http", "mailto:")):
                continue
            # Strip anchors and line ranges for existence test
            cleaned = path.split("#")[0]
            if not (REPO_ROOT / cleaned).exists():
                out.append({"id": eid, "kind": "external_ref_missing", "path": cleaned})
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="machine output")
    parser.add_argument("--fix", action="store_true", help="apply safe fixes (INDEX rebuild)")
    args = parser.parse_args(argv)

    if args.fix:
        db.rebuild_index()
        print(f"FIX: rebuilt {db._registry.index_path}")

    errors = db.validate_all()
    ext_misses = check_external_refs()

    structured = {
        "validation_errors": [{"id": e.entity_id, "message": e.message} for e in errors],
        "external_ref_warnings": ext_misses,
        "summary": {
            "validation_errors": len(errors),
            "external_ref_warnings": len(ext_misses),
        },
    }

    if args.json:
        print(json.dumps(structured, indent=2))
    else:
        if errors:
            print(f"{len(errors)} validation error(s):")
            for e in errors:
                print(f"  FAIL  {e}")
        if ext_misses:
            print(f"\n{len(ext_misses)} external_ref warning(s) (non-blocking):")
            for w in ext_misses[:30]:
                print(f"  WARN  {w['id']}  {w['kind']}  {w['path']}")
            if len(ext_misses) > 30:
                print(f"  ... ({len(ext_misses) - 30} more)")
        if not errors and not ext_misses:
            print("OK: backlog clean")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
