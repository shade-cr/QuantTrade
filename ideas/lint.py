"""Ideas repo-level lint.

CLI:
    uv run python -m ideas.lint
    uv run python -m ideas.lint --json
    uv run python -m ideas.lint --fix
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ideas import db
from ideas.schema import ROOT


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args(argv)

    if args.fix:
        db.rebuild_index()
        print(f"FIX: rebuilt {db._registry.index_path}")

    errors = db.validate_all()

    # Cross-prefix integrity: an idea's links.promoted_to must point at a
    # real B-entry. We import backlog lazily so ideas/ does not hard-require it.
    cross_misses: list[dict] = []
    try:
        from backlog import db as bdb  # noqa: F401

        for iid in db.list_all():
            idea = db.get_entry(iid)
            pt = idea.links.get("promoted_to") if isinstance(idea.links, dict) else None
            if pt and not bdb.exists(pt):
                cross_misses.append({"id": iid, "kind": "promoted_to_missing", "target": pt})
    except Exception as e:  # backlog tree may not exist yet during bootstrap
        cross_misses.append({"id": "<bootstrap>", "kind": "backlog_unavailable", "detail": str(e)})

    structured = {
        "validation_errors": [{"id": e.entity_id, "message": e.message} for e in errors],
        "cross_prefix_errors": cross_misses,
        "summary": {
            "validation_errors": len(errors),
            "cross_prefix_errors": len(cross_misses),
        },
    }

    if args.json:
        print(json.dumps(structured, indent=2))
    else:
        if errors:
            print(f"{len(errors)} validation error(s):")
            for e in errors:
                print(f"  FAIL  {e}")
        if cross_misses:
            print(f"\n{len(cross_misses)} cross-prefix issue(s):")
            for w in cross_misses:
                print(f"  WARN  {w}")
        if not errors and not cross_misses:
            print("OK: ideas clean")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
