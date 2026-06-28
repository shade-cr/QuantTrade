"""PostToolUse hook for `git commit`.

Reads the just-made commit, finds B-ID references in subject + body,
appends history events to the referenced entries, and prints suggested
status transitions to stdout. NEVER blocks the commit (exits 0 always).

Recognized prefix patterns in commit subjects:
    feat(B0040): ...       -> suggests proposed -> in_progress
    fix(B0040): ...        -> suggests proposed -> in_progress
    done(B0040): ...       -> suggests * -> done
    close(B0040): ...      -> suggests * -> done
    archive(B0040): ...    -> suggests * -> done
    discard(B0040): ...    -> suggests * -> discarded

Plain references (B0040 mentioned without a prefix) append a history event
but suggest no transition.

CLI:
    uv run python -m backlog.commit_sync [<sha>]
        sha defaults to HEAD
"""
from __future__ import annotations

import re
import subprocess
import sys
from typing import Optional

from backlog import db
from backlog.schema import STATUSES

B_ID_RE = re.compile(r"\b(B\d{4})\b")

# subject-line prefix patterns -> suggested transition
PREFIX_PATTERNS = [
    (re.compile(r"\b(?:feat|fix|refactor|impl|wip)\s*\(\s*(B\d{4})", re.IGNORECASE), "in_progress"),
    (re.compile(r"\b(?:done|close[ds]?|complete[ds]?)\s*\(\s*(B\d{4})", re.IGNORECASE), "done"),
    (re.compile(r"\b(?:archive[ds]?)\s*\(\s*(B\d{4})", re.IGNORECASE), "done"),
    (re.compile(r"\b(?:discard[s]?|reject[s]?)\s*\(\s*(B\d{4})", re.IGNORECASE), "discarded"),
]


def git_show(sha: str) -> tuple[str, str]:
    """Return (subject, full_body) for the given commit SHA."""
    subject = subprocess.run(
        ["git", "log", "-1", "--pretty=%s", sha],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    body = subprocess.run(
        ["git", "log", "-1", "--pretty=%B", sha],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return subject, body


def changed_files(sha: str) -> list[str]:
    """Return the repo-relative paths changed by the given commit."""
    out = subprocess.run(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", sha],
        check=True, capture_output=True, text=True,
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


# Registry DATA files (not the registry's Python code): per-entry JSON under a
# status subfolder, or the index. backlog/db.py etc. are code and do NOT match.
_REGISTRY_DATA_RE = re.compile(r"^(?:backlog|ideas)/(?:[^/]+/[A-Z]\d{4}\.json|index\.json)$")


def is_bookkeeping_only(files: list[str]) -> bool:
    """True if the commit touches ONLY backlog/ or ideas/ registry DATA files.

    These are the registry's own annotation churn (e.g. the git_commit history
    events this very hook writes). Annotating them would re-reference the same
    B-IDs and retrigger another round of churn -> a self-feeding loop. A commit
    that links code/docs to a B-ID always touches files outside the registry
    data (and registry CODE like backlog/db.py is not bookkeeping either).
    """
    if not files:
        return False
    return all(_REGISTRY_DATA_RE.match(f) for f in files)


def detect_intent(subject: str) -> dict[str, str]:
    """Map B-ID -> suggested new status from subject prefix patterns."""
    out: dict[str, str] = {}
    for pattern, new_status in PREFIX_PATTERNS:
        for m in pattern.finditer(subject):
            b_id = m.group(1)
            # First-match-wins per B-ID (patterns are ordered: active before terminal)
            out.setdefault(b_id, new_status)
    return out


def run(sha: str) -> int:
    try:
        subject, body = git_show(sha)
    except subprocess.CalledProcessError as e:
        print(f"backlog.commit_sync: git show failed for {sha!r}: {e}", file=sys.stderr)
        return 0  # never block

    # Skip commits that ONLY touch backlog/ or ideas/ bookkeeping. Annotating
    # them would re-reference the same B-IDs and retrigger annotation -> an
    # endless churn loop. See is_bookkeeping_only().
    try:
        if is_bookkeeping_only(changed_files(sha)):
            return 0
    except subprocess.CalledProcessError:
        pass  # diff-tree failed (e.g. root commit) -> fall through, never block

    all_b_ids = sorted(set(B_ID_RE.findall(subject + "\n" + body)))
    if not all_b_ids:
        return 0

    intent = detect_intent(subject)

    print(f"\nbacklog.commit_sync — commit {sha[:8]} references {len(all_b_ids)} B-ID(s):")
    for b_id in all_b_ids:
        if not db.exists(b_id):
            print(f"  WARN  {b_id}: referenced but no entry exists")
            continue

        # Option A — print suggestions only; NEVER mutate the registry JSON.
        # The old git_commit history event is fully redundant with git itself
        # (`git log --grep=<B-ID>` reconstructs commit<->B-ID), and writing it
        # back into the per-entry JSON dirtied the working tree after every code
        # commit that referenced a B-ID, forcing a follow-up bookkeeping commit.
        # Surfacing the linkage + transition hint to stdout keeps all the value
        # with zero working-tree churn.

        # Suggest transitions where appropriate
        if b_id in intent:
            suggested = intent[b_id]
            current = db.status_of(b_id)
            if current == suggested:
                continue
            if suggested == "in_progress" and current == "proposed":
                print(f"  SUGGEST  {b_id}: '{current}' -> 'in_progress'  ({subject})")
            elif suggested == "done" and current not in ("done", "discarded"):
                print(f"  SUGGEST  {b_id}: '{current}' -> 'done'  ({subject})")
            elif suggested == "discarded" and current not in ("done", "discarded"):
                print(f"  SUGGEST  {b_id}: '{current}' -> 'discarded'  ({subject})")
            else:
                print(f"  INFO     {b_id}: '{current}' (no transition; pattern matched but inappropriate)")
        else:
            print(f"  INFO     {b_id}: referenced (no status transition suggested)")

    print()
    return 0


def main(argv: list[str]) -> int:
    sha = argv[0] if argv else "HEAD"
    return run(sha)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
