"""Stop hook — session-end audit.

Inspects commits made since the session started and verifies each
referenced B-ID's history has a `git_commit` event citing one of those
SHAs. Surfaces missed cases to stdout. Non-blocking.

The "session started" marker is best-effort: use the last commit prior to
the session as the baseline, OR fall back to "last 10 commits" if no
explicit marker is set.

CLI:
    uv run python -m backlog.session_audit
    uv run python -m backlog.session_audit --since <ref>
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from typing import Optional

from backlog import db
from backlog.commit_sync import B_ID_RE


def git_log(since: Optional[str] = None, max_count: int = 10) -> list[tuple[str, str]]:
    """Return [(sha, subject), ...] from most recent first."""
    args = ["git", "log", f"--max-count={max_count}", "--pretty=%H%x09%s"]
    if since:
        args.append(f"{since}..HEAD")
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return []
    out: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        if "\t" in line:
            sha, subject = line.split("\t", 1)
            out.append((sha, subject))
    return out


def run(since: Optional[str], max_count: int) -> int:
    commits = git_log(since=since, max_count=max_count)
    if not commits:
        return 0

    issues: list[str] = []
    for sha, subject in commits:
        b_ids = sorted(set(B_ID_RE.findall(subject)))
        if not b_ids:
            continue
        for b_id in b_ids:
            if not db.exists(b_id):
                issues.append(f"commit {sha[:8]} references {b_id} but no entry exists")
                continue
            entry = db.get_entry(b_id)
            history_shas = {
                ev.get("sha", "")
                for ev in entry.history
                if ev.get("event") == "git_commit"
            }
            if sha[:12] not in history_shas and sha[:8] not in {s[:8] for s in history_shas}:
                issues.append(f"commit {sha[:8]} mentions {b_id} but entry history has no matching git_commit event")

    if issues:
        print("backlog.session_audit — possible forgotten updates:")
        for msg in issues:
            print(f"  {msg}")
        print()
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default=None, help="git ref; default = check last N commits")
    parser.add_argument("--max-count", type=int, default=10)
    args = parser.parse_args(argv)
    return run(since=args.since, max_count=args.max_count)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
