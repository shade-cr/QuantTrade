"""commit_sync must NOT mutate registry JSON (Option A).

The git_commit history events the hook used to append are redundant with git
itself (`git log --grep=B0177`), and writing them back into the per-entry JSON
dirtied the working tree after every code commit that referenced a B-ID,
forcing a follow-up bookkeeping commit. Option A: commit_sync prints its
suggestions to stdout but never writes to the registry, so the tree stays
clean. This pins that no-write contract.
"""
from __future__ import annotations

import backlog.commit_sync as cs


def _patch_commit(monkeypatch, *, subject, body, files):
    monkeypatch.setattr(cs, "git_show", lambda sha: (subject, body))
    monkeypatch.setattr(cs, "changed_files", lambda sha: files)
    monkeypatch.setattr(cs.db, "exists", lambda b_id: True)
    monkeypatch.setattr(cs.db, "status_of", lambda b_id: "proposed")

    calls = []
    # Record (don't raise) — run() swallows exceptions from append_history, so a
    # raising spy would be hidden. We assert the call list stays empty instead.
    monkeypatch.setattr(cs.db, "append_history",
                        lambda *a, **k: calls.append((a, k)))
    return calls


def test_code_commit_referencing_bids_does_not_write_registry(monkeypatch, capsys):
    """A normal code commit mentioning B-IDs annotates NOTHING on disk."""
    calls = _patch_commit(
        monkeypatch,
        subject="fix(B0177): run_loop resilience",
        body="fix(B0177): run_loop resilience\n\nAlso touches B0108 in prose.",
        files=["scripts/run_survival_live.py", "docs/submission/operations-runbook.md"],
    )
    rc = cs.run("deadbeef")
    assert rc == 0  # never blocks
    assert calls == []  # Option A: NO registry writes
    out = capsys.readouterr().out
    # behavior preserved: still surfaces the references + transition suggestion
    assert "B0177" in out
    assert "B0108" in out


def test_bookkeeping_only_commit_still_short_circuits(monkeypatch):
    """The existing guard is untouched: a registry-only commit returns early."""
    calls = _patch_commit(
        monkeypatch,
        subject="chore(backlog): annotations",
        body="chore(backlog): annotations B0177",
        files=["backlog/done/B0177.json"],
    )
    assert cs.run("deadbeef") == 0
    assert calls == []  # never reached the (now-absent) write path
