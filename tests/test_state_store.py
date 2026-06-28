"""Tests for execution.state_store — the shared bounded-state-store helper (B0104,
spec 2026-05-30). Covers the durable-write primitive, the fixed-size BoundedIndex,
and the day-partitioned JSONL archive. These are the unit tests called for in the
spec's test plan §1.
"""
from __future__ import annotations

import json

from pipeline.util.state_store import (
    BoundedIndex,
    append_archive_jsonl,
    atomic_write_text,
)


# --- atomic_write_text ------------------------------------------------------ #
def test_atomic_write_text_writes_target_and_leaves_no_tmp(tmp_path):
    target = tmp_path / "sub" / "file.json"
    atomic_write_text(target, '{"a": 1}')
    assert target.read_text(encoding="utf-8") == '{"a": 1}'
    # no stray .tmp sibling left behind after a successful replace
    assert not (target.with_suffix(target.suffix + ".tmp")).exists()
    assert list(target.parent.glob("*.tmp")) == []


def test_atomic_write_text_overwrites_existing_intact(tmp_path):
    target = tmp_path / "file.json"
    atomic_write_text(target, "old")
    atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    assert list(target.parent.glob("*.tmp")) == []


# --- BoundedIndex.load (fail-soft) ------------------------------------------ #
def test_load_fail_soft_on_missing(tmp_path):
    idx = BoundedIndex(tmp_path / "nope.json")
    assert idx.load() == {}


def test_load_fail_soft_on_corrupt(tmp_path):
    p = tmp_path / "corrupt.json"
    p.write_text("{not valid json", encoding="utf-8")
    idx = BoundedIndex(p)
    assert idx.load() == {}


# --- bump_counter ----------------------------------------------------------- #
def test_bump_counter_increments_total_and_each_bucket(tmp_path):
    idx = BoundedIndex(tmp_path / "i.json")
    idx.load()
    idx.bump_counter("CRITICAL")
    idx.bump_counter("LOW")
    idx.bump_counter("LOW")
    idx.save()
    data = json.loads((tmp_path / "i.json").read_text(encoding="utf-8"))
    assert data["counters"]["total"] == 3
    assert data["counters"]["CRITICAL"] == 1
    assert data["counters"]["LOW"] == 2


def test_bump_counter_multiple_buckets_one_call(tmp_path):
    idx = BoundedIndex(tmp_path / "i.json")
    idx.load()
    idx.bump_counter("a", "b")
    idx.save()
    data = json.loads((tmp_path / "i.json").read_text(encoding="utf-8"))
    assert data["counters"]["total"] == 1
    assert data["counters"]["a"] == 1
    assert data["counters"]["b"] == 1


# --- push_recent prunes to K ------------------------------------------------ #
def test_push_recent_prunes_to_K(tmp_path):
    idx = BoundedIndex(tmp_path / "i.json", K=5)
    idx.load()
    for n in range(12):
        idx.push_recent({"n": n})
    idx.save()
    data = json.loads((tmp_path / "i.json").read_text(encoding="utf-8"))
    assert len(data["recent"]) == 5
    assert data["recent"][-1]["n"] == 11   # newest kept
    assert data["recent"][0]["n"] == 7     # oldest of the retained window


# --- set_projection upserts and stays bounded ------------------------------- #
def test_set_projection_upserts_in_place(tmp_path):
    idx = BoundedIndex(tmp_path / "i.json")
    idx.load()
    idx.set_projection("lru_by_cell", "A|D1|BULL", "2026-01-01")
    idx.set_projection("lru_by_cell", "A|D1|BULL", "2026-02-02")  # upsert same key
    idx.set_projection("lru_by_cell", "B|D1|BEAR", "2026-03-03")
    idx.save()
    data = json.loads((tmp_path / "i.json").read_text(encoding="utf-8"))
    proj = data["lru_by_cell"]
    assert proj["A|D1|BULL"] == "2026-02-02"   # upserted, not duplicated
    assert proj["B|D1|BEAR"] == "2026-03-03"
    assert len(proj) == 2                       # bounded by the finite key set


def test_set_projection_size_constant_across_repeated_keys(tmp_path):
    idx = BoundedIndex(tmp_path / "i.json")
    idx.load()
    for n in range(200):
        idx.set_projection("m", "fixed_key", n)   # same key, 200 times
    idx.save()
    data = json.loads((tmp_path / "i.json").read_text(encoding="utf-8"))
    assert len(data["m"]) == 1
    assert data["m"]["fixed_key"] == 199


# --- save() is atomic (no stray tmp) ---------------------------------------- #
def test_save_leaves_no_tmp(tmp_path):
    p = tmp_path / "i.json"
    idx = BoundedIndex(p)
    idx.load()
    idx.bump_counter("x")
    idx.save()
    assert p.exists()
    assert list(p.parent.glob("*.tmp")) == []


# --- append_archive_jsonl --------------------------------------------------- #
def test_append_archive_jsonl_appends_one_record_per_line(tmp_path):
    root = tmp_path / "loop_a"
    p1 = append_archive_jsonl(root, "2026-05-30", {"tick": 1})
    p2 = append_archive_jsonl(root, "2026-05-30", {"tick": 2})
    assert p1 == p2 == root / "2026-05-30.jsonl"
    lines = p1.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"tick": 1}
    assert json.loads(lines[1]) == {"tick": 2}


def test_append_archive_jsonl_partitions_by_date(tmp_path):
    root = tmp_path / "loop_a"
    append_archive_jsonl(root, "2026-05-30", {"d": "a"})
    append_archive_jsonl(root, "2026-05-31", {"d": "b"})
    assert (root / "2026-05-30.jsonl").exists()
    assert (root / "2026-05-31.jsonl").exists()
