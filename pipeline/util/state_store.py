"""execution.state_store — context-safe state-file primitives (B0104).

The shared convention behind the project's "no file an agent reads into context
may grow unbounded" invariant. Three building blocks:

  * atomic_write_text   — the single durable-write primitive (tmp + flush +
                          os.fsync + os.replace). Factored from the two ad-hoc
                          copies in execution/watchdog.py and execution/state.py.
  * BoundedIndex        — a fixed-size JSON index (integer counters + recent[K]
                          + caller-defined bounded projections) over an append-
                          only archive. Never raises on a missing/corrupt index
                          (fail-soft to empty), mirroring write_finding's contract.
  * append_archive_jsonl— append one record to <archive_root>/<date>.jsonl,
                          the machine-readable, day-partitioned full history.

Pure helpers + one class; the only side effects are the explicit atomic writes.
See docs/superpowers/specs/2026-05-30-bounded-state-store-design.md.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Cap long free-text in `recent` summaries so the index stays readable in one
# glance. Matches execution/watchdog.py::_EXPLANATION_CAP exactly (399, so the
# capped string + the appended ellipsis is <= 400 chars).
_EXPLANATION_CAP = 399


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically and DURABLY (tmp + fsync + os.replace).

    The fsync forces the data to disk before the rename, so a power-loss/crash
    cannot leave the destination renamed-but-empty (a torn write). os.replace is
    atomic on both Windows and POSIX. Creates parent dirs as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def cap_text(text: str, cap: int = _EXPLANATION_CAP) -> str:
    """Cap long free-text, appending an ellipsis when truncated. Matches the
    watchdog's _EXPLANATION_CAP behavior so callers can keep `recent` summaries
    short (the index must stay context-safe)."""
    text = str(text)
    return text[:cap] + ("…" if len(text) > cap else "")


class BoundedIndex:
    """A fixed-size, context-safe JSON index over an append-only archive.

    On-disk layout (the ONLY file routinely read into agent context):
      { "version": int,
        "counters": {"total": int, <bucket>: int, ...},   # integers, never lists
        "recent":   [ <summary dict>, ... ][-K:],          # capped to K
        <caller-defined bounded projections, e.g. "lru_by_cell": {...}> }

    Never raises on a missing/corrupt index (fail-soft to empty), mirroring
    write_finding's current contract. Mutations operate on an in-memory dict;
    `save()` writes the WHOLE index in one atomic_write_text.

    Projections (set_projection) are caller-defined but MUST be bounded by
    construction (keyed by a fixed finite set). The helper does not police this;
    the caller documents the bound.
    """

    def __init__(self, path: Path, *, K: int = 50) -> None:
        self.path = Path(path)
        self.K = K
        self._data: dict | None = None

    def load(self) -> dict:
        """Read the index from disk, fail-soft to an empty dict on a missing or
        corrupt file. Caches the result for subsequent mutations."""
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._data = {}
        if not isinstance(self._data, dict):
            self._data = {}
        return self._data

    def _ensure(self) -> dict:
        if self._data is None:
            self.load()
        return self._data  # type: ignore[return-value]

    def bump_counter(self, *buckets: str) -> None:
        """Increment `counters.total` by 1, plus each named bucket by 1."""
        data = self._ensure()
        counters = data.setdefault("counters", {"total": 0})
        counters["total"] = int(counters.get("total", 0)) + 1
        for bucket in buckets:
            counters[bucket] = int(counters.get(bucket, 0)) + 1

    def push_recent(self, entry: dict) -> None:
        """Append `entry` to `recent` (newest at end) and prune to the last K."""
        data = self._ensure()
        recent: list[dict] = data.get("recent", [])
        recent.append(entry)
        data["recent"] = recent[-self.K:]

    def set_projection(self, key: str, subkey: str, value: Any) -> None:
        """Upsert `value` at `index[key][subkey]`. The map is bounded by the
        caller's finite subkey set; repeated subkeys overwrite in place."""
        data = self._ensure()
        proj = data.setdefault(key, {})
        proj[subkey] = value

    def save(self) -> None:
        """Atomically write the whole index to disk (one atomic_write_text)."""
        data = self._ensure()
        atomic_write_text(self.path, json.dumps(data, indent=2, default=str))


def append_archive_jsonl(archive_root: Path, date_str: str, record: dict) -> Path:
    """Append one record to <archive_root>/<date_str>.jsonl (machine-readable,
    day-partitioned). Returns the file path. Mirror of the watchdog .md archive
    but JSONL so the history is queryable without an LLM. Append-only (open "a")
    is the correct durability model for an event journal."""
    archive_root = Path(archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    path = archive_root / f"{date_str}.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return path
