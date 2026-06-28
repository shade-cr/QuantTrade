# Backlog schema

Agent-facing reference for the backlog filesystem database. Humans rarely
read this; it's here so agents can rebuild the mental model in one fetch.

## Layout

```
backlog/
├── INDEX.json                              # global state + summary rows
├── SCHEMA.md                               # this file
├── schema.py                               # dataclass + validate()
├── db.py                                   # public API; agents call this
├── lint.py                                 # repo-level validator
├── migrate.py                              # one-shot; deleted after stable
├── commit_sync.py                          # PostToolUse git-commit hook
├── session_audit.py                        # Stop hook
├── proposed/      B<NNNN>.json
├── in_progress/   B<NNNN>.json
├── blocked/       B<NNNN>.json
├── done/          B<NNNN>.json
└── discarded/     B<NNNN>.json
```

**Active** = `proposed + in_progress + blocked`. Recalibration loads only
these via `db.load_active_entries()`. **Archive** = `done + discarded`,
touched only by direct `db.get_entry(b_id)` when a link traversal resolves
into them.

## Entry fields

| Field | Type | Notes |
|---|---|---|
| `id` | `str` | `^B\d{4}$`; immutable; matches filename |
| `title` | `str` | Short headline; required |
| `status` | `str` | One of `proposed`, `in_progress`, `blocked`, `done`, `discarded`; MUST match parent folder |
| `effort` | `str \| dict` | `XS / S / M / L`; or `{option_a: "S", option_b: "XS", ...}` for multi-option entries (B0040) |
| `value` | `str` | `L / M / H` |
| `tags` | `list[str]` | Cross-cutting filters (e.g., `loop-a`, `phase5`, `h4`) |
| `created_at` | `str` | `YYYY-MM-DD` |
| `updated_at` | `str` | `YYYY-MM-DD`; auto-set on any write |
| `source` | `str` | Where the idea/B-ID surfaced (prose) |
| `why` | `str` | Motivation paragraph |
| `value_rationale` | `str` | Why this value rating |
| `scope` | `list` | List of `str` OR list of `dict` (multi-option entries) |
| `recommendation` | `str` | Optional |
| `out_of_scope` | `str` | Optional |
| `links` | `dict` | See below |
| `external_refs` | `list[dict]` | `[{path, relation, label?}, ...]` |
| `da_reviews` | `list[dict]` | `[{path, verdict, severity_counts, date}, ...]` |
| `falsification_clock` | `dict \| null` | See below |
| `history` | `list[dict]` | Append-only; first event must be `created` |

### `links` (dict; symmetric backlinks enforced)

| Relation | Inverse | Notes |
|---|---|---|
| `blocks` | `blocked_by` | This entry blocks the listed B-IDs |
| `related` | `related` | Self-inverse (peers) |
| `supersedes` | `superseded_by` | This entry replaces the listed B-IDs |
| `spawned` | `spawned_by` | This entry spawned the listed B-IDs |
| `spawned_from` | (cross-prefix) | Set when promoted from an I-ID (in `ideas/`) |

All values are bare B-IDs (or I-IDs for `spawned_from`). The lint enforces
target existence and symmetric backlinks (e.g., `B0040.blocks=[B0030]`
implies `B0030.blocked_by` contains `B0040`).

### `falsification_clock` (optional)

```jsonc
{
  "metric": "ticks_until_archive",
  "current": 1,
  "limit": 20,
  "trigger": "no Loop A survivor uses any B0036 feature in top-5 MDA",
  "archive_under": "B0000 §scope-reopen-trigger #5"
}
```

`db.tick_falsification_clock(b_id, delta=1, reason="...")` advances `current`.

### `history` events

```jsonc
[
  {"date": "2026-05-26", "event": "created", "reason": "..."},
  {"date": "2026-05-27", "event": "status_change", "reason": "...", "from": "proposed", "to": "in_progress"},
  {"date": "2026-05-27", "event": "field_update", "reason": "downgraded H->M", "field": "value", "from": "H", "to": "M"},
  {"date": "2026-05-28", "event": "da_review", "reason": "BLOCK with 2 high", "path": "backlog_reviews/..."}
]
```

Append-only. `db.append_history(...)`, `db.change_status(...)`, and
`db.update_field(...)` all do this automatically.

## Public API (`backlog.db`)

```python
# reads
get_entry(b_id)              # any folder; cheap
exists(b_id), status_of(b_id)
list_active(value=None, tag=None)        # proposed + in_progress + blocked
list_by_status(status)
list_archive(status=None)                # done + discarded; rare
list_all()
load_active_entries()                    # full entries from active folders only
list_blocking(b_id, transitive=False)
list_blocked_by(b_id, transitive=False)
load_index()

# writes (all validate before persist)
add_entry(entry, status=None)
change_status(b_id, new_status, reason)  # moves file; appends history
update_field(b_id, field, value, reason)
append_history(b_id, event, reason, extra=None)
add_link(b_id_a, b_id_b, relation)       # symmetric (writes both sides)
add_da_review(b_id, review_path, verdict, severity_counts=None, review_date=None)
tick_falsification_clock(b_id, delta=1, reason="")

# maintenance
rebuild_index(extra=None)
validate_all()                           # list[ValidationError]; [] = clean
```

## Invariants

1. **Filename matches `id`**: `B0040.json` MUST contain `"id": "B0040"`.
2. **Folder matches `status`**: `done/B0007.json` MUST contain `"status": "done"`.
3. **All linked B-IDs exist**: `links.related = ["B9999"]` is invalid unless `B9999.json` exists somewhere.
4. **Symmetric backlinks**: if A says it blocks B, B must say it's blocked by A.
5. **No cycles via `blocks`**: B0040.blocks → B0030.blocks → B0040 is invalid.
6. **History is non-empty + has `created`**: every entry has at least one event with `"event": "created"`.

The lint (`uv run python -m backlog.lint`) checks all six. Pre-commit hook
runs it; bad data never lands on `main`.

## Recalibration contract

A recalibration agent should:

1. Call `db.load_active_entries()` once. Snapshot.
2. Use `db.load_index()` for cross-reference resolution (lightweight).
3. Use `db.get_entry(b_id)` for the rare archive lookup (e.g., resolving `B0007` as a `superseded_by` target).
4. Apply changes via `db.change_status(...)`, `db.update_field(...)`, `db.add_link(...)`.
5. The library validates each write; bad transitions fail loudly.
6. Done. Never iterate `done/` or `discarded/` folders during recalibration.
