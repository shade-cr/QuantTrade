# Ideas schema

Agent-facing reference for the ideas filesystem database. Captures raw
hypotheses/observations before triage promotes them into the backlog
(or discards them).

## Layout

```
ideas/
├── INDEX.json
├── SCHEMA.md
├── schema.py
├── db.py
├── lint.py
├── migrate.py
├── open/        I<NNNN>.json     # raw captures awaiting triage
├── promoted/    I<NNNN>.json     # triaged into a B-ID; links.promoted_to set
└── discarded/   I<NNNN>.json     # rejected at triage; links.discarded_reason set
```

**Active** = `open`. **Archive** = `promoted + discarded`.

Promotion is terminal: when an idea becomes a B-ID, the idea moves to
`promoted/` and the B-entry takes over. The B-entry should carry
`links.spawned_from = "I<NNNN>"`.

## Entry fields

| Field | Type | Notes |
|---|---|---|
| `id` | `str` | `^I\d{4}$`; immutable |
| `title` | `str` | Short headline |
| `status` | `str` | `open / promoted / discarded`; MUST match parent folder |
| `tags` | `list[str]` | Free-form |
| `created_at` | `str` | `YYYY-MM-DD` |
| `updated_at` | `str` | `YYYY-MM-DD` |
| `description` | `str` | Free-form prose (the capture itself) |
| `source` | `str` | Where the idea came from (conversation date, paper, …) |
| `context` | `str` | Optional related-context paragraph |
| `links` | `dict` | See below |
| `history` | `list[dict]` | First event must be `captured` |

### `links`

```jsonc
{
  "related_backlog":   ["B0015"],            // existing B-IDs this idea touches
  "related_ideas":     ["I0003"],            // peer ideas
  "promoted_to":       "B0042",              // set on promotion; null otherwise
  "discarded_reason":  "schedule lookahead"  // set on discard; null otherwise
}
```

The lint enforces:
- `promoted_to` set ⇒ status is `promoted` AND the target B-ID exists.
- `discarded_reason` set ⇒ status is `discarded`.
- Both null ⇒ status is `open`.

## Public API (`ideas.db`)

```python
# reads
get_entry(i_id), exists(i_id)
list_active(tag=None)                # 'open' only
list_by_status(status)
list_archive(status=None)
list_all()
load_active_entries()
load_index()

# writes (validate before persist)
next_id()                            # allocate I<NNNN> (monotonic)
add_entry(entry)                     # default status='open'
change_status(i_id, new_status, reason)
update_field(i_id, field, value, reason)
append_history(i_id, event, reason, extra=None)
promote_to_backlog(i_id, b_id, reason)   # status -> 'promoted', links.promoted_to=b_id
discard(i_id, reason)                    # status -> 'discarded', links.discarded_reason=reason

# maintenance
rebuild_index(extra=None)
validate_all()
```

## Promotion flow

Caller responsibilities (so `ideas.db` doesn't need to write across trees):

1. Decide the next B-ID (e.g., from `backlog.db.list_all()`).
2. Create the B-entry via `backlog.db.add_entry(...)` with
   `links.spawned_from = "I<NNNN>"`.
3. Call `ideas.db.promote_to_backlog(i_id, b_id, reason)`.

`/idea-triage` skill (when it lands) automates this orchestration.

## `/capture` integration

The `/capture` skill calls:

```python
from ideas import db as idb
from ideas.schema import IdeaEntry, make_captured_event

eid = idb.next_id()
entry = IdeaEntry(
    id=eid,
    title=...,
    status="open",
    tags=...,
    created_at=today,
    updated_at=today,
    description=...,
    source=...,
    history=[make_captured_event()],
)
idb.add_entry(entry)
```

This replaces appending to the legacy `IDEAS.md` (moved to `legacy/`).
