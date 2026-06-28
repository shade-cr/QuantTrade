"""Filesystem-as-database registry for ticketed entities (backlog, ideas).

Generic, parameterized base used by `backlog/db.py` and `ideas/db.py`.
Both subsystems share file layout + index + lint logic; only schema and
ID prefix differ.

Design contract — every entity stored via Registry must expose:
  - `.id: str` matching `id_regex`
  - `.status: str` in `statuses`
  - `.history: list[dict]` (append-only audit trail; updated by Registry)
  - `.validate()` -> None; raises on schema violation
  - `.to_dict()` -> dict (JSON-serializable)
  - classmethod `.from_dict(payload: dict)` (reconstructs from JSON)

File layout (parameterized by root + statuses):
    <root>/INDEX.json                  # summary rows + global state
    <root>/<status>/<id>.json          # one entity per file

Status changes move files between folders (shutil.move = git-mv-equivalent).
The `status` field inside the JSON must match the parent folder name; the
validator enforces this match.

Stdlib only (no Pydantic) per project convention.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol


class RegistryError(Exception):
    """Raised on registry-level integrity violations."""


class EntityNotFoundError(RegistryError):
    pass


class ValidationError(RegistryError):
    """Validation failure — bundles entity id + the failing message."""

    def __init__(self, entity_id: str, message: str):
        super().__init__(f"{entity_id}: {message}")
        self.entity_id = entity_id
        self.message = message


class EntityProtocol(Protocol):
    id: str
    status: str
    history: list[dict]

    def validate(self) -> None: ...
    def to_dict(self) -> dict: ...


def today_iso() -> str:
    """Return today's date as YYYY-MM-DD."""
    return date.today().isoformat()


def utc_now_iso() -> str:
    """Return current UTC time as ISO 8601 with timezone."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class HistoryEvent:
    date: str
    event: str
    reason: str = ""
    extra: dict | None = None

    def to_dict(self) -> dict:
        out = {"date": self.date, "event": self.event, "reason": self.reason}
        if self.extra:
            out.update(self.extra)
        return out


class Registry:
    """Generic entity registry over a status-folder filesystem layout."""

    def __init__(
        self,
        root: Path,
        id_prefix: str,
        id_width: int,
        statuses: tuple[str, ...],
        active_statuses: tuple[str, ...],
        archive_statuses: tuple[str, ...],
        from_dict: Callable[[dict], Any],
        index_path: Optional[Path] = None,
        default_new_status: str = "proposed",
    ):
        """
        Args:
            root: directory containing the status subfolders + INDEX.json
            id_prefix: 'B' or 'I'; first char of every entity ID
            id_width: zero-padded digit count (4 -> "B0040")
            statuses: every legal status; each must have a matching subfolder
            active_statuses: subset; recalibration / load_active touch only these
            archive_statuses: subset; touched only via direct get() or link resolution
            from_dict: factory to reconstruct an entity from a JSON payload
            index_path: path to INDEX.json (defaults to root / INDEX.json)
            default_new_status: where add() writes by default
        """
        self.root = Path(root)
        self.id_prefix = id_prefix
        self.id_width = id_width
        self.id_regex = re.compile(rf"^{re.escape(id_prefix)}\d{{{id_width}}}$")
        self.statuses = statuses
        self.active_statuses = active_statuses
        self.archive_statuses = archive_statuses
        self.from_dict = from_dict
        self.index_path = Path(index_path) if index_path else self.root / "INDEX.json"
        self.default_new_status = default_new_status

    # ------------------------------------------------------------------ paths

    def _status_dir(self, status: str) -> Path:
        if status not in self.statuses:
            raise RegistryError(f"unknown status {status!r}; allowed: {self.statuses}")
        return self.root / status

    def _entity_path(self, entity_id: str, status: str) -> Path:
        return self._status_dir(status) / f"{entity_id}.json"

    def _find_path(self, entity_id: str) -> Path:
        """Locate the file for an entity by searching across all status folders."""
        if not self.id_regex.match(entity_id):
            raise RegistryError(
                f"id {entity_id!r} does not match {self.id_prefix}<{self.id_width} digits>"
            )
        for status in self.statuses:
            candidate = self._entity_path(entity_id, status)
            if candidate.exists():
                return candidate
        raise EntityNotFoundError(f"{entity_id} not found in any status folder under {self.root}")

    def ensure_layout(self) -> None:
        """Create all status subfolders if missing. Idempotent."""
        self.root.mkdir(parents=True, exist_ok=True)
        for status in self.statuses:
            (self.root / status).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ reads

    def get(self, entity_id: str) -> Any:
        path = self._find_path(entity_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return self.from_dict(payload)

    def exists(self, entity_id: str) -> bool:
        try:
            self._find_path(entity_id)
            return True
        except (EntityNotFoundError, RegistryError):
            return False

    def status_of(self, entity_id: str) -> str:
        path = self._find_path(entity_id)
        return path.parent.name

    def list_by_status(self, status: str) -> list[str]:
        d = self._status_dir(status)
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob(f"{self.id_prefix}*.json"))

    def list_active(self) -> list[str]:
        out: list[str] = []
        for status in self.active_statuses:
            out.extend(self.list_by_status(status))
        return sorted(out)

    def list_archive(self, status: Optional[str] = None) -> list[str]:
        if status is not None:
            if status not in self.archive_statuses:
                raise RegistryError(f"{status!r} is not an archive status")
            return self.list_by_status(status)
        out: list[str] = []
        for s in self.archive_statuses:
            out.extend(self.list_by_status(s))
        return sorted(out)

    def list_all(self) -> list[str]:
        out: list[str] = []
        for status in self.statuses:
            out.extend(self.list_by_status(status))
        return sorted(out)

    def load_active_entries(self) -> list[Any]:
        """Load full entities from active folders only. Recalibration entrypoint."""
        return [self.get(eid) for eid in self.list_active()]

    # ----------------------------------------------------------------- writes

    def _validate_against_path(self, entity: Any, path: Path) -> None:
        """Check schema + folder/status agreement before persisting."""
        if not self.id_regex.match(entity.id):
            raise ValidationError(entity.id, f"id does not match {self.id_prefix}<{self.id_width} digits>")
        if path.stem != entity.id:
            raise ValidationError(entity.id, f"filename {path.stem!r} != entity.id {entity.id!r}")
        folder_status = path.parent.name
        if folder_status not in self.statuses:
            raise ValidationError(entity.id, f"folder {folder_status!r} not a known status")
        if entity.status != folder_status:
            raise ValidationError(
                entity.id,
                f"entity.status={entity.status!r} does not match folder={folder_status!r}",
            )
        try:
            entity.validate()
        except Exception as e:
            raise ValidationError(entity.id, str(e)) from e

    def _save(self, entity: Any, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(entity.to_dict(), indent=2, ensure_ascii=False, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    def add(self, entity: Any, status: Optional[str] = None, skip_index: bool = False) -> None:
        """Write a new entity. Default status is `default_new_status`."""
        if status is None:
            status = entity.status or self.default_new_status
        if entity.status != status:
            # the caller passed an explicit status; mirror it onto the entity
            entity.status = status
        path = self._entity_path(entity.id, status)
        if path.exists():
            raise RegistryError(f"{entity.id} already exists at {path}")
        self._validate_against_path(entity, path)
        self._save(entity, path)
        if not skip_index:
            self.rebuild_index()

    def change_status(self, entity_id: str, new_status: str, reason: str) -> None:
        """Move the entity's file to a new status folder + append history."""
        if new_status not in self.statuses:
            raise RegistryError(f"unknown status {new_status!r}")
        old_path = self._find_path(entity_id)
        old_status = old_path.parent.name
        if old_status == new_status:
            return  # idempotent
        entity = self.get(entity_id)
        entity.status = new_status
        entity.history.append(
            HistoryEvent(
                date=today_iso(),
                event="status_change",
                reason=reason,
                extra={"from": old_status, "to": new_status},
            ).to_dict()
        )
        # Update updated_at if the entity carries the field
        if hasattr(entity, "updated_at"):
            entity.updated_at = today_iso()
        new_path = self._entity_path(entity_id, new_status)
        self._validate_against_path(entity, new_path)
        self._save(entity, new_path)
        old_path.unlink()
        self.rebuild_index()

    def update_field(self, entity_id: str, field: str, value: Any, reason: str) -> None:
        """Update a top-level field on the entity + append history."""
        path = self._find_path(entity_id)
        entity = self.get(entity_id)
        if not hasattr(entity, field):
            raise RegistryError(f"{entity_id} has no field {field!r}")
        old_value = getattr(entity, field)
        setattr(entity, field, value)
        entity.history.append(
            HistoryEvent(
                date=today_iso(),
                event="field_update",
                reason=reason,
                extra={"field": field, "from": old_value, "to": value},
            ).to_dict()
        )
        if hasattr(entity, "updated_at"):
            entity.updated_at = today_iso()
        self._validate_against_path(entity, path)
        self._save(entity, path)
        self.rebuild_index()

    def append_history(self, entity_id: str, event: str, reason: str, extra: Optional[dict] = None) -> None:
        """Append a history event without changing other fields. Used by hooks."""
        path = self._find_path(entity_id)
        entity = self.get(entity_id)
        entity.history.append(
            HistoryEvent(date=today_iso(), event=event, reason=reason, extra=extra).to_dict()
        )
        if hasattr(entity, "updated_at"):
            entity.updated_at = today_iso()
        self._validate_against_path(entity, path)
        self._save(entity, path)

    # ------------------------------------------------------------------ index

    def _index_row(self, entity: Any) -> dict:
        row = {
            "id": entity.id,
            "status": entity.status,
            "title": getattr(entity, "title", ""),
            "updated_at": getattr(entity, "updated_at", ""),
        }
        for opt in ("effort", "value", "tags"):
            if hasattr(entity, opt):
                row[opt] = getattr(entity, opt)
        if hasattr(entity, "links") and isinstance(entity.links, dict):
            row["blocks_count"] = len(entity.links.get("blocks", []) or [])
            row["blocked_by_count"] = len(entity.links.get("blocked_by", []) or [])
        return row

    def rebuild_index(self, extra: Optional[dict] = None) -> None:
        """Regenerate INDEX.json from per-file truth.

        `extra` lets the specialization inject global state (priorities,
        falsification clocks, recalibration history) atop the entry rows.
        """
        entries: list[dict] = []
        for entity_id in self.list_all():
            entity = self.get(entity_id)
            entries.append(self._index_row(entity))
        existing: dict[str, Any] = {}
        if self.index_path.exists():
            try:
                existing = json.loads(self.index_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
        out = {
            "version": existing.get("version", 1),
            "schema_version": existing.get("schema_version", "1.0"),
            "updated_at": utc_now_iso(),
            "entries": entries,
        }
        # Preserve global keys the specialization owns
        for k, v in existing.items():
            if k not in ("version", "schema_version", "updated_at", "entries"):
                out[k] = v
        if extra:
            out.update(extra)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(out, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def load_index(self) -> dict:
        if not self.index_path.exists():
            return {}
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ links

    def list_blocking(self, entity_id: str, transitive: bool = False) -> list[str]:
        """IDs of entities that THIS one blocks. Follows `links.blocks`."""
        return self._traverse_link(entity_id, "blocks", transitive)

    def list_blocked_by(self, entity_id: str, transitive: bool = False) -> list[str]:
        return self._traverse_link(entity_id, "blocked_by", transitive)

    def _traverse_link(self, entity_id: str, relation: str, transitive: bool) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        frontier = [entity_id]
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            try:
                entity = self.get(current)
            except EntityNotFoundError:
                continue
            links = getattr(entity, "links", {}) or {}
            targets = links.get(relation, []) or []
            for t in targets:
                if t not in seen:
                    out.append(t)
                    if transitive:
                        frontier.append(t)
        return out

    # ----------------------------------------------------------- validation

    def validate_all(self) -> list[ValidationError]:
        """Walk every entity + INDEX; return list of violations (empty = clean)."""
        errors: list[ValidationError] = []
        ids_seen: dict[str, Path] = {}
        for status in self.statuses:
            d = self._status_dir(status)
            if not d.exists():
                continue
            for path in sorted(d.glob("*.json")):
                # JSON parse
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as e:
                    errors.append(ValidationError(path.stem, f"invalid JSON: {e}"))
                    continue
                # filename format
                if not self.id_regex.match(path.stem):
                    errors.append(ValidationError(path.stem, f"filename does not match {self.id_regex.pattern}"))
                    continue
                # id field
                pid = payload.get("id")
                if pid != path.stem:
                    errors.append(ValidationError(path.stem, f"id field={pid!r} != filename"))
                # uniqueness
                if pid in ids_seen:
                    errors.append(
                        ValidationError(pid, f"duplicate id: also at {ids_seen[pid]}")
                    )
                else:
                    ids_seen[pid] = path
                # schema
                try:
                    entity = self.from_dict(payload)
                    self._validate_against_path(entity, path)
                except ValidationError as e:
                    errors.append(e)
                except Exception as e:
                    errors.append(ValidationError(pid or path.stem, str(e)))
        # link targets exist + symmetric backlinks
        errors.extend(self._validate_links(ids_seen.keys()))
        # cycle detection on `blocks`
        errors.extend(self._validate_no_cycles(ids_seen.keys()))
        # INDEX consistency
        errors.extend(self._validate_index(ids_seen))
        return errors

    def _validate_links(self, ids_present: set[str] | Any) -> list[ValidationError]:
        ids_present = set(ids_present)
        errors: list[ValidationError] = []
        symmetric_pairs = {"blocks": "blocked_by", "supersedes": "superseded_by", "spawned": "spawned_by"}
        for entity_id in sorted(ids_present):
            try:
                entity = self.get(entity_id)
            except Exception:
                continue
            links = getattr(entity, "links", {}) or {}
            for relation, targets in links.items():
                if not isinstance(targets, list):
                    continue
                for target in targets:
                    if not isinstance(target, str):
                        errors.append(ValidationError(entity_id, f"links.{relation} contains non-string {target!r}"))
                        continue
                    # only validate same-prefix B-IDs / I-IDs against THIS registry's id space
                    if target.startswith(self.id_prefix) and target not in ids_present:
                        errors.append(ValidationError(entity_id, f"links.{relation} points to missing {target}"))
                # symmetry
                if relation in symmetric_pairs:
                    back = symmetric_pairs[relation]
                    for target in targets:
                        if not target.startswith(self.id_prefix) or target not in ids_present:
                            continue
                        target_entity = self.get(target)
                        target_links = getattr(target_entity, "links", {}) or {}
                        if entity_id not in (target_links.get(back, []) or []):
                            errors.append(
                                ValidationError(
                                    entity_id,
                                    f"links.{relation}=[{target}] but {target}.links.{back} missing back-ref",
                                )
                            )
        return errors

    def _validate_no_cycles(self, ids_present: set[str] | Any) -> list[ValidationError]:
        ids_present = set(ids_present)
        errors: list[ValidationError] = []
        for entity_id in sorted(ids_present):
            visited: set[str] = set()
            stack = [(entity_id, [entity_id])]
            while stack:
                current, path = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                try:
                    entity = self.get(current)
                except Exception:
                    continue
                links = getattr(entity, "links", {}) or {}
                for target in links.get("blocks", []) or []:
                    if target == entity_id and len(path) > 1:
                        errors.append(
                            ValidationError(entity_id, f"cycle via blocks: {' -> '.join(path + [target])}")
                        )
                    elif target in ids_present and target not in path:
                        stack.append((target, path + [target]))
        return errors

    def _validate_index(self, ids_seen: dict[str, Path]) -> list[ValidationError]:
        errors: list[ValidationError] = []
        if not self.index_path.exists():
            errors.append(ValidationError("INDEX", "INDEX.json missing"))
            return errors
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errors.append(ValidationError("INDEX", f"invalid JSON: {e}"))
            return errors
        index_ids = {row["id"]: row for row in payload.get("entries", []) if "id" in row}
        for eid in index_ids.keys() - set(ids_seen.keys()):
            errors.append(ValidationError("INDEX", f"row for {eid} but no file on disk"))
        for eid in set(ids_seen.keys()) - index_ids.keys():
            errors.append(ValidationError("INDEX", f"file {eid} on disk but no row"))
        # field staleness
        for eid, row in index_ids.items():
            if eid not in ids_seen:
                continue
            disk_status = ids_seen[eid].parent.name
            if row.get("status") != disk_status:
                errors.append(
                    ValidationError("INDEX", f"{eid} row status={row.get('status')} but folder={disk_status}")
                )
        return errors
