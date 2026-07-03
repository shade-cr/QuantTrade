"""Loader/validator for the frozen M3 equity universe (B0003).

The YAML is the single frozen artifact of the OQ1 universe decision; the
fetch script, asset registry, and pooled runner all consume it through
`load_universe` so no ticker list is ever duplicated in code.
"""
from __future__ import annotations

from pathlib import Path

import yaml

_REQUIRED_KEYS = ("selection_rule", "selected_at", "stocks", "etfs",
                  "alternates", "excluded_delistees")


def load_universe(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"universe file {path}: top level must be a mapping")
    missing = [k for k in _REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"universe file {path}: missing keys {missing}")
    for key in ("stocks", "etfs", "alternates"):
        vals = payload[key]
        if not isinstance(vals, list) or not all(isinstance(t, str) and t for t in vals):
            raise ValueError(f"universe file {path}: {key} must be a list of non-empty strings")
        if len(vals) != len(set(vals)):
            raise ValueError(f"universe file {path}: duplicate tickers in {key}")
    overlap = set(payload["stocks"]) & set(payload["etfs"])
    if overlap:
        raise ValueError(f"universe file {path}: stocks/etfs overlap: {sorted(overlap)}")
    if not isinstance(payload["excluded_delistees"], dict):
        raise ValueError(f"universe file {path}: excluded_delistees must be a mapping")
    if not str(payload["selection_rule"]).strip():
        raise ValueError(f"universe file {path}: selection_rule must be non-empty")
    return payload
