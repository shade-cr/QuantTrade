"""Tests for pipeline.friction — per-asset cost_bps resolution.

Invariants:
  1. Backward compat: if cfg has no 'friction' block, resolve_cost_bps returns
     cfg['metrics']['cost_per_trade_bps'] (the legacy global). Phase 1 v4
     baseline depends on this — DO NOT break.
  2. When 'friction.config_path' is set, the resolver loads the YAML and looks
     up per_asset_bps[asset]. Asset case sensitive.
  3. Assets not listed in per_asset_bps fall back to the friction's own
     'default_bps' field (NOT the legacy metrics.cost_per_trade_bps).
  4. Malformed friction configs raise FrictionError with a clear message.
  5. resolve_cost_bps returns float (downstream PnL math expects float).
"""
from __future__ import annotations
import textwrap
from pathlib import Path
import pytest

from pipeline.friction import (
    FrictionError,
    load_friction_config,
    resolve_cost_bps,
)


def _write_yaml(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    return p


def test_resolve_cost_bps_falls_back_to_legacy_global():
    """No friction block → use cfg['metrics']['cost_per_trade_bps']."""
    cfg = {"metrics": {"cost_per_trade_bps": 10}}
    assert resolve_cost_bps("XAUUSD", cfg) == 10.0


def test_resolve_cost_bps_returns_float():
    """Even when source value is int, output must be float (PnL math)."""
    cfg = {"metrics": {"cost_per_trade_bps": 10}}
    out = resolve_cost_bps("XAUUSD", cfg)
    assert isinstance(out, float)


def test_resolve_cost_bps_uses_per_asset_when_friction_present(tmp_path):
    """friction.config_path → load YAML → per_asset_bps lookup wins over global."""
    fpath = _write_yaml(tmp_path, "friction.yaml", """
        per_asset_bps:
          EURUSD: 1.5
          BTCUSD: 8.0
          XAUUSD: 5.0
        default_bps: 10.0
    """)
    cfg = {
        "metrics": {"cost_per_trade_bps": 10},  # legacy, should be IGNORED
        "friction": {"config_path": str(fpath)},
    }
    assert resolve_cost_bps("EURUSD", cfg) == 1.5
    assert resolve_cost_bps("BTCUSD", cfg) == 8.0
    assert resolve_cost_bps("XAUUSD", cfg) == 5.0


def test_resolve_cost_bps_unknown_asset_falls_back_to_friction_default(tmp_path):
    """Asset not in per_asset_bps → use friction's default_bps, NOT legacy global."""
    fpath = _write_yaml(tmp_path, "friction.yaml", """
        per_asset_bps:
          EURUSD: 1.5
        default_bps: 7.5
    """)
    cfg = {
        "metrics": {"cost_per_trade_bps": 10},  # MUST be ignored — friction wins
        "friction": {"config_path": str(fpath)},
    }
    assert resolve_cost_bps("BTCUSD", cfg) == 7.5
    assert resolve_cost_bps("UNKNOWN", cfg) == 7.5


def test_load_friction_config_rejects_missing_file(tmp_path):
    with pytest.raises(FrictionError, match="not found"):
        load_friction_config(tmp_path / "nope.yaml")


def test_load_friction_config_rejects_missing_per_asset_key(tmp_path):
    fpath = _write_yaml(tmp_path, "bad.yaml", """
        default_bps: 10.0
    """)
    with pytest.raises(FrictionError, match="per_asset_bps"):
        load_friction_config(fpath)


def test_load_friction_config_rejects_missing_default_key(tmp_path):
    fpath = _write_yaml(tmp_path, "bad.yaml", """
        per_asset_bps:
          EURUSD: 1.5
    """)
    with pytest.raises(FrictionError, match="default_bps"):
        load_friction_config(fpath)


def test_load_friction_config_rejects_non_dict_per_asset(tmp_path):
    fpath = _write_yaml(tmp_path, "bad.yaml", """
        per_asset_bps:
          - EURUSD
          - BTCUSD
        default_bps: 10.0
    """)
    with pytest.raises(FrictionError, match="per_asset_bps"):
        load_friction_config(fpath)


def test_resolve_cost_bps_friction_block_without_config_path_uses_legacy():
    """Empty friction block (no config_path) → backward compat fallback."""
    cfg = {
        "metrics": {"cost_per_trade_bps": 10},
        "friction": {},  # empty block — shouldn't break
    }
    assert resolve_cost_bps("XAUUSD", cfg) == 10.0
