"""Per-asset friction config loader.

Resolves transaction costs (round-trip bid-ask spread + slippage) per asset,
with backward-compatible fallback to the legacy global
`cfg['metrics']['cost_per_trade_bps']` so Phase 1 v4 bit-exact regression
is preserved.

Usage in orchestrators:

    from pipeline.friction import resolve_cost_bps
    cost_bps = resolve_cost_bps(asset_name, cfg)

When `cfg` contains `friction.config_path`, the YAML is loaded and the asset's
entry from `per_asset_bps` is returned (falling back to the friction YAML's
own `default_bps` if the asset is unlisted).

When no friction block is present, the legacy global `metrics.cost_per_trade_bps`
is returned unchanged. This is the path Phase 1 v4 and the Phase 2 H4 runs
took, and is the default behavior for backward compat.

Why per-asset friction matters: a single global 10 bps over-penalizes FX
(real cost ~1.5 bps) and under-penalizes thin-liquidity crypto like SOL (~10+
bps). The wrong cost mis-selects thresholds in inner-CV and mis-scores OOF
metrics. Both effects compound — a primary signal that looks dead at 10 bps
may have edge at 1.5 bps for EUR/USD.
"""
from __future__ import annotations
from pathlib import Path

import yaml


class FrictionError(ValueError):
    """Raised when the friction config is missing or malformed."""


def load_friction_config(path: str | Path) -> dict:
    """Load and validate a friction YAML.

    Schema:
        per_asset_bps: {ASSET: float}  # round-trip bps per asset
        default_bps:   float           # fallback for unlisted assets

    Both keys are required; values must have the right shapes.
    """
    path = Path(path)
    if not path.exists():
        raise FrictionError(f"friction config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "per_asset_bps" not in raw:
        raise FrictionError(
            f"friction config {path} missing required key 'per_asset_bps'"
        )
    if not isinstance(raw["per_asset_bps"], dict):
        raise FrictionError(
            f"friction config {path}: 'per_asset_bps' must be a mapping "
            f"of {{asset: bps}}, got {type(raw['per_asset_bps']).__name__}"
        )
    if "default_bps" not in raw:
        raise FrictionError(
            f"friction config {path} missing required key 'default_bps' "
            f"(fallback for unlisted assets)"
        )
    return raw


def resolve_cost_bps(asset: str, cfg: dict) -> float:
    """Resolve cost_bps for `asset` from a pipeline config dict.

    Resolution order:
      1. If `cfg['friction']['config_path']` is set → load friction YAML.
         Use `per_asset_bps[asset]` if listed; else `default_bps` from
         the same YAML.
      2. Otherwise → fall back to `cfg['metrics']['cost_per_trade_bps']`
         (legacy behavior preserved for Phase 1 v4 regression).
    """
    fr = cfg.get("friction") or {}
    config_path = fr.get("config_path")
    if config_path:
        loaded = load_friction_config(config_path)
        return float(
            loaded["per_asset_bps"].get(asset, loaded["default_bps"])
        )
    return float(cfg["metrics"]["cost_per_trade_bps"])
