"""Tests for the deployment-config aggregator (T9.B.3).

The aggregator reads `results/clf_multi_h4/{asset}/{primary}/` outputs and
composes them into a single deployment_config.yaml via:
  - T13 best_model selector (DSR-aware with median-Sharpe fallback)
  - T12 asset_deployment_tier (DSR bands + hard gates)

Tests use synthetic summary.json/psr_dsr.json fixtures written into a tmp_path.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_multi_h4 import (  # noqa: E402
    _build_deployment_entry,
    _max_dd_for_model,
    aggregate_to_deployment_config,
)


def _write_summary(path: Path, summary: dict, psr_dsr: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (path / "psr_dsr.json").write_text(json.dumps(psr_dsr, indent=2), encoding="utf-8")


def _cfg() -> dict:
    """Minimum cfg required by aggregator."""
    return {
        "primary": {
            "ema_cross": {"fast": 60, "slow": 240, "dead_zone_atr": 0.25},
            "momentum_zscore": {"lookback": 24, "threshold": 0.3},
        },
        "best_model": {
            "min_trades_per_fold": 30,
            "min_folds_with_trades": 2,
        },
    }


def _summary_template(asset: str, primary: str, models: list[str]) -> dict:
    """Build a summary.json-shape dict with default per-fold lists."""
    return {
        "primary": primary,
        "asset": asset,
        "asset_class": "fx",
        "n_events": 1000,
        "n_folds": 4,
        "baseline_sharpe_per_fold": [-0.2, -0.1, 0.0, 0.1],
        "sharpe_per_fold_per_model": {m: [0.5, 0.6, 0.4, 0.5] for m in models},
        "n_trades_per_fold_per_model": {m: [120, 130, 100, 110] for m in models},
        "max_dd_per_fold_per_model": {m: [-0.05, -0.04, -0.08, -0.03] for m in models},
        "selected_threshold_per_fold_per_model": {m: [0.50, 0.52, 0.50, 0.54] for m in models},
        "median_selected_threshold_per_model": {m: 0.51 for m in models},
        "median_sharpe": {m: 0.5 for m in models},
        "best_model": models[0],
        "stack_decision": {"stack": False, "reason": "NO STACK test", "n_models_passing": 0, "max_pair_corr": 0.5},
    }


def _psr_dsr_template(models: list[str], psr_vals: dict, dsr_vals: dict) -> dict:
    return {
        "psr": {m: psr_vals.get(m, 0.5) for m in models},
        "dsr": {m: dsr_vals.get(m, 0.1) for m in models},
        "n_trials_global": len(models) * 4,
        "per_model_aggregate": {
            m: {"n_trades": 460, "sr_per_trade": 0.3, "skew": 0.0, "kurt": 3.0}
            for m in models
        },
        "trial_pool_sr_per_trade": [0.1, 0.2, 0.3] * len(models),
    }


# ---------------------------------------------------------------------------
# _max_dd_for_model
# ---------------------------------------------------------------------------

def test_max_dd_returns_absolute_worst_drawdown_for_named_model():
    summary = {"max_dd_per_fold_per_model": {"xgb": [-0.05, -0.20, -0.10, -0.08]}}
    assert _max_dd_for_model(summary, "xgb") == pytest.approx(0.20)


def test_max_dd_returns_zero_when_no_per_fold_data():
    summary = {"max_dd_per_fold_per_model": {}}
    assert _max_dd_for_model(summary, "xgb") == 0.0


# ---------------------------------------------------------------------------
# _build_deployment_entry
# ---------------------------------------------------------------------------

def test_build_entry_assigns_disabled_tier_for_low_dsr():
    """DSR=0.001 (Phase 1 v4 catboost on XAU D1) → tier=disabled."""
    summary = _summary_template("XAUUSD", "ema_cross", ["xgb", "catboost", "rf"])
    # Override n_trades so the best_model selector picks catboost via fallback.
    summary["n_trades_per_fold_per_model"] = {
        "xgb":      [48, 0, 23, 0],
        "catboost": [92, 0,  7, 1],
        "rf":       [25, 0, 21, 0],
    }
    summary["sharpe_per_fold_per_model"] = {
        "xgb":      [-0.23, float("nan"), float("nan"), float("nan")],
        "catboost": [1.07,  float("nan"), float("nan"), float("nan")],
        "rf":       [float("nan"), float("nan"), float("nan"), float("nan")],
    }
    summary["max_dd_per_fold_per_model"] = {
        "xgb": [-0.05] * 4, "catboost": [-0.18] * 4, "rf": [-0.10] * 4,
    }
    psr_dsr = _psr_dsr_template(
        ["xgb", "catboost", "rf"],
        psr_vals={"xgb": 0.952, "catboost": 0.981, "rf": 0.995},
        dsr_vals={"xgb": 0.004, "catboost": 0.001, "rf": 0.257},
    )

    entry = _build_deployment_entry("XAUUSD", "metal", summary, psr_dsr, _cfg())

    # With default min_folds_with_trades=2, no model qualifies → median Sharpe fallback → catboost.
    assert entry["best_model"] == "catboost"
    assert entry["selection_reason"] == "median_sharpe_fallback"
    # catboost has DSR=0.001 < 0.05 → disabled tier.
    assert entry["deployment_tier"] == "disabled"
    assert entry["kelly_fraction"] == 0.0
    assert entry["enabled"] is False


def test_build_entry_assigns_quarter_tier_with_h4_dsr():
    """Simulated H4 scenario: rf has DSR=0.30 → tier=quarter, kelly=0.25."""
    models = ["xgb", "catboost", "rf"]
    summary = _summary_template("EURUSD", "ema_cross", models)
    summary["n_trades_per_fold_per_model"] = {
        m: [620, 580, 560, 540] for m in models
    }
    psr_dsr = _psr_dsr_template(
        models,
        psr_vals={"xgb": 0.98, "catboost": 0.97, "rf": 0.96},
        dsr_vals={"xgb": 0.10, "catboost": 0.15, "rf": 0.30},
    )

    entry = _build_deployment_entry("EURUSD", "fx", summary, psr_dsr, _cfg())

    assert entry["best_model"] == "rf"
    assert entry["selection_reason"] == "dsr_aware"
    assert entry["deployment_tier"] == "quarter"
    assert entry["kelly_fraction"] == 0.25
    assert entry["enabled"] is True


def test_build_entry_carries_primary_params_through():
    summary = _summary_template("BTCUSD", "ema_cross", ["xgb"])
    psr_dsr = _psr_dsr_template(["xgb"], {"xgb": 0.99}, {"xgb": 0.60})
    entry = _build_deployment_entry("BTCUSD", "crypto", summary, psr_dsr, _cfg())
    assert entry["primary"] == "ema_cross"
    assert entry["primary_params"] == {"fast": 60, "slow": 240, "dead_zone_atr": 0.25}


# ---------------------------------------------------------------------------
# aggregate_to_deployment_config: full flow
# ---------------------------------------------------------------------------

def test_aggregator_writes_yaml_with_correct_structure(tmp_path):
    output_dir = tmp_path
    models = ["xgb", "catboost", "rf"]

    # 2 assets, each with one primary's summary.
    for asset, dsr in [("EURUSD", 0.30), ("BTCUSD", 0.60)]:
        s = _summary_template(asset, "ema_cross", models)
        p = _psr_dsr_template(
            models,
            psr_vals={m: 0.95 for m in models},
            dsr_vals={"xgb": 0.10, "catboost": 0.15, "rf": dsr},
        )
        _write_summary(output_dir / asset / "ema_cross", s, p)

    plan = [
        {"asset": "EURUSD", "primary": "ema_cross", "asset_class": "fx"},
        {"asset": "BTCUSD", "primary": "ema_cross", "asset_class": "crypto"},
    ]

    deployment = aggregate_to_deployment_config(output_dir, plan, _cfg())

    # Top-level keys
    assert {"last_modified", "trial_pool_size", "assets", "skipped"} <= set(deployment.keys())
    # Two assets present
    assert set(deployment["assets"].keys()) == {"EURUSD", "BTCUSD"}
    # YAML file written
    yaml_path = output_dir / "deployment_config.yaml"
    assert yaml_path.exists()
    re_read = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert re_read["assets"]["EURUSD"]["deployment_tier"] == "quarter"
    assert re_read["assets"]["BTCUSD"]["deployment_tier"] == "half"


def test_aggregator_handles_missing_summary_gracefully(tmp_path):
    """Plan references an asset whose summary.json doesn't exist (training
    error / dry-run only). The aggregator must skip it and not crash."""
    output_dir = tmp_path
    plan = [{"asset": "SOLUSD", "primary": "ema_cross", "asset_class": "crypto"}]
    deployment = aggregate_to_deployment_config(output_dir, plan, _cfg())
    assert deployment["assets"] == {}
    assert "SOLUSD" in deployment["skipped"][0]


def test_aggregator_skips_assets_with_skip_reason_in_summary(tmp_path):
    """When a training pair recorded `skip_reason` (e.g. 0 events), the
    aggregator must NOT try to build a deployment entry from incomplete data."""
    output_dir = tmp_path
    pair_dir = output_dir / "SOLUSD" / "ema_cross"
    pair_dir.mkdir(parents=True)
    (pair_dir / "summary.json").write_text(json.dumps({
        "primary": "ema_cross", "asset": "SOLUSD",
        "n_events": 0, "skip_reason": "no primary signals",
    }), encoding="utf-8")

    plan = [{"asset": "SOLUSD", "primary": "ema_cross", "asset_class": "crypto"}]
    deployment = aggregate_to_deployment_config(output_dir, plan, _cfg())
    assert deployment["assets"] == {}
    assert any("SOLUSD" in s for s in deployment["skipped"])


def test_aggregator_picks_higher_tier_when_asset_has_multiple_primaries(tmp_path):
    """Rare case: an asset has BOTH primaries trained (force_both_primaries).
    The primary with the better tier wins for that asset."""
    output_dir = tmp_path
    models = ["xgb", "catboost", "rf"]

    # ema_cross → tier=disabled (low DSR for all)
    s1 = _summary_template("XAUUSD", "ema_cross", models)
    p1 = _psr_dsr_template(
        models,
        psr_vals={m: 0.95 for m in models},
        dsr_vals={m: 0.01 for m in models},
    )
    _write_summary(output_dir / "XAUUSD" / "ema_cross", s1, p1)

    # momentum_zscore → tier=quarter (rf has DSR=0.30)
    s2 = _summary_template("XAUUSD", "momentum_zscore", models)
    p2 = _psr_dsr_template(
        models,
        psr_vals={m: 0.95 for m in models},
        dsr_vals={"xgb": 0.10, "catboost": 0.15, "rf": 0.30},
    )
    _write_summary(output_dir / "XAUUSD" / "momentum_zscore", s2, p2)

    plan = [
        {"asset": "XAUUSD", "primary": "ema_cross", "asset_class": "metal"},
        {"asset": "XAUUSD", "primary": "momentum_zscore", "asset_class": "metal"},
    ]
    deployment = aggregate_to_deployment_config(output_dir, plan, _cfg())

    # The momentum_zscore variant wins because tier=quarter > tier=disabled.
    assert deployment["assets"]["XAUUSD"]["primary"] == "momentum_zscore"
    assert deployment["assets"]["XAUUSD"]["deployment_tier"] == "quarter"
