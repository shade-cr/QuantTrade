"""Tests for the planning helpers of the multi-asset orchestrator (T9.B.1).

Training itself is a heavy integration step — covered manually by running
`scripts/run_multi_h4.py`. The planning phase (config loading, asset
overrides, pre-screening, plan composition) is unit-testable and gated
here so a config edit can't silently break the plan structure.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure repo root is on sys.path so scripts.* imports work in tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.run_multi_h4 import (  # noqa: E402
    ASSET_CLASS_BY_NAME,
    apply_asset_overrides,
    build_training_plan,
    run_pre_screening,
)


def _trending_close(n: int = 3500) -> pd.DataFrame:
    """Synthetic upward-drift price series. Hurst > 0.52 → trending regime."""
    rng = np.random.default_rng(0)
    inc = rng.standard_normal(n) * 0.01 + 0.005
    close = 100.0 + np.cumsum(inc)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"close": close}, index=idx)


# ---------------------------------------------------------------------------
# apply_asset_overrides
# ---------------------------------------------------------------------------

def test_apply_overrides_replaces_nested_values_for_listed_asset():
    cfg = {
        "walk_forward": {"train_min_bars": 3000, "n_folds": 4},
        "asset_overrides": {
            "SOLUSD": {"walk_forward": {"train_min_bars": 1200, "n_folds": 3}},
        },
    }
    out = apply_asset_overrides(cfg, "SOLUSD")
    assert out["walk_forward"]["train_min_bars"] == 1200
    assert out["walk_forward"]["n_folds"] == 3
    # Original must be untouched (deep-copied internally).
    assert cfg["walk_forward"]["train_min_bars"] == 3000


def test_apply_overrides_returns_input_when_asset_not_overridden():
    cfg = {
        "walk_forward": {"train_min_bars": 3000, "n_folds": 4},
        "asset_overrides": {"SOLUSD": {"walk_forward": {"train_min_bars": 1200}}},
    }
    out = apply_asset_overrides(cfg, "EURUSD")
    assert out["walk_forward"]["train_min_bars"] == 3000


def test_apply_overrides_with_no_overrides_section():
    cfg = {"walk_forward": {"train_min_bars": 3000}}
    out = apply_asset_overrides(cfg, "SOLUSD")
    assert out["walk_forward"]["train_min_bars"] == 3000


# ---------------------------------------------------------------------------
# run_pre_screening
# ---------------------------------------------------------------------------

def test_run_pre_screening_invokes_hurst_per_asset():
    asset_dfs = {"EURUSD": _trending_close(), "GBPUSD": _trending_close()}
    cfg = {
        "primary_screening": {
            "enabled": True,
            "hurst_threshold_trending": 0.52,
            "hurst_threshold_mr": 0.48,
        },
        "primary": {"candidates": ["ema_cross", "momentum_zscore"]},
        "walk_forward": {"train_min_bars": 3000},
    }
    screening = run_pre_screening(asset_dfs, cfg)
    assert set(screening.keys()) == {"EURUSD", "GBPUSD"}
    for asset, info in screening.items():
        assert "screened" in info
        assert "regime" in info
        assert info["regime"] in ("trending", "mean_reverting", "mixed", "insufficient_data")


def test_run_pre_screening_disabled_returns_all_candidates():
    asset_dfs = {"EURUSD": _trending_close()}
    cfg = {
        "primary_screening": {"enabled": False},
        "primary": {"candidates": ["ema_cross", "momentum_zscore"]},
        "walk_forward": {"train_min_bars": 3000},
    }
    screening = run_pre_screening(asset_dfs, cfg)
    assert screening["EURUSD"]["regime"] == "screening_disabled"
    assert set(screening["EURUSD"]["screened"]) == {"ema_cross", "momentum_zscore"}


def test_run_pre_screening_force_all_overrides_hurst():
    """force_all_primaries=true bypasses Hurst even if regime is clearly trending."""
    asset_dfs = {"EURUSD": _trending_close()}
    cfg = {
        "primary_screening": {"enabled": True, "force_all_primaries": True},
        "primary": {"candidates": ["ema_cross", "momentum_zscore"]},
        "walk_forward": {"train_min_bars": 3000},
    }
    screening = run_pre_screening(asset_dfs, cfg)
    assert screening["EURUSD"]["regime"] == "force_all_primaries"
    assert set(screening["EURUSD"]["screened"]) == {"ema_cross", "momentum_zscore"}


def test_run_pre_screening_applies_per_asset_train_min_bars():
    """SOLUSD with train_min_bars=1200 must screen on the first 1200 bars."""
    asset_dfs = {"SOLUSD": _trending_close(n=2000)}  # only 2000 bars total
    cfg = {
        "primary_screening": {"enabled": True},
        "primary": {"candidates": ["ema_cross", "momentum_zscore"]},
        "walk_forward": {"train_min_bars": 3000},  # default
        "asset_overrides": {
            "SOLUSD": {"walk_forward": {"train_min_bars": 1200}},
        },
    }
    screening = run_pre_screening(asset_dfs, cfg)
    # With train_min_bars=1200 and 2000 available, screening should pick a
    # regime (not "insufficient_data" — that would mean we used the full
    # 2000 or the default 3000 without honoring the override).
    assert screening["SOLUSD"]["regime"] in ("trending", "mean_reverting", "mixed")
    assert screening["SOLUSD"]["n_bars_used"] == 1200


# ---------------------------------------------------------------------------
# build_training_plan
# ---------------------------------------------------------------------------

def test_build_training_plan_emits_one_entry_per_screened_pair():
    screening = {
        "EURUSD": {"screened": ["ema_cross"], "regime": "trending", "hurst": 0.6},
        "BTCUSD": {"screened": ["ema_cross", "momentum_zscore"], "regime": "mixed", "hurst": 0.5},
    }
    cfg = {
        "walk_forward": {"train_min_bars": 3000, "n_folds": 4},
        "bars_per_year_by_class": {"fx": 1560, "metal": 1560, "crypto": 2190},
    }
    plan = build_training_plan(screening, cfg)
    assert len(plan) == 3   # EUR: 1 + BTC: 2
    eur_entries = [p for p in plan if p["asset"] == "EURUSD"]
    btc_entries = [p for p in plan if p["asset"] == "BTCUSD"]
    assert len(eur_entries) == 1 and eur_entries[0]["primary"] == "ema_cross"
    assert len(btc_entries) == 2
    assert {e["primary"] for e in btc_entries} == {"ema_cross", "momentum_zscore"}


def test_build_training_plan_applies_per_class_bars_per_year():
    screening = {
        "EURUSD": {"screened": ["ema_cross"], "regime": "trending", "hurst": 0.6},
        "BTCUSD": {"screened": ["ema_cross"], "regime": "trending", "hurst": 0.65},
        "XAUUSD": {"screened": ["ema_cross"], "regime": "trending", "hurst": 0.55},
    }
    cfg = {
        "walk_forward": {"train_min_bars": 3000, "n_folds": 4},
        "bars_per_year_by_class": {"fx": 1560, "metal": 1560, "crypto": 2190},
    }
    plan = build_training_plan(screening, cfg)
    by_asset = {p["asset"]: p for p in plan}
    assert by_asset["EURUSD"]["bars_per_year"] == 1560
    assert by_asset["BTCUSD"]["bars_per_year"] == 2190
    assert by_asset["XAUUSD"]["bars_per_year"] == 1560
    # asset_class also propagated
    assert by_asset["EURUSD"]["asset_class"] == "fx"
    assert by_asset["BTCUSD"]["asset_class"] == "crypto"
    assert by_asset["XAUUSD"]["asset_class"] == "metal"


def test_build_training_plan_applies_asset_specific_overrides():
    """SOLUSD's plan entry must reflect its asset_overrides (train_min_bars=1200,
    n_folds=3) not the defaults."""
    screening = {
        "SOLUSD": {"screened": ["ema_cross"], "regime": "trending", "hurst": 0.65},
    }
    cfg = {
        "walk_forward": {"train_min_bars": 3000, "n_folds": 4},
        "bars_per_year_by_class": {"crypto": 2190, "fx": 1560, "metal": 1560},
        "asset_overrides": {
            "SOLUSD": {"walk_forward": {"train_min_bars": 1200, "n_folds": 3}},
        },
    }
    plan = build_training_plan(screening, cfg)
    assert plan[0]["train_min_bars"] == 1200
    assert plan[0]["n_folds"] == 3


# ---------------------------------------------------------------------------
# Asset class lookup
# ---------------------------------------------------------------------------

def test_asset_class_lookup_covers_all_8_hackathon_assets():
    """No silent fallbacks for the hackathon set."""
    expected = {
        "EURUSD": "fx", "GBPUSD": "fx", "USDJPY": "fx",
        "XAUUSD": "metal", "XAGUSD": "metal",
        "BTCUSD": "crypto", "ETHUSD": "crypto", "SOLUSD": "crypto",
    }
    for asset, cls in expected.items():
        assert ASSET_CLASS_BY_NAME[asset] == cls, f"{asset} → {cls} expected"
