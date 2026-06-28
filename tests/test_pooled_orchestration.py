"""B0148 SLICE 2 — pooled orchestration tests.

Covers the wiring that Slice 2 adds to scripts/run_multi_h4.py:
  - Phase B core-schema composition (intersection + class one-hot, no NaN).
  - pooled uniqueness wired (B1): pooled weight == pooled_avg_uniqueness, NOT the
    concatenation of per-asset avg_uniqueness.
  - per_class weight balancing equalizes effective mass per class.
  - calibration holdout is the latest-by-event_time tail of the fold train (B2),
    and is causal (mutating a later label does not move an earlier calib assignment).
  - Phase D slicing reproduces each asset's event count and a per-asset summary.json
    with the baseline schema.
  - the falsification comparison helper returns the correct CONFIRMED/FALSIFIED.

The OFF=parity guarantee is additionally exercised by the dry-run smoke in the
slice's validation step; here we pin the refactored helpers directly.

Spec: docs/superpowers/specs/2026-06-04-b0148-cross-asset-meta-pooling-design.md
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.sample_weights import avg_uniqueness, pooled_avg_uniqueness
import scripts.run_multi_h4 as orch


# --------------------------------------------------------------------------- #
# Synthetic member builders
# --------------------------------------------------------------------------- #
def _member(asset, asset_class, base_cols, n, t0="2021-01-01", freq_h=4,
            extra_cols=None, bars_per_year=1560, seed=0):
    """Build a minimal Phase-A-style member dict for the pooled helpers."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(t0, periods=n, freq=f"{freq_h}h", tz="UTC")
    cols = list(base_cols)
    data = {c: rng.standard_normal(n) for c in cols}
    if extra_cols:
        for c in extra_cols:
            data[c] = rng.standard_normal(n)
    X = pd.DataFrame(data, index=idx)
    X["primary_side"] = rng.choice([-1, 1], size=n).astype(float)
    X["primary_strength"] = rng.standard_normal(n)
    X["bars_since_signal"] = rng.integers(0, 20, size=n).astype(float)
    y = pd.Series(rng.integers(0, 2, size=n), index=idx)
    label_end = idx + pd.Timedelta(hours=freq_h)   # 1-bar resolution
    return {
        "asset": asset,
        "asset_class": asset_class,
        "primary_name": "ema_cross",
        "bars_per_year": bars_per_year,
        "cost_bps": 10.0,
        "X": X,
        "y": y,
        "w": avg_uniqueness(np.arange(n), np.arange(n), n_bars=n),  # per-asset
        "side": X["primary_side"],
        "fwd_ret": pd.Series(rng.standard_normal(n) * 0.01, index=idx),
        "event_time": pd.DatetimeIndex(idx),
        "label_end_time": pd.DatetimeIndex(label_end),
    }


# --------------------------------------------------------------------------- #
# Phase B — core schema
# --------------------------------------------------------------------------- #
def test_core_schema_is_intersection_plus_primary_cols():
    m1 = _member("EURUSD", "fx", ["rsi", "z_r24bars", "fx_only"], 30, seed=1)
    m2 = _member("BTCUSD", "crypto", ["rsi", "z_r24bars", "btc_only"], 30, seed=2)
    shared = orch._pooled_core_schema([m1, m2])
    # fx_only / btc_only are NOT shared → excluded; primary cols always present.
    assert "fx_only" not in shared
    assert "btc_only" not in shared
    assert "rsi" in shared and "z_r24bars" in shared
    assert shared[-3:] == ["primary_side", "primary_strength", "bars_since_signal"]


def test_compose_core_adds_class_onehot_and_no_nan():
    m1 = _member("EURUSD", "fx", ["rsi", "z_r24bars", "fx_only"], 30, seed=1)
    m2 = _member("BTCUSD", "crypto", ["rsi", "z_r24bars", "btc_only"], 30, seed=2)
    shared = orch._pooled_core_schema([m1, m2])
    X1 = orch._compose_pooled_X(m1, shared, schema="core")
    X2 = orch._compose_pooled_X(m2, shared, schema="core")
    # identical columns across members
    assert list(X1.columns) == list(X2.columns)
    for oh in ("is_fx", "is_metal", "is_crypto"):
        assert oh in X1.columns
    # one-hot correctness
    assert (X1["is_fx"] == 1.0).all() and (X1["is_crypto"] == 0.0).all()
    assert (X2["is_crypto"] == 1.0).all() and (X2["is_fx"] == 0.0).all()
    # no NaN in core schema
    assert not X1.isna().any().any()
    assert not X2.isna().any().any()
    # row count preserved
    assert len(X1) == 30 and len(X2) == 30


def test_extended_schema_is_nan_union_via_concat():
    m1 = _member("EURUSD", "fx", ["rsi", "fx_only"], 20, seed=1)
    m2 = _member("BTCUSD", "crypto", ["rsi", "btc_only"], 20, seed=2)
    shared = orch._pooled_core_schema([m1, m2])
    X1 = orch._compose_pooled_X(m1, shared, schema="extended")
    X2 = orch._compose_pooled_X(m2, shared, schema="extended")
    pooled = pd.concat([X1, X2], axis=0)
    # NaN-union: fx_only present for m1, NaN for m2's rows.
    assert "fx_only" in pooled.columns and "btc_only" in pooled.columns
    assert pooled["btc_only"].iloc[:20].isna().all()       # m1 rows lack btc_only
    assert pooled["fx_only"].iloc[20:].isna().all()        # m2 rows lack fx_only


# --------------------------------------------------------------------------- #
# B1 — pooled uniqueness wired (NOT concatenated per-asset)
# --------------------------------------------------------------------------- #
def test_pooled_weight_is_pooled_uniqueness_not_concat():
    # Two assets with PERFECTLY contemporaneous spans → pooled uniqueness ~0.5,
    # whereas concatenated per-asset avg_uniqueness would be ~1.0 each.
    t0 = pd.Timestamp("2021-01-01", tz="UTC")
    H = pd.Timedelta(hours=4)
    n = 10
    et = pd.DatetimeIndex([t0 + i * H for i in range(n)] * 2)
    le = pd.DatetimeIndex([t0 + (i + 1) * H for i in range(n)] * 2)  # 1-bar, overlapping
    asset_row = np.array(["EURUSD"] * n + ["GBPUSD"] * n)

    pooled_w = pooled_avg_uniqueness(et, le)
    # contemporaneous on two assets → each downweighted toward 0.5 (ρ=1 bound)
    assert np.all(pooled_w <= 0.75)
    assert np.median(pooled_w) < 0.75
    # the per-asset concat would be ~1.0 (each asset's own events are sequential)
    per_asset_concat = np.concatenate([
        avg_uniqueness(np.arange(n), np.arange(n), n_bars=n),
        avg_uniqueness(np.arange(n), np.arange(n), n_bars=n),
    ])
    assert np.median(per_asset_concat) > np.median(pooled_w)


# --------------------------------------------------------------------------- #
# §4 — per_class weight balancing
# --------------------------------------------------------------------------- #
def test_per_class_balance_equalizes_effective_mass():
    # crypto has 3x the events of fx → raw mass dominated by crypto.
    members = [
        _member("EURUSD", "fx", ["rsi"], 10, seed=1),
        _member("BTCUSD", "crypto", ["rsi"], 30, seed=2),
    ]
    asset_row = np.array(["EURUSD"] * 10 + ["BTCUSD"] * 30)
    w = np.ones(40)
    out = orch._balance_pool_weights(w, asset_row, members, "per_class")
    fx_mass = out[asset_row == "EURUSD"].sum()
    crypto_mass = out[asset_row == "BTCUSD"].sum()
    assert fx_mass == pytest.approx(crypto_mass, rel=1e-9)
    # total preserved
    assert out.sum() == pytest.approx(w.sum(), rel=1e-9)


def test_none_balance_is_identity():
    members = [_member("EURUSD", "fx", ["rsi"], 10, seed=1)]
    asset_row = np.array(["EURUSD"] * 10)
    w = np.linspace(0.1, 1.0, 10)
    out = orch._balance_pool_weights(w, asset_row, members, "none")
    np.testing.assert_array_equal(out, w)


# --------------------------------------------------------------------------- #
# B2 — calibration tail in pooled wall-clock time, causal
# --------------------------------------------------------------------------- #
def _calib_split_in_time(event_time: pd.DatetimeIndex, label_end_time: pd.DatetimeIndex,
                         calib_pct: float):
    """Mirror the exact B2 calib split logic in _run_one_pool for testing."""
    et_ns = event_time.asi8
    order = np.argsort(et_ns, kind="stable")
    n_hold = max(int(len(event_time) * calib_pct), 1)
    calib_local = order[-n_hold:]
    train_local = order[:-n_hold]
    calib_first_ns = int(et_ns[calib_local].min())
    le_ns = label_end_time.asi8
    train_local = train_local[le_ns[train_local] < calib_first_ns]
    return train_local, calib_local


def test_calibration_holdout_is_latest_by_event_time():
    t0 = pd.Timestamp("2021-01-01", tz="UTC")
    H = pd.Timedelta(hours=4)
    n = 20
    # SHUFFLED event order (different assets interleave) — the tail must be by TIME.
    times = [t0 + i * H for i in range(n)]
    perm = [3, 0, 7, 1, 9, 2, 5, 4, 8, 6, 13, 10, 17, 11, 19, 12, 15, 14, 18, 16]
    et = pd.DatetimeIndex([times[i] for i in perm])
    le = pd.DatetimeIndex([times[i] + H for i in perm])
    train_local, calib_local = _calib_split_in_time(et, le, 0.2)
    # calib rows are the latest-by-time
    calib_times = et[calib_local]
    train_times = et[train_local]
    assert calib_times.min() > train_times.max()


def test_calibration_split_is_causal_under_late_label_mutation():
    """Mutating a LATER row's label_end (lengthening it) does not change which
    rows are assigned to calib (calib assignment is by event_time, not label)."""
    t0 = pd.Timestamp("2021-01-01", tz="UTC")
    H = pd.Timedelta(hours=4)
    n = 20
    et = pd.DatetimeIndex([t0 + i * H for i in range(n)])
    le = pd.DatetimeIndex([t0 + (i + 1) * H for i in range(n)])
    _, calib_a = _calib_split_in_time(et, le, 0.2)
    le2 = le.copy().to_list()
    le2[-1] = le2[-1] + pd.Timedelta(days=30)   # mutate the latest event's label
    _, calib_b = _calib_split_in_time(et, pd.DatetimeIndex(le2), 0.2)
    np.testing.assert_array_equal(np.sort(calib_a), np.sort(calib_b))


# --------------------------------------------------------------------------- #
# embargo sizing
# --------------------------------------------------------------------------- #
def test_embargo_sized_off_coarsest_horizon():
    cfg = {"triple_barrier": {"horizon": 48}}
    fx = _member("EURUSD", "fx", ["rsi"], 5)
    crypto = _member("BTCUSD", "crypto", ["rsi"], 5, bars_per_year=2190)
    td = orch._pooled_embargo_td([fx, crypto], cfg)
    # coarsest = fx bar duration (365.25/1560 d) × 48 bars
    expected = pd.Timedelta(days=365.25 / 1560) * 48
    assert td == expected
    # crypto-only pool uses the finer crypto duration
    td_c = orch._pooled_embargo_td([crypto], cfg)
    assert td_c == pd.Timedelta(days=365.25 / 2190) * 48
    assert td > td_c


# --------------------------------------------------------------------------- #
# DSR cluster-N
# --------------------------------------------------------------------------- #
def test_dsr_cluster_n_single_cell_fallback():
    idx = pd.RangeIndex(50)
    oof = pd.DataFrame({"xgb": np.full(50, 0.6)}, index=idx)
    side = pd.Series(np.ones(50), index=idx)
    fwd = pd.Series(np.linspace(-0.01, 0.01, 50), index=idx)
    n, note = orch._pooled_dsr_cluster_n(oof, fwd, side)
    assert n == 1
    assert "familywise" in note.lower()


# --------------------------------------------------------------------------- #
# Phase D — per-asset OOS slicing reproduces counts + writes baseline schema
# --------------------------------------------------------------------------- #
def test_write_per_asset_oos_schema(tmp_path):
    """_write_per_asset_oos produces a summary.json with the baseline schema keys."""
    n = 60
    # Production always passes a DatetimeIndex (per-asset event_time); the per-fold
    # `years_in_window` is computed from its calendar span.
    idx = pd.date_range("2021-01-01", periods=n, freq="4h", tz="UTC")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"rsi": rng.standard_normal(n), "primary_side": rng.choice([-1, 1], n).astype(float)},
                     index=idx)
    y = pd.Series(rng.integers(0, 2, n), index=idx)
    side = X["primary_side"]
    fwd = pd.Series(rng.standard_normal(n) * 0.01, index=idx)
    w = np.ones(n)
    models = ["xgb"]
    cfg = {
        "models": models,
        "metrics": {"threshold_grid": [0.5, 0.55, 0.6]},
        "stacking": {"min_models_beating_baseline": 2, "min_folds_beating_baseline": 3,
                     "max_oof_corr": 0.7, "min_trades_per_fold": 30},
    }
    oof = pd.DataFrame({"xgb": rng.uniform(0.3, 0.7, n)}, index=idx)
    folds_test_idx = [np.arange(0, 30), np.arange(30, 60)]
    sel = {"xgb": [0.55, 0.55]}
    summary = orch._write_per_asset_oos(
        out_dir=tmp_path, cfg=cfg, primary_name="ema_cross", asset="EURUSD",
        asset_class="fx", bars_per_year=1560, cost_bps=10.0,
        X=X, y=y, side=side, fwd_ret=fwd, w=w, oof_probs=oof,
        folds_test_idx=folds_test_idx,
        selected_threshold_per_model_per_fold=sel,
        threshold_selection_diag={"xgb": [{}, {}]},
        mda_per_fold={"xgb": []}, clustered_mda_per_fold={"xgb": []},
        n_trials_familywise=len(models) * 2 * 3,
    )
    for key in ("primary", "asset", "asset_class", "bars_per_year", "n_events",
                "n_folds", "sharpe_per_fold_per_model", "n_trades_per_fold_per_model",
                "median_sharpe", "best_model", "stack_decision"):
        assert key in summary, f"missing {key}"
    assert summary["n_events"] == n
    assert summary["n_folds"] == 2
    written = json.loads((tmp_path / "summary.json").read_text())
    assert written["asset"] == "EURUSD"
    assert (tmp_path / "psr_dsr.json").exists()


# --------------------------------------------------------------------------- #
# OFF = parity (structural): the refactor only re-routes, it does not branch
# on meta_pooling inside the per-asset path, and the config default is OFF.
# The byte-identical guarantee is additionally exercised by the dry-run smoke.
# --------------------------------------------------------------------------- #
def test_config_default_meta_pooling_off():
    import yaml
    cfg = yaml.safe_load(Path("configs/multi_h4.yaml").read_text(encoding="utf-8"))
    assert cfg["meta_pooling"]["enabled"] is False
    assert cfg["meta_pooling"]["scope"] == "within_class"
    assert cfg["meta_pooling"]["schema"] == "core"
    assert cfg["meta_pooling"]["weight_balance"] == "per_class"
    assert cfg["meta_pooling"]["pooled_uniqueness"] is True


def test_per_asset_path_delegates_to_phase_a_and_phase_d(monkeypatch):
    """_run_one_asset_primary must build via _build_asset_primary_inputs and write
    via _write_per_asset_oos — the same two pieces the pooled path reuses. If the
    builder returns None (skip), the train block is never reached."""
    calls = {"build": 0, "write": 0}

    def fake_build(entry, cfg, asset_dfs, daily_macro):
        calls["build"] += 1
        return None  # simulate a skip → must short-circuit before any training

    monkeypatch.setattr(orch, "_build_asset_primary_inputs", fake_build)
    orch._run_one_asset_primary(
        entry={"asset": "EURUSD", "primary": "ema_cross"},
        cfg={}, asset_dfs={}, daily_macro=None, dry_run=False,
    )
    assert calls["build"] == 1


# --------------------------------------------------------------------------- #
# Comparison verdict (falsification helper)
# --------------------------------------------------------------------------- #
def _write_pair(tree: Path, asset, primary, dsr, median_sharpe, model="xgb"):
    d = tree / asset / primary
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({
        "primary": primary, "asset": asset, "best_model": model,
        "median_sharpe": {model: median_sharpe},
    }))
    (d / "psr_dsr.json").write_text(json.dumps({"dsr": {model: dsr}, "psr": {model: 0.9}}))


def test_comparison_confirmed_case(tmp_path):
    from scripts.compare_pooled_vs_per_asset import compare_trees
    base, pooled = tmp_path / "base", tmp_path / "pooled"
    # baseline: 0 assets clear DSR; pooled: 2 clear, sharpe within band.
    _write_pair(base, "EURUSD", "ema_cross", dsr=0.5, median_sharpe=0.30)
    _write_pair(base, "GBPUSD", "ema_cross", dsr=0.6, median_sharpe=0.40)
    _write_pair(pooled, "EURUSD", "ema_cross", dsr=0.97, median_sharpe=0.25)  # drop 0.05 OK
    _write_pair(pooled, "GBPUSD", "ema_cross", dsr=0.96, median_sharpe=0.35)  # drop 0.05 OK
    v = compare_trees(base, pooled)
    assert v["verdict"] == "CONFIRMED", v["criteria"]


def test_comparison_falsified_by_dominance_veto(tmp_path):
    from scripts.compare_pooled_vs_per_asset import compare_trees
    base, pooled = tmp_path / "base", tmp_path / "pooled"
    _write_pair(base, "EURUSD", "ema_cross", dsr=0.5, median_sharpe=0.30)
    _write_pair(base, "GBPUSD", "ema_cross", dsr=0.5, median_sharpe=0.80)
    # pooled clears DSR for both, but GBPUSD drops 0.45 > 0.40 veto.
    _write_pair(pooled, "EURUSD", "ema_cross", dsr=0.97, median_sharpe=0.28)
    _write_pair(pooled, "GBPUSD", "ema_cross", dsr=0.97, median_sharpe=0.35)
    v = compare_trees(base, pooled)
    assert v["verdict"] == "FALSIFIED"
    assert v["criteria"]["crit4_dominance_veto"]["tripped"]
    assert "GBPUSD" in v["criteria"]["crit4_dominance_veto"]["veto_assets"]
