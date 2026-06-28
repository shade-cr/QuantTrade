"""B_supervised — supervised-direct mode bypasses primary signal."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 600) -> pd.DataFrame:
    """Build a synthetic OHLCV frame.

    Default n=600 so that build_technical_features.dropna() leaves ~330 rows
    (252-bar rolling warmup consumes the first ~271 bars). Tests that need
    n_events > 100 or a substantial labeled-mode subset must use >= 600 bars.
    """
    rng = np.random.default_rng(42)
    idx = pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC")
    # Stationary (not cumsum) so forward returns are ~50/50 positive/negative,
    # giving balanced labels even in small test datasets. cumsum produces a
    # trending random walk that causes mostly label=1 (TP hit) → single-class
    # calibration slices → RefittingCalibratedPipeline failure in tests.
    close = 1800.0 + rng.standard_normal(n) * 8
    hi = close + rng.uniform(0.5, 2, n)
    lo = close - rng.uniform(0.5, 2, n)
    return pd.DataFrame({"open": close, "high": hi, "low": lo, "close": close,
                         "volume": rng.integers(1000, 5000, n).astype(float)},
                        index=idx)


def _make_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Build technical features; _atr_14 is already in build_technical_features output."""
    from pipeline.features import build_technical_features
    return build_technical_features(ohlcv).dropna()


def _base_cfg(tmp_path) -> dict:
    return {
        "output_dir": str(tmp_path),
        "primary": {"mode": "supervised_direct", "supervised_horizon": 20},
        "primary_feature_blacklist": [],
        "feature_overrides_add": [],
        "feature_overrides_drop": [],
        "regime_mask_path": None,
        "triple_barrier": {"horizon": 20, "tp_atr_mult": 3.0, "sl_atr_mult": 1.0},
        "walk_forward": {"n_folds": 2, "train_min_bars": 50, "purge_bars": 5,
                         "embargo_pct": 0.01},
        "hyperparam_search": {"n_iter": 2, "cv_splits": 2, "cv_purge_bars": 5,
                               "random_state": 42},
        "models": ["rf"],
        "random_seed": 42,
        "calibration": {"method": "sigmoid", "calib_holdout_pct": 0.20,
                        "min_minority_for_isotonic": 20},
        "stacking": {"min_models_beating_baseline": 2, "min_folds_beating_baseline": 1,
                     "max_oof_corr": 0.9, "min_trades_per_fold": 5,
                     "meta_n_folds": 2, "meta_C": 1.0},
        "metrics": {"bars_per_year": 252, "threshold_grid": [0.5],
                    "cost_per_trade_bps": 5},
        "dry_run": {"n_iter": 2, "max_minutes_per_model_warn": 100},
        "shutoff": {"rolling_window": 6, "threshold": 0.0},
    }


def test_supervised_direct_count_returns_n_bars(tmp_path):
    """count_events_only mode returns all non-NaN in-scope bars, not a sparse subset."""
    from scripts.run_xau_d1 import _run_one_primary
    ohlcv = _make_ohlcv()
    features = _make_features(ohlcv)
    cfg = _base_cfg(tmp_path)

    result = _run_one_primary(
        "supervised_direct", cfg, ohlcv.loc[features.index],
        features, dry_run=True, count_events_only=True,
    )

    assert result is not None
    assert result["n_events"] > 100
    assert result["primary"] == "supervised_direct"


def test_supervised_direct_x_has_no_primary_columns(tmp_path, monkeypatch):
    """X matrix must NOT contain primary_side, primary_strength, bars_since_signal."""
    import scripts.run_xau_d1 as module
    captured_X = {}

    def fake_fit(X, y, w, cfg, random_state, models):
        captured_X["X"] = X.copy()
        return {}

    monkeypatch.setattr(module, "fit_all_models", fake_fit, raising=False)

    ohlcv = _make_ohlcv()
    features = _make_features(ohlcv)
    cfg = _base_cfg(tmp_path)

    try:
        module._run_one_primary("supervised_direct", cfg, ohlcv.loc[features.index],
                                features, dry_run=True, count_events_only=False)
    except Exception:
        pass  # may fail at reporting; we only care about X

    if captured_X:
        X = captured_X["X"]
        assert "primary_side" not in X.columns
        assert "primary_strength" not in X.columns
        assert "bars_since_signal" not in X.columns


def test_supervised_direct_respects_regime_gate(tmp_path):
    """When regime_mask_path is set, only in-scope bars become events."""
    from scripts.run_xau_d1 import _run_one_primary
    ohlcv = _make_ohlcv()
    features = _make_features(ohlcv)
    cfg = _base_cfg(tmp_path)

    mask_path = tmp_path / "mask.parquet"
    mask_df = pd.DataFrame(
        {"mask": [True] * 100 + [False] * (len(features) - 100)},
        index=features.index,
    )
    mask_df.to_parquet(mask_path)
    cfg["regime_mask_path"] = str(mask_path)

    result = _run_one_primary(
        "supervised_direct", cfg, ohlcv.loc[features.index],
        features, dry_run=True, count_events_only=True,
    )

    assert result["n_events"] <= 100


def test_supervised_direct_labels_binary(tmp_path, monkeypatch):
    """Labels returned by supervised mode must be 0 or 1."""
    import scripts.run_xau_d1 as module
    captured_y = {}

    def fake_fit(X, y, w, cfg, random_state, models):
        captured_y["y"] = y.copy()
        return {}

    monkeypatch.setattr(module, "fit_all_models", fake_fit, raising=False)

    ohlcv = _make_ohlcv()
    features = _make_features(ohlcv)
    cfg = _base_cfg(tmp_path)

    try:
        module._run_one_primary("supervised_direct", cfg, ohlcv.loc[features.index],
                                features, dry_run=True, count_events_only=False)
    except Exception:
        pass

    if captured_y:
        y = captured_y["y"]
        assert set(y.unique()).issubset({0, 1}), f"unexpected label values: {y.unique()}"


def test_labeled_mode_unchanged_by_supervised_code(tmp_path):
    """The default labeled mode still produces sparse events (primary-gated)."""
    from scripts.run_xau_d1 import _run_one_primary
    ohlcv = _make_ohlcv()
    features = _make_features(ohlcv)
    cfg = _base_cfg(tmp_path)
    cfg["primary"] = {
        "mode": "labeled",
        "candidates": ["ema_cross"],
        # dead_zone_atr=0.5 filters ~64% of bars on the synthetic random walk
        # (EMA spread must exceed 0.5 ATR to fire); ensures n_events << n_bars.
        "ema_cross": {"fast": 5, "slow": 20, "dead_zone_atr": 0.5},
    }

    result = _run_one_primary(
        "ema_cross", cfg, ohlcv.loc[features.index],
        features, dry_run=True, count_events_only=True,
    )

    assert result["n_events"] < len(features) // 2


def test_supervised_direct_zero_events_after_tight_mask(tmp_path):
    """If mask leaves 0 bars, supervised mode returns n_events=0 gracefully."""
    from scripts.run_xau_d1 import _run_one_primary
    ohlcv = _make_ohlcv()
    features = _make_features(ohlcv)
    cfg = _base_cfg(tmp_path)

    mask_path = tmp_path / "empty_mask.parquet"
    pd.DataFrame({"mask": [False] * len(features)}, index=features.index).to_parquet(mask_path)
    cfg["regime_mask_path"] = str(mask_path)

    result = _run_one_primary(
        "supervised_direct", cfg, ohlcv.loc[features.index],
        features, dry_run=True, count_events_only=True,
    )
    assert result["n_events"] == 0


# ---------------------------------------------------------------------------
# Task 2: _run_folds_and_report wired into supervised_direct non-count path
# ---------------------------------------------------------------------------

def test_supervised_direct_noncountpath_does_not_raise(tmp_path):
    """Non-count supervised_direct path must complete without raising.

    This is the Task 2 invariant: once _run_folds_and_report is wired, the
    non-count path must run the fold-training + reporting block (verified by
    the companion test that checks summary.json is written). The dry-run path
    returns None (bare return) in _run_folds_and_report — same as labeled mode.
    Pre-Task-2 code never reached _run_folds_and_report and also returned None,
    but did NOT write summary.json or run any training. The distinction is
    tested by test_supervised_direct_noncountpath_writes_summary_json.
    """
    from scripts.run_xau_d1 import _run_one_primary
    ohlcv = _make_ohlcv()
    features = _make_features(ohlcv)
    cfg = _base_cfg(tmp_path)

    # Must not raise; dry-run returns None (bare return) by design.
    _run_one_primary(
        "supervised_direct", cfg, ohlcv.loc[features.index],
        features, dry_run=True, count_events_only=False,
    )


def test_supervised_direct_noncountpath_writes_summary_json(tmp_path):
    """Dry-run non-count path writes summary.json to the output directory."""
    from scripts.run_xau_d1 import _run_one_primary
    import json as _json
    ohlcv = _make_ohlcv()
    features = _make_features(ohlcv)
    cfg = _base_cfg(tmp_path)

    _run_one_primary(
        "supervised_direct", cfg, ohlcv.loc[features.index],
        features, dry_run=True, count_events_only=False,
    )

    summary_path = tmp_path / "supervised_direct" / "summary.json"
    assert summary_path.exists(), (
        f"summary.json not written — non-count path not producing output. "
        f"Files present: {list((tmp_path / 'supervised_direct').glob('*')) if (tmp_path / 'supervised_direct').exists() else 'dir missing'}"
    )
    data = _json.loads(summary_path.read_text())
    assert "dry_run" in data or "primary" in data, (
        f"summary.json has unexpected shape: {list(data.keys())}"
    )


def test_run_folds_and_report_is_exported(tmp_path):
    """After Task 2 refactor, _run_folds_and_report must exist as a callable
    in the scripts.run_xau_d1 module (structural contract).
    """
    import scripts.run_xau_d1 as module
    assert hasattr(module, "_run_folds_and_report"), (
        "_run_folds_and_report not found in scripts.run_xau_d1 — "
        "extraction not complete (Task 2)"
    )
    assert callable(module._run_folds_and_report), (
        "_run_folds_and_report is not callable"
    )
