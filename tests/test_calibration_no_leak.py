"""Verify calibration uses 'prefit' (not KFold int) and sample_weight propagates."""
from __future__ import annotations
import inspect
import re
import numpy as np
import pandas as pd
import pytest

from pipeline.train import build_calibrated_classifier, fit_calibrated, MODEL_FACTORIES


def test_train_module_does_not_use_cv_int():
    """Static check: the training code must NOT call CalibratedClassifierCV(cv=<int>).

    With sklearn 1.8+ where cv='prefit' was removed, the implementation must wrap the
    base estimator in FrozenEstimator to prevent CalibratedClassifierCV from using a
    default KFold that would leak across folds in time-series data.
    """
    src = inspect.getsource(__import__("pipeline.train", fromlist=["*"]))
    # The only acceptable `cv=` argument values are the string 'prefit'.
    matches = re.findall(r"CalibratedClassifierCV\([^)]*cv\s*=\s*([^,)\s]+)", src)
    for m in matches:
        assert m.strip().strip('"').strip("'") == "prefit", (
            f"Found CalibratedClassifierCV(cv={m}); only cv='prefit' is allowed (see spec)"
        )

    # Enforce FrozenEstimator presence (sklearn 1.8 replacement for cv='prefit')
    assert "FrozenEstimator(" in src, (
        "pipeline/train.py must wrap base in FrozenEstimator (sklearn-1.8 replacement "
        "for cv='prefit'); without this, CalibratedClassifierCV uses a default KFold "
        "that leaks across folds in time-series data."
    )


def test_sample_weight_changes_predictions_through_calibration():
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    y = pd.Series((X["x0"] > 0).astype(int).values)
    holdout_pct = 0.15
    n_train = int(n * (1 - holdout_pct))
    # Weights that zero out one class in the training half.
    w_skewed = np.where(y.values == 1, 1e-6, 1.0).astype(float)
    w_equal = np.ones(n, dtype=float)

    clf_a = fit_calibrated("xgb", X.iloc[:n_train], y.iloc[:n_train], w_skewed[:n_train],
                           X.iloc[n_train:], y.iloc[n_train:], w_skewed[n_train:], random_state=42)
    clf_b = fit_calibrated("xgb", X.iloc[:n_train], y.iloc[:n_train], w_equal[:n_train],
                           X.iloc[n_train:], y.iloc[n_train:], w_equal[n_train:], random_state=42)
    p_a = clf_a.predict_proba(X.iloc[n_train:])[:, 1]
    p_b = clf_b.predict_proba(X.iloc[n_train:])[:, 1]
    assert not np.allclose(p_a, p_b, atol=1e-3), (
        "sample_weight did not propagate through CalibratedClassifierCV — "
        "predictions are identical when training weights differ drastically"
    )


def test_model_factories_present():
    # B0053: lgbm + lr added as active models; catboost kept for backward compat.
    assert {"xgb", "lgbm", "rf", "lr"}.issubset(set(MODEL_FACTORIES.keys()))
    assert "catboost" in MODEL_FACTORIES  # compat factory — removed from active config


def test_boosting_factories_are_single_threaded_for_determinism():
    """B0064: xgb (tree_method='hist') and lgbm (leaf-wise) are thread-
    nondeterministic — multithreaded float reductions vary run-to-run, which
    makes the Phase-1 regression baseline flaky. Pin n_jobs=1 so a fixed seed
    yields bit-reproducible fits (the regression test depends on this)."""
    assert MODEL_FACTORIES["xgb"]().get_params()["n_jobs"] == 1
    assert MODEL_FACTORIES["lgbm"]().get_params()["n_jobs"] == 1
