"""Tests for RefittingCalibratedPipeline (Phase 2, T9.A.1).

This file is part of the TDD scaffold for the wrapper that decouples inner-CV
training from threshold scoring. The most important test here is the no-leak
test: it asserts that `cross_val_predict` over `PurgedTimeSeriesSplit` truly
hands the wrapper inner_train-only rows.

RED phase: with the deliberately-leaking stub in pipeline/train.py, the no-leak
test MUST fail with a ROC-AUC ≈ 1.0. If it passes, the test is broken.

GREEN phase (T9.A.1.GREEN): with the real wrapper, ROC-AUC must drop below
the 0.95 threshold (the real wrapper re-fits the base from scratch on each
inner_train, so the base never sees inner_val).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pipeline.train import RefittingCalibratedPipeline
from pipeline.walk_forward import PurgedTimeSeriesSplit, inner_oof_predict_proba


def _make_lag_dataset(n: int = 800, lag: int = 10, seed: int = 42, noise_std: float = 0.8):
    """Synthetic dataset where y[t] depends on a lagged feature visible at X[t].

    Design constraints:
      - Signal must be IN X (so a model that trains on (X_train, y_train)
        and predicts on X_val can actually learn the rule). A signal
        encoded as "y depends on a feature absent from X" trivially scores
        AUC ≈ 0.5 regardless of leak.
      - Signal must be LEARNABLE but NOT TRIVIAL — a small model on partial
        train data should score around 0.70-0.85. If the rule is "y =
        sign(x_lag)" cleanly, even 30 trees memorise it and AUC ≈ 1.0,
        which makes the no-leak test (upper-bound 0.95) brittle.
      - Noise calibrated so AUC lands comfortably between 0.60 (the lower
        bound that detects "wrapper isn't learning") and 0.95 (the upper
        bound that detects leak).

    Construction:
      - x[t] is a stationary AR(1)-ish signal
      - X[t] = {"x_lag": x[t-lag], "x_now": x[t], "noise": independent N(0,1)}
      - p(y[t]=1) = sigmoid(beta * x[t-lag] + noise_std * eps[t])
      - y[t] ~ Bernoulli(p)
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    x_lag = np.roll(x, lag)
    eps = rng.standard_normal(n)
    logits = 2.0 * x_lag + noise_std * eps
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_vals = (rng.uniform(size=n) < probs).astype(int)
    X = pd.DataFrame(
        {"x_lag": x_lag, "x_now": x, "noise": rng.standard_normal(n)},
        index=idx,
    )
    y = pd.Series(y_vals, index=idx)
    # Drop the first `lag` rows where the roll wrapped around (no real past).
    return X.iloc[lag:].copy(), y.iloc[lag:].copy()


def test_manual_cv_no_leak():
    """No-leak gate.

    With y = sign(X[t-lag]) and a CV that purges `lag` bars between
    inner_train and inner_val, a non-leaking wrapper cannot achieve
    perfect OOF probabilities — the base, trained on inner_train, has
    never seen the inner_val rows, so it can only generalise the rule
    learned from inner_train.

    A leaking wrapper (the RED stub: memorises index → y_true) will
    return perfect probabilities on any row it saw at fit-time.
    Crucially, the manual CV loop calls .fit(inner_train) THEN
    .predict_proba(inner_val) — so the stub only "saw" inner_train rows.
    If the wrapper somehow gets inner_val rows fed into its fit, it
    will memorise them too and ROC-AUC will be ≈ 1.0.

    Threshold 0.95 is permissive — a non-leaking model on this synthetic
    signal typically scores ~0.85; perfect-from-leak scores 1.0.

    Note: we use a manual loop instead of sklearn.cross_val_predict because
    PurgedTimeSeriesSplit is not a partition (early rows never appear in any
    val fold). The production threshold pipeline will do the same.
    """
    lag = 10
    X, y = _make_lag_dataset(n=800, lag=lag, seed=42)

    rcp = RefittingCalibratedPipeline(
        model_name="rf",
        base_kwargs={"n_estimators": 30, "max_depth": 6, "min_samples_leaf": 5},
        method="sigmoid",
        random_state=0,
    )

    cv = PurgedTimeSeriesSplit(n_splits=3, purge=lag)
    oof_full = inner_oof_predict_proba(rcp, X, y, cv)
    oof = oof_full[:, 1]

    # Manual CV leaves NaN for indices outside any val fold. Score only on
    # the rows that were actually predicted.
    predicted_mask = ~np.isnan(oof)
    assert predicted_mask.any(), "manual CV produced no predictions"

    auc = roc_auc_score(y.values[predicted_mask], oof[predicted_mask])

    # Dual bound discriminates three cases:
    #   - RED stub (returns 0.5 for unseen rows): AUC ≈ 0.5 → fails LOWER bound
    #     (the wrapper isn't learning — it's still a stub)
    #   - GREEN real wrapper: AUC ~ 0.75-0.90 → passes BOTH bounds
    #     (wrapper learns the temporal rule without leaking)
    #   - BROKEN wrapper with leak (e.g. base sees inner_val via FrozenEstimator
    #     bug): AUC ≈ 1.0 → fails UPPER bound (leak detected)
    assert 0.60 < auc < 0.95, (
        f"ROC-AUC={auc:.4f} outside expected range (0.60, 0.95). "
        f"If auc <= 0.60: the wrapper isn't learning the temporal rule "
        f"y = sign(X[t-{lag}]) — likely still a stub or untrained. "
        f"If auc >= 0.95: leak detected — the wrapper is predicting inner_val "
        f"rows with information from outside its training set."
    )


# ---------------------------------------------------------------------------
# Property tests of the wrapper (T9.A.1 additional invariants)
# ---------------------------------------------------------------------------

def test_fit_predict_proba_shape_and_range():
    """predict_proba returns shape (n, 2) with values in [0, 1] summing to 1."""
    X, y = _make_lag_dataset(n=300, lag=5, seed=1)
    rcp = RefittingCalibratedPipeline(
        model_name="rf", base_kwargs={"n_estimators": 20}, method="sigmoid",
        random_state=0,
    )
    rcp.fit(X, y)
    probas = rcp.predict_proba(X.iloc[:50])
    assert probas.shape == (50, 2)
    assert (probas >= 0).all() and (probas <= 1).all()
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)


def test_clone_produces_independent_instance():
    """sklearn.clone(rcp) must yield a fresh estimator with no fitted state
    and hyperparams that survive. Re-fitting one must not affect the other."""
    from sklearn.base import clone

    X, y = _make_lag_dataset(n=300, lag=5, seed=1)
    rcp_fitted = RefittingCalibratedPipeline(
        model_name="rf", base_kwargs={"n_estimators": 20}, random_state=0,
    ).fit(X, y)
    assert hasattr(rcp_fitted, "calibrator_")

    clone_rcp = clone(rcp_fitted)
    assert not hasattr(clone_rcp, "calibrator_"), "clone must not preserve fitted state"
    assert clone_rcp.model_name == "rf"
    assert clone_rcp.random_state == 0

    # Fit the clone on different data; original must remain unaffected.
    X2, y2 = _make_lag_dataset(n=300, lag=5, seed=99)
    clone_rcp.fit(X2, y2)
    rcp_fresh = RefittingCalibratedPipeline(
        model_name="rf", base_kwargs={"n_estimators": 20}, random_state=0,
    ).fit(X, y)
    np.testing.assert_allclose(
        rcp_fitted.predict_proba(X.iloc[:10]),
        rcp_fresh.predict_proba(X.iloc[:10]),
        atol=1e-9,
    )


def test_single_class_calibration_slice_degrades_gracefully():
    """Single-class calibration slice now degrades gracefully (supervised-direct support).

    Previously raised ValueError; now emits RuntimeWarning and returns uncalibrated
    base model probabilities so strongly-trending supervised-direct audits don't crash.
    """
    import warnings as _warnings
    n = 100
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    X = pd.DataFrame({"x": np.arange(n, dtype=float)}, index=idx)
    y = pd.Series(np.array([1] * 85 + [0] * 15), index=idx)
    rcp = RefittingCalibratedPipeline(
        model_name="rf", base_kwargs={"n_estimators": 10}, calib_holdout_pct=0.15,
    )
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        rcp.fit(X, y)
        assert any("Calibration skipped" in str(warning.message) for warning in w)
    proba = rcp.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_chronological_holdout_is_tail():
    """Calibration holdout is the LAST int(n * calib_holdout_pct) rows.
    Verified by intercepting MODEL_FACTORIES — the fake base records
    which rows it receives, and we assert they're the chronological head."""
    from unittest.mock import patch

    n = 100
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.standard_normal(n)}, index=idx)
    y = pd.Series((rng.uniform(size=n) > 0.5).astype(int), index=idx)

    captured: dict = {}

    from sklearn.base import BaseEstimator, ClassifierMixin

    class _FakeBase(ClassifierMixin, BaseEstimator):
        classes_ = np.array([0, 1])

        def fit(self, X, y, sample_weight=None):
            captured["base_X_first"] = X.index[0]
            captured["base_X_last"] = X.index[-1]
            captured["base_n"] = len(X)
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            # Non-degenerate probas so CalibratedClassifierCV is happy.
            return np.column_stack([
                1 - np.linspace(0.1, 0.9, len(X)),
                np.linspace(0.1, 0.9, len(X)),
            ])

        def predict(self, X):
            return self.predict_proba(X).argmax(axis=1)

    rcp = RefittingCalibratedPipeline(
        model_name="rf", base_kwargs={"n_estimators": 10},
        calib_holdout_pct=0.20,  # last 20 rows → base sees first 80
    )

    with patch.dict("pipeline.train.MODEL_FACTORIES", {"rf": lambda **kw: _FakeBase()}):
        rcp.fit(X, y)

    assert captured["base_n"] == 80
    assert captured["base_X_first"] == idx[0]
    assert captured["base_X_last"] == idx[79]  # last row BEFORE the 20-row tail


def test_sample_weight_propagates_to_base():
    """sample_weight, when provided, must be sliced and passed to base.fit.
    Verified via a fake base that records what it received."""
    from unittest.mock import patch

    n = 100
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.standard_normal(n)}, index=idx)
    y = pd.Series((rng.uniform(size=n) > 0.5).astype(int), index=idx)
    w = np.arange(n, dtype=float) * 0.1  # distinguishable weights

    captured: dict = {}

    from sklearn.base import BaseEstimator, ClassifierMixin

    class _FakeBase(ClassifierMixin, BaseEstimator):
        classes_ = np.array([0, 1])

        def fit(self, X, y, sample_weight=None):
            captured["base_n"] = len(X)
            captured["base_sw"] = (
                None if sample_weight is None else np.asarray(sample_weight).copy()
            )
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            return np.column_stack([
                1 - np.linspace(0.1, 0.9, len(X)),
                np.linspace(0.1, 0.9, len(X)),
            ])

        def predict(self, X):
            return self.predict_proba(X).argmax(axis=1)

    rcp = RefittingCalibratedPipeline(
        model_name="rf", base_kwargs={"n_estimators": 10},
        calib_holdout_pct=0.15,  # last 15 rows → base sees first 85
    )
    with patch.dict("pipeline.train.MODEL_FACTORIES", {"rf": lambda **kw: _FakeBase()}):
        rcp.fit(X, y, sample_weight=w)

    assert captured["base_n"] == 85
    np.testing.assert_array_equal(captured["base_sw"], w[:85])


def test_b0032_small_minority_calibration_slice_succeeds_at_n_minority_2():
    """B0032 — minority=2 calibration slice must succeed via cv-cap.

    sklearn's CalibratedClassifierCV(FrozenEstimator(...)) partitions
    X_calib via StratifiedKFold(n_splits=cv). The default cv=5 quietly
    succeeds on n_minority=2 but the path is brittle; the explicit cv
    cap to max(2, min(5, n_minority_calib))=2 makes the success deterministic.
    n_minority=2 is the smallest supported case (n_minority<2 raises an
    explicit project-side ValueError; see the sibling test).
    """
    rng = np.random.default_rng(0)
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    X = pd.DataFrame({"x": rng.standard_normal(n), "x2": rng.standard_normal(n)}, index=idx)
    # Calibration tail = last 6 rows (30 * 0.20 = 6). Engineer 2 minority + 4 majority.
    y = np.concatenate([
        np.tile([0, 1], 12),                # 24 base rows, balanced
        np.array([1, 1, 0, 0, 0, 0]),       # 6 calib rows, 2 minority
    ])
    y = pd.Series(y, index=idx)
    assert int(min((y.iloc[-6:] == 0).sum(), (y.iloc[-6:] == 1).sum())) == 2

    rcp = RefittingCalibratedPipeline(
        model_name="rf", base_kwargs={"n_estimators": 10}, calib_holdout_pct=0.20,
    )
    rcp.fit(X, y)
    proba = rcp.predict_proba(X.iloc[:10])
    assert proba.shape == (10, 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_b0032_n_minority_1_degrades_gracefully():
    """B0032 update — n_minority=1 now degrades gracefully (supervised-direct support).

    Pre-first-fix: sklearn raised a cryptic error deep in cross_val_predict.
    Post-first-fix: explicit ValueError raised.
    Post-second-fix: calibration is SKIPPED with a RuntimeWarning so
    supervised-direct audits in strongly trending regimes (single-class calib
    slices) don't crash. The pipeline returns uncalibrated base probabilities.
    """
    import warnings as _warnings
    rng = np.random.default_rng(0)
    n = 35
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    X = pd.DataFrame({"x": rng.standard_normal(n), "x2": rng.standard_normal(n)}, index=idx)
    y = np.concatenate([
        np.tile([0, 1], 14),              # 28 base rows, balanced
        np.array([1, 0, 0, 0, 0, 0, 0]), # 7 calib rows, 1 minority
    ])
    y = pd.Series(y, index=idx)

    rcp = RefittingCalibratedPipeline(
        model_name="rf", base_kwargs={"n_estimators": 10}, calib_holdout_pct=0.20,
    )
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        rcp.fit(X, y)
        assert any("Calibration skipped" in str(warning.message) for warning in w), (
            "Expected RuntimeWarning about skipped calibration"
        )
    proba = rcp.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
