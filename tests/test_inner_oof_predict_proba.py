"""Tests for inner_oof_predict_proba (T9.A.0).

Helper that iterates a CV (typically PurgedTimeSeriesSplit, which is NOT a
partition) calling clone(estimator).fit() on each train_idx and storing
predict_proba(val_idx) into a pre-allocated output array. Rows outside any
val fold remain NaN.

This exists because sklearn's cross_val_predict requires the CV to be a
partition and rejects PurgedTimeSeriesSplit with
"cross_val_predict only works for partitions" (discovered during T9.A.1.RED
on commit c59093f).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from pipeline.walk_forward import PurgedTimeSeriesSplit, inner_oof_predict_proba


class _RecordingEstimator(BaseEstimator, ClassifierMixin):
    """Test double: records every fit() and predict_proba() call.

    Each cloned instance gets its own `fit_calls` and `predict_calls` lists
    (sklearn's clone discards instance attributes set in fit, but the lists
    we assign in __init__ via copy.deepcopy are preserved as hyperparams —
    so to record across clones we use a class-level registry instead).
    """

    _registry: list[dict] = []  # class-level: accumulates across all clones

    def __init__(self, n_classes: int = 2):
        self.n_classes = n_classes

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.arange(self.n_classes)
        # Snapshot what fit saw, so the test can assert per-fold inputs.
        self._registry.append({
            "kind": "fit",
            "n_rows": len(X),
            "train_index_first": X.index[0] if hasattr(X, "index") else None,
            "train_index_last": X.index[-1] if hasattr(X, "index") else None,
            "sample_weight": None if sample_weight is None else np.asarray(sample_weight).copy(),
            "y_sum": int(np.asarray(y).sum()),
        })
        return self

    def predict_proba(self, X):
        self._registry.append({
            "kind": "predict",
            "n_rows": len(X),
            "val_index_first": X.index[0] if hasattr(X, "index") else None,
            "val_index_last": X.index[-1] if hasattr(X, "index") else None,
        })
        # Return uniform 0.5 — content doesn't matter for these structural tests.
        return np.full((len(X), self.n_classes), 0.5)


@pytest.fixture(autouse=True)
def _clear_registry():
    _RecordingEstimator._registry.clear()
    yield
    _RecordingEstimator._registry.clear()


def _make_dataset(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    X = pd.DataFrame({"x": rng.standard_normal(n)}, index=idx)
    y = pd.Series((rng.standard_normal(n) > 0).astype(int), index=idx)
    return X, y


def test_returns_nan_for_rows_outside_val_folds():
    """PurgedTimeSeriesSplit leaves early rows outside any val fold.

    Layout for n=200, n_splits=3, purge=5 → step=50:
      split 1: train=[0,50),   val=[55,105)
      split 2: train=[0,100),  val=[105,155)
      split 3: train=[0,150),  val=[155,200)

    Val ranges are contiguous and partition [55, 200). The only NaN region
    is the head [0, 55):
      - Rows [0, 50) are train-only across all splits.
      - Rows [50, 55) are the initial purge zone for split 1 (excluded
        from split 1's train AND val, and never in any later val).

    Intermediate "purge zones" between consecutive splits do NOT exist as
    NaN — e.g. rows [100, 105) are excluded from split 2's train/val but
    ARE inside split 1's val [55, 105), so they get predicted by split 1.

    The fact that PurgedTimeSeriesSplit IS a partition of [55, 200) but
    NOT a partition of [0, 200) is precisely what makes sklearn's
    cross_val_predict reject it — and what motivates this helper.
    """
    X, y = _make_dataset(n=200)
    cv = PurgedTimeSeriesSplit(n_splits=3, purge=5)
    est = _RecordingEstimator()

    out = inner_oof_predict_proba(est, X, y, cv)

    assert out.shape == (200, 2), f"expected shape (200, 2), got {out.shape}"
    # Head gap [0, step+purge) = [0, 55) is NaN — never in any val fold.
    assert np.isnan(out[:55]).all(), "head gap [0, 55) should be NaN"
    # Everything from row 55 onwards IS predicted (val ranges partition this).
    assert not np.isnan(out[55:]).any(), (
        "rows [55, 200) should all be predicted (val ranges partition this region)"
    )


def test_predicts_val_rows_via_clone_fit():
    """For each (train_idx, val_idx) yielded by cv.split, the helper must
    call est.fit on train_idx-rows only (not val_idx) and est.predict_proba
    on val_idx-rows only. Verified via a recording test double."""
    X, y = _make_dataset(n=200)
    cv = PurgedTimeSeriesSplit(n_splits=3, purge=5)
    est = _RecordingEstimator()

    inner_oof_predict_proba(est, X, y, cv)

    fit_calls = [c for c in _RecordingEstimator._registry if c["kind"] == "fit"]
    predict_calls = [c for c in _RecordingEstimator._registry if c["kind"] == "predict"]

    # 3 splits → 3 fits and 3 predicts.
    assert len(fit_calls) == 3, f"expected 3 fit calls, got {len(fit_calls)}"
    assert len(predict_calls) == 3, f"expected 3 predict calls, got {len(predict_calls)}"

    # For each split, fit_idx and predict_idx must be disjoint
    # (val_start > train_end thanks to purge).
    for fit_c, pred_c in zip(fit_calls, predict_calls):
        assert fit_c["train_index_last"] < pred_c["val_index_first"], (
            f"train must end strictly before val starts; "
            f"got train_end={fit_c['train_index_last']}, val_start={pred_c['val_index_first']}"
        )

    # Each fit must grow (expanding window).
    fit_sizes = [c["n_rows"] for c in fit_calls]
    assert fit_sizes == sorted(fit_sizes), f"train sizes should grow: {fit_sizes}"


def test_return_val_indices_yields_the_cv_val_arrays_in_order():
    """When return_val_indices=True, the helper also returns the list of
    val_idx arrays in the order the CV produced them. Downstream code
    (e.g. select_threshold_inner_cv with sub_block_indices=) uses these
    to evaluate metrics over exactly the OOF regions."""
    X, y = _make_dataset(n=200)
    cv = PurgedTimeSeriesSplit(n_splits=3, purge=5)
    est = _RecordingEstimator()

    result = inner_oof_predict_proba(est, X, y, cv, return_val_indices=True)
    assert isinstance(result, tuple) and len(result) == 2
    out, val_indices = result

    assert out.shape == (200, 2)
    assert len(val_indices) == 3, "expected one val_idx array per split"

    # Cross-check against splitting cv directly: the val_indices returned
    # must match cv.split exactly (same arrays, same order).
    expected = [v for _, v in cv.split(X, y)]
    for i, (got, exp) in enumerate(zip(val_indices, expected)):
        np.testing.assert_array_equal(
            got, exp,
            err_msg=f"split {i}: val_indices mismatch",
        )


def test_sample_weight_passes_through():
    """sample_weight, when provided, must be sliced by train_idx and passed
    to est.fit. Without sample_weight, est.fit receives None."""
    X, y = _make_dataset(n=200)
    cv = PurgedTimeSeriesSplit(n_splits=3, purge=5)
    est = _RecordingEstimator()
    w = np.arange(200, dtype=float) * 0.01  # distinguishable weights

    inner_oof_predict_proba(est, X, y, cv, sample_weight=w)

    fit_calls = [c for c in _RecordingEstimator._registry if c["kind"] == "fit"]
    assert len(fit_calls) == 3

    # Each fit should have received a sample_weight slice of length == n_rows.
    # The slice values must come from `w` at the train_idx positions.
    # PurgedTimeSeriesSplit train_idx for split k is np.arange(0, k*step).
    step = 200 // (3 + 1)  # = 50
    for k, fit_c in enumerate(fit_calls, start=1):
        expected_train_end = k * step
        sw = fit_c["sample_weight"]
        assert sw is not None, f"split {k}: sample_weight missing"
        assert len(sw) == expected_train_end, (
            f"split {k}: sample_weight length {len(sw)} != train size {expected_train_end}"
        )
        # Weights should be the first `expected_train_end` entries of w
        # (since train_idx = np.arange(0, expected_train_end)).
        np.testing.assert_array_equal(sw, w[:expected_train_end])
