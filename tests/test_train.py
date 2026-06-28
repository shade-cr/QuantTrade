"""Tests for pipeline.train calibration-method defaults (B0133).

Invariant under test (CLAUDE.md): calibration MUST default to `sigmoid`
(Platt), NOT `isotonic`. Isotonic produces step-function probabilities that
collapse threshold selection (pct_signals_kept < 5%). Two builders historically
leaned isotonic when a `method` config key was dropped; B0133 hardens both
defaults to sigmoid while keeping explicit overrides intact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.train import build_calibrated_classifier, fit_calibrated


def _synth_classification(n: int = 400, minority_frac: float = 0.45, seed: int = 0):
    """Small linearly-separable-ish binary set with a controllable minority.

    Returns (X, y, w) as DataFrame / Series / ndarray, matching fit_calibrated's
    signature.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(round(n * minority_frac))
    n_neg = n - n_pos
    # Two gaussians with overlap so the calibrator has a non-trivial mapping.
    X_pos = rng.normal(0.8, 1.0, size=(n_pos, 3))
    X_neg = rng.normal(-0.8, 1.0, size=(n_neg, 3))
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
    # Shuffle so train/calib splits below are not class-ordered.
    order = rng.permutation(n)
    X, y = X[order], y[order]
    Xdf = pd.DataFrame(X, columns=["f0", "f1", "f2"])
    ys = pd.Series(y)
    w = np.ones(n, dtype=float)
    return Xdf, ys, w


def _split(X, y, w, calib_frac: float = 0.5):
    n = len(X)
    cut = int(n * (1 - calib_frac))
    return (
        X.iloc[:cut], y.iloc[:cut], w[:cut],
        X.iloc[cut:], y.iloc[cut:], w[cut:],
    )


def _fitted_method(clf) -> str:
    """Extract the calibration method actually used by a fitted
    CalibratedClassifierCV, robust across sklearn versions: prefer the per-fold
    _CalibratedClassifier.method, fall back to the top-level attribute."""
    cc = getattr(clf, "calibrated_classifiers_", None)
    if cc:
        m = getattr(cc[0], "method", None)
        if m is not None:
            return m
    return clf.method


# ---------------------------------------------------------------------------
# fit_calibrated — degenerate-slice degradation (B0032/B0156 family, F008)
# ---------------------------------------------------------------------------
def test_fit_calibrated_single_class_calib_slice_degrades():
    """Single-class calibration holdout (selective primary on a regime-gated
    fold) must NOT crash CalibratedClassifierCV's internal cross_val_predict
    (IndexError in _enforce_prediction_order) — it returns uncalibrated base
    probabilities with a warning."""
    X, y, w = _synth_classification()
    Xt, yt, wt, Xc, yc, wc = _split(X, y, w)
    yc_single = pd.Series(np.zeros(len(yc), dtype=int), index=yc.index)
    with pytest.warns(RuntimeWarning, match="skipping calibration"):
        clf = fit_calibrated("lr", Xt, yt, wt, Xc, yc_single, wc)
    p = clf.predict_proba(Xc)
    assert p.shape == (len(Xc), 2)
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1.0)


def test_fit_calibrated_one_minority_calib_sample_degrades():
    """n_minority=1 in the calibration slice also crashes sklearn's internal
    partition; must degrade, not crash."""
    X, y, w = _synth_classification()
    Xt, yt, wt, Xc, yc, wc = _split(X, y, w)
    yc_one = pd.Series(np.zeros(len(yc), dtype=int), index=yc.index)
    yc_one.iloc[0] = 1
    with pytest.warns(RuntimeWarning, match="skipping calibration"):
        clf = fit_calibrated("lr", Xt, yt, wt, Xc, yc_one, wc)
    assert clf.predict_proba(Xc).shape == (len(Xc), 2)


def test_fit_calibrated_single_class_train_slice_degrades():
    """Single-class TRAIN slice: no model can be fit — constant majority
    predictor, not a crash."""
    X, y, w = _synth_classification()
    Xt, yt, wt, Xc, yc, wc = _split(X, y, w)
    yt_single = pd.Series(np.zeros(len(yt), dtype=int), index=yt.index)
    with pytest.warns(RuntimeWarning, match="single-class"):
        clf = fit_calibrated("lr", Xt, yt_single, wt, Xc, yc, wc)
    p = clf.predict_proba(Xc)
    assert (p[:, 1] == 0.0).all()


# ---------------------------------------------------------------------------
# fit_calibrated
# ---------------------------------------------------------------------------
def test_fit_calibrated_default_is_sigmoid_attribute():
    """No method arg -> the fitted calibrator uses sigmoid (not isotonic)."""
    X, y, w = _synth_classification()
    Xt, yt, wt, Xc, yc, wc = _split(X, y, w)
    clf = fit_calibrated("lr", Xt, yt, wt, Xc, yc, wc)
    assert _fitted_method(clf) == "sigmoid"


def test_fit_calibrated_default_produces_continuous_probs():
    """Behavioral signature: sigmoid -> many distinct probabilities; isotonic
    would pile predictions onto a few plateau values."""
    X, y, w = _synth_classification()
    Xt, yt, wt, Xc, yc, wc = _split(X, y, w)
    clf = fit_calibrated("lr", Xt, yt, wt, Xc, yc, wc)
    probs = clf.predict_proba(Xc)[:, 1]
    n_unique = len(np.unique(np.round(probs, 6)))
    # Sigmoid maps each distinct decision score to a distinct probability, so
    # the count should be close to n_samples. Isotonic would collapse to a
    # handful of plateaus. A generous floor still separates the two regimes.
    assert n_unique > len(probs) * 0.5


def test_fit_calibrated_isotonic_override_respected():
    """Explicit method='isotonic' still forces isotonic."""
    X, y, w = _synth_classification()
    Xt, yt, wt, Xc, yc, wc = _split(X, y, w)
    clf = fit_calibrated("lr", Xt, yt, wt, Xc, yc, wc, method="isotonic")
    assert _fitted_method(clf) == "isotonic"


def test_fit_calibrated_auto_resolves_isotonic_when_minority_high():
    """auto -> isotonic when calib-set minority >= isotonic_min_minority."""
    X, y, w = _synth_classification(n=600, minority_frac=0.45)
    Xt, yt, wt, Xc, yc, wc = _split(X, y, w)
    minority = int(min((yc == 0).sum(), (yc == 1).sum()))
    assert minority >= 50  # precondition for this branch
    clf = fit_calibrated("lr", Xt, yt, wt, Xc, yc, wc, method="auto", isotonic_min_minority=50)
    assert _fitted_method(clf) == "isotonic"


def test_fit_calibrated_auto_resolves_sigmoid_when_minority_low():
    """auto -> sigmoid when calib-set minority < isotonic_min_minority."""
    X, y, w = _synth_classification(n=400, minority_frac=0.45)
    Xt, yt, wt, Xc, yc, wc = _split(X, y, w)
    minority = int(min((yc == 0).sum(), (yc == 1).sum()))
    # Force the low-minority branch via a high threshold regardless of split.
    clf = fit_calibrated(
        "lr", Xt, yt, wt, Xc, yc, wc,
        method="auto", isotonic_min_minority=minority + 1000,
    )
    assert _fitted_method(clf) == "sigmoid"


def test_fit_calibrated_rejects_bad_method():
    X, y, w = _synth_classification()
    Xt, yt, wt, Xc, yc, wc = _split(X, y, w)
    with pytest.raises(ValueError):
        fit_calibrated("lr", Xt, yt, wt, Xc, yc, wc, method="bogus")


# ---------------------------------------------------------------------------
# build_calibrated_classifier (UNFITTED builder)
# ---------------------------------------------------------------------------
def test_build_calibrated_classifier_default_is_sigmoid():
    clf = build_calibrated_classifier("lr")
    assert clf.method == "sigmoid"


def test_build_calibrated_classifier_isotonic_override():
    clf = build_calibrated_classifier("lr", method="isotonic")
    assert clf.method == "isotonic"


def test_build_calibrated_classifier_auto_treated_as_sigmoid():
    """'auto' on the UNFITTED builder cannot resolve minority (no calib data),
    so it is treated as sigmoid — the safe default per the CLAUDE.md invariant."""
    clf = build_calibrated_classifier("lr", method="auto")
    assert clf.method == "sigmoid"


def test_build_calibrated_classifier_rejects_bad_method():
    with pytest.raises(ValueError):
        build_calibrated_classifier("lr", method="bogus")
