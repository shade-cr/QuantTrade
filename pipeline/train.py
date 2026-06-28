"""Model factories + calibrated fit with chronological prefit holdout."""
from __future__ import annotations
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def _make_xgb(random_state: int = 42, **params):
    from xgboost import XGBClassifier
    defaults = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=1,  # B0064: single-threaded -> deterministic 'hist' (multithread float-reduce is not bit-reproducible)
        random_state=random_state,
    )
    defaults.update(params)
    return XGBClassifier(**defaults)


def _make_catboost(random_state: int = 42, **params):
    from catboost import CatBoostClassifier
    defaults = dict(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5,
        loss_function="Logloss",
        verbose=False,
        random_seed=random_state,
        thread_count=-1,
    )
    defaults.update(params)
    return CatBoostClassifier(**defaults)


def _make_rf(random_state: int = 42, **params):
    from sklearn.ensemble import RandomForestClassifier
    defaults = dict(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )
    defaults.update(params)
    return RandomForestClassifier(**defaults)


def _make_lgbm(random_state: int = 42, **params):
    """LightGBM classifier — leaf-wise tree growth gives genuine diversity vs XGB.

    B0053: replaces CatBoost in the active model config. CatBoost factory
    remains for backward compat but is removed from configs/xau_d1.yaml.
    Key differences from XGB: leaf-wise (vs level-wise) growth, histogram-
    based binning, faster training, different regularization surface.
    verbose=-1 suppresses per-iteration output that floods fold logs.
    """
    from lightgbm import LGBMClassifier
    defaults = dict(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_lambda=1.0,
        objective="binary",
        n_jobs=1,  # B0064: single-threaded -> deterministic leaf-wise boosting (reproducible baseline)
        random_state=random_state,
        verbose=-1,
    )
    defaults.update(params)
    return LGBMClassifier(**defaults)


def _make_lr(random_state: int = 42, **params):
    """Logistic Regression (L2) — linear base learner for stack diversity.

    B0053: trees approximate linear signals noisily; a dedicated linear model
    captures them cleanly and is typically least-correlated with tree ensembles,
    improving the stacking gate (max_oof_corr < 0.7). C=0.1 = stronger L2
    regularization appropriate for high-dimensional tier2 feature spaces.
    sample_weight is natively supported by sklearn's LogisticRegression.fit().
    """
    from sklearn.linear_model import LogisticRegression
    defaults = dict(
        C=0.1,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )
    defaults.update(params)
    return LogisticRegression(**defaults)


MODEL_FACTORIES: dict[str, Callable] = {
    "xgb": _make_xgb,
    "catboost": _make_catboost,   # kept for backward compat; not in active config (B0053)
    "lgbm": _make_lgbm,
    "rf": _make_rf,
    "lr": _make_lr,
}


def build_calibrated_classifier(
    model_name: str,
    base_kwargs: dict | None = None,
    random_state: int = 42,
    method: str = "sigmoid",
):
    """Return an UNFITTED CalibratedClassifierCV using prefit semantics (FrozenEstimator).

    The base estimator must be fitted by the caller before using this wrapper.
    Uses FrozenEstimator (sklearn 1.4+ equivalent of cv='prefit') to avoid KFold
    temporal leakage.

    `method` controls the calibration mapping and DEFAULTS TO "sigmoid" (B0133).
    This hardens the CLAUDE.md invariant: calibration must default to sigmoid
    (Platt), NOT isotonic. Isotonic produces step-function probabilities that
    pile onto a few plateaus, collapsing per-fold threshold selection
    (every threshold in [0.55, 0.65] selects the same trade set; pct_signals_kept
    dropped below 5%). A dropped config key must NOT silently fall back to
    isotonic.

    "auto" is accepted for signature symmetry with fit_calibrated, but this is
    an UNFITTED builder with no calibration data, so the minority count that
    "auto" depends on cannot be resolved here. "auto" is therefore treated as
    "sigmoid" (the safe default). Callers that genuinely want the minority-based
    auto resolution must use fit_calibrated, which has the calibration set.
    """
    if method not in ("isotonic", "sigmoid", "auto"):
        raise ValueError(f"calibration method must be 'auto'|'isotonic'|'sigmoid', got {method!r}")
    if method == "auto":
        # No calibration data here to count minority -> fall back to the safe default.
        method = "sigmoid"
    base = MODEL_FACTORIES[model_name](random_state=random_state, **(base_kwargs or {}))
    # FrozenEstimator is the sklearn 1.4+ replacement for cv='prefit':
    # it prevents CalibratedClassifierCV from re-fitting the base estimator.
    return CalibratedClassifierCV(estimator=FrozenEstimator(base), method=method)


def fit_calibrated(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: np.ndarray,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    w_calib: np.ndarray,
    random_state: int = 42,
    isotonic_min_minority: int = 50,
    base_kwargs: dict | None = None,
    method: str = "sigmoid",
):
    """Fit the base on (X_train, y_train, w_train), then calibrate on (X_calib, y_calib, w_calib).

    `method` controls the calibration mapping. The DEFAULT IS "sigmoid" (B0133),
    the safe value per the CLAUDE.md invariant: calibration must default to
    sigmoid (Platt), NOT isotonic. Isotonic produces step-function probabilities
    that pile onto a few plateaus, collapsing per-fold threshold selection
    (every threshold in [0.55, 0.65] selects the same trade set; pct_signals_kept
    dropped below 5%). A dropped config key must NOT silently fall back to
    isotonic — hence the default is sigmoid, not "auto".

      - "sigmoid" (DEFAULT): force Platt scaling. Produces continuous
        probabilities so the threshold grid actually differentiates trades.
      - "isotonic": force isotonic regardless of minority count.
      - "auto": isotonic if calibration-set minority count ≥ isotonic_min_minority,
        else fall back to sigmoid (Platt). Kept for callers that explicitly opt
        in; no longer the default. The full auto/isotonic/sigmoid resolution
        below is unchanged — only the default value of `method` moved to sigmoid.
    """
    if method == "auto":
        minority = int(min((y_calib == 0).sum(), (y_calib == 1).sum()))
        method = "isotonic" if minority >= isotonic_min_minority else "sigmoid"
    elif method not in ("isotonic", "sigmoid"):
        raise ValueError(f"calibration method must be 'auto'|'isotonic'|'sigmoid', got {method!r}")

    # B0032/B0156 family — graceful degradation for degenerate slices, same
    # policy as RefittingCalibratedPipeline.fit below. Selective primaries on
    # regime-gated folds (e.g. F008 short-only on BEAR_QUIET) can hand this a
    # single-class train slice or a single-class / 1-minority calibration
    # holdout; CalibratedClassifierCV(FrozenEstimator).fit then crashes inside
    # cross_val_predict (_enforce_prediction_order IndexError). Skip what
    # cannot be fit honestly instead of crashing the whole audit.
    if len(np.unique(y_train)) < 2:
        import warnings
        warnings.warn(
            "fit_calibrated: train slice is single-class; returning a constant "
            "majority-class predictor (no model can be fit).",
            RuntimeWarning, stacklevel=2,
        )
        return _ConstantProbaClassifier(int(y_train.iloc[0] if hasattr(y_train, "iloc")
                                            else y_train[0]))

    base = MODEL_FACTORIES[model_name](random_state=random_state, **(base_kwargs or {}))
    # XGBoost and CatBoost both accept sample_weight in fit().
    if model_name == "catboost":
        base.fit(X_train, y_train, sample_weight=w_train)
    else:
        base.fit(X_train, y_train, sample_weight=w_train)

    n_minority_calib = int(min((y_calib == 0).sum(), (y_calib == 1).sum()))
    if n_minority_calib < 2:
        import warnings
        warnings.warn(
            f"fit_calibrated: calibration slice has {n_minority_calib} "
            f"minority-class sample(s); skipping calibration and returning "
            f"uncalibrated base probabilities.",
            RuntimeWarning, stacklevel=2,
        )
        return _UncalibratedBaseWrapper(base)

    # FrozenEstimator (sklearn 1.4+) is the replacement for cv='prefit':
    # prevents CalibratedClassifierCV from re-fitting the base on calibration data.
    calib = CalibratedClassifierCV(estimator=FrozenEstimator(base), method=method)
    calib.fit(X_calib, y_calib, sample_weight=w_calib)
    return calib


class _ConstantProbaClassifier:
    """Degenerate fallback: train slice had one class; predicts it with p=1."""

    def __init__(self, majority_class: int):
        self.majority_class = int(majority_class)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        n = len(X)
        p1 = np.full(n, 1.0 if self.majority_class == 1 else 0.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.majority_class)


class _UncalibratedBaseWrapper:
    """Degenerate fallback: calibration slice unusable; raw base probabilities.

    Normalizes predict_proba to two columns ordered [P(0), P(1)] regardless of
    the base's classes_ layout, so downstream `[:, 1]` indexing stays correct.
    """

    def __init__(self, base):
        self.base_ = base
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        p = np.asarray(self.base_.predict_proba(X))
        classes = list(getattr(self.base_, "classes_", [0, 1]))
        p1 = p[:, classes.index(1)] if 1 in classes else np.zeros(p.shape[0])
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Phase 2 — RefittingCalibratedPipeline (T9.A.1)
# ---------------------------------------------------------------------------
class RefittingCalibratedPipeline(BaseEstimator, ClassifierMixin):
    """sklearn-compatible estimator that refits base + calibration per .fit().

    Designed for `pipeline.walk_forward.inner_oof_predict_proba` (manual CV
    loop with clone): each fold of the outer CV receives a cloned instance
    that trains from scratch on its own inner-train. NO leak — the base
    never sees inner-val.

    Why this exists instead of passing the existing
    `CalibratedClassifierCV(FrozenEstimator(base_fitted), 'sigmoid')` to the
    inner CV directly: `FrozenEstimator.fit()` is a no-op by design, so a
    cloned instance would keep its base trained on the outer_train (which
    includes inner_val rows) and only re-fit the sigmoid on inner_train.
    That is a leak. This wrapper sidesteps it by re-training the base from
    scratch in every .fit() call.

    Hyperparams (preserved by sklearn.clone()):
      - model_name: 'xgb' | 'catboost' | 'lgbm' | 'rf' | 'lr'
      - base_kwargs: dict passed to MODEL_FACTORIES[model_name] (None → {})
      - calib_holdout_pct: float, chronological tail fraction for calibration
      - method: 'sigmoid' | 'isotonic' | 'auto'
      - random_state: int
      - isotonic_min_minority: int (used when method='auto')

    Fitted state (NOT preserved by clone — fresh on every .fit()):
      - calibrator_: CalibratedClassifierCV
      - classes_:    delegated from calibrator
    """

    def __init__(self, model_name, base_kwargs=None, calib_holdout_pct=0.15,
                 method="sigmoid", random_state=42, isotonic_min_minority=50):
        self.model_name = model_name
        self.base_kwargs = base_kwargs
        self.calib_holdout_pct = calib_holdout_pct
        self.method = method
        self.random_state = random_state
        self.isotonic_min_minority = isotonic_min_minority

    def fit(self, X, y, sample_weight=None):
        n = len(X)
        hold = max(int(n * self.calib_holdout_pct), 1)
        if hold >= n:
            raise ValueError(
                f"calib_holdout_pct={self.calib_holdout_pct} consumes all {n} rows"
            )
        # Chronological tail split — NO shuffling.
        slc_base = slice(0, n - hold)
        slc_calib = slice(n - hold, n)
        X_base = X.iloc[slc_base] if hasattr(X, "iloc") else X[slc_base]
        X_calib = X.iloc[slc_calib] if hasattr(X, "iloc") else X[slc_calib]
        y_base = y.iloc[slc_base] if hasattr(y, "iloc") else y[slc_base]
        y_calib = y.iloc[slc_calib] if hasattr(y, "iloc") else y[slc_calib]
        if sample_weight is None:
            w_base, w_calib = None, None
        else:
            sw = np.asarray(sample_weight)
            w_base, w_calib = sw[slc_base], sw[slc_calib]

        # Both classes required in base AND calibration. If a fold's slice
        # is degenerate, propagate as ValueError so the caller surfaces it.
        # B0032: tighten the calibration guard to require n_minority >= 2.
        # sklearn's CalibratedClassifierCV(FrozenEstimator(...)) runs
        # cross_val_predict(cv=k) internally; even with k=2 the partition
        # needs >=2 samples of each class. n_minority=1 hard-crashes with
        # "n_splits=2 cannot be greater than the number of members in each
        # class" — empirically the H4-with-selectivity audit failure mode.
        if len(np.unique(y_base)) < 2 or len(np.unique(y_calib)) < 2:
            # Graceful degradation (same as n_minority < 2 below): single-class
            # slices occur in supervised-direct mode for strongly trending regimes
            # where all forward returns share the same sign. Skip calibration and
            # fall back to uncalibrated base model predictions.
            import warnings
            _n_base_cls = len(np.unique(y_base))
            _n_cal_cls  = len(np.unique(y_calib))
            warnings.warn(
                f"Calibration skipped: base slice has {_n_base_cls} class(es), "
                f"calib slice has {_n_cal_cls} class(es). "
                f"Returning uncalibrated base estimator probabilities.",
                RuntimeWarning, stacklevel=2,
            )
            self.calibrator_ = None
            self.classes_ = np.array([0, 1])
            if len(np.unique(y_base)) < 2:
                # Even the base training slice is single-class — cannot fit any
                # model. Return a dummy predictor that always predicts the
                # majority class with probability 1.0.
                majority_class = int(y_base.iloc[0] if hasattr(y_base, 'iloc') else y_base[0])
                self._dummy_proba = majority_class  # 0 or 1
                self.base_ = None
            else:
                base = MODEL_FACTORIES[self.model_name](
                    random_state=self.random_state, **(self.base_kwargs or {})
                )
                base.fit(X_base, y_base, sample_weight=w_base)
                self.base_ = base
            return self
        n_minority_calib = int(min((y_calib == 0).sum(), (y_calib == 1).sum()))
        if n_minority_calib < 2:
            # Graceful degradation for supervised-direct mode in trend regimes:
            # the calibration slice may have only 1 minority-class sample when
            # the regime strongly concentrates positive forward returns (e.g.
            # USDJPY/EURUSD BULL_QUIET). Skip sigmoid calibration and return the
            # base estimator's raw probabilities — less precise but not a crash.
            import warnings
            warnings.warn(
                f"Calibration skipped: only {n_minority_calib} minority-class "
                f"sample(s) in calibration slice. Returning uncalibrated "
                f"base estimator probabilities.",
                RuntimeWarning, stacklevel=2,
            )
            self.calibrator_ = None   # sentinel: predict_proba falls back to base_
            base = MODEL_FACTORIES[self.model_name](
                random_state=self.random_state, **(self.base_kwargs or {})
            )
            if self.model_name == "catboost":
                base.fit(X_base, y_base, sample_weight=w_base)
            elif self.model_name == "xgb":
                base.fit(X_base, y_base, sample_weight=w_base)
            else:
                base.fit(X_base, y_base, sample_weight=w_base)
            self.base_ = base
            self.classes_ = np.array([0, 1])
            return self

        method = self.method
        if method == "auto":
            minority = int(min((y_calib == 0).sum(), (y_calib == 1).sum()))
            method = "isotonic" if minority >= self.isotonic_min_minority else "sigmoid"
        elif method not in ("isotonic", "sigmoid"):
            raise ValueError(
                f"calibration method must be 'auto'|'isotonic'|'sigmoid', got {method!r}"
            )

        # Build + fit base FRESH (the leak-killer move).
        base = MODEL_FACTORIES[self.model_name](
            random_state=self.random_state, **(self.base_kwargs or {})
        )
        base.fit(X_base, y_base, sample_weight=w_base)

        # Calibrate via FrozenEstimator (sklearn 1.4+ replacement for cv='prefit').
        # B0032: cap cv at the minority-class count to keep StratifiedKFold
        # feasible on small calibration slices (n_minority_calib is computed
        # above as part of the guard).
        calib_cv = max(2, min(5, n_minority_calib))
        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(base), method=method, cv=calib_cv,
        )
        calibrator.fit(X_calib, y_calib, sample_weight=w_calib)

        self.calibrator_ = calibrator
        self.classes_ = calibrator.classes_
        return self

    def predict_proba(self, X):
        # calibrator_ is None when calibration was skipped (single-class calib slice)
        if self.calibrator_ is None:
            if self.base_ is None:
                # Dummy predictor: entire training fold was single-class
                n = len(X)
                c = getattr(self, "_dummy_proba", 1)
                proba = np.zeros((n, 2))
                proba[:, c] = 1.0
                return proba
            proba = self.base_.predict_proba(X)
            if proba.ndim == 1 or proba.shape[1] == 1:
                proba = np.column_stack([1 - proba[:, 0], proba[:, 0]])
            return proba
        return self.calibrator_.predict_proba(X)

    def predict(self, X):
        if self.calibrator_ is None:
            if self.base_ is None:
                c = getattr(self, "_dummy_proba", 1)
                return np.full(len(X), c, dtype=int)
            return self.base_.predict(X)
        return self.calibrator_.predict(X)
