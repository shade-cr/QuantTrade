"""Stack decision + LogReg meta-learner with nested WF.

The decision criteria are evaluated ex ante: before looking at the meta-learner's
predictions, two filters gate whether stacking is allowed at all.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class StackDecision:
    stack: bool
    n_models_passing: int
    max_pair_corr: float
    reason: str


def should_stack(
    sharpe_per_fold_per_model: dict[str, list[float]],
    sharpe_baseline_per_fold: list[float],
    oof_corr: np.ndarray,
    n_trades_per_fold_per_model: dict[str, list[int]] | None = None,
    min_models: int = 2,
    min_folds: int = 3,
    max_corr: float = 0.7,
    min_trades_per_fold: int = 30,
) -> StackDecision:
    """Apply the stack-gate criteria. Returns a `StackDecision` with provenance.

    A fold only counts toward "beats baseline" if it has at least
    `min_trades_per_fold` trades AND a finite (non-NaN) Sharpe. Without this
    gate, a model that takes 3 trades all-winners can earn a Sharpe in the
    tens that has no statistical meaning. (See Phase 1 XAU D1 report —
    `momentum_zscore` had models with 1–3 trades reporting Sharpes of 31–62.)
    """
    n_passing = 0
    for model, sharpes in sharpe_per_fold_per_model.items():
        trades = (n_trades_per_fold_per_model.get(model, [None] * len(sharpes))
                  if n_trades_per_fold_per_model is not None
                  else [None] * len(sharpes))
        beats = 0
        for s, b, n in zip(sharpes, sharpe_baseline_per_fold, trades):
            if n is not None and n < min_trades_per_fold:
                continue
            if not np.isfinite(s) or not np.isfinite(b):
                continue
            if s > b:
                beats += 1
        if beats >= min_folds:
            n_passing += 1
    if n_passing < min_models:
        return StackDecision(
            stack=False,
            n_models_passing=n_passing,
            max_pair_corr=float(_max_offdiag(oof_corr)),
            reason=(f"competence: only {n_passing}/{len(sharpe_per_fold_per_model)} "
                    f"models beat baseline in ≥{min_folds}/4 folds with ≥{min_trades_per_fold} "
                    f"trades each (need ≥{min_models})"),
        )
    pair_corr = _max_offdiag(oof_corr)
    if pair_corr >= max_corr:
        return StackDecision(
            stack=False,
            n_models_passing=n_passing,
            max_pair_corr=float(pair_corr),
            reason=f"correlation: max pairwise OOF corr = {pair_corr:.3f} ≥ {max_corr}",
        )
    return StackDecision(
        stack=True,
        n_models_passing=n_passing,
        max_pair_corr=float(pair_corr),
        reason=f"passed: {n_passing} models compete, max corr {pair_corr:.3f} < {max_corr}",
    )


def _max_offdiag(corr: np.ndarray) -> float:
    if corr.shape[0] != corr.shape[1] or corr.ndim != 2:
        raise ValueError("corr must be square 2-D")
    m = corr.copy()
    np.fill_diagonal(m, -np.inf)
    return float(m.max())


def fit_meta_nested_wf(
    oof_probs: pd.DataFrame,            # rows=events, cols=model names (probabilities)
    extra_features: pd.DataFrame | None,  # e.g. rv_regime as side feature; same index
    y: pd.Series,
    n_folds: int = 4,
    purge: int = 20,
    embargo_pct: float = 0.01,
    C: float = 1.0,
) -> tuple[pd.Series, list[LogisticRegression]]:
    """Run nested 4-fold expanding-window WF on the OOF probs and return a meta-OOF series.

    Each sub-train fits LogisticRegression(C, l2) on probs(+extra). Each sub-test
    yields meta-OOF predictions. The full meta-OOF series (over all events) is the
    honest OOS evaluation of the stack.
    """
    from pipeline.walk_forward import make_folds  # local import to avoid cycles
    n = len(oof_probs)
    folds = make_folds(n, n_folds=n_folds, train_min=n // (n_folds + 1),
                       purge=purge, embargo_pct=embargo_pct)
    meta_oof = pd.Series(np.nan, index=oof_probs.index)
    models: list[LogisticRegression] = []
    X = oof_probs.copy()
    if extra_features is not None:
        X = pd.concat([X, extra_features], axis=1)
    for f in folds:
        m = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=500)
        m.fit(X.iloc[f.train_idx], y.iloc[f.train_idx])
        meta_oof.iloc[f.test_idx] = m.predict_proba(X.iloc[f.test_idx])[:, 1]
        models.append(m)
    return meta_oof, models
