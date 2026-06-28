"""Tests for pipeline.feature_importance.

Covers the original MDA permutation importance plus the Clustered Feature
Importance (MLfAM Ch6 §6.5.2, Snippet 6.5 / B0131) and the reusable
neg-log-loss scorer (AFML §8.3 / MLfAM Snippet 6.3 / B0130).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from pipeline.feature_importance import (
    mda_importance,
    neg_log_loss_scorer,
    neg_log_loss_estimator_scorer,
    cluster_features,
    clustered_mda_importance,
    aggregate_clustered_mda_across_folds,
)


def _scoring(y_true: np.ndarray, y_prob: np.ndarray, sample_weight=None) -> float:
    return accuracy_score(y_true, (y_prob >= 0.5).astype(int), sample_weight=sample_weight)


# --------------------------------------------------------------------------- #
# Existing MDA tests (must keep passing)
# --------------------------------------------------------------------------- #
def test_dominant_feature_has_highest_mda():
    rng = np.random.default_rng(0)
    n = 600
    x0 = rng.normal(size=n)
    y = (x0 > 0).astype(int)  # x0 fully determines the label
    noise = rng.normal(size=(n, 4))
    X = pd.DataFrame(np.column_stack([x0, noise]),
                     columns=["signal", "n1", "n2", "n3", "n4"])
    rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X.iloc[:400], y[:400])
    imp = mda_importance(rf, X.iloc[400:], pd.Series(y[400:]), _scoring,
                         n_repeats=5, random_state=42)
    assert imp["signal"] == max(imp.values())
    for col in ("n1", "n2", "n3", "n4"):
        assert imp[col] < imp["signal"] / 2


def test_mda_returns_dict_of_floats_keyed_by_columns():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    y = pd.Series(rng.integers(0, 2, size=100))
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    imp = mda_importance(rf, X, y, _scoring, n_repeats=3, random_state=1)
    assert set(imp.keys()) == {"a", "b", "c"}
    assert all(isinstance(v, float) for v in imp.values())


def test_mda_n_repeats_reduces_variance():
    """More repeats → smaller variance in the reported importance (sanity)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] > 0).astype(int).values)
    rf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    imps_1 = [mda_importance(rf, X, y, _scoring, n_repeats=1, random_state=k)["a"] for k in range(10)]
    imps_20 = [mda_importance(rf, X, y, _scoring, n_repeats=20, random_state=k)["a"] for k in range(10)]
    assert np.std(imps_20) <= np.std(imps_1)


def test_mda_propagates_sample_weight():
    """If sample_weight is plumbed through, drastically different weights must change MDA."""
    rng = np.random.default_rng(0)
    n = 600
    x0 = rng.normal(size=n)
    y = (x0 > 0).astype(int)
    X = pd.DataFrame({"signal": x0, "noise": rng.normal(size=n)})
    rf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X.iloc[:400], y[:400])
    w_uniform = np.ones(200, dtype=float)
    w_skewed = np.where(y[400:] == 1, 100.0, 1.0).astype(float)
    imp_u = mda_importance(rf, X.iloc[400:], pd.Series(y[400:]), _scoring,
                           sample_weight=w_uniform, n_repeats=5, random_state=42)
    imp_s = mda_importance(rf, X.iloc[400:], pd.Series(y[400:]), _scoring,
                           sample_weight=w_skewed, n_repeats=5, random_state=42)
    assert imp_u["signal"] != imp_s["signal"], "sample_weight did not propagate into MDA scoring"


# --------------------------------------------------------------------------- #
# B0130: neg_log_loss_scorer
# --------------------------------------------------------------------------- #
def test_neg_log_loss_perfect_probs_near_zero():
    y = np.array([0, 1, 1, 0, 1])
    # near-perfect confident probs for the true class
    p = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    score = neg_log_loss_scorer(y, p)
    assert score <= 0.0
    assert score > -1e-6, f"perfect probs should give ~0, got {score}"


def test_neg_log_loss_worse_probs_more_negative():
    y = np.array([0, 1, 1, 0, 1])
    good = np.array([0.05, 0.95, 0.95, 0.05, 0.95])
    bad = np.array([0.45, 0.55, 0.55, 0.45, 0.55])
    worse = np.array([0.9, 0.1, 0.1, 0.9, 0.1])  # confidently wrong
    s_good = neg_log_loss_scorer(y, good)
    s_bad = neg_log_loss_scorer(y, bad)
    s_worse = neg_log_loss_scorer(y, worse)
    assert s_good > s_bad > s_worse


def test_neg_log_loss_handles_exact_0_and_1():
    """Exact 0/1 probs (even when wrong) must not produce inf/nan due to clipping."""
    y = np.array([1, 0])
    p = np.array([0.0, 1.0])  # confidently WRONG on both -> finite but very negative
    score = neg_log_loss_scorer(y, p)
    assert np.isfinite(score)
    assert score < -10.0


def test_neg_log_loss_respects_sample_weight():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.4, 0.6, 0.6, 0.4])
    w_uniform = np.ones(4)
    w_skewed = np.array([10.0, 1.0, 1.0, 10.0])
    s_u = neg_log_loss_scorer(y, p, sample_weight=w_uniform)
    s_s = neg_log_loss_scorer(y, p, sample_weight=w_skewed)
    assert s_u != s_s, "sample_weight not plumbed into neg_log_loss"


# --------------------------------------------------------------------------- #
# B0156: neg_log_loss_estimator_scorer — single-class validation slices
# --------------------------------------------------------------------------- #
def _fit_small_clf(y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(len(y_train), 3))
    if len(np.unique(y_train)) < 2:
        clf = DummyClassifier(strategy="prior")
    else:
        clf = LogisticRegression()
    return clf.fit(X, y_train), X


def test_estimator_scorer_single_class_validation_slice_is_finite():
    """The B0156 crash: string scoring='neg_log_loss' raises when the inner-CV
    validation slice is single-class. The callable scorer must score finitely."""
    clf, _ = _fit_small_clf(np.array([0, 1, 0, 1, 0, 1, 0, 1]))
    rng = np.random.default_rng(1)
    X_val = rng.normal(size=(5, 3))
    for single in (np.zeros(5, dtype=int), np.ones(5, dtype=int)):
        score = neg_log_loss_estimator_scorer(clf, X_val, single)
        assert np.isfinite(score) and score <= 0.0


def test_estimator_scorer_single_class_estimator_is_finite():
    """Estimator that never saw class 1 (predict_proba has one column)."""
    clf, _ = _fit_small_clf(np.zeros(8, dtype=int))
    rng = np.random.default_rng(2)
    X_val = rng.normal(size=(4, 3))
    score = neg_log_loss_estimator_scorer(clf, X_val, np.array([0, 1, 0, 1]))
    assert np.isfinite(score)


def test_estimator_scorer_matches_metric_scorer_on_two_class_data():
    clf, _ = _fit_small_clf(np.array([0, 1, 0, 1, 0, 1, 0, 1]))
    rng = np.random.default_rng(3)
    X_val = rng.normal(size=(6, 3))
    y_val = np.array([0, 1, 1, 0, 1, 0])
    p1 = clf.predict_proba(X_val)[:, list(clf.classes_).index(1)]
    assert neg_log_loss_estimator_scorer(clf, X_val, y_val) == pytest.approx(
        neg_log_loss_scorer(y_val, p1))


# --------------------------------------------------------------------------- #
# B0131: cluster_features
# --------------------------------------------------------------------------- #
def test_cluster_duplicate_columns_share_cluster():
    rng = np.random.default_rng(0)
    n = 300
    f1 = rng.normal(size=n)
    X = pd.DataFrame({
        "f1": f1,
        "f1_dup": f1.copy(),           # perfectly correlated duplicate
        "g": rng.normal(size=n),
        "h": rng.normal(size=n),
    })
    clusters = cluster_features(X, random_state=0)
    # find cluster containing f1
    cid = next(cid for cid, members in clusters.items() if "f1" in members)
    assert "f1_dup" in clusters[cid], "perfect duplicates must land in the same cluster"


def test_cluster_uncorrelated_columns_separate():
    rng = np.random.default_rng(1)
    n = 400
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
    })
    clusters = cluster_features(X, random_state=0)
    # every column assigned exactly once, full coverage
    all_members = [m for members in clusters.values() for m in members]
    assert sorted(all_members) == ["a", "b", "c"]
    assert len(all_members) == len(set(all_members))


def test_cluster_constant_column_is_singleton():
    rng = np.random.default_rng(2)
    n = 200
    f1 = rng.normal(size=n)
    X = pd.DataFrame({
        "f1": f1,
        "f1_dup": f1.copy(),
        "const": np.full(n, 3.14),
    })
    clusters = cluster_features(X, random_state=0)
    const_cid = next(cid for cid, members in clusters.items() if "const" in members)
    assert clusters[const_cid] == ["const"], "constant column must be its own singleton cluster"


# --------------------------------------------------------------------------- #
# B0131: clustered_mda_importance — the substitution-resolution headline
# --------------------------------------------------------------------------- #
def _build_substitution_dataset(seed=0, n=800):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    y = (f1 + 0.25 * rng.normal(size=n) > 0).astype(int)
    X = pd.DataFrame({
        "f1": f1,
        "f2": f1.copy(),                  # perfect duplicate of the informative feature
        "noise1": rng.normal(size=n),
        "noise2": rng.normal(size=n),
    })
    return X, pd.Series(y)


def test_cfi_resolves_substitution_effect():
    """Two perfectly correlated informative features split MDA credit individually,
    but their shared cluster recaptures the full importance (MLfAM §6.5.2)."""
    X, y = _build_substitution_dataset(seed=0)
    Xtr, ytr = X.iloc[:500], y.iloc[:500]
    Xte, yte = X.iloc[500:], y.iloc[500:]
    rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)

    per_feat = mda_importance(rf, Xte, yte, neg_log_loss_scorer, n_repeats=10, random_state=7)

    clusters = cluster_features(X, random_state=0)
    # f1 and f2 must be clustered together
    cid_f1 = next(cid for cid, m in clusters.items() if "f1" in m)
    assert "f2" in clusters[cid_f1]

    clustered = clustered_mda_importance(rf, Xte, yte, neg_log_loss_scorer, clusters,
                                         n_repeats=10, random_state=7)
    cluster_imp = clustered[cid_f1]

    # Substitution effect: each individual feature under-rates because shuffling one
    # leaves the duplicate to compensate.
    indiv_max = max(per_feat["f1"], per_feat["f2"])
    assert cluster_imp > 1.5 * indiv_max, (
        f"clustered importance ({cluster_imp:.4f}) should dominate either individual "
        f"MDA (f1={per_feat['f1']:.4f}, f2={per_feat['f2']:.4f})"
    )


def test_clustered_mda_one_entry_per_cluster():
    X, y = _build_substitution_dataset(seed=1)
    rf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X.iloc[:500], y.iloc[:500])
    clusters = cluster_features(X, random_state=0)
    clustered = clustered_mda_importance(rf, X.iloc[500:], y.iloc[500:],
                                         neg_log_loss_scorer, clusters, n_repeats=3, random_state=1)
    assert set(clustered.keys()) == set(clusters.keys())
    assert all(isinstance(v, float) for v in clustered.values())


def test_clustered_mda_propagates_sample_weight():
    X, y = _build_substitution_dataset(seed=2)
    Xte, yte = X.iloc[500:], y.iloc[500:]
    rf = RandomForestClassifier(n_estimators=80, random_state=0).fit(X.iloc[:500], y.iloc[:500])
    clusters = cluster_features(X, random_state=0)
    w_uniform = np.ones(len(yte))
    w_skewed = np.where(yte.to_numpy() == 1, 50.0, 1.0).astype(float)
    cid = next(cid for cid, m in clusters.items() if "f1" in m)
    c_u = clustered_mda_importance(rf, Xte, yte, neg_log_loss_scorer, clusters,
                                   sample_weight=w_uniform, n_repeats=5, random_state=3)
    c_s = clustered_mda_importance(rf, Xte, yte, neg_log_loss_scorer, clusters,
                                   sample_weight=w_skewed, n_repeats=5, random_state=3)
    assert c_u[cid] != c_s[cid], "sample_weight not plumbed into clustered MDA"


# --------------------------------------------------------------------------- #
# B0131: aggregate_clustered_mda_across_folds
# --------------------------------------------------------------------------- #
def test_aggregate_clustered_mda_keys_by_members():
    # Cluster ids differ across folds but members are stable -> canonical key.
    fold0 = {0: 0.5, 1: 0.1}
    fold1 = {7: 0.3, 9: 0.2}  # different ids, same member sets
    labels = {0: ["f1", "f2"], 1: ["noise"]}
    labels2 = {7: ["f1", "f2"], 9: ["noise"]}
    agg = aggregate_clustered_mda_across_folds([fold0, fold1], cluster_labels=labels)
    # without per-fold labels for fold1 we still need a deterministic story; pass merged labels
    agg2 = aggregate_clustered_mda_across_folds(
        [fold0, fold1],
        cluster_labels={**labels, **labels2},
    )
    # f1/f2 cluster aggregates across both folds: mean of 0.5 and 0.3
    key = next(k for k, v in agg2.items() if set(v["members"]) == {"f1", "f2"})
    assert agg2[key]["mean"] == 0.4
    assert agg2[key]["std"] >= 0.0
    assert set(agg2[key]["members"]) == {"f1", "f2"}
    # smoke check the labels-only-for-fold0 path returns structured dicts
    assert all({"mean", "std", "members"} <= set(v.keys()) for v in agg.values())
