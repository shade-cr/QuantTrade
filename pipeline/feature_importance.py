"""Mean Decrease Accuracy (permutation) feature importance.

Used per fold of the outer WF: a model fitted on train_fold is evaluated on test_fold;
each feature column is shuffled in turn, and the score drop is averaged over n_repeats.
This respects temporal validity because permutation happens entirely within the test fold.
"""
from __future__ import annotations
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def mda_importance(
    fitted_estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: Callable[..., float],
    sample_weight: np.ndarray | None = None,
    n_repeats: int = 5,
    random_state: int | None = None,
) -> dict[str, float]:
    """Compute permutation importance on a single test fold for a fitted estimator.

    AFML cap.4 sample weights (`mp_sample_tw`) are passed through to scoring so that
    high-overlap labels contribute proportionally less to the importance estimate.

    scoring signature: `scoring(y_true, y_prob, sample_weight=None) -> float`.

    Returns: `{column_name -> mean(baseline_score - permuted_score)}` across n_repeats shuffles.
    Higher value = more important. Negative values can occur for irrelevant features
    (interpret as ≈ 0).
    """
    rng = np.random.default_rng(random_state)
    baseline_proba = fitted_estimator.predict_proba(X_test)[:, 1]
    baseline = scoring(np.asarray(y_test), baseline_proba, sample_weight=sample_weight)
    out: dict[str, float] = {}
    for col in X_test.columns:
        drops: list[float] = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[col] = rng.permutation(X_perm[col].to_numpy())
            permuted_proba = fitted_estimator.predict_proba(X_perm)[:, 1]
            drops.append(baseline - scoring(np.asarray(y_test), permuted_proba, sample_weight=sample_weight))
        out[col] = float(np.mean(drops))
    return out


def aggregate_mda_across_folds(per_fold: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate per-fold MDA dicts into {column -> {'mean': μ, 'std': σ}} across folds."""
    cols = set().union(*[d.keys() for d in per_fold])
    out = {}
    for c in cols:
        values = np.array([d.get(c, 0.0) for d in per_fold], dtype=float)
        out[c] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
    return out


# --------------------------------------------------------------------------- #
# B0130: reusable neg-log-loss scorer (AFML §8.3 / MLfAM Snippet 6.3)
# --------------------------------------------------------------------------- #
def neg_log_loss_scorer(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Negative log-loss scorer for binary meta-labels.

    Returns ``-log_loss(y_true, y_prob, labels=[0, 1])``. Higher (closer to 0) is
    better, so it composes with ``mda_importance``'s ``baseline - permuted`` drop
    convention (a feature whose shuffle worsens log-loss yields a positive drop).

    AFML §8.3 / MLfAM Snippet 6.3 prefer neg-log-loss over accuracy@0.5 as the MDA
    scorer: accuracy is misleading on imbalanced meta-labels and ignores the
    classifier's confidence. This is B0130.

    ``y_prob`` is the P(class==1) vector. It is clipped to ``[1e-15, 1-1e-15]`` so
    that exact 0/1 probabilities (even when confidently wrong) stay finite.
    """
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1.0 - 1e-15)
    return float(-log_loss(np.asarray(y_true), p, labels=[0, 1], sample_weight=sample_weight))


# --------------------------------------------------------------------------- #
# B0156: search-CV scorer robust to single-class validation slices
# --------------------------------------------------------------------------- #
def neg_log_loss_estimator_scorer(estimator, X, y) -> float:
    """``RandomizedSearchCV``-style scorer (estimator, X, y) -> float.

    sklearn's string ``scoring="neg_log_loss"`` does NOT pass ``labels`` to
    ``log_loss``, so it raises when an inner-CV validation slice contains a
    single class — which dense H4 regime cells (e.g. BTC/ETH BEAR_QUIET, B0156)
    hit routinely under purged time-series splits. This wrapper extracts
    P(class==1) from ``estimator.classes_`` (0.0 if the estimator never saw
    class 1) and delegates to :func:`neg_log_loss_scorer`, which pins
    ``labels=[0, 1]`` and clips probabilities, so single-class slices score
    finitely instead of poisoning the search with NaN.
    """
    proba = np.asarray(estimator.predict_proba(X))
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if 1 in classes:
        p1 = proba[:, classes.index(1)]
    else:
        p1 = np.zeros(proba.shape[0])
    return neg_log_loss_scorer(np.asarray(y), p1)


# --------------------------------------------------------------------------- #
# B0131: clustered feature importance (MLfAM Ch6 §6.5.2, Snippet 6.5)
# --------------------------------------------------------------------------- #
def _correlation_distance(X: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Codependence distance matrix d_ij = sqrt(0.5 * (1 - corr_ij)) (MLfAM)."""
    corr = X[cols].corr().to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    # symmetrize against floating-point asymmetry; zero the diagonal
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


def cluster_features(
    X: pd.DataFrame,
    max_clusters: int | None = None,
    random_state: int | None = None,
) -> dict[int, list[str]]:
    """Cluster feature COLUMNS by correlation codependence distance (MLfAM §6.5.2).

    Distance is ``d = sqrt(0.5 * (1 - corr))``. Hierarchical (average-linkage)
    clustering via scipy is used; the number of clusters ``k`` is chosen by a
    deterministic silhouette sweep over ``k ∈ [2, min(10, n_var-1)]`` on the
    precomputed distance matrix. Constant / zero-variance columns (corr undefined)
    are removed before clustering and each is returned as its own singleton cluster.

    Two perfectly correlated features have ``d = 0`` and therefore always land in
    the same cluster.

    Returns ``{cluster_id: [colnames]}``. ``random_state`` is accepted for API
    stability; the procedure is deterministic.
    """
    cols = list(X.columns)
    # Separate constant / zero-variance columns: corr is undefined for them.
    variances = X.var(axis=0, ddof=0)
    var_cols = [c for c in cols if variances.get(c, 0.0) > 0.0 and np.isfinite(variances.get(c, 0.0))]
    const_cols = [c for c in cols if c not in var_cols]

    clusters: dict[int, list[str]] = {}
    next_id = 0

    if len(var_cols) == 1:
        clusters[next_id] = [var_cols[0]]
        next_id += 1
    elif len(var_cols) >= 2:
        dist = _correlation_distance(X, var_cols)
        labels = _hierarchical_labels(dist, max_clusters=max_clusters)
        for lab in sorted(set(labels.tolist())):
            members = [var_cols[i] for i in range(len(var_cols)) if labels[i] == lab]
            clusters[next_id] = members
            next_id += 1

    # Each constant column is its own singleton cluster.
    for c in const_cols:
        clusters[next_id] = [c]
        next_id += 1

    return clusters


def _hierarchical_labels(dist: np.ndarray, max_clusters: int | None = None) -> np.ndarray:
    """Return integer cluster labels for a precomputed distance matrix.

    Uses scipy average-linkage hierarchical clustering when available, choosing the
    cut that maximizes the mean silhouette over ``k ∈ [2, k_max]``. Falls back to
    sklearn ``AgglomerativeClustering`` with a precomputed metric if scipy is absent.
    """
    n = dist.shape[0]
    k_upper = min(10, n - 1)
    if max_clusters is not None:
        k_upper = min(k_upper, max_clusters)
    k_upper = max(k_upper, 2)

    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="average")
        best_labels = None
        best_score = -np.inf
        for k in range(2, k_upper + 1):
            labels = fcluster(Z, t=k, criterion="maxclust")
            if len(set(labels.tolist())) < 2:
                continue
            score = _mean_silhouette(dist, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
        if best_labels is None:
            best_labels = fcluster(Z, t=2, criterion="maxclust")
        return np.asarray(best_labels)
    except ImportError:  # pragma: no cover - scipy is a project dependency
        from sklearn.cluster import AgglomerativeClustering

        best_labels = None
        best_score = -np.inf
        for k in range(2, k_upper + 1):
            model = AgglomerativeClustering(
                n_clusters=k, metric="precomputed", linkage="average"
            )
            labels = model.fit_predict(dist)
            if len(set(labels.tolist())) < 2:
                continue
            score = _mean_silhouette(dist, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
        if best_labels is None:
            best_labels = AgglomerativeClustering(
                n_clusters=2, metric="precomputed", linkage="average"
            ).fit_predict(dist)
        return np.asarray(best_labels)


def _mean_silhouette(dist: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette score on a precomputed distance matrix (deterministic)."""
    from sklearn.metrics import silhouette_score

    if len(set(labels.tolist())) < 2:
        return -1.0
    try:
        return float(silhouette_score(dist, labels, metric="precomputed"))
    except ValueError:
        return -1.0


def clustered_mda_importance(
    fitted_estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: Callable[..., float],
    clusters: dict[int, list[str]],
    sample_weight: np.ndarray | None = None,
    n_repeats: int = 5,
    random_state: int | None = None,
) -> dict[int, float]:
    """Clustered MDA on a single test fold (MLfAM Snippet 6.5 / B0131).

    Instead of shuffling one feature at a time, every column belonging to a cluster
    is permuted TOGETHER: per repeat a single permutation index is drawn and applied
    to each member column, destroying the cluster's joint structure while leaving the
    rest of ``X_test`` intact. The score drop ``baseline - permuted`` is averaged over
    ``n_repeats``. This resolves the substitution effect that makes individual MDA
    under-rate mutually-redundant informative features.

    Permutation happens entirely within ``X_test`` (no refit), mirroring
    ``mda_importance``. Returns ``{cluster_id -> mean importance}``.
    """
    rng = np.random.default_rng(random_state)
    baseline_proba = fitted_estimator.predict_proba(X_test)[:, 1]
    baseline = scoring(np.asarray(y_test), baseline_proba, sample_weight=sample_weight)
    n = len(X_test)
    out: dict[int, float] = {}
    for cid, members in clusters.items():
        drops: list[float] = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            perm = rng.permutation(n)  # SAME index for every column in the cluster
            for col in members:
                X_perm[col] = X_test[col].to_numpy()[perm]
            permuted_proba = fitted_estimator.predict_proba(X_perm)[:, 1]
            drops.append(baseline - scoring(np.asarray(y_test), permuted_proba, sample_weight=sample_weight))
        out[cid] = float(np.mean(drops))
    return out


def aggregate_clustered_mda_across_folds(
    per_fold: list[dict[int, float]],
    cluster_labels: dict[int, list[str]] | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate per-fold clustered MDA into ``{cluster_key -> {mean, std, members}}``.

    Cluster ids are NOT guaranteed stable across folds (hierarchical labelling can
    relabel), so when ``cluster_labels`` (an id→members mapping covering the ids seen
    across folds) is supplied, entries are keyed by a canonical sorted-tuple of member
    names. That groups the "same" cluster across folds even if its numeric id differs.

    If ``cluster_labels`` is None, ids are used directly as the key and ``members`` is
    left empty. ``mean``/``std`` are computed over the folds in which the cluster
    appears (population std, ddof=0).
    """
    # Build key -> (members, [values]) accumulator.
    buckets: dict[str, tuple[list[str], list[float]]] = {}
    for fold in per_fold:
        for cid, val in fold.items():
            if cluster_labels is not None and cid in cluster_labels:
                members = sorted(cluster_labels[cid])
                key = "|".join(members)
            else:
                members = []
                key = str(cid)
            if key not in buckets:
                buckets[key] = (members, [])
            buckets[key][1].append(float(val))

    out: dict[str, dict[str, float]] = {}
    for key, (members, values) in buckets.items():
        arr = np.asarray(values, dtype=float)
        out[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "members": members,
        }
    return out
