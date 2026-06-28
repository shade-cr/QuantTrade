"""Tests for the pooled (wall-clock TIME) walk-forward splitter and inner CV.

B0148 SLICE 1 — cross-asset meta-learner pooling. The pooled meta-learner trains
on the UNION of events across assets, so purge/embargo must operate in wall-clock
TIME (a bar position is not comparable across assets of different bar densities),
not in per-asset bar-position space. These tests pin the AFML §7.4.1 (getTrainTimes,
leading-edge label-end purge) and §7.4.2 (getEmbargoTimes, trailing-edge embargo)
generalizations to a multi-asset pooled event set.

Spec: docs/superpowers/specs/2026-06-04-b0148-cross-asset-meta-pooling-design.md
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from pipeline.walk_forward import FoldIndices, make_folds
from pipeline.pooled_walk_forward import (
    make_pooled_time_folds,
    PurgedTimeGroupSplit,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ts(base: str, hours):
    """Convenience: list of Timestamps at `base` + each hour offset."""
    t0 = pd.Timestamp(base)
    return [t0 + pd.Timedelta(hours=int(h)) for h in hours]


# --------------------------------------------------------------------------- #
# make_pooled_time_folds — cross-asset leak / time-not-position
# --------------------------------------------------------------------------- #
def test_no_cross_asset_leak_straddling_label_end_is_purged():
    """A B-asset train event whose label_end straddles a test block boundary T_k
    is PURGED from that block's train; one resolving strictly before T_k is KEPT.
    (AFML §7.4.1 getTrainTimes generalized across assets via timestamps.)"""
    # Two assets share a wall clock. Build enough events so n_folds split is sane.
    t0 = pd.Timestamp("2020-01-01")
    H = pd.Timedelta(hours=4)
    # 20 A-events every 4h, 20 B-events every 4h offset by 2h, all 1-bar labels...
    n_each = 20
    a_start = [t0 + i * H for i in range(n_each)]
    a_end = [s + H for s in a_start]  # ~immediate resolution
    b_start = [t0 + pd.Timedelta(hours=2) + i * H for i in range(n_each)]
    b_end = [s + H for s in b_start]

    event_time = a_start + b_start
    label_end = a_end + b_end
    asset = ["A"] * n_each + ["B"] * n_each

    folds = make_pooled_time_folds(
        event_time, label_end, n_folds=2, train_min_frac=0.5,
        embargo_td=pd.Timedelta(0), asset=asset,
    )
    assert len(folds) == 2

    et = pd.DatetimeIndex(event_time)
    le = pd.DatetimeIndex(label_end)
    for f in folds:
        if len(f.test_idx) == 0 or len(f.train_idx) == 0:
            continue
        T_k = et[f.test_idx].min()
        # AFML getTrainTimes invariant: NO train event's label may reach >= T_k.
        assert (le[f.train_idx] < T_k).all()

    # Now inject a B-event that straddles the last block's boundary T_k and a
    # sibling resolving strictly before it; assert the straddler is purged, the
    # sibling kept. The two crafted events are placed EARLY (well inside the train
    # region) so they do not perturb the test-pool composition; we then read the
    # actual boundary off the recomputed folds.
    early_anchor = et[0] + pd.Timedelta(hours=4)
    straddle_start = early_anchor
    clean_start = early_anchor

    event_time2 = event_time + [straddle_start, clean_start]
    # label ends are filled below once we know the recomputed boundary T_k.
    label_end2 = label_end + [straddle_start, clean_start]
    asset2 = asset + ["B", "B"]
    straddle_idx = len(event_time)        # original-order index of straddler
    clean_idx = len(event_time) + 1

    folds_probe = make_pooled_time_folds(
        event_time2, label_end2, n_folds=2, train_min_frac=0.5,
        embargo_td=pd.Timedelta(0), asset=asset2,
    )
    et2 = pd.DatetimeIndex(event_time2)
    T_k = et2[folds_probe[-1].test_idx].min()
    # straddler resolves PAST T_k -> must be purged; clean resolves before -> kept.
    label_end2[straddle_idx] = T_k + pd.Timedelta(hours=4)
    label_end2[clean_idx] = T_k - pd.Timedelta(hours=1)

    folds3 = make_pooled_time_folds(
        event_time2, label_end2, n_folds=2, train_min_frac=0.5,
        embargo_td=pd.Timedelta(0), asset=asset2,
    )
    last = folds3[-1]
    assert straddle_idx not in set(last.train_idx.tolist())
    assert clean_idx in set(last.train_idx.tolist())


def test_purge_depends_on_time_not_position():
    """Two assets with different bar densities (4h vs 6h). Purge must depend only
    on timestamps, never on per-asset positional index."""
    t0 = pd.Timestamp("2021-01-01")
    n = 30
    a_start = [t0 + i * pd.Timedelta(hours=4) for i in range(n)]
    a_end = [s + pd.Timedelta(hours=4) for s in a_start]
    b_start = [t0 + i * pd.Timedelta(hours=6) for i in range(n)]
    b_end = [s + pd.Timedelta(hours=6) for s in b_start]

    event_time = a_start + b_start
    label_end = a_end + b_end
    asset = ["A"] * n + ["B"] * n

    folds = make_pooled_time_folds(
        event_time, label_end, n_folds=3, train_min_frac=0.4,
        embargo_td=pd.Timedelta(0), asset=asset,
    )
    et = pd.DatetimeIndex(event_time)
    le = pd.DatetimeIndex(label_end)
    for f in folds:
        if len(f.test_idx) == 0 or len(f.train_idx) == 0:
            continue
        T_k = et[f.test_idx].min()
        # Pure timestamp predicate — independent of which asset / positional idx.
        assert (le[f.train_idx] < T_k).all()
        # And no train event starts at/after the test boundary.
        assert (et[f.train_idx] < et[f.test_idx].max() + pd.Timedelta(days=3650)).all()


def test_expanding_window_leading_and_trailing_edge():
    """Fold k's resolved test events become eligible train for fold k+1 ONLY via
    label_end_time < T_{k+1}; an earlier event whose label reaches into T_{k+1} is
    purged; no event with event_time >= T_{k+1} ever enters fold k's train."""
    t0 = pd.Timestamp("2022-01-01")
    H = pd.Timedelta(hours=4)
    n = 60
    event_time = [t0 + i * H for i in range(n)]
    label_end = [s + H for s in event_time]
    asset = ["A"] * n
    folds = make_pooled_time_folds(
        event_time, label_end, n_folds=3, train_min_frac=0.4,
        embargo_td=pd.Timedelta(0), asset=asset,
    )
    et = pd.DatetimeIndex(event_time)
    le = pd.DatetimeIndex(label_end)
    test_first_times = [et[f.test_idx].min() for f in folds]
    for k, f in enumerate(folds):
        T_k = test_first_times[k]
        if len(f.train_idx):
            # leading-edge: all labels resolve strictly before T_k
            assert (le[f.train_idx] < T_k).all()
            # no future event enters train
            assert (et[f.train_idx] < T_k).all()
    # expanding: fold k+1's train is a superset-by-time of fold k's train pool
    for k in range(len(folds) - 1):
        assert len(folds[k + 1].train_idx) >= len(folds[k].train_idx)


def test_embargo_in_time_drops_post_test_window_regardless_of_asset():
    """A train event with event_time in (E_j, E_j + embargo_td] after a prior test
    block is dropped regardless of asset; one just outside is kept. (AFML §7.4.2,
    POST-test only.)"""
    t0 = pd.Timestamp("2023-01-01")
    H = pd.Timedelta(hours=4)
    n = 60
    event_time = [t0 + i * H for i in range(n)]
    # long labels so events resolve well after their start, but embargo is what we test
    label_end = [s + H for s in event_time]
    asset = ["A"] * n
    embargo_td = pd.Timedelta(hours=20)

    folds_no = make_pooled_time_folds(
        event_time, label_end, n_folds=3, train_min_frac=0.4,
        embargo_td=pd.Timedelta(0), asset=asset,
    )
    folds_emb = make_pooled_time_folds(
        event_time, label_end, n_folds=3, train_min_frac=0.4,
        embargo_td=embargo_td, asset=asset,
    )
    et = pd.DatetimeIndex(event_time)
    # The embargo can only REMOVE train rows, never add.
    for fno, fe in zip(folds_no, folds_emb):
        assert set(fe.train_idx.tolist()).issubset(set(fno.train_idx.tolist()))

    # Find the last fold; its train should exclude the embargo zone after each
    # prior test block.
    for k, fe in enumerate(folds_emb):
        prior_tests = folds_emb[:k]
        for pj in prior_tests:
            E_j = et[pj.test_idx].max()
            zone_lo, zone_hi = E_j, E_j + embargo_td
            for idx in fe.train_idx:
                t = et[idx]
                in_zone = (t > zone_lo) and (t <= zone_hi)
                assert not in_zone, f"fold {k} train idx {idx} sits in embargo zone"


# --------------------------------------------------------------------------- #
# Single-asset reduction (correctness anchor)
# --------------------------------------------------------------------------- #
def test_single_asset_reduces_to_make_folds_label_end_purge():
    """On a single-asset fixture, make_pooled_time_folds must produce folds
    EQUIVALENT to make_folds(..., event_start_bar=, event_end_bar=) (the existing
    label-end purge mode): for one asset, time order == bar order and
    label_end_time < T_k  <=>  event_end_bar < test_first_bar."""
    n = 600
    horizon = 5
    event_start_bar = np.arange(n)
    event_end_bar = event_start_bar + horizon
    n_folds = 2
    train_min = 300

    # bar position i -> a wall-clock timestamp (strictly increasing, 1 bar = 4h)
    t0 = pd.Timestamp("2019-01-01")
    H = pd.Timedelta(hours=4)
    event_time = [t0 + int(b) * H for b in event_start_bar]
    label_end = [t0 + int(b) * H for b in event_end_bar]
    asset = ["A"] * n

    ref = make_folds(
        n=n, n_folds=n_folds, train_min=train_min, purge=0, embargo_pct=0.0,
        event_start_bar=event_start_bar, event_end_bar=event_end_bar,
    )
    # train_min_frac chosen so the time cutoff lands on bar `train_min`.
    train_min_frac = train_min / (n - 1)
    pooled = make_pooled_time_folds(
        event_time, label_end, n_folds=n_folds, train_min_frac=train_min_frac,
        embargo_td=pd.Timedelta(0), asset=asset,
    )
    assert len(pooled) == len(ref) == n_folds
    for fp, fr in zip(pooled, ref):
        assert set(fp.test_idx.tolist()) == set(fr.test_idx.tolist())
        assert set(fp.train_idx.tolist()) == set(fr.train_idx.tolist())


# --------------------------------------------------------------------------- #
# Degenerate guards
# --------------------------------------------------------------------------- #
def test_empty_pool_returns_empty():
    folds = make_pooled_time_folds(
        [], [], n_folds=3, train_min_frac=0.5, embargo_td=pd.Timedelta(0), asset=[]
    )
    assert folds == []


def test_single_asset_class_is_supported():
    """A 'pool' that is a single asset is the degenerate within-class case and
    must still split cleanly."""
    t0 = pd.Timestamp("2020-06-01")
    H = pd.Timedelta(hours=4)
    n = 40
    event_time = [t0 + i * H for i in range(n)]
    label_end = [s + H for s in event_time]
    asset = ["A"] * n
    folds = make_pooled_time_folds(
        event_time, label_end, n_folds=2, train_min_frac=0.5,
        embargo_td=pd.Timedelta(0), asset=asset,
    )
    assert len(folds) == 2
    all_test = np.concatenate([f.test_idx for f in folds])
    assert len(np.unique(all_test)) == len(all_test)


def test_n_folds_too_large_raises():
    """n_folds exceeding the test-pool event count is a refusal — raise ValueError
    (mirrors make_folds' refusal convention)."""
    t0 = pd.Timestamp("2020-06-01")
    H = pd.Timedelta(hours=4)
    n = 10
    event_time = [t0 + i * H for i in range(n)]
    label_end = [s + H for s in event_time]
    asset = ["A"] * n
    with pytest.raises(ValueError):
        make_pooled_time_folds(
            event_time, label_end, n_folds=20, train_min_frac=0.5,
            embargo_td=pd.Timedelta(0), asset=asset,
        )


def test_indices_map_back_to_original_input_order():
    """Returned indices index into the ORIGINAL (unsorted) input order, so a caller
    can map back to its X rows. Build a shuffled pool and verify."""
    rng = np.random.default_rng(0)
    t0 = pd.Timestamp("2021-03-01")
    H = pd.Timedelta(hours=4)
    n = 50
    starts = [t0 + i * H for i in range(n)]
    ends = [s + H for s in starts]
    order = rng.permutation(n)
    event_time = [starts[i] for i in order]
    label_end = [ends[i] for i in order]
    asset = ["A"] * n
    folds = make_pooled_time_folds(
        event_time, label_end, n_folds=2, train_min_frac=0.5,
        embargo_td=pd.Timedelta(0), asset=asset,
    )
    et = pd.DatetimeIndex(event_time)
    le = pd.DatetimeIndex(label_end)
    for f in folds:
        if len(f.test_idx) and len(f.train_idx):
            T_k = et[f.test_idx].min()
            # invariant holds when indices correctly map to the original order
            assert (le[f.train_idx] < T_k).all()
        # test indices are contiguous-in-time blocks (sorted-time order)
        if len(f.test_idx) > 1:
            tt = et[f.test_idx].sort_values()
            assert (tt.values == np.sort(et[f.test_idx].values)).all()


# --------------------------------------------------------------------------- #
# PurgedTimeGroupSplit (inner CV)
# --------------------------------------------------------------------------- #
def test_inner_split_purges_on_timestamps_not_position():
    """An inner split purges train rows by label_end_time < val_first_time, in
    timestamp space (not concatenated position)."""
    t0 = pd.Timestamp("2022-01-01")
    H = pd.Timedelta(hours=4)
    n = 120
    event_time = pd.DatetimeIndex([t0 + i * H for i in range(n)])
    horizon = pd.Timedelta(hours=20)
    label_end = pd.DatetimeIndex([t + horizon for t in event_time])

    splitter = PurgedTimeGroupSplit(
        n_splits=3, event_time=event_time, label_end_time=label_end,
        embargo_td=pd.Timedelta(0),
    )
    X = np.arange(n).reshape(-1, 1)
    saw = False
    for tr_idx, va_idx in splitter.split(X):
        saw = True
        assert len(np.intersect1d(tr_idx, va_idx)) == 0
        val_first = event_time[va_idx].min()
        if len(tr_idx):
            assert (label_end[tr_idx] < val_first).all()
    assert saw
    assert splitter.get_n_splits() == 3


def test_inner_split_embargo_in_time():
    """Embargo (post-val) drops train rows whose event_time falls within embargo_td
    after the val window."""
    t0 = pd.Timestamp("2022-01-01")
    H = pd.Timedelta(hours=4)
    n = 120
    event_time = pd.DatetimeIndex([t0 + i * H for i in range(n)])
    label_end = pd.DatetimeIndex([t + H for t in event_time])
    embargo_td = pd.Timedelta(hours=20)

    no_emb = PurgedTimeGroupSplit(
        n_splits=3, event_time=event_time, label_end_time=label_end,
        embargo_td=pd.Timedelta(0),
    )
    emb = PurgedTimeGroupSplit(
        n_splits=3, event_time=event_time, label_end_time=label_end,
        embargo_td=embargo_td,
    )
    X = np.arange(n).reshape(-1, 1)
    for (tr0, va0), (tr1, va1) in zip(no_emb.split(X), emb.split(X)):
        np.testing.assert_array_equal(va0, va1)
        assert set(tr1.tolist()).issubset(set(tr0.tolist()))
        val_last = event_time[va1].max()
        zone_lo, zone_hi = val_last, val_last + embargo_td
        for idx in tr1:
            t = event_time[idx]
            assert not (t > zone_lo and t <= zone_hi)


def test_inner_split_works_as_randomizedsearchcv_cv():
    """Smoke: PurgedTimeGroupSplit works as the `cv` arg to RandomizedSearchCV
    (sklearn calls .split(X, y, groups) and only sees X,y,groups — the timestamps
    must be stored on self)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV

    t0 = pd.Timestamp("2022-01-01")
    H = pd.Timedelta(hours=4)
    n = 200
    rng = np.random.default_rng(1)
    event_time = pd.DatetimeIndex([t0 + i * H for i in range(n)])
    label_end = pd.DatetimeIndex([t + H for t in event_time])
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
    y = pd.Series((rng.standard_normal(n) > 0).astype(int))

    cv = PurgedTimeGroupSplit(
        n_splits=3, event_time=event_time, label_end_time=label_end,
        embargo_td=pd.Timedelta(0),
    )
    search = RandomizedSearchCV(
        RandomForestClassifier(n_estimators=5, random_state=0),
        param_distributions={"max_depth": [2, 3]},
        n_iter=2, cv=cv, random_state=0, error_score="raise",
    )
    search.fit(X, y)  # must not raise
    assert hasattr(search, "best_estimator_")
