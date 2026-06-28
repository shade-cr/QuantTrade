"""Tests for the walk-forward splitter and PurgedTimeSeriesSplit."""
from __future__ import annotations
import numpy as np
import pytest

from pipeline.walk_forward import (
    make_folds,
    PurgedTimeSeriesSplit,
    WalkForwardGeometryError,
    resolve_train_min,
    wf_event_floor,
)


def test_make_folds_no_overlap_and_purge_respected():
    folds = make_folds(n=5000, n_folds=4, train_min=1500, purge=20, embargo_pct=0.01)
    assert len(folds) == 4
    for f in folds:
        # disjoint
        assert len(np.intersect1d(f.train_idx, f.test_idx)) == 0
        train_before_test = f.train_idx[f.train_idx < f.test_idx.min()]
        assert f.test_idx.min() - train_before_test.max() >= 20  # purge


def test_make_folds_embargo_applied_between_folds():
    folds = make_folds(n=5000, n_folds=4, train_min=1500, purge=20, embargo_pct=0.01)
    embargo = int(np.ceil(0.01 * 5000))
    for k in range(1, 4):
        for j in range(k):
            zone = set(range(folds[j].test_idx.max() + 1, folds[j].test_idx.max() + 1 + embargo))
            assert zone.isdisjoint(set(folds[k].train_idx.tolist()))


def test_make_folds_test_coverage_exhaustive_with_remainder():
    folds = make_folds(n=5003, n_folds=4, train_min=1500, purge=20, embargo_pct=0.01)
    all_test = np.concatenate([f.test_idx for f in folds])
    assert len(np.unique(all_test)) == len(all_test)  # no dupes
    assert all_test.min() == 1500
    assert all_test.max() == 5002  # last fold absorbs the remainder


def test_make_folds_monotonic_test_folds():
    folds = make_folds(n=5000, n_folds=4, train_min=1500, purge=20, embargo_pct=0.01)
    for k in range(3):
        assert folds[k].test_idx.max() < folds[k + 1].test_idx.min()


def test_purged_time_series_split_gap():
    splitter = PurgedTimeSeriesSplit(n_splits=3, purge=20)
    X = np.arange(900).reshape(-1, 1)
    for tr_idx, va_idx in splitter.split(X):
        assert va_idx.min() - tr_idx.max() >= 20
        assert len(np.intersect1d(tr_idx, va_idx)) == 0


def test_make_folds_rejects_n_too_small():
    # B0013 — still IS-A ValueError so existing handlers keep working.
    with pytest.raises(ValueError, match="refusal cliff"):
        make_folds(n=1600, n_folds=4, train_min=1500, purge=20, embargo_pct=0.01)


def test_make_folds_refusal_cliff_is_structured_valueerror():
    """B0013 — the refusal cliff raises WalkForwardGeometryError (IS-A ValueError)
    carrying structured fields that name the relaxable axis."""
    with pytest.raises(WalkForwardGeometryError) as ei:
        make_folds(n=1600, n_folds=4, train_min=1500, purge=20, embargo_pct=0.01)
    err = ei.value
    assert isinstance(err, ValueError)            # contract preserved
    assert err.events == 1600
    assert err.n_folds == 4
    assert err.train_min == 1500
    assert err.test_pool_needed == 400            # 4 * 100
    assert err.required_events == 1900            # 1500 + 400
    assert err.shortfall == 300                   # 1900 - 1600
    # Message names the cliff + the relaxable axes
    msg = str(err)
    assert "refusal cliff" in msg
    assert "lower n_folds" in msg and "lower train_min" in msg


def test_make_folds_just_above_cliff_succeeds():
    """n - train_min == n_folds*100 exactly is the inclusive boundary (no raise)."""
    folds = make_folds(n=1900, n_folds=4, train_min=1500, purge=20, embargo_pct=0.01)
    assert len(folds) == 4


# --- B0048: pre-flight event-floor gate -------------------------------------

def test_resolve_train_min_caps_at_half_events():
    """train_min passed to make_folds is min(train_min_bars, n_events // 2).
    Centralized here so the orchestrator caller and the floor projection agree."""
    assert resolve_train_min(100, 35) == 17   # T011D2: 35 events -> 17
    assert resolve_train_min(100, 73) == 36    # T015D2: 73 events -> 36
    assert resolve_train_min(100, 1000) == 100  # cap binds when events plentiful
    assert resolve_train_min(500, 800) == 400   # n//2 < cap


def test_wf_event_floor_known_geometries():
    """The minimum event count for which make_folds will NOT raise, given the
    geometry build_transient_config uses. Values account for train_min = n//2
    in the scarce regime (the reason T011D2's floor was 217, not 300)."""
    assert wf_event_floor(n_folds=2, train_min_bars=100) == 300   # D1 phase5_*
    assert wf_event_floor(n_folds=2, train_min_bars=50) == 250    # H4
    assert wf_event_floor(n_folds=2, train_min_bars=20) == 220    # diagnostic
    assert wf_event_floor(n_folds=3, train_min_bars=500) == 599   # D1 built-in template


@pytest.mark.parametrize(
    "n_folds,train_min_bars",
    [(2, 100), (2, 50), (2, 20), (3, 500), (1, 100), (4, 1500)],
)
def test_wf_event_floor_matches_make_folds_boundary(n_folds, train_min_bars):
    """Anti-drift: the floor must be the exact boundary of make_folds' real
    refusal cliff, using the SAME resolved train_min the orchestrator passes.
    At the floor make_folds succeeds; one event below it raises."""
    floor = wf_event_floor(n_folds=n_folds, train_min_bars=train_min_bars)

    folds = make_folds(
        n=floor, n_folds=n_folds,
        train_min=resolve_train_min(train_min_bars, floor),
        purge=2, embargo_pct=0.0,
    )
    assert len(folds) == n_folds

    with pytest.raises(WalkForwardGeometryError):
        make_folds(
            n=floor - 1, n_folds=n_folds,
            train_min=resolve_train_min(train_min_bars, floor - 1),
            purge=2, embargo_pct=0.0,
        )


# --- B0129: AFML §7.4.1 Snippet 7.1 exact label-end purge --------------------

def _overlap_in_fold(fold, event_start_bar, event_end_bar):
    """Return True if any TRAIN event before the test window has a label that
    reaches into or past the test window's first BAR (AFML getTrainTimes leak)."""
    test_first = int(fold.test_idx.min())
    test_first_bar = int(event_start_bar[test_first])
    train_before = fold.train_idx[fold.train_idx < test_first]
    if len(train_before) == 0:
        return False
    return bool(np.any(event_end_bar[train_before] >= test_first_bar))


def test_make_folds_dense_events_label_end_purge_closes_leak():
    """B0129 — dense events (1 bar apart) with horizon 40 and a too-small scalar
    purge=20. With t1 arrays passed, NO train event's label may overlap the test
    window's first bar (AFML §7.4.1 Snippet 7.1 getTrainTimes)."""
    n = 5000
    horizon = 40
    event_start_bar = np.arange(n)
    event_end_bar = event_start_bar + horizon
    folds = make_folds(
        n=n, n_folds=4, train_min=1500, purge=20, embargo_pct=0.0,
        event_start_bar=event_start_bar, event_end_bar=event_end_bar,
    )
    # Exact purge: zero overlap in every fold.
    for f in folds:
        assert not _overlap_in_fold(f, event_start_bar, event_end_bar)


def test_make_folds_dense_events_scalar_purge_leaks():
    """B0129 — WITHOUT the t1 arrays, scalar purge=20 < horizon=40 UNDER-purges:
    at least one fold retains a training event whose label reaches into the test
    window's first bar. This is the look-ahead leak B0129 closes."""
    n = 5000
    horizon = 40
    event_start_bar = np.arange(n)
    event_end_bar = event_start_bar + horizon
    folds = make_folds(n=n, n_folds=4, train_min=1500, purge=20, embargo_pct=0.0)
    leaks = [_overlap_in_fold(f, event_start_bar, event_end_bar) for f in folds]
    assert any(leaks), "scalar purge=20 should under-purge a horizon-40 dense scenario"


def test_make_folds_sparse_events_both_paths_no_overlap():
    """B0129 — sparse events ~8 bars apart, horizon 40, scalar purge=20. Here the
    scalar purge happens to OVER-cover (20 events × 8 bars/event = 160 bars >> 40),
    so both the scalar path and the exact label-end path produce zero overlap.
    Demonstrates the density dependence the exact purge removes."""
    n = 2000
    spacing = 8
    horizon = 40
    event_start_bar = np.arange(n) * spacing
    event_end_bar = event_start_bar + horizon
    folds_exact = make_folds(
        n=n, n_folds=4, train_min=600, purge=20, embargo_pct=0.0,
        event_start_bar=event_start_bar, event_end_bar=event_end_bar,
    )
    folds_scalar = make_folds(n=n, n_folds=4, train_min=600, purge=20, embargo_pct=0.0)
    for f in folds_exact:
        assert not _overlap_in_fold(f, event_start_bar, event_end_bar)
    for f in folds_scalar:
        assert not _overlap_in_fold(f, event_start_bar, event_end_bar)


def test_make_folds_validates_t1_array_lengths():
    """B0129 — event_start_bar / event_end_bar must have length n."""
    with pytest.raises(ValueError, match="length"):
        make_folds(
            n=2000, n_folds=4, train_min=600, purge=20, embargo_pct=0.0,
            event_start_bar=np.arange(1999), event_end_bar=np.arange(2000),
        )
    with pytest.raises(ValueError, match="length"):
        make_folds(
            n=2000, n_folds=4, train_min=600, purge=20, embargo_pct=0.0,
            event_start_bar=np.arange(2000), event_end_bar=np.arange(1999),
        )


def test_make_folds_requires_both_t1_arrays():
    """B0129 — passing only one of the two arrays is ambiguous: must pass both."""
    with pytest.raises(ValueError, match="(?i)both"):
        make_folds(
            n=2000, n_folds=4, train_min=600, purge=20, embargo_pct=0.0,
            event_start_bar=np.arange(2000),
        )


def test_purged_tss_label_end_purge_no_leak():
    """B0129 — PurgedTimeSeriesSplit with event_end_bar provided: no train index's
    label-end may reach the val window's first bar (AFML §7.4.1)."""
    n = 900
    horizon = 40
    event_start_bar = np.arange(n)
    event_end_bar = event_start_bar + horizon
    splitter = PurgedTimeSeriesSplit(
        n_splits=3, purge=5,
        event_start_bar=event_start_bar, event_end_bar=event_end_bar,
    )
    X = np.arange(n).reshape(-1, 1)
    saw_split = False
    for tr_idx, va_idx in splitter.split(X):
        saw_split = True
        assert len(np.intersect1d(tr_idx, va_idx)) == 0
        val_first_bar = int(event_start_bar[va_idx.min()])
        if len(tr_idx):
            assert np.all(event_end_bar[tr_idx] < val_first_bar)
    assert saw_split


def test_purged_tss_embargo_drops_post_val_indices():
    """B0129 — embargo drops the first `embargo` train positions immediately
    following the val block (AFML Snippet 7.2 getEmbargoTimes)."""
    n = 1200
    embargo = 7
    splitter_no_emb = PurgedTimeSeriesSplit(n_splits=3, purge=10, embargo=0)
    splitter_emb = PurgedTimeSeriesSplit(n_splits=3, purge=10, embargo=embargo)
    X = np.arange(n).reshape(-1, 1)
    no_emb = list(splitter_no_emb.split(X))
    emb = list(splitter_emb.split(X))
    assert len(no_emb) == len(emb)
    for (tr0, va0), (tr1, va1) in zip(no_emb, emb):
        np.testing.assert_array_equal(va0, va1)
        # the embargo zone is the `embargo` positions right after the val block
        zone = set(range(int(va1.max()) + 1, int(va1.max()) + 1 + embargo))
        # embargoed train must exclude that zone; non-embargoed may include it
        assert zone.isdisjoint(set(tr1.tolist()))


def test_purged_tss_backward_compat_scalar_purge_unchanged():
    """B0129 — with no t1 arrays and embargo=0, behavior is identical to before."""
    n = 900
    base = PurgedTimeSeriesSplit(n_splits=3, purge=20)
    X = np.arange(n).reshape(-1, 1)
    for tr_idx, va_idx in base.split(X):
        assert va_idx.min() - tr_idx.max() >= 20
        assert len(np.intersect1d(tr_idx, va_idx)) == 0
