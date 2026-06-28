"""Wall-clock-TIME purged walk-forward for a POOLED multi-asset event set (B0148).

This is the load-bearing new infrastructure for cross-asset meta-learner pooling.
The single-asset splitters in :mod:`pipeline.walk_forward` operate in per-asset
bar-POSITION space (``make_folds`` / ``PurgedTimeSeriesSplit``): bar position 500
in EURUSD is a different wall-clock instant than bar 500 in BTCUSD, so concatenating
``event_end_bar`` across assets is meaningless and would both mis-purge and silently
leak. The pooled splitters below purge in wall-clock TIME instead.

References:
  Marcos López de Prado, *Advances in Financial Machine Learning*, Wiley 2018.
    §7.4.1 Snippet 7.1 ``getTrainTimes`` — leading-edge label-end purge.
    §7.4.2 Snippet 7.2 ``getEmbargoTimes`` — trailing-edge (POST-test) embargo.

Spec: docs/superpowers/specs/2026-06-04-b0148-cross-asset-meta-pooling-design.md
"""
from __future__ import annotations
from typing import Iterator
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from pipeline.walk_forward import FoldIndices


def make_pooled_time_folds(
    event_time,
    label_end_time,
    n_folds: int,
    train_min_frac: float = 0.5,
    embargo_td: pd.Timedelta = pd.Timedelta(0),
    asset=None,
) -> list[FoldIndices]:
    """Expanding-window WF for a pooled multi-asset event set, purged in TIME.

    Wall-clock generalization of :func:`pipeline.walk_forward.make_folds`'s exact
    label-end purge mode. Inputs are TIMESTAMPS (one per pooled event, in arbitrary
    input order), not bar positions, so the AFML §7.4.1 ``getTrainTimes`` predicate
    becomes ``label_end_time < T_k`` across ALL assets — which removes cross-asset
    temporal-overlap leakage for free.

    Algorithm
    ---------
    1. Sort events by ``event_time`` (stable). The test pool = events at or after
       the ``train_min_frac`` cutoff of the global ``[t_min, t_max]`` span.
    2. Split the test pool into ``n_folds`` contiguous-in-time blocks **by equal
       event count along the sorted order** (matches ``make_folds``' equal-chunk
       semantics; the last block absorbs the remainder). The block's first
       ``event_time`` is its boundary ``T_k``.
    3. For test block ``k`` with first time ``T_k``:
       * **leading-edge purge** (AFML §7.4.1): train = events with
         ``label_end_time < T_k`` (any asset). An outcome window straddling the
         boundary is removed for every asset.
       * **trailing-edge embargo** (AFML §7.4.2, POST-test ONLY): for each PRIOR
         test block ``j < k`` with last ``event_time`` ``E_j``, drop train events
         whose ``event_time ∈ (E_j, E_j + embargo_td]``. There is deliberately NO
         symmetric pre-test embargo (spec [R2]: pre-test training labels are
         already in the test's information set).
    4. Return :class:`FoldIndices` with indices into the ORIGINAL input order, so a
       caller can map straight back to its ``X`` rows.

    Args:
      event_time:     array-like of pandas Timestamps — each pooled event's
                      entry-bar time (any asset).
      label_end_time: array-like of pandas Timestamps — each event's
                      triple-barrier resolution time on its own asset.
      n_folds:        number of contiguous test blocks.
      train_min_frac: fraction of the global time span reserved as initial train;
                      events at/after this time cutoff form the test pool.
      embargo_td:     wall-clock ``pd.Timedelta`` for the post-test embargo. Size it
                      off the COARSEST asset's vertical-barrier horizon (spec [R]).
      asset:          optional array-like of asset labels (not used by the purge —
                      time is asset-agnostic — accepted for symmetry / future use).

    Returns:
      list[FoldIndices], one per test block, indices in ORIGINAL input order.

    Raises:
      ValueError: if inputs mismatch in length, or ``n_folds`` exceeds the number
                  of events in the test pool (refusal — mirrors ``make_folds``).
    """
    et = pd.DatetimeIndex(pd.to_datetime(list(event_time)))
    le = pd.DatetimeIndex(pd.to_datetime(list(label_end_time)))
    if len(et) != len(le):
        raise ValueError("event_time and label_end_time must have the same length")
    n = len(et)
    if n == 0:
        return []
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if not isinstance(embargo_td, pd.Timedelta):
        embargo_td = pd.Timedelta(embargo_td)

    # Stable sort by event_time; keep the mapping back to original positions.
    order = np.argsort(et.values, kind="stable")
    et_sorted = et[order]
    le_sorted = le[order]

    t_min = et_sorted[0]
    t_max = et_sorted[-1]
    span = t_max - t_min
    cutoff_time = t_min + span * float(train_min_frac)

    # Test pool = sorted positions whose event_time >= cutoff_time.
    sorted_pos = np.arange(n)
    test_pool_mask = et_sorted.values >= np.datetime64(cutoff_time)
    test_pool_sorted = sorted_pos[test_pool_mask]
    n_test = len(test_pool_sorted)

    if n_test < n_folds:
        raise ValueError(
            f"make_pooled_time_folds: test pool has {n_test} events but n_folds="
            f"{n_folds}. Relax ONE axis: lower n_folds, lower train_min_frac, or "
            f"obtain more events. (mirrors make_folds' refusal convention)"
        )

    # Split the test pool into n_folds equal-count contiguous blocks; last absorbs
    # the remainder (matches make_folds chunk semantics).
    base_size = n_test // n_folds
    sizes = [base_size] * n_folds
    sizes[-1] += n_test - base_size * n_folds

    # event_time values as int64 ns for fast comparisons.
    et_ns_sorted = et_sorted.asi8
    le_ns = le.asi8                       # original order
    et_ns = et.asi8                       # original order
    embargo_ns = np.int64(embargo_td.value)

    folds: list[FoldIndices] = []
    cursor = 0
    blocks: list[np.ndarray] = []   # sorted-space positions per test block
    for size in sizes:
        block = test_pool_sorted[cursor : cursor + size]
        blocks.append(block)
        cursor += size

    for k, block in enumerate(blocks):
        # block holds SORTED positions; map to original indices for the result.
        test_orig = order[block]
        T_k_ns = int(et_ns_sorted[block[0]])     # first event_time in the block

        # Leading-edge label-end purge (AFML §7.4.1): label resolves strictly
        # before T_k. Evaluated in ORIGINAL order so result indices map back.
        keep = le_ns < T_k_ns
        train_orig = np.flatnonzero(keep)

        # Trailing-edge embargo (AFML §7.4.2, post-test only) for each prior block.
        if embargo_ns > 0 and k > 0:
            drop = np.zeros(n, dtype=bool)
            for j in range(k):
                E_j_ns = int(et_ns_sorted[blocks[j][-1]])   # last event_time of block j
                zone = (et_ns > E_j_ns) & (et_ns <= E_j_ns + embargo_ns)
                drop |= zone
            train_orig = train_orig[~drop[train_orig]]

        folds.append(
            FoldIndices(
                train_idx=np.sort(train_orig).astype(int),
                test_idx=np.sort(test_orig).astype(int),
            )
        )
    return folds


class PurgedTimeGroupSplit(BaseCrossValidator):
    """sklearn-compatible inner CV: forward-rolling splits purged in TIMESTAMP space.

    Wall-clock-TIME analog of :class:`pipeline.walk_forward.PurgedTimeSeriesSplit`
    for the pooled train slice (B0148 blocker B2). Mirrors that class's structure
    exactly — a ``BaseCrossValidator`` whose ``split(self, X, y=None, groups=None)``
    signature is fixed by sklearn — but purges on ``label_end_time < val_first_time``
    in TIMESTAMP space, NOT positional bar arrays.

    The timestamp arrays are stored on ``self`` (the ``event_start_bar`` /
    ``event_end_bar`` pattern) because ``RandomizedSearchCV`` calls
    ``.split(X, y, groups)`` and will not pass times any other way — do NOT try to
    smuggle time through ``groups``.

    This is NOT a partition: early rows and purge-zone rows belong to no val fold,
    which is expected (and handled by :func:`pipeline.walk_forward.inner_oof_predict_proba`).

    Embargo (AFML §7.4.2 ``getEmbargoTimes``, POST-val only): drops train rows whose
    ``event_time`` falls within ``embargo_td`` AFTER the val window's last event.
    """

    def __init__(
        self,
        n_splits: int = 3,
        event_time=None,
        label_end_time=None,
        embargo_td: pd.Timedelta = pd.Timedelta(0),
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if (event_time is None) != (label_end_time is None):
            raise ValueError(
                "PurgedTimeGroupSplit: event_time and label_end_time must BOTH be "
                "provided for the AFML §7.4.1 timestamp label-end purge."
            )
        self.n_splits = n_splits
        self.event_time = event_time
        self.label_end_time = label_end_time
        self.embargo_td = embargo_td

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        use_time = self.event_time is not None
        if use_time:
            et = pd.DatetimeIndex(pd.to_datetime(list(self.event_time)))
            le = pd.DatetimeIndex(pd.to_datetime(list(self.label_end_time)))
            if len(et) != n or len(le) != n:
                raise ValueError(
                    f"PurgedTimeGroupSplit: event_time / label_end_time must each "
                    f"have length len(X)={n}."
                )
            # Sort positions by event_time so the rolling val windows are
            # contiguous-in-time even if X rows are unsorted.
            order = np.argsort(et.values, kind="stable")
            et_ns = et.asi8
            le_ns = le.asi8
            embargo_td = self.embargo_td
            if not isinstance(embargo_td, pd.Timedelta):
                embargo_td = pd.Timedelta(embargo_td)
            embargo_ns = np.int64(embargo_td.value)
        else:
            order = np.arange(n)

        step = n // (self.n_splits + 1)
        if step < 1:
            raise ValueError(
                f"Series too short ({n} rows) for n_splits={self.n_splits}"
            )
        for k in range(1, self.n_splits + 1):
            train_end = k * step                      # exclusive, in SORTED space
            val_start = train_end
            val_end = min((k + 1) * step, n)
            if val_start >= n:
                break
            val_sorted = order[val_start:val_end]
            train_sorted_candidates = order[0:train_end]
            if use_time:
                val_first_ns = int(et_ns[val_sorted].min())
                cand = train_sorted_candidates
                keep = le_ns[cand] < val_first_ns
                train_idx = cand[keep]
                if embargo_ns > 0:
                    val_last_ns = int(et_ns[val_sorted].max())
                    zone = (et_ns[train_idx] > val_last_ns) & (
                        et_ns[train_idx] <= val_last_ns + embargo_ns
                    )
                    train_idx = train_idx[~zone]
                val_idx = val_sorted
            else:
                train_idx = train_sorted_candidates
                val_idx = val_sorted
            yield np.sort(train_idx).astype(int), np.sort(val_idx).astype(int)
