"""Walk-forward splitter with purge & embargo, plus inner-CV PurgedTimeSeriesSplit.

References: AFML chapter 7 (Cross-Validation in Finance).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator
import numpy as np
from sklearn.model_selection import BaseCrossValidator


@dataclass(frozen=True)
class FoldIndices:
    train_idx: np.ndarray
    test_idx: np.ndarray


class WalkForwardGeometryError(ValueError):
    """Raised by make_folds when the event count is below the empirical refusal
    cliff `n - train_min >= n_folds * 100` (B0013).

    IS-A ValueError so existing `except ValueError:` handlers keep working
    unchanged; new callers may isinstance-check to read the structured fields
    and name the relaxable axis (lower n_folds, lower train_min, or more events).

    The cliff is documented empirically in phase5/audit_null_distribution.md §B3
    — at defaults (n_folds=3, train_min=300) it sits at n >= 600 events.
    """

    def __init__(self, events: int, n_folds: int, train_min: int):
        self.events = int(events)
        self.n_folds = int(n_folds)
        self.train_min = int(train_min)
        self.test_pool_needed = n_folds * 100
        self.required_events = train_min + self.test_pool_needed
        self.shortfall = self.required_events - self.events
        super().__init__(
            f"walk-forward refusal cliff: {self.events} events < required "
            f"{self.required_events} (= train_min {self.train_min} + n_folds {self.n_folds} "
            f"× 100 test-pool bars); short by {self.shortfall}. Relax ONE axis: "
            f"lower n_folds, lower train_min, or obtain more events (less selective "
            f"primary / wider regime scope). See phase5/audit_null_distribution.md §B3."
        )


def resolve_train_min(train_min_bars: int, n_events: int) -> int:
    """Resolve the train_min (in events) actually passed to make_folds.

    The orchestrator caps the configured `train_min_bars` at half the available
    events so a sparse primary still leaves a test pool. Centralized here (was
    inline in scripts/run_backtest.py) so the pre-flight floor projection in
    `wf_event_floor` cannot drift from what the audit actually does.
    """
    return min(int(train_min_bars), int(n_events) // 2)


def wf_event_floor(n_folds: int, train_min_bars: int) -> int:
    """Smallest in-regime event count for which make_folds will NOT raise.

    B0048: pre-flight needs to project the walk-forward refusal cliff WITHOUT
    running the audit. Because the resolved train_min is `min(train_min_bars,
    n//2)`, the floor is a clean function of the geometry alone — but NOT simply
    `train_min_bars + n_folds*100`: in the sparse regime train_min collapses to
    n//2, which is why T011D2's floor was 217 (train_min 17), not 300. Computed
    iteratively against the exact make_folds predicate to guarantee no drift.
    """
    test_pool_needed = int(n_folds) * 100
    n = 1
    while n - resolve_train_min(train_min_bars, n) < test_pool_needed:
        n += 1
    return n


def make_folds(
    n: int,
    n_folds: int,
    train_min: int,
    purge: int,
    embargo_pct: float,
    event_start_bar: np.ndarray | None = None,
    event_end_bar: np.ndarray | None = None,
) -> list[FoldIndices]:
    """Expanding-window WF with purge before each test and embargo after past tests.

    Test pool = [train_min, n-1], divided into n_folds chunks (the last absorbs the
    remainder when (n-train_min) is not divisible by n_folds).

    Event-space vs bar-space, and the AFML §7.4.1 purge (B0129)
    ----------------------------------------------------------
    This splitter operates in EVENT-INDEX space ``[0, n)`` — fold boundaries are
    event positions. But each triple-barrier event ``i`` carries a LABEL that does
    not resolve until a later BAR position ``event_end_bar[i]`` (its ``t_end_idx``),
    while it OPENS at ``event_start_bar[i]``. AFML §7.4.1 Snippet 7.1
    (``getTrainTimes``) requires purging every training observation whose
    label-outcome window ``[event_start_bar[j], event_end_bar[j]]`` overlaps the
    test span — otherwise the model trains on an event whose label was still
    "in flight" when the test window opened (look-ahead leak).

    Two purge modes
    ~~~~~~~~~~~~~~~~
    * **Exact label-end purge** (both ``event_start_bar`` and ``event_end_bar``
      given): for each fold the test window's first BAR is
      ``event_start_bar[ts_start]``. A training event ``j < ts_start`` is PURGED
      iff ``event_end_bar[j] >= event_start_bar[ts_start]`` — i.e. its label
      reaches into or past the test window's first bar. Only events whose label
      fully resolves *before* the test window opens are retained. This is the
      AFML-correct rule and is density-INDEPENDENT.

    * **Scalar fallback** (arrays ``None`` — the legacy default): purge a FIXED
      number of EVENTS, ``train_end_exclusive = ts_start - purge``. This is
      DENSITY-FRAGILE: ``purge`` events span ``purge × (bars-per-event)`` bars,
      so a scalar that over-covers a sparse primary (many bars/event) silently
      UNDER-purges a dense one (``~1`` bar/event), where ``purge=20`` events
      ``≈ 20`` bars ``<`` a 40-bar horizon — a real look-ahead leak in the
      ``_run_supervised_direct`` mode. The scalar path is preserved verbatim for
      backward compatibility; pass the t1 arrays to get the correct purge.

    Embargo (AFML Snippet 7.2 ``getEmbargoTimes``) — the removal of the first
    ``embargo`` event positions immediately following each PRIOR test block — is
    applied identically in both modes.
    """
    if n - train_min < n_folds * 100:
        raise WalkForwardGeometryError(events=n, n_folds=n_folds, train_min=train_min)

    use_label_end = event_start_bar is not None or event_end_bar is not None
    if use_label_end:
        if event_start_bar is None or event_end_bar is None:
            raise ValueError(
                "make_folds: event_start_bar and event_end_bar must BOTH be "
                "provided for the AFML §7.4.1 exact label-end purge (got only one)."
            )
        event_start_bar = np.asarray(event_start_bar)
        event_end_bar = np.asarray(event_end_bar)
        if len(event_start_bar) != n or len(event_end_bar) != n:
            raise ValueError(
                f"make_folds: event_start_bar / event_end_bar must each have "
                f"length n={n} (got {len(event_start_bar)} and {len(event_end_bar)})."
            )

    embargo = int(np.ceil(embargo_pct * n))
    test_pool = n - train_min
    base_size = test_pool // n_folds
    sizes = [base_size] * n_folds
    sizes[-1] += test_pool - base_size * n_folds

    folds: list[FoldIndices] = []
    cursor = train_min
    test_ranges: list[tuple[int, int]] = []
    for size in sizes:
        ts_start = cursor
        ts_end = cursor + size - 1
        test_ranges.append((ts_start, ts_end))
        cursor = ts_end + 1

    for k, (ts_start, ts_end) in enumerate(test_ranges):
        if use_label_end:
            # AFML §7.4.1 Snippet 7.1 getTrainTimes: retain only training events
            # whose LABEL fully resolves before the test window's first bar.
            test_first_bar = int(event_start_bar[ts_start])
            candidates = np.arange(0, ts_start, dtype=int)
            keep = event_end_bar[candidates] < test_first_bar
            train_pool = set(candidates[keep].tolist())
        else:
            # Legacy scalar-event purge (density-fragile — see docstring).
            train_end_exclusive = ts_start - purge
            train_pool = set(range(0, train_end_exclusive))
        for j in range(k):
            prev_end = test_ranges[j][1]
            embargo_zone = range(prev_end + 1, prev_end + 1 + embargo)
            train_pool -= set(embargo_zone)
        # Also remove indices that fall inside ANY test range from earlier folds.
        # (Embargo only removes the first `embargo` bars after each prior test.
        # Earlier test bars themselves are intentionally kept as train — see spec.)
        train_idx = np.array(sorted(train_pool), dtype=int)
        test_idx = np.arange(ts_start, ts_end + 1, dtype=int)
        folds.append(FoldIndices(train_idx=train_idx, test_idx=test_idx))
    return folds


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """sklearn-compatible CV: forward-rolling splits with a purge gap between tr/va.

    Each split: train = [0, k*step], val = [k*step + purge, (k+1)*step + purge].
    The last split runs to the end of the array.

    Purge modes (B0129, AFML §7.4.1 Snippet 7.1 ``getTrainTimes``)
    -------------------------------------------------------------
    This splitter indexes EVENTS positionally, but each event's triple-barrier
    LABEL resolves at a later BAR. Two purge modes mirror ``make_folds``:

    * **Exact label-end purge** (``event_start_bar`` AND ``event_end_bar``
      given, each length ``len(X)``): a candidate train position ``j`` is kept
      only if its label-end bar resolves strictly before the val window's first
      bar, ``event_end_bar[j] < event_start_bar[val_start]``. Density-independent
      and AFML-correct.

    * **Scalar fallback** (arrays ``None`` — legacy default): a fixed ``purge``
      *positions* gap between ``train_end`` and ``val_start``. DENSITY-FRAGILE:
      ``purge`` positions span ``purge × bars-per-event`` bars, so a scalar tuned
      for a sparse primary silently UNDER-purges a dense one (``~1`` bar/event,
      where ``purge`` positions ``<`` the label horizon → look-ahead leak).

    Embargo (AFML Snippet 7.2 ``getEmbargoTimes``)
    ----------------------------------------------
    ``embargo`` drops the first ``embargo`` TRAIN positions immediately FOLLOWING
    the val block (the head-gap of the next partition). Default 0 preserves the
    legacy expanding-only train window exactly. The ``split`` signature stays
    ``(self, X, y=None, groups=None)`` for sklearn compatibility.
    """

    def __init__(
        self,
        n_splits: int = 3,
        purge: int = 0,
        event_start_bar: np.ndarray | None = None,
        event_end_bar: np.ndarray | None = None,
        embargo: int = 0,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if (event_start_bar is None) != (event_end_bar is None):
            raise ValueError(
                "PurgedTimeSeriesSplit: event_start_bar and event_end_bar must "
                "BOTH be provided for the AFML §7.4.1 exact label-end purge."
            )
        self.n_splits = n_splits
        self.purge = purge
        self.event_start_bar = event_start_bar
        self.event_end_bar = event_end_bar
        self.embargo = embargo

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        use_label_end = self.event_start_bar is not None
        if use_label_end:
            esb = np.asarray(self.event_start_bar)
            eeb = np.asarray(self.event_end_bar)
            if len(esb) != n or len(eeb) != n:
                raise ValueError(
                    f"PurgedTimeSeriesSplit: event_start_bar / event_end_bar must "
                    f"each have length len(X)={n}."
                )
        # Reserve at least 1/(n_splits+1) of the data for val on the first split.
        step = n // (self.n_splits + 1)
        if step <= self.purge:
            raise ValueError(f"Series too short for purge={self.purge} with n_splits={self.n_splits}")
        for k in range(1, self.n_splits + 1):
            train_end = k * step
            val_start = train_end + self.purge
            val_end = min((k + 1) * step + self.purge, n)
            if val_start >= n:
                break
            val_idx = np.arange(val_start, val_end)
            if use_label_end:
                # AFML §7.4.1 getTrainTimes: keep train positions whose label-end
                # bar resolves before the val window's first bar.
                val_first_bar = int(esb[val_start])
                candidates = np.arange(0, train_end)
                train_idx = candidates[eeb[candidates] < val_first_bar]
            else:
                train_idx = np.arange(0, train_end)
            if self.embargo > 0:
                # Drop the first `embargo` train positions immediately following
                # the val block (AFML Snippet 7.2 getEmbargoTimes head-gap).
                embargo_zone = set(range(val_end, val_end + self.embargo))
                if embargo_zone:
                    train_idx = train_idx[~np.isin(train_idx, list(embargo_zone))]
            yield train_idx, val_idx


def inner_oof_predict_proba(estimator, X, y, cv, sample_weight=None,
                             return_val_indices: bool = False):
    """Manual replacement for sklearn.cross_val_predict tolerant of non-partition CVs.

    For each (train_idx, val_idx) yielded by `cv.split(X, y)`, fits a fresh
    `clone(estimator)` on the train rows and stores its `predict_proba` on
    the val rows. Rows outside any val fold remain NaN.

    Why this exists instead of `cross_val_predict`: sklearn requires the CV
    to be a partition (every input row appears in exactly one val fold) and
    raises `ValueError("cross_val_predict only works for partitions")`
    otherwise. `PurgedTimeSeriesSplit` is intentionally not a partition —
    early rows and purge-zone rows are never assigned to a val fold by
    design. This helper accepts that and propagates NaN, which downstream
    code (`select_threshold_inner_cv`'s NaN-masking sub-blocks) already
    handles.

    Parameters
    ----------
    estimator : sklearn-compatible estimator with `fit` and `predict_proba`.
        Must accept `sample_weight=` in `fit` if `sample_weight` is provided.
        Cloned per fold via `sklearn.base.clone`, so any fitted state is
        discarded between folds.
    X : pd.DataFrame
        Feature matrix. Indexed by `cv.split` via `.iloc[idx]`.
    y : pd.Series
        Labels. Same indexing.
    cv : sklearn-compatible splitter
        Typically `PurgedTimeSeriesSplit`. Anything with a `split(X, y)`
        method yielding `(train_idx, val_idx)` arrays works.
    sample_weight : np.ndarray | None
        If provided, sliced by `train_idx` per fold and passed to
        `est.fit(..., sample_weight=w[train_idx])`.
    return_val_indices : bool
        If True, also return the list of val_idx arrays (one per CV split).
        Downstream code (e.g. `select_threshold_inner_cv`) can use these
        as sub-blocks to evaluate metrics over exactly the regions where
        OOF probs were produced, matching the per-CV-fold averaging that
        the Phase 1 path used.

    Returns
    -------
    out : np.ndarray, shape (n, n_classes)
        OOF probabilities indexed positionally against `X`. Rows outside
        any val fold are NaN across all class columns.
    val_indices : list[np.ndarray]  (only when return_val_indices=True)
        The val_idx arrays as yielded by `cv.split(X, y)`, in iteration order.
    """
    from sklearn.base import clone

    n = len(X)
    n_classes = len(np.unique(np.asarray(y)))
    out = np.full((n, n_classes), np.nan)
    val_indices_list: list[np.ndarray] = []
    for train_idx, val_idx in cv.split(X, y):
        est = clone(estimator)
        if sample_weight is None:
            est.fit(X.iloc[train_idx], y.iloc[train_idx])
        else:
            est.fit(
                X.iloc[train_idx], y.iloc[train_idx],
                sample_weight=np.asarray(sample_weight)[train_idx],
            )
        out[val_idx] = est.predict_proba(X.iloc[val_idx])
        val_indices_list.append(val_idx)
    if return_val_indices:
        return out, val_indices_list
    return out
