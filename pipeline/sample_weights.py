"""Compute López-de-Prado average-uniqueness sample weights.

References:
  Marcos López de Prado, *Advances in Financial Machine Learning*, Wiley 2018,
  Chapter 4 (Sample Weights). The function mp_sample_tw computes, for each
  event i, the mean of (1 / num_co_events(t)) over the outcome bars
  t ∈ [t_start_i, t_end_i].
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def avg_uniqueness(t_starts: np.ndarray, t_ends: np.ndarray, n_bars: int) -> np.ndarray:
    """Return per-event sample weights ∈ (0, 1] via AFML §4 average-uniqueness.

    Args:
      t_starts: int array, position of each event's entry bar in the ohlcv frame.
      t_ends:   int array, position of each event's outcome resolution bar.
      n_bars:   total number of bars (sets the size of the concurrency count vector).
    """
    t_starts = np.asarray(t_starts, dtype=int)
    t_ends = np.asarray(t_ends, dtype=int)
    if t_starts.shape != t_ends.shape:
        raise ValueError("t_starts and t_ends must have the same shape")
    if (t_ends < t_starts).any():
        raise ValueError("t_ends must be >= t_starts")

    # Concurrency: how many open events cover each bar.
    co_events = np.zeros(n_bars, dtype=int)
    for s, e in zip(t_starts, t_ends):
        co_events[s : e + 1] += 1

    weights = np.empty(len(t_starts), dtype=float)
    for k, (s, e) in enumerate(zip(t_starts, t_ends)):
        weights[k] = (1.0 / co_events[s : e + 1]).mean()
    return weights


def pooled_avg_uniqueness(
    event_time, label_end_time
) -> np.ndarray:
    """Cross-asset wall-clock average-uniqueness weights ∈ (0, 1] (AFML §4.3, B0148).

    Generalizes :func:`avg_uniqueness` from single-asset bar positions to a SHARED
    wall-clock timeline built from the union of every event's ``[event_time,
    label_end_time]`` span across the whole pool. This is the load-bearing
    correctness fix for cross-asset pooling (spec blocker B1): concatenating
    per-asset weight vectors does NOT down-weight contemporaneous events on
    different assets, which inflates the effective-N premise behind the pooled
    DSR/MinBTL read.

    Concurrency model (the ρ=1 conservative span-overlap bound)
    -----------------------------------------------------------
    The union of all span endpoints partitions the timeline into atomic
    sub-intervals. Concurrency on each sub-interval = number of events whose span
    covers it. Each event's weight = the DURATION-WEIGHTED mean of ``1/concurrency``
    over the sub-intervals it covers.

    AFML §4.3 defines concurrency as labels sharing a common return ``r_{t-1,t}``;
    within one asset overlapping labels literally share the draw. Across assets,
    contemporaneous events have *correlated but distinct* returns (ρ≈0.8, not 1.0),
    so pure span-overlap counting (two contemporaneous events → u≈0.5) treats them
    as fully redundant — the ρ=1 limit. This OVER-penalizes, which is the SAFE
    direction for a validation gate (shrinks effective-N → harder to clear DSR,
    never easier → no leak). The faithful correlation-weighted generalization is
    deferred (spec [R2 — corpus caveat]).

    Args:
      event_time:     array-like of pandas Timestamps — each event's entry-bar time.
      label_end_time: array-like of pandas Timestamps — each event's triple-barrier
                      resolution time (its own asset's ``t_end_idx`` bar timestamp).
                      Must satisfy ``label_end_time[i] >= event_time[i]``.

    Returns:
      np.ndarray of per-event weights in the ORIGINAL input order. Empty input
      returns an empty float array.
    """
    starts = pd.DatetimeIndex(pd.to_datetime(list(event_time)))
    ends = pd.DatetimeIndex(pd.to_datetime(list(label_end_time)))
    if len(starts) != len(ends):
        raise ValueError("event_time and label_end_time must have the same length")
    n = len(starts)
    if n == 0:
        return np.empty(0, dtype=float)
    if (ends < starts).any():
        raise ValueError("label_end_time must be >= event_time for every event")

    s = starts.asi8.astype(np.int64)   # ns since epoch
    e = ends.asi8.astype(np.int64)

    # Atomic sub-interval boundaries: the sorted union of all span endpoints.
    boundaries = np.unique(np.concatenate([s, e]))
    # Sub-interval i spans [boundaries[i], boundaries[i+1]); its duration in ns.
    seg_lo = boundaries[:-1]
    seg_hi = boundaries[1:]
    seg_dur = (seg_hi - seg_lo).astype(np.float64)
    n_seg = len(seg_lo)
    if n_seg == 0:
        # All spans are zero-duration single instants → treat each as fully unique
        # unless they coincide. Fall back to instant-point concurrency.
        return _instant_point_uniqueness(s, e)

    # Concurrency per sub-interval: an event covers sub-interval i iff its span
    # [s_k, e_k] overlaps [seg_lo[i], seg_hi[i]). Using half-open segments with
    # an inclusive event end means coverage = (s_k <= seg_lo[i]) & (e_k >= seg_hi[i]).
    concurrency = np.zeros(n_seg, dtype=np.int64)
    covers = np.empty((n, n_seg), dtype=bool)
    for k in range(n):
        c = (s[k] <= seg_lo) & (e[k] >= seg_hi)
        covers[k] = c
        concurrency += c.astype(np.int64)

    # Guard against zero concurrency on a covered segment (shouldn't happen since
    # every covered segment is covered by >=1 event, but be safe).
    inv_conc = np.where(concurrency > 0, 1.0 / np.maximum(concurrency, 1), 0.0)

    weights = np.empty(n, dtype=float)
    for k in range(n):
        cov = covers[k]
        dur = seg_dur[cov]
        total = dur.sum()
        if total <= 0:
            # zero-duration event (s_k == e_k): single instant → unique
            weights[k] = 1.0
        else:
            weights[k] = float((inv_conc[cov] * dur).sum() / total)
    return weights


def _instant_point_uniqueness(s: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Degenerate fallback: all spans are single instants (s == e). Concurrency at
    each instant = number of events sharing that exact timestamp."""
    n = len(s)
    weights = np.empty(n, dtype=float)
    for k in range(n):
        same = int((s == s[k]).sum())
        weights[k] = 1.0 / same
    return weights
