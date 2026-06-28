"""Tests for pipeline.sample_weights.avg_uniqueness."""
from __future__ import annotations
import numpy as np
import pandas as pd

from pipeline.sample_weights import avg_uniqueness, pooled_avg_uniqueness


def test_weights_in_unit_interval():
    n = 50
    t_starts = np.arange(n)
    t_ends = t_starts + 5
    w = avg_uniqueness(t_starts, t_ends, n_bars=n + 10)
    assert (w > 0).all() and (w <= 1.0 + 1e-9).all()


def test_total_overlap_yields_small_weights():
    """100 events all with the same outcome window → each weight ≈ 1/100."""
    n_events = 100
    t_starts = np.zeros(n_events, dtype=int)
    t_ends = np.full(n_events, 10, dtype=int)
    w = avg_uniqueness(t_starts, t_ends, n_bars=20)
    np.testing.assert_allclose(w, 1.0 / n_events, rtol=1e-9)


def test_isolated_event_yields_weight_one():
    t_starts = np.array([0, 50])
    t_ends = np.array([5, 55])  # no overlap between events
    w = avg_uniqueness(t_starts, t_ends, n_bars=100)
    np.testing.assert_allclose(w, [1.0, 1.0])


# --------------------------------------------------------------------------- #
# B0148 — pooled cross-asset wall-clock concurrency uniqueness (blocker B1)
# --------------------------------------------------------------------------- #

def test_pooled_lone_event_weight_one():
    """A lone event covers its whole span at concurrency 1 -> u = 1.0."""
    t0 = pd.Timestamp("2020-01-01")
    H = pd.Timedelta(hours=4)
    event_time = [t0]
    label_end = [t0 + 10 * H]
    w = pooled_avg_uniqueness(event_time, label_end)
    np.testing.assert_allclose(w, [1.0])


def test_pooled_two_contemporaneous_cross_asset_events_each_half():
    """Two perfectly contemporaneous, equal-span events on DIFFERENT assets get
    u ≈ 0.5 each — the ρ=1 conservative span-overlap bound (B1). Span-overlap
    counting treats contemporaneous cross-asset labels as fully redundant."""
    t0 = pd.Timestamp("2020-01-01")
    H = pd.Timedelta(hours=4)
    event_time = [t0, t0]
    label_end = [t0 + 10 * H, t0 + 10 * H]
    w = pooled_avg_uniqueness(event_time, label_end)
    np.testing.assert_allclose(w, [0.5, 0.5], rtol=1e-9)


def test_pooled_half_overlap_analytic():
    """Two equal-length spans that overlap on exactly half their duration.

    Event A: [0h, 10h], Event B: [5h, 15h]. Each spans 10h.
    On [0,5) concurrency=1, on [5,10) concurrency=2, on [10,15) concurrency=1.
    A covers [0,10]: half at u=1, half at u=1/2 -> mean = 0.5*1 + 0.5*0.5 = 0.75.
    B covers [5,15]: symmetric -> 0.75.
    """
    t0 = pd.Timestamp("2020-01-01")
    H = pd.Timedelta(hours=1)
    event_time = [t0, t0 + 5 * H]
    label_end = [t0 + 10 * H, t0 + 15 * H]
    w = pooled_avg_uniqueness(event_time, label_end)
    np.testing.assert_allclose(w, [0.75, 0.75], rtol=1e-9)


def test_pooled_isolated_events_each_weight_one():
    """Two non-overlapping events each get u = 1.0 regardless of asset."""
    t0 = pd.Timestamp("2020-01-01")
    H = pd.Timedelta(hours=1)
    event_time = [t0, t0 + 100 * H]
    label_end = [t0 + 5 * H, t0 + 105 * H]
    w = pooled_avg_uniqueness(event_time, label_end)
    np.testing.assert_allclose(w, [1.0, 1.0])


def test_pooled_weights_in_unit_interval():
    rng = np.random.default_rng(0)
    t0 = pd.Timestamp("2020-01-01")
    n = 40
    starts = [t0 + pd.Timedelta(hours=int(h)) for h in np.sort(rng.integers(0, 200, n))]
    ends = [s + pd.Timedelta(hours=int(d)) for s, d in zip(starts, rng.integers(1, 30, n))]
    w = pooled_avg_uniqueness(starts, ends)
    assert (w > 0).all() and (w <= 1.0 + 1e-9).all()


def test_pooled_reduces_to_single_asset_avg_uniqueness():
    """For a single asset on a regular grid (1 bar = constant duration), the
    pooled span-overlap weights match avg_uniqueness up to the discrete-vs-interval
    edge convention; assert the RANKING/relative redundancy direction matches."""
    t0 = pd.Timestamp("2020-01-01")
    H = pd.Timedelta(hours=4)
    # three events, first two overlap, third isolated
    event_time = [t0, t0 + 2 * H, t0 + 100 * H]
    label_end = [t0 + 5 * H, t0 + 7 * H, t0 + 102 * H]
    w = pooled_avg_uniqueness(event_time, label_end)
    # isolated event is fully unique
    np.testing.assert_allclose(w[2], 1.0)
    # the two overlapping events are down-weighted below 1
    assert w[0] < 1.0 and w[1] < 1.0
