"""Tests for UTC session filtering and statistical-power gating (T4).

The module maps UTC timestamps to one of 4 session labels and provides a
viability gate used by the multi-asset orchestrator (T9.B) to decide
which per-session models are worth training.

Edge-cases that matter:
  - Session boundaries are half-open intervals: 13:00 UTC is OVERLAP, not LONDON
  - ASIA wraps around midnight (22:00 → 07:00 next day) — handled by negation
  - Vector path (`get_session_series`) returns one label per row in O(n)
  - The filter is asset-agnostic: crypto weekend bars and FX weekday bars
    are mapped by their UTC hour only; weekend exclusion is the caller's job
"""
from __future__ import annotations
import pandas as pd
import pytest

from pipeline.session_filter import (
    ALL_SESSIONS,
    SESSION_LONDON, SESSION_OVERLAP, SESSION_NY, SESSION_ASIA,
    get_session,
    get_session_series,
    filter_by_session,
    evaluate_session_viability,
)


# ---------------------------------------------------------------------------
# get_session: scalar mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hour, expected", [
    (7,  SESSION_LONDON),     # inclusive start of LONDON
    (10, SESSION_LONDON),
    (12, SESSION_LONDON),     # last hour of LONDON
    (13, SESSION_OVERLAP),    # start of OVERLAP, end of LONDON
    (15, SESSION_OVERLAP),
    (16, SESSION_OVERLAP),    # last hour of OVERLAP
    (17, SESSION_NY),         # start of NY
    (20, SESSION_NY),
    (21, SESSION_NY),         # last hour of NY
    (22, SESSION_ASIA),       # ASIA starts here, wraps midnight
    (3,  SESSION_ASIA),
    (5,  SESSION_ASIA),
    (6,  SESSION_ASIA),       # last hour of ASIA before LONDON opens
])
def test_get_session_maps_utc_hour_to_session(hour, expected):
    ts = pd.Timestamp(f"2024-03-15 {hour:02d}:00:00", tz="UTC")
    assert get_session(ts) == expected


def test_get_session_accepts_tz_naive_timestamps_as_utc():
    """tz-naive timestamps are interpreted as UTC (project convention)."""
    ts = pd.Timestamp("2024-03-15 09:00:00")  # no tz
    assert get_session(ts) == SESSION_LONDON


# ---------------------------------------------------------------------------
# get_session_series: vector mapping
# ---------------------------------------------------------------------------

def test_get_session_series_returns_one_label_per_row():
    idx = pd.date_range("2024-03-15 00:00", periods=24, freq="h", tz="UTC")
    s = get_session_series(idx)
    assert len(s) == 24
    assert set(s.unique()) <= set(ALL_SESSIONS)
    assert s.index.equals(idx)


def test_get_session_series_consistent_with_scalar():
    idx = pd.date_range("2024-03-15 00:00", periods=24, freq="h", tz="UTC")
    s = get_session_series(idx)
    for i in range(24):
        assert s.iloc[i] == get_session(idx[i]), f"hour={i}: vector != scalar"


# ---------------------------------------------------------------------------
# filter_by_session: subset by session
# ---------------------------------------------------------------------------

def test_filter_by_session_returns_only_matching_bars():
    """Build a 24-hour index, filter for OVERLAP; only hours 13..16 survive."""
    idx = pd.date_range("2024-03-15 00:00", periods=24, freq="h", tz="UTC")
    events = pd.DataFrame({"x": range(24)}, index=idx)
    filtered = filter_by_session(events, SESSION_OVERLAP)
    expected_hours = {13, 14, 15, 16}
    assert set(filtered.index.hour) == expected_hours
    assert len(filtered) == 4


def test_filter_by_session_for_asia_includes_pre_and_post_midnight():
    """ASIA spans [22:00, 07:00), which crosses midnight. The filter must
    include both the late-night (22:00, 23:00) and early-morning (00:00–06:00)
    bars."""
    idx = pd.date_range("2024-03-15 00:00", periods=24, freq="h", tz="UTC")
    events = pd.DataFrame({"x": range(24)}, index=idx)
    filtered = filter_by_session(events, SESSION_ASIA)
    expected_hours = {22, 23, 0, 1, 2, 3, 4, 5, 6}
    assert set(filtered.index.hour) == expected_hours


def test_filter_by_session_raises_on_unknown_session_name():
    idx = pd.date_range("2024-03-15 00:00", periods=24, freq="h", tz="UTC")
    events = pd.DataFrame({"x": range(24)}, index=idx)
    with pytest.raises(ValueError, match="unknown session"):
        filter_by_session(events, "ROGUE_SESSION")


def test_filter_preserves_event_metadata():
    """The filter is index-based — column data passes through unchanged
    for surviving rows."""
    idx = pd.date_range("2024-03-15 00:00", periods=24, freq="h", tz="UTC")
    events = pd.DataFrame({
        "side": [1] * 24,
        "fwd_return": [0.001 * i for i in range(24)],
    }, index=idx)
    filtered = filter_by_session(events, SESSION_LONDON)
    # LONDON = hours 7..12 inclusive → 6 rows
    assert len(filtered) == 6
    # fwd_return at hour 7 should be 0.007, at hour 12 should be 0.012
    assert filtered.loc[pd.Timestamp("2024-03-15 07:00", tz="UTC"), "fwd_return"] == pytest.approx(0.007)
    assert filtered.loc[pd.Timestamp("2024-03-15 12:00", tz="UTC"), "fwd_return"] == pytest.approx(0.012)


# ---------------------------------------------------------------------------
# evaluate_session_viability: power-check gate
# ---------------------------------------------------------------------------

def _events(n: int) -> pd.DataFrame:
    """Build a dummy events frame of length n. Content irrelevant — gate
    only inspects length."""
    idx = pd.date_range("2024-03-15 00:00", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"x": range(n)}, index=idx)


def test_viability_below_total_events_floor_is_not_viable():
    """Default floor is 250 total events. 100 events fails."""
    result = evaluate_session_viability(_events(100))
    assert result["viable"] is False
    assert result["n_events"] == 100
    assert "n_events" in result["reason"]


def test_viability_below_per_fold_floor_is_not_viable():
    """For n_folds=6 and 250 events, events_per_fold ≈ 41.7 < 50 → fails the
    per-fold gate even though total events ≥ 250."""
    result = evaluate_session_viability(_events(250), n_folds=6)
    assert result["viable"] is False
    assert "events_per_fold" in result["reason"]


def test_viability_with_enough_events_is_viable():
    """250 total events, 4 folds → 62.5 events/fold → passes both gates."""
    result = evaluate_session_viability(_events(250), n_folds=4)
    assert result["viable"] is True
    assert result["n_events"] == 250
    assert result["events_per_fold"] == pytest.approx(62.5)
    assert "reason" not in result or result.get("reason") in ("", None)


def test_viability_respects_overrides():
    """The thresholds are configurable so callers (e.g. crypto with more
    events) can tune them per asset class."""
    result = evaluate_session_viability(
        _events(150), n_folds=4,
        min_events_total=100, min_events_per_fold=20,
    )
    assert result["viable"] is True
