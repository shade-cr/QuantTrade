"""UTC session filtering and statistical-power gating (Phase 2 T4).

Session ranges (UTC, all closed on the left, open on the right):
  - LONDON:             [07:00, 13:00)
  - LONDON_NY_OVERLAP:  [13:00, 17:00)
  - NEW_YORK:           [17:00, 22:00)
  - ASIA:               everything else (i.e., [22:00, 07:00) crossing midnight)

The module is asset-agnostic — it does NOT know about FX weekends or
crypto 24/7. Callers (the multi-asset orchestrator) decide whether to
exclude weekend bars before passing data to `filter_by_session`.
"""
from __future__ import annotations
from typing import Final
import pandas as pd


SESSION_LONDON: Final[str] = "LONDON"
SESSION_OVERLAP: Final[str] = "LONDON_NY_OVERLAP"
SESSION_NY: Final[str] = "NEW_YORK"
SESSION_ASIA: Final[str] = "ASIA"

ALL_SESSIONS: Final[tuple[str, ...]] = (
    SESSION_LONDON, SESSION_OVERLAP, SESSION_NY, SESSION_ASIA,
)


def get_session(timestamp: pd.Timestamp) -> str:
    """Map a single UTC timestamp to its session label.

    tz-naive timestamps are interpreted as UTC (project convention —
    `pipeline/data.py::load_dataset` already enforces UTC tz-aware
    indices, so this fallback only affects synthetic / test data).
    """
    hour = timestamp.hour
    if 7 <= hour < 13:
        return SESSION_LONDON
    if 13 <= hour < 17:
        return SESSION_OVERLAP
    if 17 <= hour < 22:
        return SESSION_NY
    return SESSION_ASIA


def get_session_series(index: pd.DatetimeIndex) -> pd.Series:
    """Map a DatetimeIndex to a Series of session labels (vectorised).

    O(n) — uses pandas hour accessor + boolean masks, not a Python loop.
    Output index equals the input index.
    """
    hours = index.hour
    labels = pd.Series(SESSION_ASIA, index=index, dtype=object)
    labels[(hours >= 7) & (hours < 13)] = SESSION_LONDON
    labels[(hours >= 13) & (hours < 17)] = SESSION_OVERLAP
    labels[(hours >= 17) & (hours < 22)] = SESSION_NY
    return labels


def filter_by_session(events: pd.DataFrame, session_name: str) -> pd.DataFrame:
    """Return the subset of `events` whose timestamps fall in the named session.

    The events frame must have a DatetimeIndex. Column data passes through
    unchanged for surviving rows.
    """
    if session_name not in ALL_SESSIONS:
        raise ValueError(
            f"unknown session {session_name!r}; expected one of {ALL_SESSIONS}"
        )
    labels = get_session_series(events.index)
    return events.loc[labels == session_name]


def evaluate_session_viability(
    events_session: pd.DataFrame,
    n_folds: int = 4,
    min_events_total: int = 250,
    min_events_per_fold: int = 50,
) -> dict:
    """Statistical-power gate: is this session worth training on?

    Two criteria (both must hold):
      - n_events ≥ min_events_total: López de Prado §14 approximation —
        to detect Sharpe > 0.3 with power 0.8 at alpha 0.05 requires
        ~250 events.
      - events_per_fold ≥ min_events_per_fold: each WF fold needs enough
        data to estimate Sharpe at the per-fold level.

    Returns
    -------
    dict with keys:
      - viable: bool
      - n_events: int
      - events_per_fold: float
      - reason: str (only when viable=False)
    """
    n_events = len(events_session)
    events_per_fold = n_events / n_folds if n_folds > 0 else 0.0

    if n_events < min_events_total:
        return {
            "viable": False,
            "n_events": n_events,
            "events_per_fold": events_per_fold,
            "reason": (
                f"n_events={n_events} < {min_events_total} "
                f"(insufficient statistical power)"
            ),
        }
    if events_per_fold < min_events_per_fold:
        return {
            "viable": False,
            "n_events": n_events,
            "events_per_fold": events_per_fold,
            "reason": (
                f"events_per_fold={events_per_fold:.1f} < {min_events_per_fold} "
                f"({n_events} events / {n_folds} folds insufficient)"
            ),
        }
    return {
        "viable": True,
        "n_events": n_events,
        "events_per_fold": events_per_fold,
    }
