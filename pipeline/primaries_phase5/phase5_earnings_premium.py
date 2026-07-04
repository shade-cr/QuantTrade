"""B0017 — earnings announcement PREMIUM primary (pre-registered 2026-07-04).

Frozen rule (docs/superpowers/specs/2026-07-04-b0017-earnings-premium-preregistration.md,
committed BEFORE any data look; parameters are NOT tunable without a new DSR trial):

  +1 (long) on the FIRST bar whose date falls within
  [expected_date - ENTRY_DAYS_BEFORE business days, expected_date - 1 business day],
  one signal per expected event, no short leg, 0 otherwise.

where expected_date comes from pipeline.earnings_events.expected_announcement_schedule —
the PIT scheduler that predicts announcement k's date using ONLY announcements <= k-1
(median of trailing gaps; each expectation carries its `known_asof` timestamp).

Causal-window discipline:
  - A signal for expected event k may only fire on bars strictly AFTER that
    expectation's known_asof (the k-1 announcement's EDGAR acceptance time).
    This is enforced explicitly, not assumed.
  - The realized k-th announcement date is NEVER read by the signal — only
    expectations generated from the past. A mis-scheduled event simply misses
    its window (thinner sample, never lookahead).
  - Announcement knowledge timestamps are EDGAR acceptance times (UTC);
    bar dates are compared at day granularity with the bar's session date.

Asset identity: the pooled/backtest runner injects cfg["_current_asset"]
before calling signal(); this module fails loud without it (a silent default
would compute another ticker's calendar).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.earnings_events import (
    effective_knowledge_day,
    expected_announcement_schedule,
    filter_amendments,
    load_earnings_announcements,
)

INPUT_COLUMNS: tuple[str, ...] = ()  # reads only ohlcv index + earnings cache

ENTRY_DAYS_BEFORE = 3  # business days; FROZEN by pre-registration


def signal(ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: dict) -> pd.Series:
    asset = cfg.get("_current_asset")
    if not asset:
        raise ValueError(
            "phase5_earnings_premium requires cfg['_current_asset'] (injected by the "
            "runner) to load the ticker's earnings cache — refusing to guess"
        )
    # DA review 2026-07-04 high #2: strip 8-K/A amendments / same-quarter
    # re-disclosures BEFORE scheduling — they corrupt the median gap and
    # spawn bogus expectation windows.
    ann_clean = filter_amendments(load_earnings_announcements(asset).index)
    sched = expected_announcement_schedule(ann_clean)

    out = pd.Series(0, index=ohlcv.index, dtype=int)
    if sched.empty:
        return out

    # Normalized session dates of the bars (tz handled by comparing dates).
    bar_dates = pd.DatetimeIndex(ohlcv.index).tz_localize(None).normalize()
    # First session that could know each announcement (AMC rolls to next day).
    ann_known_days = effective_knowledge_day(ann_clean)

    for i, (expected_date, row) in enumerate(sched.iterrows()):
        exp = pd.Timestamp(expected_date).tz_localize(None).normalize()
        window_start = exp - pd.tseries.offsets.BusinessDay(ENTRY_DAYS_BEFORE)
        window_end = exp - pd.tseries.offsets.BusinessDay(1)
        known = pd.Timestamp(row["known_asof"]).tz_localize(None)
        # Eligible bars: inside the window AND strictly after the expectation
        # became known (belt-and-braces: known_asof precedes the window by ~a
        # quarter under the median-gap rule, but never assume).
        mask = (bar_dates >= window_start) & (bar_dates <= window_end) & (
            bar_dates > known.normalize()
        )
        # DA review 2026-07-04 medium: if the ACTUAL next announcement lands
        # early (inside/before the window), entering after it no longer
        # expresses the pre-announcement premium — cap eligibility at the
        # session that first knows announcement i+1.
        if i + 1 < len(ann_known_days):
            mask &= bar_dates < ann_known_days[i + 1]
        idx = np.flatnonzero(mask)
        if len(idx):
            out.iloc[idx[0]] = 1  # first bar in window only — one shot per event
    return out
