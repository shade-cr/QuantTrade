"""B0017 — earnings announcement events from SEC EDGAR 8-K (Item 2.02).

Point-in-time discipline
------------------------
The KNOWLEDGE timestamp of an earnings announcement is the EDGAR acceptance
timestamp of the 8-K that discloses it (Item 2.02 "Results of Operations and
Financial Condition"). Vendor earnings calendars are backfilled from current
data and are NOT point-in-time; EDGAR acceptance timestamps are what the
market could actually observe, to the second, since mid-2003.

Two distinct objects live here:

1. ``load_earnings_announcements`` — the realized announcement history
   (per ticker, cached parquet written by scripts/fetch_earnings_dates.py).
2. ``expected_announcement_schedule`` — the PIT-clean *ex ante* schedule.
   Trading "N days before the announcement" requires knowing the date in
   advance; the pre-registered scheduling rule uses ONLY past announcements:

       expected_next = last_announcement + median(historical gaps)

   with the median over the trailing ``lookback`` gaps (default 8 ≈ 2 years)
   and a fallback gap of 91 calendar days when fewer than ``min_gaps`` are
   observed. Each expected date is stamped with ``known_asof`` = the
   announcement that generated it, so a backtest can verify no forward
   knowledge is used. Empirically large caps announce on a stable quarterly
   cadence; scheduling error is a few days — the event window must tolerate
   that (it is part of the hypothesis, not a bug to be fixed ex post).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EARNINGS_CACHE_DIR = Path("data/earnings")

# Canonical names of the config-gated calendar features joined by the pooled
# runner (features.earnings_calendar) — unioned into the tier2 registry test
# exactly like GLD_VOLUME_FEATURES (same-commit sync enforcement).
EARNINGS_CALENDAR_FEATURES = ("days_since_last_announcement", "days_to_expected_earnings")
FALLBACK_GAP_DAYS = 91
DEFAULT_LOOKBACK_GAPS = 8
MIN_GAPS = 3

# US session close in UTC. 16:00 ET is 20:00 UTC (EDT) / 21:00 UTC (EST); the
# EARLIER bound (20:00) is used so ambiguity always defers knowledge to the
# next session — conservative, never a leak. DA review 2026-07-04, high #1.
SESSION_CLOSE_UTC_HOUR = 20
# Announcements closer than this to the previously kept one are 8-K/A
# amendments / duplicate disclosures, not new earnings events (quarterly
# cadence is ~63 BD). DA review 2026-07-04, high #2.
AMENDMENT_MIN_GAP_DAYS = 30


def effective_knowledge_day(acceptance: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Map EDGAR acceptance timestamps (UTC) to the first SESSION DATE whose
    close could know them: acceptances at/after the session close roll to the
    next calendar day. Direction-safe: may defer knowledge, never advance it."""
    ts = pd.DatetimeIndex(acceptance)
    naive = ts.tz_localize(None) if ts.tz is not None else ts
    rolled = naive.normalize() + pd.to_timedelta(
        (naive.hour >= SESSION_CLOSE_UTC_HOUR).astype(int), unit="D"
    )
    return pd.DatetimeIndex(rolled)


def filter_amendments(
    announcements: pd.DatetimeIndex, min_gap_days: int = AMENDMENT_MIN_GAP_DAYS
) -> pd.DatetimeIndex:
    """Drop announcements closer than ``min_gap_days`` to the previously KEPT
    one (keep the earliest of each cluster). Removes 8-K/A amendments and
    same-quarter re-disclosures that would otherwise corrupt the median-gap
    scheduler and spawn bogus entry windows."""
    ts = pd.DatetimeIndex(announcements).sort_values()
    if len(ts) == 0:
        return ts
    kept = [ts[0]]
    for t in ts[1:]:
        if (t - kept[-1]).days >= min_gap_days:
            kept.append(t)
    return pd.DatetimeIndex(kept)


def load_earnings_announcements(
    ticker: str, cache_dir: Path = EARNINGS_CACHE_DIR
) -> pd.DataFrame:
    """Read the cached 8-K Item 2.02 history for one ticker.

    Returns a DataFrame indexed by UTC acceptance timestamp (tz-aware,
    ascending, unique) with columns ``filing_date`` (date) and ``items``
    (str). Raises FileNotFoundError with a remediation hint if the cache is
    absent — fail loud, never silently return an empty frame.
    """
    p = Path(cache_dir) / f"{ticker}_8k202.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"No earnings cache at {p} — run "
            f"`uv run python scripts/fetch_earnings_dates.py --ticker {ticker}` first."
        )
    df = pd.read_parquet(p)
    if df.index.tz is None:
        raise ValueError(f"{p}: acceptance index must be tz-aware UTC")
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError(f"{p}: acceptance index must be ascending and unique")
    return df


def expected_announcement_schedule(
    announcements: pd.DatetimeIndex,
    lookback: int = DEFAULT_LOOKBACK_GAPS,
    min_gaps: int = MIN_GAPS,
    fallback_gap_days: int = FALLBACK_GAP_DAYS,
) -> pd.DataFrame:
    """PIT-clean ex-ante schedule of the NEXT expected announcement.

    For each realized announcement at t_i, predict the next one:

        expected[i] = t_i + median(gaps of the trailing ``lookback``
                      announcements ending at t_i)

    using ``fallback_gap_days`` while fewer than ``min_gaps`` gaps exist.
    Output rows: index = expected announcement date (normalized, tz-aware);
    columns ``known_asof`` (the announcement timestamp that generated the
    prediction — the earliest moment this expectation exists) and
    ``predicted_gap_days``. Only PAST information feeds each row.
    """
    if len(announcements) == 0:
        return pd.DataFrame(columns=["known_asof", "predicted_gap_days"])
    ts = pd.DatetimeIndex(announcements).sort_values()
    dates = ts.normalize()
    rows = []
    for i in range(len(ts)):
        past = dates[: i + 1]
        gaps = np.diff(past.view("int64")) / (86_400 * 1_000_000_000)
        gaps = gaps[-lookback:]
        if len(gaps) >= min_gaps:
            gap = float(np.median(gaps))
        else:
            gap = float(fallback_gap_days)
        expected = dates[i] + pd.Timedelta(days=round(gap))
        rows.append({"expected_date": expected, "known_asof": ts[i],
                     "predicted_gap_days": gap})
    out = pd.DataFrame(rows).set_index("expected_date")
    return out


def earnings_calendar_features(
    bar_index: pd.DatetimeIndex,
    announcements: pd.DatetimeIndex,
    cap_days: int = 63,
) -> pd.DataFrame:
    """B0017 meta-features: PIT earnings-calendar state per bar.

    - ``days_since_last_announcement``: calendar days since the most recent
      announcement with acceptance <= bar date (NaN before the first one).
    - ``days_to_expected_earnings``: calendar days until the nearest expected
      announcement whose expectation was ALREADY KNOWN at the bar (known_asof
      <= bar) and whose expected date is >= bar. NaN when no live expectation.

    Both capped at ``cap_days`` (a bar three months from any announcement
    carries no event information worth distinguishing). Uses ONLY past
    announcements per bar — truncation-invariant by construction.
    """
    bars = pd.DatetimeIndex(bar_index).tz_localize(None).normalize()
    ann = filter_amendments(pd.DatetimeIndex(announcements).sort_values())
    # AMC filings (accepted at/after the session close) are knowable only from
    # the NEXT session — a bar's features are close-of-session information.
    ann_days = effective_knowledge_day(ann)
    out = pd.DataFrame(index=bar_index)

    # days since last announcement (searchsorted on normalized days)
    pos = np.searchsorted(ann_days.values, bars.values, side="right") - 1
    since = np.full(len(bars), np.nan)
    valid = pos >= 0
    since[valid] = (
        (bars.values[valid] - ann_days.values[pos[valid]]) / np.timedelta64(1, "D")
    )
    out["days_since_last_announcement"] = np.minimum(since, cap_days)

    # days to the expected next announcement, using the schedule row known as
    # of each bar: the expectation generated by the latest announcement <= bar.
    sched = expected_announcement_schedule(ann)
    to_next = np.full(len(bars), np.nan)
    if not sched.empty:
        exp_days = pd.DatetimeIndex(sched.index).tz_localize(None).normalize()
        # row i of sched corresponds to announcement i (known_asof == ann[i])
        for j in range(len(bars)):
            i = pos[j]
            if i < 0:
                continue
            delta = (exp_days[i] - bars[j]).days
            # delta < 0: the expected date passed but the announcement hasn't
            # been filed yet (scheduler error window) — the event is OVERDUE,
            # i.e. imminent: clamp to 0, never NaN (dropna would silently
            # remove exactly the bars nearest the event).
            to_next[j] = min(max(delta, 0), cap_days)
    out["days_to_expected_earnings"] = to_next
    return out
