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
FALLBACK_GAP_DAYS = 91
DEFAULT_LOOKBACK_GAPS = 8
MIN_GAPS = 3


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
