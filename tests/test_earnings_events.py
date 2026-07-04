"""B0017 — earnings events: PIT scheduler + cache contract + 8-K extraction."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.earnings_events import (
    expected_announcement_schedule,
    load_earnings_announcements,
)
from scripts.fetch_earnings_dates import _extract_8k202


def _quarterly(start: str, n: int, gap_days: int = 91) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        [pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=gap_days * i) for i in range(n)]
    )


# ---------------------------------------------------------------- scheduler #

def test_schedule_uses_median_gap():
    ann = _quarterly("2010-01-15 12:00", 6, gap_days=91)
    sched = expected_announcement_schedule(ann)
    assert len(sched) == 6
    # After enough history, prediction = last + 91d.
    last = sched.iloc[-1]
    assert last.name == ann[-1].normalize() + pd.Timedelta(days=91)
    assert last["predicted_gap_days"] == 91


def test_schedule_fallback_gap_when_history_short():
    ann = _quarterly("2010-01-15 12:00", 2, gap_days=80)  # only 1 gap < min_gaps
    sched = expected_announcement_schedule(ann, fallback_gap_days=91)
    assert (sched["predicted_gap_days"] == 91).all()


def test_schedule_is_strictly_pit():
    """Each expected date must be generated ONLY from announcements <= known_asof,
    and truncating the future must not change past rows."""
    ann = _quarterly("2010-01-15 12:00", 10, gap_days=91)
    # Perturb the LAST gap heavily; earlier predictions must be unaffected.
    perturbed = ann[:-1].append(pd.DatetimeIndex([ann[-1] + pd.Timedelta(days=30)]))
    full = expected_announcement_schedule(perturbed)
    trunc = expected_announcement_schedule(perturbed[:-1])
    pd.testing.assert_frame_equal(full.iloc[:-1], trunc)
    assert (full["known_asof"].values == perturbed.values).all()


def test_schedule_empty_input():
    out = expected_announcement_schedule(pd.DatetimeIndex([]))
    assert out.empty


# ------------------------------------------------------------------- cache #

def test_load_missing_cache_fails_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_earnings_dates"):
        load_earnings_announcements("ZZZZ", cache_dir=tmp_path)


def test_load_rejects_tz_naive_cache(tmp_path):
    idx = pd.DatetimeIndex([pd.Timestamp("2020-01-01 12:00")])  # naive
    pd.DataFrame({"filing_date": ["2020-01-01"], "items": ["2.02"]}, index=idx).to_parquet(
        tmp_path / "AAA_8k202.parquet"
    )
    with pytest.raises(ValueError, match="tz-aware"):
        load_earnings_announcements("AAA", cache_dir=tmp_path)


def test_load_roundtrip(tmp_path):
    idx = pd.DatetimeIndex([pd.Timestamp("2020-01-01 12:00", tz="UTC"),
                            pd.Timestamp("2020-04-01 12:00", tz="UTC")])
    pd.DataFrame({"filing_date": ["2020-01-01", "2020-04-01"], "items": ["2.02", "2.02,9.01"]},
                 index=idx).rename_axis("acceptance").to_parquet(tmp_path / "AAA_8k202.parquet")
    df = load_earnings_announcements("AAA", cache_dir=tmp_path)
    assert len(df) == 2 and df.index.tz is not None


# ---------------------------------------------------------- calendar feats #

def test_calendar_features_pit_and_caps():
    from pipeline.earnings_events import earnings_calendar_features

    ann = _quarterly("2010-01-15 12:00", 5, gap_days=91)
    bars = pd.bdate_range("2010-01-01", periods=400, tz="UTC")
    f = earnings_calendar_features(bars, ann)

    naive = bars.tz_localize(None).normalize()
    # Before the first announcement: both NaN.
    pre = naive < pd.Timestamp("2010-01-15")
    assert f.loc[bars[pre], "days_since_last_announcement"].isna().all()
    assert f.loc[bars[pre], "days_to_expected_earnings"].isna().all()

    # Day after the first announcement: since == 1.
    day_after = bars[naive == pd.Timestamp("2010-01-18")]  # Monday after Fri 15th
    assert f.loc[day_after, "days_since_last_announcement"].iloc[0] == 3

    # Caps respected.
    assert (f["days_since_last_announcement"].dropna() <= 63).all()
    assert (f["days_to_expected_earnings"].dropna() <= 63).all()

    # Overdue expectation (expected date passed, announcement not yet filed)
    # clamps to 0 — never NaN, which dropna would silently remove.
    post = ~pre
    assert (f.loc[bars[post], "days_to_expected_earnings"] >= 0).all()
    assert not f.loc[bars[post], "days_to_expected_earnings"].isna().any()

    # PIT: truncating future announcements must not change earlier bars.
    cutoff = ann[2]
    early = bars[naive <= cutoff.tz_localize(None).normalize()]
    f_trunc = earnings_calendar_features(bars, ann[:3])
    pd.testing.assert_frame_equal(f.loc[early], f_trunc.loc[early])


# ----------------------------------------------- DA-review fixes (2026-07-04) #

def test_amc_acceptance_not_known_same_session():
    """DA high #1: an 8-K accepted at/after the session close (>=20:00 UTC)
    must NOT be visible to that session's features — days_since > 0 on the
    filing day, 0 only on the next session."""
    from pipeline.earnings_events import earnings_calendar_features

    early = [pd.Timestamp("2010-01-15 12:00", tz="UTC") + pd.Timedelta(days=91 * i)
             for i in range(4)]
    amc = pd.Timestamp("2011-01-14 21:30", tz="UTC")  # after 16:00 ET close
    ann = pd.DatetimeIndex(early + [amc])
    bars = pd.bdate_range("2010-01-01", periods=300, tz="UTC")
    f = earnings_calendar_features(bars, ann)

    naive = bars.tz_localize(None).normalize()
    on_day = f.loc[bars[naive == pd.Timestamp("2011-01-14")], "days_since_last_announcement"]
    next_day = f.loc[bars[naive == pd.Timestamp("2011-01-17")], "days_since_last_announcement"]
    assert (on_day > 0).all(), "AMC filing leaked into same-session features"
    assert (next_day <= 3).all()  # known by the following session (weekend gap)


def test_filter_amendments_drops_near_duplicates():
    from pipeline.earnings_events import filter_amendments

    base = pd.Timestamp("2010-01-15 12:00", tz="UTC")
    ann = pd.DatetimeIndex([
        base,
        base + pd.Timedelta(days=5),    # 8-K/A amendment — dropped
        base + pd.Timedelta(days=91),   # next real quarter — kept
        base + pd.Timedelta(days=92),   # re-disclosure — dropped
        base + pd.Timedelta(days=182),  # kept
    ])
    kept = filter_amendments(ann)
    assert list(kept) == [base, base + pd.Timedelta(days=91), base + pd.Timedelta(days=182)]


def test_effective_knowledge_day_rolls_amc():
    from pipeline.earnings_events import effective_knowledge_day

    ts = pd.DatetimeIndex([
        pd.Timestamp("2010-01-15 12:00", tz="UTC"),  # BMO -> same day
        pd.Timestamp("2010-01-15 21:30", tz="UTC"),  # AMC -> next day
    ])
    eff = effective_knowledge_day(ts)
    assert eff[0] == pd.Timestamp("2010-01-15")
    assert eff[1] == pd.Timestamp("2010-01-16")


# -------------------------------------------------------------- extraction #

def test_extract_8k202_filters_forms_and_items():
    block = {
        "form": ["8-K", "10-Q", "8-K", "8-K"],
        "items": ["2.02,9.01", "", "5.02", "2.02"],
        "acceptanceDateTime": [
            "2020-01-22T12:35:32.000Z", "2020-02-01T10:00:00.000Z",
            "2020-03-01T10:00:00.000Z", "2020-04-16T11:36:25.000Z",
        ],
        "filingDate": ["2020-01-22", "2020-02-01", "2020-03-01", "2020-04-16"],
    }
    rows = _extract_8k202(block)
    assert [r["filing_date"] for r in rows] == ["2020-01-22", "2020-04-16"]
    assert all("2.02" in r["items"] for r in rows)
