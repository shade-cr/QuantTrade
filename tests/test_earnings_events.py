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
