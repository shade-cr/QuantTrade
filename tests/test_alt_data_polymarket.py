"""Tests for pipeline.alt_data.polymarket (B0170) — stitcher + PIT loader.

All tests run on synthetic parquet caches + manifest in tmp_path (no network).

Invariants:
  1. build_fomc_stitched orders events by meeting date; each event's active
     (front) window is (prev_event_end, event_end]; pm_roll tags the first
     covered day of every non-first event.
  2. cut_prob = SUM of all cut-bucket legs; a day with a missing cut leg is
     NaN (never a silent partial sum) and pm_coverage < 1.
  3. Gaps between events stay NaN — no ffill across data the market did not
     produce; entropy is computed on the NORMALIZED outcome vector with
     prob_sum emitted as a data-quality column.
  4. PIT: a market bar at calendar date t must NOT see the probability stamped
     at t (mirror of tests/test_alt_data_gld_holdings.py no-leak test); ffill
     is bounded (limit) so dead zones don't propagate stale probabilities.
  5. roll_masked_diff NaNs any diff spanning a roll boundary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.alt_data.polymarket import (
    PolymarketCacheMissing,
    build_event_series,
    build_fomc_stitched,
    load_polymarket_features,
    roll_masked_diff,
)


# ---------------------------------------------------------------- synthetic cache
def _write_market(cache_dir: Path, name: str, event_slug: str, end_date: str,
                  bucket: str, token: str, dates: list[str], probs: list[float],
                  label: str = "") -> dict:
    """Write one market parquet + return its manifest record."""
    rel = Path("history") / name / f"{event_slug}__{bucket}__{token[:8]}.parquet"
    path = cache_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.to_datetime(dates, utc=True)
    pd.DataFrame({"p": probs}, index=pd.DatetimeIndex(idx, name="t")).to_parquet(path)
    return {
        "name": name, "event_slug": event_slug, "event_end_date": end_date,
        "market_slug": f"mkt-{token}", "market_question": label or bucket,
        "outcome_label": label or bucket, "outcome_bucket": bucket,
        "token_id": token, "closed": True, "fidelity_min": 720,
        "file": str(rel), "rows": len(dates),
        "t_min": str(idx.min()), "t_max": str(idx.max()),
        "fetched_at": "2026-06-12T00:00:00",
    }


def _save_manifest(cache_dir: Path, records: list[dict]) -> None:
    (cache_dir / "manifest.json").write_text(json.dumps(records), encoding="utf-8")


def _two_event_cache(tmp_path: Path) -> Path:
    """Event A (end 2026-01-10): legs cut25, cut50, nochange — data Jan 5-10.
    Event B (end 2026-02-10): same legs — data Feb 1-10 (gap Jan 11-31).
    B also has pre-window data (Jan 8-9) that must be ignored while A is front."""
    recs = []
    a_days = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10"]
    recs.append(_write_market(tmp_path, "fomc", "ev-a", "2026-01-10", "cut", "a-cut25",
                              a_days, [0.10, 0.12, 0.14, 0.16, 0.18, 0.20], label="25 bps decrease"))
    recs.append(_write_market(tmp_path, "fomc", "ev-a", "2026-01-10", "cut", "a-cut50",
                              a_days, [0.05] * 6, label="50+ bps decrease"))
    recs.append(_write_market(tmp_path, "fomc", "ev-a", "2026-01-10", "nochange", "a-noch",
                              a_days, [0.85, 0.83, 0.81, 0.79, 0.77, 0.75], label="No change"))
    b_days = ["2026-01-08", "2026-01-09"] + [f"2026-02-{d:02d}" for d in range(1, 11)]
    recs.append(_write_market(tmp_path, "fomc", "ev-b", "2026-02-10", "cut", "b-cut25",
                              b_days, [0.30, 0.30] + [0.40] * 10, label="25 bps decrease"))
    recs.append(_write_market(tmp_path, "fomc", "ev-b", "2026-02-10", "cut", "b-cut50",
                              b_days, [0.10, 0.10] + [0.10] * 10, label="50+ bps decrease"))
    recs.append(_write_market(tmp_path, "fomc", "ev-b", "2026-02-10", "nochange", "b-noch",
                              b_days, [0.60, 0.60] + [0.50] * 10, label="No change"))
    _save_manifest(tmp_path, recs)
    return tmp_path


def _ts(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


# ---------------------------------------------------------------- stitching
def test_stitched_orders_events_and_tags_roll(tmp_path):
    out = build_fomc_stitched(_two_event_cache(tmp_path))
    assert out.index.is_monotonic_increasing
    # Roll = first covered day of event B (Feb 1); event A days are not rolls.
    assert bool(out.loc[_ts("2026-02-01"), "pm_roll"]) is True
    assert not out.loc[[_ts("2026-01-05"), _ts("2026-01-10"), _ts("2026-02-02")], "pm_roll"].any()


def test_stitched_cut_prob_sums_cut_legs(tmp_path):
    out = build_fomc_stitched(_two_event_cache(tmp_path))
    # Event A on Jan 5: cut25=0.10 + cut50=0.05.
    assert out.loc[_ts("2026-01-05"), "pm_fomc_cut_prob"] == pytest.approx(0.15)
    # Event B on Feb 1: 0.40 + 0.10.
    assert out.loc[_ts("2026-02-01"), "pm_fomc_cut_prob"] == pytest.approx(0.50)


def test_stitched_front_window_excludes_next_events_premature_data(tmp_path):
    """Event B has data on Jan 8-9 (while A is still front) — must be ignored."""
    out = build_fomc_stitched(_two_event_cache(tmp_path))
    # Jan 8: A's values (0.16+0.05), NOT B's (0.30+0.10).
    assert out.loc[_ts("2026-01-08"), "pm_fomc_cut_prob"] == pytest.approx(0.21)


def test_stitched_gap_between_events_stays_nan(tmp_path):
    out = build_fomc_stitched(_two_event_cache(tmp_path))
    gap_days = pd.date_range("2026-01-11", "2026-01-31", freq="D", tz="UTC")
    assert out.loc[gap_days, "pm_fomc_cut_prob"].isna().all()


def test_stitched_missing_cut_leg_day_is_nan_with_coverage_flag(tmp_path):
    cache = _two_event_cache(tmp_path)
    # Rewrite a-cut50 WITHOUT Jan 7 — a missing cut leg that day.
    recs = json.loads((cache / "manifest.json").read_text())
    days = ["2026-01-05", "2026-01-06", "2026-01-08", "2026-01-09", "2026-01-10"]
    new = _write_market(cache, "fomc", "ev-a", "2026-01-10", "cut", "a-cut50",
                        days, [0.05] * 5, label="50+ bps decrease")
    recs = [r if r["token_id"] != "a-cut50" else new for r in recs]
    _save_manifest(cache, recs)

    out = build_fomc_stitched(cache)
    assert np.isnan(out.loc[_ts("2026-01-07"), "pm_fomc_cut_prob"])
    assert out.loc[_ts("2026-01-07"), "pm_coverage"] < 1.0
    # Other days unaffected.
    assert out.loc[_ts("2026-01-06"), "pm_fomc_cut_prob"] == pytest.approx(0.17)
    assert out.loc[_ts("2026-01-06"), "pm_coverage"] == pytest.approx(1.0)


def test_stitched_entropy_normalized_and_prob_sum_reported(tmp_path):
    out = build_fomc_stitched(_two_event_cache(tmp_path))
    # Jan 5 legs: 0.10, 0.05, 0.85 -> prob_sum 1.0; entropy of normalized 3-vector.
    assert out.loc[_ts("2026-01-05"), "prob_sum"] == pytest.approx(1.0)
    p = np.array([0.10, 0.05, 0.85])
    expected = -(p / p.sum() * np.log(p / p.sum())).sum()
    assert out.loc[_ts("2026-01-05"), "pm_fomc_entropy"] == pytest.approx(expected)
    assert (out["pm_fomc_entropy"].dropna() >= 0).all()
    assert (out["pm_fomc_entropy"].dropna() <= np.log(3) + 1e-9).all()


def test_stitched_days_to_meeting(tmp_path):
    out = build_fomc_stitched(_two_event_cache(tmp_path))
    assert out.loc[_ts("2026-01-05"), "pm_fomc_days_to_meeting"] == 5
    assert out.loc[_ts("2026-01-10"), "pm_fomc_days_to_meeting"] == 0
    assert out.loc[_ts("2026-02-01"), "pm_fomc_days_to_meeting"] == 9


def test_missing_cache_raises(tmp_path):
    with pytest.raises(PolymarketCacheMissing, match="ingest_polymarket"):
        build_fomc_stitched(tmp_path / "nope")


# ---------------------------------------------------------------- event series (cuts2026)
def test_build_event_series_expected_cuts_mean(tmp_path):
    recs = [
        _write_market(tmp_path, "cuts2026", "how-many", "2026-12-31", "0-0-bps", "t0",
                      ["2026-06-01", "2026-06-02"], [0.20, 0.25]),
        _write_market(tmp_path, "cuts2026", "how-many", "2026-12-31", "2-50-bps", "t2",
                      ["2026-06-01", "2026-06-02"], [0.50, 0.50]),
        _write_market(tmp_path, "cuts2026", "how-many", "2026-12-31", "4-100-bps", "t4",
                      ["2026-06-01", "2026-06-02"], [0.30, 0.25]),
    ]
    _save_manifest(tmp_path, recs)
    out = build_event_series(tmp_path, name="cuts2026")
    # Jun 1: normalized probs (.2,.5,.3) over k=(0,2,4) -> E[k] = 0+1.0+1.2 = 2.2
    assert out.loc[_ts("2026-06-01"), "pm_cuts2026_exp_cuts"] == pytest.approx(2.2)
    assert out.loc[_ts("2026-06-01"), "prob_sum"] == pytest.approx(1.0)


# ---------------------------------------------------------------- PIT loader
def _simple_cache(tmp_path: Path) -> Path:
    recs = [
        _write_market(tmp_path, "fomc", "ev-a", "2026-01-10", "cut", "a-cut25",
                      ["2026-01-05", "2026-01-06", "2026-01-07"], [0.10, 0.20, 0.30]),
        _write_market(tmp_path, "fomc", "ev-a", "2026-01-10", "nochange", "a-noch",
                      ["2026-01-05", "2026-01-06", "2026-01-07"], [0.90, 0.80, 0.70]),
    ]
    _save_manifest(tmp_path, recs)
    return tmp_path


def test_pit_no_leak_same_day(tmp_path):
    """A market bar at calendar date t must NOT see the probability stamped at t."""
    cache = _simple_cache(tmp_path)
    target = pd.date_range("2026-01-05", periods=4, freq="D", tz="UTC")
    out = load_polymarket_features(target, cache)
    # Jan 5 bar: first cache stamp is Jan 5 -> shifted to Jan 6 -> NaN on Jan 5.
    assert np.isnan(out.loc[_ts("2026-01-05"), "pm_fomc_cut_prob"])
    # Jan 6 bar sees Jan 5's value.
    assert out.loc[_ts("2026-01-06"), "pm_fomc_cut_prob"] == pytest.approx(0.10)
    assert out.loc[_ts("2026-01-07"), "pm_fomc_cut_prob"] == pytest.approx(0.20)


def test_pit_ffill_is_bounded(tmp_path):
    """A long dead zone must NOT propagate stale probabilities past the limit."""
    cache = _simple_cache(tmp_path)
    target = pd.date_range("2026-01-06", periods=15, freq="D", tz="UTC")
    out = load_polymarket_features(target, cache, ffill_limit=5)
    # Last stamp Jan 7 -> visible Jan 8, ffill through Jan 13 (5 steps), NaN after.
    assert out.loc[_ts("2026-01-13"), "pm_fomc_cut_prob"] == pytest.approx(0.30)
    assert np.isnan(out.loc[_ts("2026-01-14"), "pm_fomc_cut_prob"])


def test_pit_loader_empty_target_index(tmp_path):
    cache = _simple_cache(tmp_path)
    out = load_polymarket_features(pd.DatetimeIndex([], tz="UTC"), cache)
    assert len(out) == 0
    assert "pm_fomc_cut_prob" in out.columns


def test_pit_loader_works_on_h4_index(tmp_path):
    """All H4 bars of day t+1 see day t's stamp (conservative)."""
    cache = _simple_cache(tmp_path)
    target = pd.date_range("2026-01-06 00:00", "2026-01-06 20:00", freq="4h", tz="UTC")
    out = load_polymarket_features(target, cache)
    assert np.allclose(out["pm_fomc_cut_prob"].to_numpy(), 0.10)


# ---------------------------------------------------------------- roll-masked diff
def test_roll_masked_diff_nans_diffs_spanning_rolls():
    idx = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")
    s = pd.Series([0.1, 0.2, 0.5, 0.6, 0.7], index=idx)
    roll = pd.Series([False, False, True, False, False], index=idx)
    d = roll_masked_diff(s, roll)
    assert np.isnan(d.iloc[0])               # ordinary first diff
    assert d.iloc[1] == pytest.approx(0.1)
    assert np.isnan(d.iloc[2])               # spans the roll boundary -> masked
    assert d.iloc[3] == pytest.approx(0.1)
