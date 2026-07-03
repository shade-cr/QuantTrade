"""Tests for B0010: dossier primary-baseline audit-window event counts.

Loop A ticks 1-2 both died at the M3 audit event floor because
primary_baseline_summary.n_events is measured over a ticker's FULL history
while audits run on the AUDIT_WINDOW_START (2006+) config window. This module
adds an additive n_events_audit_window field so hypothesizer-side event-count
floors are designed against the number the audit will actually see.
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase5.regime_stats import (
    AUDIT_WINDOW_START,
    _primary_raw_metrics,
    build_primary_baselines,
)


def _long_span_ohlc(n: int = 6000) -> pd.DataFrame:
    """Rising-close OHLC spanning 2000-01-01 .. ~2016 (D1 bars), straddling
    the 2006-01-01 audit-window boundary. Mirrors the `_rising_ohlc` fixture
    pattern used elsewhere in tests/phase5/test_build_all_regimes.py."""
    idx = pd.date_range("2000-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100.0, 100.0 + n / 10.0, n), index=idx)
    return pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1.0},
        index=idx,
    )


def test_n_events_audit_window_counts_only_post_2006_events():
    """Events straddle 2006-01-01: n_events_audit_window must equal only the
    post-boundary subset and be strictly less than the full-history n_events."""
    ohlc = _long_span_ohlc(6000)
    atr = pd.Series(1.0, index=ohlc.index)
    audit_start = pd.Timestamp(AUDIT_WINDOW_START, tz="UTC")

    # 2000-01-01 + 2192 days == 2006-01-01 (2000 and 2004 are leap years).
    boundary_iloc = int(ohlc.index.get_indexer([audit_start])[0])
    assert ohlc.index[boundary_iloc] == audit_start  # sanity-check the fixture math

    pre_positions = [100, 500, 1000, 1500, 2000]      # all strictly before 2006-01-01
    post_positions = [2200, 3000, 4000, 5000, 5800]   # all on/after 2006-01-01
    assert all(p < boundary_iloc for p in pre_positions)
    assert all(p >= boundary_iloc for p in post_positions)

    sig = pd.Series(0.0, index=ohlc.index)
    for p in pre_positions + post_positions:
        sig.iloc[p] = 1.0

    regimes_df = pd.DataFrame({"regime_id": ["BULL_QUIET"] * len(ohlc)}, index=ohlc.index)
    raw = _primary_raw_metrics(
        ohlc, atr, regimes_df, sig, frequency="D1", tp_mult=2.0, sl_mult=1.0, horizon=20,
    )
    bq = raw["BULL_QUIET"]
    assert bq["n_events"] == len(pre_positions) + len(post_positions)
    assert bq["n_events_audit_window"] == len(post_positions)
    assert bq["n_events_audit_window"] < bq["n_events"]

    # Independently recompute the expected count directly from entry timestamps.
    entry_dates = ohlc.index[pre_positions + post_positions]
    expected = int((entry_dates >= audit_start).sum())
    assert bq["n_events_audit_window"] == expected


def test_build_primary_baselines_threads_audit_window_field():
    """The encoded per-primary baseline (what lands in the dossier JSON) carries
    n_events_audit_window alongside n_events — additive, existing keys unchanged."""
    ohlc = _long_span_ohlc(6000)
    atr = pd.Series(1.0, index=ohlc.index)
    regimes_df = pd.DataFrame({"regime_id": ["BULL_QUIET"] * len(ohlc)}, index=ohlc.index)
    baselines = build_primary_baselines(ohlc, atr, regimes_df, frequency="D1")
    ema = baselines["BULL_QUIET"]["ema_crossover"]
    assert "n_events" in ema  # existing field untouched
    assert "n_events_audit_window" in ema
    assert isinstance(ema["n_events_audit_window"], int)
    assert 0 <= ema["n_events_audit_window"] <= ema["n_events"]


def test_zero_events_regime_reports_zero_audit_window_count():
    """A regime with zero events must report n_events_audit_window == 0, not
    None or a missing key (mirrors the existing n_events==0 default)."""
    ohlc = _long_span_ohlc(200)
    atr = pd.Series(1.0, index=ohlc.index)
    sig = pd.Series(0.0, index=ohlc.index)  # no entries anywhere
    regimes_df = pd.DataFrame({"regime_id": ["BULL_QUIET"] * len(ohlc)}, index=ohlc.index)
    raw = _primary_raw_metrics(
        ohlc, atr, regimes_df, sig, frequency="D1", tp_mult=2.0, sl_mult=1.0, horizon=20,
    )
    assert raw["BULL_QUIET"]["n_events"] == 0
    assert raw["BULL_QUIET"]["n_events_audit_window"] == 0
