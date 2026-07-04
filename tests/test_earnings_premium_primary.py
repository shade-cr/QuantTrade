"""B0017 — phase5_earnings_premium: frozen-rule conformance + PIT discipline."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline.primaries_phase5.phase5_earnings_premium as epp


def _ohlcv(start="2010-01-04", periods=400):
    idx = pd.bdate_range(start, periods=periods, tz="UTC")
    close = pd.Series(100 + np.arange(periods, dtype=float) * 0.1, index=idx)
    return pd.DataFrame({"open": close, "high": close, "low": close,
                         "close": close, "volume": 1e6}, index=idx)


def _install_announcements(monkeypatch, dates_utc):
    ann = pd.DataFrame(
        {"filing_date": [d.date() for d in dates_utc], "items": "2.02"},
        index=pd.DatetimeIndex(dates_utc),
    )
    monkeypatch.setattr(epp, "load_earnings_announcements", lambda asset: ann)


def test_requires_current_asset():
    with pytest.raises(ValueError, match="_current_asset"):
        epp.signal(_ohlcv(), pd.DataFrame(), {})


def test_fires_once_inside_window_before_expected_date(monkeypatch):
    # Quarterly announcements every 91 days; after 4+ the scheduler predicts +91d.
    dates = [pd.Timestamp("2010-01-15 12:00", tz="UTC") + pd.Timedelta(days=91 * i)
             for i in range(5)]
    _install_announcements(monkeypatch, dates)
    ohlcv = _ohlcv()
    sig = epp.signal(ohlcv, pd.DataFrame(), {"_current_asset": "TEST"})

    assert set(sig.unique()) <= {0, 1}
    fired = sig[sig == 1]
    assert len(fired) >= 3  # several expected events land inside the bar range
    # Every firing bar must lie in [expected-3BD, expected-1BD] for SOME expectation.
    from pipeline.earnings_events import expected_announcement_schedule
    sched = expected_announcement_schedule(pd.DatetimeIndex(dates))
    ok = []
    for ts in fired.index:
        d = ts.tz_localize(None).normalize()
        hit = any(
            (pd.Timestamp(e).tz_localize(None).normalize()
             - pd.tseries.offsets.BusinessDay(epp.ENTRY_DAYS_BEFORE)) <= d
            <= (pd.Timestamp(e).tz_localize(None).normalize()
                - pd.tseries.offsets.BusinessDay(1))
            for e in sched.index
        )
        ok.append(hit)
    assert all(ok)


def test_one_signal_per_event(monkeypatch):
    dates = [pd.Timestamp("2010-01-15 12:00", tz="UTC") + pd.Timedelta(days=91 * i)
             for i in range(5)]
    _install_announcements(monkeypatch, dates)
    sig = epp.signal(_ohlcv(), pd.DataFrame(), {"_current_asset": "TEST"})
    # No two consecutive-bar firings from the same 3-day window.
    fired = np.flatnonzero(sig.values == 1)
    assert (np.diff(fired) > 3).all()


def test_pit_no_fire_before_knowledge(monkeypatch):
    """A signal generated from expectation k (known at announcement k-1) must
    never fire on/before the k-1 announcement date itself."""
    dates = [pd.Timestamp("2010-01-15 12:00", tz="UTC") + pd.Timedelta(days=91 * i)
             for i in range(5)]
    _install_announcements(monkeypatch, dates)
    sig = epp.signal(_ohlcv(), pd.DataFrame(), {"_current_asset": "TEST"})
    ann_days = {pd.Timestamp(d).tz_localize(None).normalize() for d in dates}
    for ts in sig[sig == 1].index:
        d = ts.tz_localize(None).normalize()
        # Firing on an announcement day would mean the window was computed
        # from that same announcement (zero-gap knowledge) — forbidden.
        assert d not in ann_days


def test_truncation_invariance(monkeypatch):
    """Signals on early bars must not change when later announcements exist."""
    dates = [pd.Timestamp("2010-01-15 12:00", tz="UTC") + pd.Timedelta(days=91 * i)
             for i in range(6)]
    ohlcv = _ohlcv(periods=300)
    _install_announcements(monkeypatch, dates)
    full = epp.signal(ohlcv, pd.DataFrame(), {"_current_asset": "TEST"})
    _install_announcements(monkeypatch, dates[:4])
    trunc_ann = epp.signal(ohlcv, pd.DataFrame(), {"_current_asset": "TEST"})
    # Up to the 4th announcement date, both must agree exactly.
    cutoff = pd.Timestamp(dates[3]).tz_localize(None).normalize()
    early = ohlcv.index[pd.DatetimeIndex(ohlcv.index).tz_localize(None).normalize() <= cutoff]
    pd.testing.assert_series_equal(full.loc[early], trunc_ann.loc[early])
