"""B0018: phase5_insider_event primary contract."""
import numpy as np
import pandas as pd
import pytest

from pipeline.primaries_phase5 import phase5_insider_event as pie


@pytest.fixture
def bars():
    idx = pd.date_range("2010-01-04", periods=120, freq="B", tz="UTC")
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
         "volume": 1e6},
        index=idx,
    )


def _patch_kd(monkeypatch, days):
    monkeypatch.setattr(pie, "load_insider_purchases", lambda *_a, **_k: pd.DataFrame({"x": []}))
    monkeypatch.setattr(
        pie, "opportunistic_knowledge_days",
        lambda *_a, **_k: pd.DatetimeIndex(pd.to_datetime(days)),
    )


def test_requires_current_asset(bars):
    with pytest.raises(ValueError, match="_current_asset"):
        pie.signal(bars, pd.DataFrame(index=bars.index), {})


def test_fires_first_bar_at_or_after_knowledge_day(bars, monkeypatch):
    _patch_kd(monkeypatch, ["2010-02-06"])  # a Saturday -> Monday 2010-02-08
    s = pie.signal(bars, pd.DataFrame(index=bars.index), {"_current_asset": "XOM"})
    fired = s[s == 1].index
    assert len(fired) == 1
    assert fired[0].normalize().tz_localize(None) == pd.Timestamp("2010-02-08")
    assert set(s.unique()) <= {0, 1}  # long-only, never -1


def test_refire_gap_suppresses_clustered_events(bars, monkeypatch):
    _patch_kd(monkeypatch, ["2010-02-08", "2010-02-10", "2010-03-08"])
    s = pie.signal(bars, pd.DataFrame(index=bars.index), {"_current_asset": "XOM"})
    pos = np.flatnonzero(s.values)
    assert len(pos) == 2  # 02-10 is within 10 bars of 02-08 -> suppressed
    assert pos[1] - pos[0] >= pie.REFIRE_GAP_BARS


def test_event_after_last_bar_is_dropped(bars, monkeypatch):
    _patch_kd(monkeypatch, ["2030-01-01"])
    s = pie.signal(bars, pd.DataFrame(index=bars.index), {"_current_asset": "XOM"})
    assert (s == 0).all()


def test_truncation_invariance_no_lookahead(bars, monkeypatch):
    """Signal on early bars must be unchanged when later bars are removed."""
    _patch_kd(monkeypatch, ["2010-02-08", "2010-04-12"])
    full = pie.signal(bars, pd.DataFrame(index=bars.index), {"_current_asset": "XOM"})
    cut = bars.iloc[:60]
    part = pie.signal(cut, pd.DataFrame(index=cut.index), {"_current_asset": "XOM"})
    pd.testing.assert_series_equal(full.iloc[:60], part)


def test_input_columns_empty_contract():
    assert pie.INPUT_COLUMNS == ()
