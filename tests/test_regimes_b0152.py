"""B0152 — regimes CLI stub-trap regression tests.

data/D1_22y/GBPUSD_D1.csv was a 1310-bar 5y stub next to the 27-year
data/D1 file; directory-priority inference picked the stub, labeled 0 bars,
and sanity_report crashed with ZeroDivisionError. Two fixes under test:
more-bars path preference and the explicit 0-labeled sanity verdict.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline import regimes
from pipeline.regimes import _resolve_data_path, sanity_report, REGIMES


def _write_csv(path, n_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("time,open,high,low,close,volume\n")
        for i in range(n_rows):
            fh.write(f"2020-01-{(i % 28) + 1:02d},1,1,1,1,1\n")


@pytest.fixture
def _data_dirs(tmp_path, monkeypatch):
    primary = tmp_path / "D1_22y"
    fallback = tmp_path / "D1"
    monkeypatch.setattr(regimes, "DEFAULT_DATA_DIRS", {"D1": str(primary), "H4": str(tmp_path / "H4")})
    monkeypatch.setattr(regimes, "DEFAULT_DATA_DIRS_FALLBACK", {"D1": str(fallback)})
    return primary, fallback


def test_deeper_fallback_beats_stub_primary(_data_dirs):
    """The B0152 trap: short stub in the primary dir must lose to the deeper
    fallback archive."""
    primary, fallback = _data_dirs
    _write_csv(primary / "GBPUSD_D1.csv", 1310)
    _write_csv(fallback / "GBPUSD_D1.csv", 7000)
    chosen = _resolve_data_path("GBPUSD", "D1", explicit=None)
    assert chosen == fallback / "GBPUSD_D1.csv"


def test_deeper_primary_still_preferred(_data_dirs):
    """Metals keep their 22y preference when the primary IS the deep archive."""
    primary, fallback = _data_dirs
    _write_csv(primary / "XAUUSD_D1.csv", 5650)
    _write_csv(fallback / "XAUUSD_D1.csv", 1300)
    chosen = _resolve_data_path("XAUUSD", "D1", explicit=None)
    assert chosen == primary / "XAUUSD_D1.csv"


def test_tie_goes_to_primary(_data_dirs):
    primary, fallback = _data_dirs
    _write_csv(primary / "XAGUSD_D1.csv", 100)
    _write_csv(fallback / "XAGUSD_D1.csv", 100)
    assert _resolve_data_path("XAGUSD", "D1", None) == primary / "XAGUSD_D1.csv"


def test_explicit_path_always_wins(_data_dirs):
    primary, fallback = _data_dirs
    _write_csv(primary / "EURUSD_D1.csv", 9000)
    explicit = fallback / "EURUSD_D1.csv"
    _write_csv(explicit, 10)
    assert _resolve_data_path("EURUSD", "D1", str(explicit)) == explicit


def test_single_candidate_used(_data_dirs):
    primary, fallback = _data_dirs
    _write_csv(fallback / "USDJPY_D1.csv", 500)
    assert _resolve_data_path("USDJPY", "D1", None) == fallback / "USDJPY_D1.csv"


def test_missing_everywhere_raises(_data_dirs):
    with pytest.raises(FileNotFoundError):
        _resolve_data_path("NOPEUSD", "D1", None)


def test_sanity_report_zero_labeled_is_explicit_not_crash():
    """SOLUSD-H4 / stub signature: every regime_id NaN -> readable verdict."""
    df = pd.DataFrame({"regime_id": pd.Series([None] * 50, dtype="object")})
    rep = sanity_report(df)
    assert "insufficient_history" in rep
    assert rep["n_episodes"] == 0
    assert rep["regime_counts"] == {r: 0 for r in REGIMES}
    assert rep["regimes_below_5pct"] == list(REGIMES)
