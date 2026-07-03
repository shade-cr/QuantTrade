"""--universe batch mode: fetches every stock+ETF, gates short/late histories."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import fetch_equity_daily  # noqa: E402


def _fake_df(start: str, n: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    base = np.linspace(100.0, 110.0, n)
    return pd.DataFrame({
        "open": base, "high": base + 1.0, "low": base - 1.0,
        "close": base + 0.5, "volume": np.full(n, 1e6),
    }, index=idx)


def _write_universe(tmp_path: Path) -> Path:
    p = tmp_path / "u.yaml"
    p.write_text(
        "selection_rule: test\nselected_at: '2026-07-03'\n"
        "stocks: [AAAA, BBBB]\netfs: [XLTT]\nalternates: []\n"
        "excluded_delistees: {}\n",
        encoding="utf-8",
    )
    return p


def test_universe_mode_writes_all_tickers(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetch_equity_daily.YFinanceSource, "fetch_daily",
        lambda self, t, s, e: _fake_df("2000-01-03", 1500),
    )
    written = []
    monkeypatch.setattr(
        fetch_equity_daily, "write_contract_csv",
        lambda df, out: written.append(Path(out).name),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["fetch_equity_daily.py", "--universe", str(_write_universe(tmp_path))],
    )
    assert fetch_equity_daily.main() == 0
    assert sorted(written) == ["AAAA_D1.csv", "BBBB_D1.csv", "XLTT_D1.csv"]


def test_universe_mode_gates_late_history_and_exits_nonzero(tmp_path, monkeypatch):
    # BBBB starts 2010 → fails the ≤2006-01-03 gate; others succeed.
    def fake_fetch(self, t, s, e):
        return _fake_df("2010-01-04" if t == "BBBB" else "2000-01-03", 1500)

    monkeypatch.setattr(fetch_equity_daily.YFinanceSource, "fetch_daily", fake_fetch)
    written = []
    monkeypatch.setattr(
        fetch_equity_daily, "write_contract_csv",
        lambda df, out: written.append(Path(out).name),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["fetch_equity_daily.py", "--universe", str(_write_universe(tmp_path))],
    )
    assert fetch_equity_daily.main() == 1
    assert "BBBB_D1.csv" not in written
    assert len(written) == 2
