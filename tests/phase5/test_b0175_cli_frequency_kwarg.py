"""B0175: the regime_stats CLI main() must pass frequency= through to
build_regime_dossiers.

B0154 rewrote the CLI call site and dropped frequency=args.frequency, so a
standalone `python -m phase5.regime_stats --frequency H4 ...` run silently fell
back to the D1 divisor when computing effective_n (FREQ_BARS_PER_DAY default).
The batch path (build_all_regimes) was always correct; only the standalone CLI
regressed. This test pins the CLI behaviour.
"""
from __future__ import annotations
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

import phase5.regime_stats as rs
import pipeline.data as pdata
import pipeline.regimes as pregimes


def _run_cli_capturing_frequency(tmp_path, monkeypatch, frequency: str) -> dict:
    """Invoke main() with the heavy data path mocked; capture build_regime_dossiers kwargs."""
    captured: dict = {}

    # A real (but empty) regimes parquet so the existence check passes.
    regimes_path = tmp_path / "regimes.parquet"
    idx = pd.date_range("2020-01-01", periods=8, freq="D", tz="UTC")
    regimes_df = pd.DataFrame({"regime_id": ["BULL_QUIET"] * 8}, index=idx)
    regimes_df.to_parquet(regimes_path)

    df = pd.DataFrame(
        {
            "open": np.ones(8),
            "high": np.ones(8) * 1.1,
            "low": np.ones(8) * 0.9,
            "close": np.ones(8),
            "volume": np.ones(8),
        },
        index=idx,
    )

    monkeypatch.setattr(pregimes, "_resolve_data_path", lambda *a, **k: tmp_path / "x.csv")
    monkeypatch.setattr(pdata, "load_dataset", lambda *a, **k: df)
    monkeypatch.setattr(rs, "build_dossier_features", lambda *a, **k: df[["close"]])
    monkeypatch.setattr(rs, "_atr", lambda *a, **k: pd.Series(np.ones(8), index=idx))
    monkeypatch.setattr(rs, "build_primary_baselines", lambda *a, **k: {})

    def _fake_build_regime_dossiers(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(rs, "build_regime_dossiers", _fake_build_regime_dossiers)

    argv = [
        "regime_stats",
        "--asset", "XAUUSD",
        "--frequency", frequency,
        "--asset-class", "metal",
        "--regimes-path", str(regimes_path),
        "--out", str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = rs.main()
    assert rc == 0
    return captured


def test_cli_main_passes_frequency_h4(tmp_path, monkeypatch):
    captured = _run_cli_capturing_frequency(tmp_path, monkeypatch, "H4")
    assert captured.get("frequency") == "H4", (
        "CLI main() must forward --frequency to build_regime_dossiers; "
        "otherwise H4 effective_n falls back to the D1 divisor."
    )


def test_cli_main_passes_frequency_d1(tmp_path, monkeypatch):
    captured = _run_cli_capturing_frequency(tmp_path, monkeypatch, "D1")
    assert captured.get("frequency") == "D1"
