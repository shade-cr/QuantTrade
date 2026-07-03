"""Universe file contract: frozen M3 universe loads and validates (B0003)."""
import pytest

from pipeline.equity_universe import load_universe


def test_frozen_m3_universe_loads_and_is_well_formed():
    u = load_universe("configs/universe_equity_m3.yaml")
    assert len(u["stocks"]) == 35
    assert len(u["etfs"]) == 9
    assert len(set(u["stocks"])) == 35, "duplicate stock tickers"
    assert not set(u["stocks"]) & set(u["etfs"]), "stocks/etfs overlap"
    assert "NVDA" not in u["stocks"], (
        "NVDA must NOT be in the M3 universe — it fails the t0-2006 top-cap rule; "
        "keeping it out is the anti-overfit property of the selection"
    )
    assert u["selection_rule"].strip(), "selection rule must be recorded"
    # Distress delistees must be enumerated (B0003 caveat 1: bounded residual bias).
    for t in ("LEH", "WB", "MER", "FNM"):
        assert t in u["excluded_delistees"]


def test_load_universe_rejects_malformed(tmp_path):
    p = tmp_path / "u.yaml"
    p.write_text("stocks: [AAA, AAA]\netfs: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_universe(p)
