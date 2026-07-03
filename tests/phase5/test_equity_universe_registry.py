"""M3 universe tickers register as D1 equity/equity_index specs (B0003)."""
from phase5.asset_registry import ASSET_REGISTRY
from pipeline.equity_universe import load_universe


def test_universe_tickers_registered_with_correct_class():
    u = load_universe("configs/universe_equity_m3.yaml")
    for t in u["stocks"]:
        spec = ASSET_REGISTRY[t]
        assert spec.asset_class == "equity"
        assert spec.frequencies == ("D1",)
        assert "cs_spread_21" in spec.feature_pack
    for t in u["etfs"]:
        spec = ASSET_REGISTRY[t]
        assert spec.asset_class == "equity_index"
        assert spec.frequencies == ("D1",)


def test_legacy_entries_untouched():
    assert ASSET_REGISTRY["NVDA"].asset_class == "equity"
    assert ASSET_REGISTRY["XAUUSD"].asset_class == "metal"
