"""Asset registry validation — keys are full tickers, classes valid, CSVs resolve."""
from __future__ import annotations
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase5.asset_registry import ASSET_REGISTRY, AssetSpec, default_min_bars
from phase5.regime_stats import SAMPLE_SUFFICIENT_BARS_MIN
from pipeline.regimes import FREQ_BARS_PER_YEAR, _resolve_data_path

VALID_CLASSES = {"fx", "metal", "crypto", "commodity", "equity", "equity_index"}

# Skip data-presence tests when the data/ directory hasn't been populated yet.
# This is expected in a fresh repo — data is gitignored and must be populated separately.
_DATA_PRESENT = (REPO_ROOT / "data" / "D1_22y").exists() or (REPO_ROOT / "data" / "D1").exists()


def test_keys_are_full_tickers_and_match_id():
    for key, spec in ASSET_REGISTRY.items():
        assert key == spec.ticker, f"registry key {key!r} != spec.ticker {spec.ticker!r}"
        # keys must match their spec.ticker exactly; length varies by asset class
        # (FX/metal/crypto use 6-char broker symbols, equities use standard tickers)
        assert 4 <= len(key) <= 6, f"{key!r} ticker length must be 4–6 chars"


def test_asset_class_valid():
    for spec in ASSET_REGISTRY.values():
        assert spec.asset_class in VALID_CLASSES, f"{spec.ticker}: bad class {spec.asset_class!r}"


def test_usdjpy_marked_inverted():
    assert ASSET_REGISTRY["USDJPY"].invert_for_own_regime is True


def test_default_min_bars_is_burnin_plus_floor():
    # D1: 252*5 + 200 = 1460 ; H4: 1512*5 + 200 = 7760
    assert default_min_bars("D1") == FREQ_BARS_PER_YEAR["D1"] * 5 + SAMPLE_SUFFICIENT_BARS_MIN
    assert default_min_bars("H4") == FREQ_BARS_PER_YEAR["H4"] * 5 + SAMPLE_SUFFICIENT_BARS_MIN
    assert default_min_bars("D1") == 1460


def test_spec_min_bars_for_attempt_uses_default_when_no_override():
    spec = ASSET_REGISTRY["XAUUSD"]
    assert spec.min_bars_for_attempt("D1") == default_min_bars("D1")


@pytest.mark.skipif(not _DATA_PRESENT, reason="data/ directory not populated — run scripts/fetch_equity_daily.py first")
@pytest.mark.parametrize("ticker", list(ASSET_REGISTRY))
def test_every_asset_resolves_a_d1_csv(ticker):
    """Registry must agree with _resolve_data_path (single source of truth, L1).

    Skips gracefully for tickers whose CSV hasn't been fetched yet — this repo
    is equity-focused and only populates data for assets under active research;
    non-equity tickers inherited from the seed are skipped until their data lands.
    """
    # Some assets may legitimately lack H4; D1 must always resolve.
    try:
        path = _resolve_data_path(ticker, "D1", None)
    except FileNotFoundError:
        pytest.skip(f"{ticker}: no D1 CSV in data/D1 or data/D1_22y — fetch it first")
    assert path.exists(), f"{ticker}: no D1 CSV at {path}"


def test_dossier_dirname_lowercases_frequency():
    from phase5.asset_registry import dossier_dirname
    assert dossier_dirname("XAUUSD", "D1") == "XAUUSD_d1"
    assert dossier_dirname("BTCUSD", "H4") == "BTCUSD_h4"
    # idempotent vs already-lower input
    assert dossier_dirname("ETHUSD", "h4") == "ETHUSD_h4"


# B0068 Tier-1: macro pack tests
from phase5.asset_registry import ASSET_REGISTRY, _MACRO_PACK


def test_macro_pack_applied_to_non_fx_with_dedup():
    metal = ASSET_REGISTRY["XAUUSD"].dossier_feature_pack()
    assert metal.count("real_yield_5y_z252d") == 1
    assert "vix_level" in metal and "us_5y2y_z252" in metal
    assert "cot_net_noncomm_z52w" in metal


def test_fx_pack_is_declared_not_auto_widened():
    """B0154: fx now carries an explicitly DECLARED exogenous pack (rates
    differential, risk sentiment, breakevens, dxy — dxy auto-vetted
    quasi_circular by the dossier's Spearman check). The auto macro-widening
    branch still applies only to non-fx: fx gets exactly its declared pack,
    nothing implicit."""
    spec = ASSET_REGISTRY["EURUSD"]
    fx = spec.dossier_feature_pack()
    assert fx == spec.feature_pack  # no implicit widening for fx
    assert "us_5y2y_z252" in fx and "vix_level" in fx and "dxy_z252" in fx
    assert "cot_net_noncomm_z52w" not in fx  # metal-only alt feature


def test_crypto_gets_macro_pack():
    btc = ASSET_REGISTRY["BTCUSD"].dossier_feature_pack()
    assert "vix_level" in btc and "dxy_z252" in btc
