"""Single source of truth for which assets the regime/dossier batch builds.

Keys are FULL tickers (XAUUSD, EURUSD, USDJPY, ...) because
pipeline.regimes._resolve_data_path builds f"{ticker}_{freq}.csv" — short keys
like 'XAG' resolve to nothing. asset_class flows into the dossier and the
lookahead firewall. min_bars_for_attempt is the burn-in + sample floor below
which the batch refuses to write an empty parquet.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.regimes import FREQ_BARS_PER_YEAR, _resolve_data_path
from phase5.regime_stats import SAMPLE_SUFFICIENT_BARS_MIN, _MACRO_MEMBERS as _MACRO_PACK

# Mirrors label_regimes' vol_pct_window_years default. The trailing 5y vol
# percentile window is the burn-in: FREQ_BARS_PER_YEAR[freq] * 5 bars are NaN.
VOL_PCT_WINDOW_YEARS = 5


def default_min_bars(frequency: str) -> int:
    """Burn-in + sample floor: below this a cell cannot produce a sufficient regime."""
    burn_in = int(FREQ_BARS_PER_YEAR[frequency] * VOL_PCT_WINDOW_YEARS)
    return burn_in + SAMPLE_SUFFICIENT_BARS_MIN


@dataclass(frozen=True)
class AssetSpec:
    ticker: str
    asset_class: str  # fx | metal | crypto | commodity | equity_index
    frequencies: tuple[str, ...]
    feature_pack: tuple[str, ...]  # alt-features to inject into the dossier
    invert_for_own_regime: bool = False  # USDJPY: own close is the dollar's trend
    min_bars_overrides: dict = field(default_factory=dict)

    def min_bars_for_attempt(self, frequency: str) -> int:
        return self.min_bars_overrides.get(frequency, default_min_bars(frequency))

    def dossier_feature_pack(self) -> tuple[str, ...]:
        """Base pack + macro pack for non-fx assets, order-preserving dedup."""
        if self.asset_class == "fx":
            return self.feature_pack
        return tuple(dict.fromkeys(self.feature_pack + _MACRO_PACK))


# Feature packs: metals get COT + real-yield (when caches exist). B0154: fx
# now gets the exogenous macro drivers (rates differential proxy, risk
# sentiment, breakevens, dollar index — dxy is auto-vetted quasi_circular for
# EUR/GBP by the dossier's Spearman check and labeled as such). Crypto gets
# the macro pack via dossier_feature_pack(). The builder still checks cache
# existence, so a declared feature is silently skipped if its cache is absent.
# B0135: cs_spread_21 (Corwin-Schultz liquidity) is computed from each asset's
# own high/low — declared for every class. B0147: GLD real-volume features are
# gold-domain alt-data — metals packs only.
_METAL_PACK = ("cot_net_noncomm_z52w", "real_yield_5y_z252d",
               "gld_dvol_z42", "gld_amihud_z252", "cs_spread_21")
_FX_PACK = ("us_5y2y_z252", "vix_level", "vix_chg_5", "breakeven_5y_chg5",
            "dxy_z252", "cs_spread_21")
_CRYPTO_PACK = ("cs_spread_21",)

ASSET_REGISTRY: dict[str, AssetSpec] = {
    "XAUUSD": AssetSpec("XAUUSD", "metal", ("D1", "H4"), _METAL_PACK),
    "XAGUSD": AssetSpec("XAGUSD", "metal", ("D1", "H4"), _METAL_PACK),
    "EURUSD": AssetSpec("EURUSD", "fx", ("D1", "H4"), _FX_PACK),
    "GBPUSD": AssetSpec("GBPUSD", "fx", ("D1", "H4"), _FX_PACK),
    "USDJPY": AssetSpec("USDJPY", "fx", ("D1", "H4"), _FX_PACK, invert_for_own_regime=True),
    "BTCUSD": AssetSpec("BTCUSD", "crypto", ("D1", "H4"), _CRYPTO_PACK),
    "ETHUSD": AssetSpec("ETHUSD", "crypto", ("D1", "H4"), _CRYPTO_PACK),
    "SOLUSD": AssetSpec("SOLUSD", "crypto", ("D1", "H4"), _CRYPTO_PACK),
}


def resolve_csv(ticker: str, frequency: str) -> Path:
    """Delegate to the labeler's resolver so there is one source of truth (L1)."""
    return _resolve_data_path(ticker, frequency, None)


def dossier_dirname(ticker: str, frequency: str) -> str:
    """Single source of truth for the per-(ticker, frequency) dossier directory name.

    Mirrors the regime parquet naming (`<ticker>_<freq_lower>_regimes.parquet`). Every
    writer and reader of signals/regime_stats/ MUST use this so the path string never
    drifts (e.g. 'D1' vs 'd1'). B0070.
    """
    return f"{ticker}_{frequency.lower()}"
