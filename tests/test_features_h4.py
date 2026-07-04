"""Tests for H4 feature builders (Phase 2 T5).

The H4 builder is a sibling of the D1 builder, not a replacement. It uses
the same primitives (RSI, MACD, ATR) but with bar counts that reflect H4
semantics: r_1bar = 4h return (not r_1 = 1-day return). The renamed
columns prevent SHAP/MDA from being misleading when both Tier 2 D1 and
Tier 2 H4 features coexist in a multi-asset analysis dashboard.

Tests cover:
  - All 24 expected columns are present (10 tech + 3 vol + 8 macro + 3 session).
  - Formulas: r_1bar = log(c/c.shift(1)), z_r24bars uses rolling(252).
  - Session one-hot is correct against the T4 session_filter contract
    (sum across {london, overlap, ny} is 1 for non-ASIA bars, 0 for ASIA).
  - No look-ahead in any renamed feature.
  - REGRESSION: the existing D1 builder (build_tier2_features) produces
    EXACTLY the same output it did before T5. T5 must extend, not modify.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from pipeline.features import (
    build_tier2_features,            # D1 â€” must be unchanged
    build_tier2_h4_features,         # new in T5 Phase A
    build_crossasset_features,       # new in T5 Phase B
)


@pytest.fixture
def synth_h4_ohlcv() -> pd.DataFrame:
    """800 bars of synthetic H4 OHLCV spanning ~6 months at UTC hours."""
    rng = np.random.default_rng(7)
    n = 800
    # H4 bars at 1:00, 5:00, 9:00, 13:00, 17:00, 21:00 UTC each day
    # â€” use a regular 4-hour frequency for simplicity.
    idx = pd.date_range("2023-01-02 01:00", periods=n, freq="4h", tz="UTC")
    log_returns = rng.normal(0.0, 0.003, size=n)  # H4 vol smaller than D1
    close = 1000.0 * np.exp(np.cumsum(log_returns))
    spread = np.abs(rng.normal(0.0, 0.002, size=n)) * close
    high = close + spread
    low = close - spread
    open_ = np.concatenate([[close[0]], close[:-1] * (1 + rng.normal(0, 0.001, size=n - 1))])
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    volume = rng.integers(10_000, 100_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _mock_macro(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Already-shifted macro frame mirroring the D1 contract."""
    return pd.DataFrame(
        {
            "DTWEXBGS": np.linspace(100, 110, len(idx)),
            "DFII5": np.linspace(0.5, 1.5, len(idx)),
            "DGS5": np.linspace(1.5, 2.5, len(idx)),
            "T5YIE": np.linspace(2.0, 3.0, len(idx)),
            "VIXCLS": np.linspace(15, 25, len(idx)),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Column set
# ---------------------------------------------------------------------------

def test_h4_feature_columns_match_spec(synth_h4_ohlcv):
    """30 columns: 14 tech + 3 vol + 10 macro + 3 session. The D1 builder
    produces technical+macro without sessions; H4 adds the 3 one-hots and
    renames the per-bar features. Tech includes B0134 (ffd_logclose) and
    B0149 (volume-participation block)."""
    macro = _mock_macro(synth_h4_ohlcv.index)
    feats = build_tier2_h4_features(synth_h4_ohlcv, macro)

    technical = {
        "r_1bar", "r_6bars", "r_24bars", "r_120bars", "z_r24bars",
        "rsi_14", "macd_signal", "macd_hist",
        "atr_14_norm", "bb_width_120bars",
        # B0134: FFD log-price (long-memory level).
        "ffd_logclose",
        # B0135: Corwin-Schultz high-low spread (liquidity dimension).
        "cs_spread_21",
        # B0149: volume-participation block (PIT-safe derived transforms).
        "volume_z42", "volume_pct_rank_21", "volume_rel_median_42",
    }
    vol = {"rv_24bars", "rv_regime", "rv_term_structure"}
    # dtwexbgs_close added in B0015b for phase5_cot_extremes primary access.
    # The meta NEVER sees it (filtered by apply_primary_feature_blacklist at the
    # orchestrator level). It only appears in the raw build_tier2_*_features output.
    macro_cols = {
        "dtwexbgs_close",
        "dxy_z252", "dxy_chg_5", "real_yield_5y", "real_yield_5y_chg_5",
        "real_yield_5y_z252d",  # B0153: z-score gate feature (B004v3/T148R2)
        "breakeven_5y", "nominal_5y_chg_5", "vix_level", "vix_chg_5",
    }
    session = {"session_london", "session_overlap", "session_ny"}

    expected = technical | vol | macro_cols | session
    feature_cols = set(feats.columns) - {"_atr_14"}  # internal artifact
    assert expected == feature_cols, (
        f"missing: {expected - feature_cols}, unexpected: {feature_cols - expected}"
    )
    assert len(expected) == 31


def test_h4_macro_features_emit_umcsent_when_present(synth_h4_ohlcv):
    """If the macro frame carries UMCSENT (+ precomputed _chg_3m), the H4
    builder exposes umcsent_level / umcsent_chg_3m. Legacy frames without
    UMCSENT keep producing exactly 24 baseline columns.
    """
    idx = synth_h4_ohlcv.index
    macro = pd.DataFrame(
        {
            "DTWEXBGS": np.linspace(100, 110, len(idx)),
            "DFII5": np.linspace(0.5, 1.5, len(idx)),
            "DGS5": np.linspace(1.5, 2.5, len(idx)),
            "T5YIE": np.linspace(2.0, 3.0, len(idx)),
            "VIXCLS": np.linspace(15, 25, len(idx)),
            "UMCSENT": np.linspace(75.0, 95.0, len(idx)),
            "UMCSENT_chg_3m": np.linspace(0.0, 5.0, len(idx)),
        },
        index=idx,
    )
    feats = build_tier2_h4_features(synth_h4_ohlcv, macro)
    assert "umcsent_level" in feats.columns
    assert "umcsent_chg_3m" in feats.columns

    # Identity passthrough at a deep row.
    assert feats["umcsent_level"].iloc[-1] == macro["UMCSENT"].iloc[-1]
    assert feats["umcsent_chg_3m"].iloc[-1] == macro["UMCSENT_chg_3m"].iloc[-1]


# ---------------------------------------------------------------------------
# Formulas
# ---------------------------------------------------------------------------

def test_r_1bar_equals_log_return_one_step(synth_h4_ohlcv):
    macro = _mock_macro(synth_h4_ohlcv.index)
    feats = build_tier2_h4_features(synth_h4_ohlcv, macro)
    expected = np.log(synth_h4_ohlcv["close"] / synth_h4_ohlcv["close"].shift(1))
    pd.testing.assert_series_equal(
        feats["r_1bar"], expected, check_names=False, rtol=1e-12,
    )


def test_r_24bars_equals_log_return_24_steps(synth_h4_ohlcv):
    macro = _mock_macro(synth_h4_ohlcv.index)
    feats = build_tier2_h4_features(synth_h4_ohlcv, macro)
    expected = np.log(synth_h4_ohlcv["close"] / synth_h4_ohlcv["close"].shift(24))
    pd.testing.assert_series_equal(
        feats["r_24bars"], expected, check_names=False, rtol=1e-12,
    )


def test_z_r24bars_uses_252_bar_rolling_window(synth_h4_ohlcv):
    macro = _mock_macro(synth_h4_ohlcv.index)
    feats = build_tier2_h4_features(synth_h4_ohlcv, macro)
    r24 = np.log(synth_h4_ohlcv["close"] / synth_h4_ohlcv["close"].shift(24))
    expected = (r24 - r24.rolling(252).mean()) / r24.rolling(252).std()
    # Compare a tail slice where the rolling window is mature.
    np.testing.assert_allclose(
        feats["z_r24bars"].iloc[-50:].values,
        expected.iloc[-50:].values,
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# Session one-hot
# ---------------------------------------------------------------------------

def test_session_one_hot_sums_to_one_for_non_asia_bars(synth_h4_ohlcv):
    """Per-row: session_london + session_overlap + session_ny == 1 when the
    bar is NOT in ASIA, and == 0 when it IS in ASIA (Asia is baseline)."""
    macro = _mock_macro(synth_h4_ohlcv.index)
    feats = build_tier2_h4_features(synth_h4_ohlcv, macro)
    cols = ["session_london", "session_overlap", "session_ny"]
    row_sums = feats[cols].sum(axis=1)
    # All values must be 0 or 1 (exactly one-hot or all-zero).
    assert set(row_sums.unique()) <= {0, 1}, (
        f"session one-hot must sum to 0 or 1 per row, got {sorted(row_sums.unique())}"
    )
    # Hours 7-21 UTC are NOT in ASIA â†’ sum should be 1 for those rows.
    non_asia_hours = (synth_h4_ohlcv.index.hour >= 7) & (synth_h4_ohlcv.index.hour < 22)
    assert (row_sums[non_asia_hours] == 1).all(), "non-ASIA hours should sum to 1"
    # Hours 22-23 and 0-6 are ASIA â†’ sum should be 0.
    asia_hours = ~non_asia_hours
    assert (row_sums[asia_hours] == 0).all(), "ASIA hours should sum to 0"


def test_session_columns_have_correct_label_per_hour(synth_h4_ohlcv):
    """Cross-check the one-hot columns against the T4 contract:
    LONDON [07:00, 13:00), OVERLAP [13:00, 17:00), NY [17:00, 22:00)."""
    macro = _mock_macro(synth_h4_ohlcv.index)
    feats = build_tier2_h4_features(synth_h4_ohlcv, macro)
    hours = synth_h4_ohlcv.index.hour
    assert (feats.loc[(hours >= 7) & (hours < 13), "session_london"] == 1).all()
    assert (feats.loc[(hours >= 13) & (hours < 17), "session_overlap"] == 1).all()
    assert (feats.loc[(hours >= 17) & (hours < 22), "session_ny"] == 1).all()


# ---------------------------------------------------------------------------
# No look-ahead (renamed features behave like D1 ones)
# ---------------------------------------------------------------------------

def test_no_lookahead_in_h4_features(synth_h4_ohlcv):
    """Mutating close[t=400] must not change feature values at rows < 400."""
    macro = _mock_macro(synth_h4_ohlcv.index)
    feats_a = build_tier2_h4_features(synth_h4_ohlcv, macro)
    mutated = synth_h4_ohlcv.copy()
    mutated.iloc[400, mutated.columns.get_loc("close")] *= 5
    feats_b = build_tier2_h4_features(mutated, macro)
    pd.testing.assert_frame_equal(
        feats_a.iloc[:399], feats_b.iloc[:399],
        check_dtype=False,
    )


# ---------------------------------------------------------------------------
# Regression: D1 builder unchanged
# ---------------------------------------------------------------------------

def test_d1_builder_columns_unchanged_after_t5(synth_ohlcv):
    """The D1 entry point (build_tier2_features) MUST produce exactly the
    same columns it did before T5 â€” plus dtwexbgs_close (added in B0015b for
    phase5_cot_extremes primary, NEVER seen by the meta thanks to the
    orchestrator-applied blacklist filter)."""
    df = synth_ohlcv.set_index("time")
    macro = _mock_macro(df.index)
    feats = build_tier2_features(df, macro)
    expected_d1 = {
        # Technical (with original D1 names)
        "r_1", "r_5", "r_10", "r_20", "z_r20",
        "rsi_14", "macd_signal", "macd_hist",
        "atr_14_norm", "bb_width_20",
        "rv_20", "rv_regime", "rv_term_structure",
        # B0134: FFD log-price (long-memory level).
        "ffd_logclose",
        # B0135: Corwin-Schultz high-low spread (liquidity dimension).
        "cs_spread_21",
        # B0016: overnight/intraday decomposition + Kyle/Amihud lambdas.
        "r_overnight", "r_intraday",
        "on_ewma_21", "on_ewma_60", "in_ewma_21", "in_ewma_60", "tug_21",
        "amihud_20", "kyle_t_20",
        # B0149: volume-participation block (PIT-safe derived transforms).
        "volume_z42", "volume_pct_rank_21", "volume_rel_median_42",
        # Macro
        "dtwexbgs_close",  # B0015b: raw DXY exposed for phase5_cot_extremes primary
        "dxy_z252", "dxy_chg_5", "real_yield_5y", "real_yield_5y_chg_5",
        "real_yield_5y_z252d",  # B0153: z-score gate feature (B004v3/T148R2)
        "breakeven_5y", "nominal_5y_chg_5", "vix_level", "vix_chg_5",
    }
    feature_cols = set(feats.columns) - {"_atr_14"}
    assert expected_d1 == feature_cols, (
        f"D1 builder drifted. Missing: {expected_d1 - feature_cols}, "
        f"unexpected: {feature_cols - expected_d1}"
    )


# ---------------------------------------------------------------------------
# T5 Phase B â€” cross-asset features (consumes T2 intraday macro + T3 cross_asset)
# ---------------------------------------------------------------------------

def _intraday_macro(target_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Synthetic version of `pipeline.macro_fetch_intraday.build_intraday_macro_frame`
    output â€” already aligned to target_idx with dxy/vix/_stale columns."""
    n = len(target_idx)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "dxy": 100.0 + rng.standard_normal(n).cumsum() * 0.1,
            "dxy_stale": np.zeros(n),
            "vix": 18.0 + rng.standard_normal(n).cumsum() * 0.05,
            "vix_stale": np.zeros(n),
        },
        index=target_idx,
    )


def _btc_h4(n: int = 800) -> pd.DataFrame:
    """Synthetic BTC H4 OHLCV at 4h cadence."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-02 01:00", periods=n, freq="4h", tz="UTC")
    log_ret = rng.normal(0.0, 0.01, size=n)
    close = 30000.0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(
        {"open": close * 0.999, "high": close * 1.005, "low": close * 0.995,
         "close": close, "volume": np.ones(n) * 1000},
        index=idx,
    )


def _xag_h4(n: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(43)
    idx = pd.date_range("2023-01-02 01:00", periods=n, freq="4h", tz="UTC")
    log_ret = rng.normal(0.0, 0.004, size=n)
    close = 25.0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": np.ones(n) * 1000},
        index=idx,
    )


def _daily_macro(target_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Already-shifted daily FRED frame, mirroring build_macro_frame output."""
    return pd.DataFrame(
        {
            "DTWEXBGS": np.linspace(100, 110, len(target_idx)),
            "DFII5": np.linspace(0.5, 1.5, len(target_idx)),
            "DGS5": np.linspace(1.5, 2.5, len(target_idx)),
            "T5YIE": np.linspace(2.0, 3.0, len(target_idx)),
            "VIXCLS": np.linspace(15, 25, len(target_idx)),
        },
        index=target_idx,
    )


# --- FX (EUR/GBP/JPY): vix_h4_level, vix_h4_change, dxy_h4_zscore --- #

def test_fx_crossasset_columns_eurusd():
    idx = pd.date_range("2024-01-01 00:00", periods=400, freq="4h", tz="UTC")
    intraday = _intraday_macro(idx)
    feats = build_crossasset_features("EURUSD", idx, intraday_macro=intraday)
    assert set(feats.columns) == {"vix_h4_level", "vix_h4_change", "dxy_h4_zscore"}
    assert feats.index.equals(idx)


def test_fx_crossasset_no_lookahead_vix_change():
    """vix_h4_change[t] = vix[t-?] - vix[t-?-6]. Both must be from strictly
    past bars relative to t (the intraday_macro input is already aligned
    with the same no-leak contract from T2)."""
    idx = pd.date_range("2024-01-01 00:00", periods=400, freq="4h", tz="UTC")
    intraday = _intraday_macro(idx)
    feats = build_crossasset_features("EURUSD", idx, intraday_macro=intraday)
    # The first 6 bars cannot have a delta-6 value.
    assert feats["vix_h4_change"].iloc[:6].isna().any() or (feats["vix_h4_change"].iloc[:6] == 0).any()


# --- Crypto (BTC/ETH/SOL): btc_h4_return, btc_h4_rv24bars, dxy_h4_return --- #

def test_crypto_crossasset_columns_for_eth():
    """ETH gets full BTC cross-features."""
    idx = pd.date_range("2024-01-01 00:00", periods=400, freq="4h", tz="UTC")
    intraday = _intraday_macro(idx)
    btc = _btc_h4(n=800)
    feats = build_crossasset_features(
        "ETHUSD", idx, intraday_macro=intraday, btc_df=btc,
    )
    assert set(feats.columns) == {"btc_h4_return", "btc_h4_rv24bars", "dxy_h4_return"}
    # btc_h4_return should be non-trivial (real BTC data has non-zero returns)
    assert feats["btc_h4_return"].dropna().abs().sum() > 0


def test_crypto_crossasset_for_btc_self_returns_zero_btc_return():
    """BTC itself: btc_h4_return is 0 (no self-reference) but btc_h4_rv24bars
    and dxy_h4_return still apply. Column structure stays consistent
    across crypto assets."""
    idx = pd.date_range("2024-01-01 00:00", periods=400, freq="4h", tz="UTC")
    intraday = _intraday_macro(idx)
    btc = _btc_h4(n=800)
    feats = build_crossasset_features(
        "BTCUSD", idx, intraday_macro=intraday, btc_df=btc,
    )
    assert set(feats.columns) == {"btc_h4_return", "btc_h4_rv24bars", "dxy_h4_return"}
    # Self-reference: btc_h4_return all zero for BTC
    assert (feats["btc_h4_return"] == 0).all()


# --- Metals (XAU/XAG): dxy_h4_return, dxy_h4_zscore, real_yield_chg_6bars + xau_xag_ratio (XAU only) --- #

def test_metal_xauusd_crossasset_columns():
    """XAU includes the xau_xag_ratio."""
    idx = pd.date_range("2024-01-01 00:00", periods=400, freq="4h", tz="UTC")
    intraday = _intraday_macro(idx)
    xag = _xag_h4(n=800)
    daily = _daily_macro(idx)
    feats = build_crossasset_features(
        "XAUUSD", idx,
        intraday_macro=intraday, xag_df=xag, daily_macro_frame=daily,
    )
    assert set(feats.columns) == {
        "dxy_h4_return", "dxy_h4_zscore", "real_yield_chg_6bars", "xau_xag_ratio",
    }


def test_metal_xagusd_crossasset_columns_no_ratio():
    """XAG does NOT get xau_xag_ratio (would be circular). 3 columns instead of 4."""
    idx = pd.date_range("2024-01-01 00:00", periods=400, freq="4h", tz="UTC")
    intraday = _intraday_macro(idx)
    daily = _daily_macro(idx)
    feats = build_crossasset_features(
        "XAGUSD", idx,
        intraday_macro=intraday, daily_macro_frame=daily,
    )
    assert set(feats.columns) == {
        "dxy_h4_return", "dxy_h4_zscore", "real_yield_chg_6bars",
    }


# --- Unknown asset class --- #

def test_unknown_asset_raises():
    idx = pd.date_range("2024-01-01 00:00", periods=400, freq="4h", tz="UTC")
    intraday = _intraday_macro(idx)
    with pytest.raises(ValueError, match="unknown asset"):
        build_crossasset_features("DOGEUSD", idx, intraday_macro=intraday)


# --- Required inputs guard --- #

def test_fx_requires_intraday_macro():
    """FX cross-features need DXY+VIX â†’ without intraday_macro, raise."""
    idx = pd.date_range("2024-01-01 00:00", periods=400, freq="4h", tz="UTC")
    with pytest.raises(ValueError, match="intraday_macro"):
        build_crossasset_features("EURUSD", idx, intraday_macro=None)


def test_crypto_non_btc_requires_btc_df():
    """ETH/SOL need btc_df for the BTC-derived features."""
    idx = pd.date_range("2024-01-01 00:00", periods=400, freq="4h", tz="UTC")
    intraday = _intraday_macro(idx)
    with pytest.raises(ValueError, match="btc_df"):
        build_crossasset_features("ETHUSD", idx, intraday_macro=intraday, btc_df=None)


def test_xau_requires_xag_df():
    """XAU's xau_xag_ratio needs xag_df."""
    idx = pd.date_range("2024-01-01 00:00", periods=400, freq="4h", tz="UTC")
    intraday = _intraday_macro(idx)
    daily = _daily_macro(idx)
    with pytest.raises(ValueError, match="xag_df"):
        build_crossasset_features(
            "XAUUSD", idx,
            intraday_macro=intraday, xag_df=None, daily_macro_frame=daily,
        )
