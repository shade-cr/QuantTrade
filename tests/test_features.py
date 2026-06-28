"""Tests for pipeline.features."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from pipeline.features import build_technical_features, build_tier2_features


def test_technical_features_columns_and_no_lookahead(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    feats = build_technical_features(df)
    expected = {
        "r_1", "r_5", "r_10", "r_20", "z_r20",
        "rsi_14", "macd_signal", "macd_hist",
        "atr_14_norm", "bb_width_20",
        "rv_20", "rv_regime", "rv_term_structure",
    }
    assert expected.issubset(feats.columns)
    # No look-ahead: changing close[t+10] must not change features at t<t+10.
    mutated = df.copy()
    mutated.iloc[400, mutated.columns.get_loc("close")] *= 5
    feats2 = build_technical_features(mutated)
    # Features up to row 399 must be identical.
    pd.testing.assert_frame_equal(
        feats.iloc[:399].drop(columns=feats.columns.difference(expected), errors="ignore"),
        feats2.iloc[:399].drop(columns=feats2.columns.difference(expected), errors="ignore"),
    )


def test_ffd_logclose_present_and_causal(synth_ohlcv):
    """B0134: ffd_logclose is wired into the D1 technical block and is causal."""
    df = synth_ohlcv.set_index("time")
    feats = build_technical_features(df)
    assert "ffd_logclose" in feats.columns
    assert feats["ffd_logclose"].iloc[-50:].notna().any()
    # No look-ahead: mutating a FUTURE close must not change earlier values.
    mutated = df.copy()
    mutated.iloc[400, mutated.columns.get_loc("close")] *= 3
    feats2 = build_technical_features(mutated)
    a = feats["ffd_logclose"].iloc[:399].to_numpy()
    b = feats2["ffd_logclose"].iloc[:399].to_numpy()
    mask = ~(np.isnan(a) | np.isnan(b))
    np.testing.assert_array_equal(a[mask], b[mask])


def test_z_r20_formula(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    feats = build_technical_features(df)
    # Direct computation of expected z_r20 at a deep row.
    r20 = np.log(df["close"] / df["close"].shift(20))
    mu = r20.rolling(252).mean()
    sd = r20.rolling(252).std()
    expected_z = (r20 - mu) / sd
    # Compare last 50 rows (where rolling stats are mature).
    np.testing.assert_allclose(feats["z_r20"].iloc[-50:].values, expected_z.iloc[-50:].values, rtol=1e-9)


def test_volume_features_present_and_causal(synth_ohlcv):
    """B0149: derived volume-participation features exist and are PIT-safe.

    Raw volume is NOT exposed as a feature (non-stationary); only rolling
    backward-looking transforms are. Mutating a FUTURE volume bar must not
    change any earlier feature row.
    """
    df = synth_ohlcv.set_index("time")
    feats = build_technical_features(df)
    cols = ["volume_z42", "volume_pct_rank_21", "volume_rel_median_42"]
    for col in cols:
        assert col in feats.columns, f"{col} missing from technical features"
        assert feats[col].iloc[-50:].notna().any(), f"{col} all-NaN at tail"
    assert "volume" not in feats.columns  # raw series never exposed
    mutated = df.copy()
    mutated.iloc[400, mutated.columns.get_loc("volume")] *= 7
    feats2 = build_technical_features(mutated)
    pd.testing.assert_frame_equal(feats[cols].iloc[:400], feats2[cols].iloc[:400])


def test_volume_z42_formula(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    feats = build_technical_features(df)
    v = df["volume"].astype(float)
    expected = (v - v.rolling(42).mean()) / v.rolling(42).std()
    np.testing.assert_allclose(
        feats["volume_z42"].iloc[-50:].values, expected.iloc[-50:].values, rtol=1e-9
    )


def test_volume_pct_rank_bounds(synth_ohlcv):
    """Rolling percentile rank of the CURRENT bar within its trailing window:
    bounded in (0, 1], NaN during warmup."""
    df = synth_ohlcv.set_index("time")
    feats = build_technical_features(df)
    tail = feats["volume_pct_rank_21"].iloc[-200:].dropna()
    assert len(tail) > 0
    assert (tail > 0).all()
    assert (tail <= 1.0).all()


def test_macro_features_columns(synth_ohlcv):
    """Macro frame is already lagged in macro_fetch; here we test alignment."""
    df = synth_ohlcv.set_index("time")
    idx = df.index
    # Mock macro frame already shifted: value at row k is "yesterday's" macro.
    macro = pd.DataFrame(
        {
            "DTWEXBGS": np.linspace(100, 110, len(idx)),
            "DFII5": np.linspace(0.5, 1.5, len(idx)),
            "DGS5": np.linspace(1.5, 2.5, len(idx)),
            "T5YIE": np.linspace(2.0, 3.0, len(idx)),
            "VIXCLS": np.linspace(15, 25, len(idx)),
        },
        index=idx,
    )
    feats = build_tier2_features(df, macro)
    expected = {
        "dxy_z252", "dxy_chg_5", "real_yield_5y", "real_yield_5y_chg_5",
        "breakeven_5y", "nominal_5y_chg_5", "vix_level", "vix_chg_5",
    }
    assert expected.issubset(feats.columns)


def test_real_yield_z252d_present_and_formula(synth_ohlcv):
    """B0153: real_yield_5y_z252d — trailing 252-bar z-score of the (already
    publication-lagged) DFII5 series. Requested by name by the B004v3 primary
    gate and the T148R2 meta feature_overrides; was silently absent."""
    df = synth_ohlcv.set_index("time")
    idx = df.index
    rng = np.random.default_rng(7)
    macro = pd.DataFrame(
        {
            "DTWEXBGS": np.linspace(100, 110, len(idx)),
            "DFII5": np.cumsum(rng.normal(0, 0.02, len(idx))) + 1.0,
            "DGS5": np.linspace(1.5, 2.5, len(idx)),
            "T5YIE": np.linspace(2.0, 3.0, len(idx)),
            "VIXCLS": np.linspace(15, 25, len(idx)),
        },
        index=idx,
    )
    feats = build_tier2_features(df, macro)
    assert "real_yield_5y_z252d" in feats.columns
    ry = macro["DFII5"]
    expected = (ry - ry.rolling(252).mean()) / ry.rolling(252).std()
    np.testing.assert_allclose(
        feats["real_yield_5y_z252d"].iloc[-50:].values,
        expected.iloc[-50:].values,
        rtol=1e-9,
    )


def test_macro_features_emit_dgs2_when_present(synth_ohlcv):
    """Audit C3 fix: DGS2 emits us2y_level, us2y_chg_5, us_2y10y_spread.

    Absent in legacy frames → columns not synthesised (backward compat).
    """
    df = synth_ohlcv.set_index("time")
    idx = df.index
    macro_with = pd.DataFrame(
        {
            "DTWEXBGS": np.linspace(100, 110, len(idx)),
            "DFII5": np.linspace(0.5, 1.5, len(idx)),
            "DGS5": np.linspace(1.5, 2.5, len(idx)),
            "DGS2": np.linspace(0.8, 1.8, len(idx)),
            "T5YIE": np.linspace(2.0, 3.0, len(idx)),
            "VIXCLS": np.linspace(15, 25, len(idx)),
        },
        index=idx,
    )
    feats = build_tier2_features(df, macro_with)
    assert "us2y_level" in feats.columns
    assert "us2y_chg_5" in feats.columns
    assert "us_2y10y_spread" in feats.columns
    # Identity / arithmetic checks at a deep row (post-warmup).
    assert feats["us2y_level"].iloc[-1] == macro_with["DGS2"].iloc[-1]
    np.testing.assert_allclose(
        feats["us_2y10y_spread"].iloc[-1],
        macro_with["DGS5"].iloc[-1] - macro_with["DGS2"].iloc[-1],
        rtol=1e-12,
    )

    # Backward compat: legacy frame without DGS2 must not crash or synthesise.
    macro_without = macro_with.drop(columns=["DGS2"])
    feats_legacy = build_tier2_features(df, macro_without)
    assert "us2y_level" not in feats_legacy.columns
    assert "us2y_chg_5" not in feats_legacy.columns
    assert "us_2y10y_spread" not in feats_legacy.columns


def test_macro_features_emit_umcsent_when_present(synth_ohlcv):
    """When the macro frame carries UMCSENT + UMCSENT_chg_3m (e.g., real
    FRED fetch), build_tier2_features (D1) exposes umcsent_level and
    umcsent_chg_3m. When absent (legacy mock), those columns are skipped."""
    df = synth_ohlcv.set_index("time")
    idx = df.index
    macro_with = pd.DataFrame(
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
    feats = build_tier2_features(df, macro_with)
    assert "umcsent_level" in feats.columns
    assert "umcsent_chg_3m" in feats.columns
    # Identity passthrough at a deep row (post-warmup).
    assert feats["umcsent_level"].iloc[-1] == macro_with["UMCSENT"].iloc[-1]
    assert feats["umcsent_chg_3m"].iloc[-1] == macro_with["UMCSENT_chg_3m"].iloc[-1]

    # Backward compat: legacy frames without UMCSENT must not crash, and
    # must not synthesise the columns.
    macro_without = macro_with.drop(columns=["UMCSENT", "UMCSENT_chg_3m"])
    feats_legacy = build_tier2_features(df, macro_without)
    assert "umcsent_level" not in feats_legacy.columns
    assert "umcsent_chg_3m" not in feats_legacy.columns


def test_macro_publication_lag_no_lookahead(synth_ohlcv):
    """If we mutate macro at t, features at the SAME t must not change (uses t-1).

    macro_fetch.build_macro_frame applies .shift(1). Here we simulate that contract:
    if the caller passes an ALREADY-shifted macro frame, features at row k use row k
    (which is yesterday's macro). To make this end-to-end, we mutate row k+1 of an
    UNshifted macro and verify it does NOT change feature row k.
    """
    df = synth_ohlcv.set_index("time")
    idx = df.index
    raw = pd.DataFrame(
        {
            "DTWEXBGS": np.linspace(100, 110, len(idx)),
            "DFII5": np.linspace(0.5, 1.5, len(idx)),
            "DGS5": np.linspace(1.5, 2.5, len(idx)),
            "T5YIE": np.linspace(2.0, 3.0, len(idx)),
            "VIXCLS": np.linspace(15, 25, len(idx)),
        },
        index=idx,
    )
    shifted = raw.shift(1)  # simulates macro_fetch output
    feats_a = build_tier2_features(df, shifted)

    raw2 = raw.copy()
    raw2.iloc[300, raw2.columns.get_loc("DTWEXBGS")] = 9999.0  # mutate "today"
    shifted2 = raw2.shift(1)
    feats_b = build_tier2_features(df, shifted2)

    # feats_a[t=300] uses macro row 299 (yesterday). Mutating macro at 300 changes
    # feats_b only from row 301 onwards. So row 300 must be identical.
    np.testing.assert_array_equal(
        feats_a.iloc[300][["dxy_z252", "real_yield_5y"]].values,
        feats_b.iloc[300][["dxy_z252", "real_yield_5y"]].values,
    )
