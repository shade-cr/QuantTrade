"""Tests for multi-asset H4 OHLCV data sanity (Phase 2 T1).

The validator separates hard errors (raise DataValidationError → caller
must fix) from soft warnings (collected in a report dict → caller can
decide whether to proceed). The cross-asset check is independent: it
only inspects the intersection of timestamps across all asset frames.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from pipeline.data_sanity import (
    DataValidationError,
    validate_asset_data,
    validate_multi_asset_data,
)


def _make_clean_h4(
    n: int = 6000,
    start: str = "2021-01-04 00:00",
    freq: str = "4h",
    is_crypto: bool = False,
) -> pd.DataFrame:
    """Build a synthetic H4 OHLCV that passes every check. For FX/metal,
    business-day filter is applied; for crypto, all 24/7 bars kept."""
    rng = np.random.default_rng(0)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    if not is_crypto:
        # Drop Saturday + Sunday bars to mimic FX weekends.
        idx = idx[~idx.dayofweek.isin([5, 6])]
    n_actual = len(idx)
    log_returns = rng.normal(0.0, 0.003, size=n_actual)
    close = 1000.0 * np.exp(np.cumsum(log_returns))
    spread = np.abs(rng.normal(0.0, 0.002, size=n_actual)) * close
    high = close + spread
    low = close - spread
    open_ = np.concatenate([[close[0]], close[:-1] * (1 + rng.normal(0, 0.001, size=n_actual - 1))])
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    volume = rng.integers(10_000, 100_000, size=n_actual).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Hard errors (must raise DataValidationError)
# ---------------------------------------------------------------------------

def test_raises_on_non_positive_close_price():
    df = _make_clean_h4(n=5500, is_crypto=True)
    df.iloc[100, df.columns.get_loc("close")] = 0.0
    with pytest.raises(DataValidationError, match="non-positive"):
        validate_asset_data("BTCUSD", df, is_crypto=True)


def test_raises_on_negative_high_price():
    df = _make_clean_h4(n=5500, is_crypto=True)
    df.iloc[100, df.columns.get_loc("high")] = -1.0
    with pytest.raises(DataValidationError, match="non-positive"):
        validate_asset_data("BTCUSD", df, is_crypto=True)


def test_raises_on_nan_in_ohlc():
    df = _make_clean_h4(n=5500, is_crypto=True)
    df.iloc[200, df.columns.get_loc("close")] = np.nan
    with pytest.raises(DataValidationError, match="NaN"):
        validate_asset_data("BTCUSD", df, is_crypto=True)


def test_raises_on_empty_frame():
    df = _make_clean_h4(n=10, is_crypto=True).iloc[:0]
    with pytest.raises(DataValidationError, match="empty"):
        validate_asset_data("BTCUSD", df, is_crypto=True)


# ---------------------------------------------------------------------------
# Soft warnings (returned in report, no raise)
# ---------------------------------------------------------------------------

def test_clean_data_returns_empty_warnings():
    """n=8000 H4 crypto bars ≈ 3.65 years — clears every default threshold."""
    df = _make_clean_h4(n=8000, is_crypto=True)
    report = validate_asset_data("BTCUSD", df, is_crypto=True)
    assert report["n_bars"] >= 5000
    assert report["warnings"] == []


def test_fewer_bars_than_min_emits_warning():
    df = _make_clean_h4(n=3000, is_crypto=True)
    report = validate_asset_data("BTCUSD", df, is_crypto=True, min_bars=5000)
    warning_keys = [w["check"] for w in report["warnings"]]
    assert "n_bars" in warning_keys


def test_high_below_open_or_close_emits_warning():
    """A bar where high < max(open, close) is structurally inconsistent —
    soft warning, not hard error, because real-world MT5 data sometimes
    has tiny precision-related inconsistencies."""
    df = _make_clean_h4(n=5500, is_crypto=True)
    # Force high to be below open and close at row 500.
    df.iloc[500, df.columns.get_loc("high")] = (
        df.iloc[500]["close"] * 0.5
    )
    df.iloc[500, df.columns.get_loc("low")] = df.iloc[500]["high"] * 0.5
    report = validate_asset_data("BTCUSD", df, is_crypto=True)
    warning_keys = [w["check"] for w in report["warnings"]]
    assert "high_lt_max_open_close" in warning_keys


def test_zero_volume_above_5pct_emits_warning():
    df = _make_clean_h4(n=5500, is_crypto=True)
    # Force 10% of volume to be zero.
    df.iloc[:550, df.columns.get_loc("volume")] = 0.0
    report = validate_asset_data("BTCUSD", df, is_crypto=True)
    warning_keys = [w["check"] for w in report["warnings"]]
    assert "zero_volume_pct" in warning_keys


def test_unexpected_gaps_emit_warning():
    """A bar gap larger than 1.5x expected (excluding weekend) is anomalous.
    For crypto (no weekends), any gap > 6h is unexpected."""
    df = _make_clean_h4(n=5500, is_crypto=True)
    # Drop a contiguous slice in the middle → creates one large gap.
    df_with_gaps = pd.concat([df.iloc[:1000], df.iloc[1100:]])
    # Add another 60 gaps by dropping random slices.
    drop_idx = []
    rng = np.random.default_rng(1)
    for _ in range(60):
        start = rng.integers(2000, 4000)
        drop_idx.extend(range(start, start + 2))
    keep_mask = ~np.isin(np.arange(len(df_with_gaps)), drop_idx)
    df_with_gaps = df_with_gaps.iloc[keep_mask]
    report = validate_asset_data("BTCUSD", df_with_gaps, is_crypto=True, max_unexpected_gaps=50)
    warning_keys = [w["check"] for w in report["warnings"]]
    assert "unexpected_gaps" in warning_keys


# ---------------------------------------------------------------------------
# Multi-asset cross-checks
# ---------------------------------------------------------------------------

def test_multi_asset_aligned_returns_per_asset_and_cross_reports():
    """Two well-aligned assets → per_asset has both reports, cross_asset
    reports the common-bar count without warnings."""
    btc = _make_clean_h4(n=6000, is_crypto=True)
    eth = _make_clean_h4(n=6000, is_crypto=True)
    report = validate_multi_asset_data(
        {"BTCUSD": btc, "ETHUSD": eth},
        crypto_assets={"BTCUSD", "ETHUSD"},
    )
    assert "per_asset" in report
    assert set(report["per_asset"].keys()) == {"BTCUSD", "ETHUSD"}
    assert "cross_asset" in report
    assert report["cross_asset"]["n_common_bars"] == 6000
    assert report["cross_asset"]["warnings"] == []


def test_multi_asset_too_few_common_bars_emits_cross_warning():
    """Even if each asset has enough bars individually, if the
    intersection is too small the cross-asset alignment fails."""
    btc = _make_clean_h4(n=6000, is_crypto=True, start="2021-01-04")
    eth = _make_clean_h4(n=6000, is_crypto=True, start="2024-01-04")
    report = validate_multi_asset_data(
        {"BTCUSD": btc, "ETHUSD": eth},
        crypto_assets={"BTCUSD", "ETHUSD"},
        min_common_bars=5000,
    )
    warning_keys = [w["check"] for w in report["cross_asset"]["warnings"]]
    assert "n_common_bars" in warning_keys


def test_multi_asset_hard_error_in_one_asset_propagates():
    """If any per-asset validation hard-errors, multi-asset must surface
    it (raise) instead of silently dropping the asset."""
    btc = _make_clean_h4(n=6000, is_crypto=True)
    eth = _make_clean_h4(n=6000, is_crypto=True)
    eth.iloc[100, eth.columns.get_loc("close")] = -5.0
    with pytest.raises(DataValidationError):
        validate_multi_asset_data(
            {"BTCUSD": btc, "ETHUSD": eth},
            crypto_assets={"BTCUSD", "ETHUSD"},
        )
