"""Tests for cross-asset alignment + dependency graph (Phase 2 T3).

The critical invariant: every cross-asset value used as a feature at
bar t comes from the source asset's value AT t-1 (one full H4 bar lag).
Without this shift, BTC features for ETH would leak: the model would
see BTC's price/return AT t when predicting ETH's outcome from t. With
the shift, the BTC info is strictly past.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from pipeline.cross_asset import (
    LEVEL_0_ASSETS,
    LEVEL_1_ASSETS,
    ALL_ASSETS,
    load_multi_asset,
    align_to_master_index,
    compute_btc_features,
    compute_xau_xag_ratio,
    level_of,
    dependencies_of,
    topological_order,
)


def _h4_ohlcv(n: int, start: str = "2024-01-01 00:00", seed: int = 0,
              base_price: float = 1000.0) -> pd.DataFrame:
    """Build synthetic H4 OHLCV indexed by tz-aware UTC timestamps."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    log_returns = rng.normal(0.0, 0.003, size=n)
    close = base_price * np.exp(np.cumsum(log_returns))
    spread = np.abs(rng.normal(0.0, 0.002, size=n)) * close
    high = close + spread
    low = close - spread
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])
    volume = rng.integers(10_000, 100_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Dependency graph (sync, no I/O)
# ---------------------------------------------------------------------------

def test_level_0_assets_are_self_contained():
    """Nivel 0 assets have no dependencies on other assets."""
    for asset in LEVEL_0_ASSETS:
        assert level_of(asset) == 0
        assert dependencies_of(asset) == frozenset()


def test_level_1_assets_depend_on_level_0():
    """ETHUSD and SOLUSD need BTCUSD; XAUUSD needs XAGUSD."""
    assert level_of("ETHUSD") == 1
    assert dependencies_of("ETHUSD") == frozenset({"BTCUSD"})
    assert level_of("SOLUSD") == 1
    assert dependencies_of("SOLUSD") == frozenset({"BTCUSD"})
    assert level_of("XAUUSD") == 1
    assert dependencies_of("XAUUSD") == frozenset({"XAGUSD"})


def test_level_of_raises_on_unknown_asset():
    with pytest.raises(ValueError, match="unknown asset"):
        level_of("DOGEUSD")


def test_topological_order_puts_dependencies_first():
    """topological_order([ETHUSD, BTCUSD]) must yield BTCUSD before ETHUSD —
    Nivel 0 always precedes Nivel 1 in the output ordering, even if the
    input order is reversed."""
    ordered = topological_order(["ETHUSD", "BTCUSD"])
    assert ordered.index("BTCUSD") < ordered.index("ETHUSD")

    ordered = topological_order(["XAUUSD", "XAGUSD"])
    assert ordered.index("XAGUSD") < ordered.index("XAUUSD")


def test_topological_order_is_stable_within_level():
    """Within a level, ordering preserves the input order — so callers
    can pin a deterministic execution order for reproducibility."""
    ordered = topological_order(["GBPUSD", "EURUSD", "USDJPY"])
    # All Level 0 → order preserved.
    assert ordered == ["GBPUSD", "EURUSD", "USDJPY"]


# ---------------------------------------------------------------------------
# load_multi_asset: I/O
# ---------------------------------------------------------------------------

def test_load_multi_asset_reads_csvs_and_returns_dict(tmp_path):
    """Each asset's CSV (with `time` column) is loaded as a DataFrame
    indexed by the time column. The dict key is the canonical asset name."""
    # Write two CSVs in the convention scripts/mt5_pull_multi_h4.py uses.
    for asset in ("EURUSD", "BTCUSD"):
        df = _h4_ohlcv(n=200, seed=1 if asset == "EURUSD" else 2).reset_index()
        df = df.rename(columns={"index": "time"})
        df.to_csv(tmp_path / f"{asset}_H4.csv", index=False)

    loaded = load_multi_asset(["EURUSD", "BTCUSD"], data_dir=tmp_path)
    assert set(loaded.keys()) == {"EURUSD", "BTCUSD"}
    for asset, df in loaded.items():
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None, f"{asset}: index must be tz-aware"
        assert {"open", "high", "low", "close", "volume"} <= set(df.columns)


def test_load_multi_asset_raises_when_csv_missing(tmp_path):
    """A missing file is a hard error — silently dropping the asset
    would corrupt downstream analysis."""
    with pytest.raises(FileNotFoundError):
        load_multi_asset(["MISSING_ASSET"], data_dir=tmp_path)


# ---------------------------------------------------------------------------
# align_to_master_index: forward-fill alignment
# ---------------------------------------------------------------------------

def test_align_forward_fills_when_target_index_finer_than_source():
    """Target has bars at finer cadence than source → source values are
    held forward until the next source bar."""
    source_idx = pd.date_range("2024-01-01 00:00", periods=5, freq="4h", tz="UTC")
    source = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=source_idx)

    # Target index with bars every hour.
    target = pd.date_range("2024-01-01 00:00", periods=20, freq="h", tz="UTC")
    aligned = align_to_master_index(source, target)

    assert aligned.index.equals(target)
    # At 00:00 → 1.0 (exact match), 01:00..03:00 → still 1.0 (forward-fill),
    # 04:00 → 2.0 (next source bar), etc.
    assert aligned["x"].iloc[0] == 1.0
    assert aligned["x"].iloc[3] == 1.0
    assert aligned["x"].iloc[4] == 2.0


def test_align_returns_nan_before_first_source_bar():
    """Target bars before the source's first timestamp should be NaN —
    forward-fill can't fabricate past data."""
    source_idx = pd.date_range("2024-01-01 12:00", periods=3, freq="4h", tz="UTC")
    source = pd.DataFrame({"x": [10.0, 20.0, 30.0]}, index=source_idx)

    target = pd.date_range("2024-01-01 00:00", periods=8, freq="4h", tz="UTC")
    aligned = align_to_master_index(source, target)

    # First 3 target bars are before source[0] → NaN.
    assert aligned["x"].iloc[:3].isna().all()
    # From the source[0] timestamp onwards values are present.
    assert aligned["x"].iloc[3] == 10.0


# ---------------------------------------------------------------------------
# compute_btc_features: shift(1) → no leak
# ---------------------------------------------------------------------------

def test_btc_features_have_expected_columns():
    """Returns the documented columns: btc_h4_return and btc_h4_rv24bars."""
    btc = _h4_ohlcv(n=500, seed=42)
    eth_idx = btc.index  # same cadence for this test
    feats = compute_btc_features(btc, eth_idx)
    assert {"btc_h4_return", "btc_h4_rv24bars"} == set(feats.columns)
    assert feats.index.equals(eth_idx)


def test_btc_features_at_target_t_use_btc_at_t_minus_one_bar():
    """The CRITICAL no-leak invariant.

    Set up: btc_close has a unique fingerprint value at bar k. ETH target
    index covers the same bars. After compute_btc_features, the
    `btc_h4_return` at the ETH timestamp == BTC's timestamp k must come
    from BTC[k-1]'s log return, NOT BTC[k]'s.

    Verification: btc_h4_return at ETH[k] equals log(BTC.close[k-1] /
    BTC.close[k-2]) — strictly past.
    """
    btc = _h4_ohlcv(n=100, seed=7)
    eth_idx = btc.index  # same timestamps as BTC
    feats = compute_btc_features(btc, eth_idx)

    # Reference: BTC's own log returns shifted by 1 bar.
    expected_at_k = (
        np.log(btc["close"].iloc[10] / btc["close"].iloc[9])  # this is BTC's return at k-1=10
    )
    # `btc_h4_return` at ETH[k=11] must equal the BTC return AT k=10
    # (which is log(close[10]/close[9])).
    actual = feats["btc_h4_return"].iloc[11]
    assert actual == pytest.approx(expected_at_k, abs=1e-12), (
        f"btc_h4_return at k=11 should be BTC return at k=10 "
        f"(log(close[10]/close[9])={expected_at_k:.6f}), got {actual:.6f}. "
        f"If they differ, the shift(1) is missing — LOOK-AHEAD LEAK."
    )


def test_btc_features_at_etb_first_bar_are_nan():
    """At the ETH index's first bar, the t-1 BTC return doesn't exist
    → NaN (the model masks it downstream)."""
    btc = _h4_ohlcv(n=100, seed=8)
    eth_idx = btc.index
    feats = compute_btc_features(btc, eth_idx)
    # The very first bar has no prior BTC data.
    assert pd.isna(feats["btc_h4_return"].iloc[0])


# ---------------------------------------------------------------------------
# compute_xau_xag_ratio: shift(1) for the metals pair
# ---------------------------------------------------------------------------

def test_xau_xag_ratio_at_t_uses_t_minus_one_close_ratio():
    """xau_xag_ratio[t] = XAU.close[t-1] / XAG.close[t-1]."""
    xau = _h4_ohlcv(n=100, seed=11, base_price=2000.0)
    xag = _h4_ohlcv(n=100, seed=12, base_price=25.0)
    target_idx = xau.index

    ratio = compute_xau_xag_ratio(xau, xag, target_idx)
    expected_k20 = float(xau["close"].iloc[19] / xag["close"].iloc[19])
    actual = ratio.iloc[20]
    assert actual == pytest.approx(expected_k20, abs=1e-12), (
        f"ratio[20] should be XAU[19]/XAG[19] = {expected_k20:.4f}, got {actual:.4f}. "
        f"If different, the shift(1) is missing — LOOK-AHEAD LEAK."
    )


def test_xau_xag_ratio_first_bar_is_nan():
    """First bar of target index has no t-1 close → NaN."""
    xau = _h4_ohlcv(n=100, seed=13, base_price=2000.0)
    xag = _h4_ohlcv(n=100, seed=14, base_price=25.0)
    target_idx = xau.index
    ratio = compute_xau_xag_ratio(xau, xag, target_idx)
    assert pd.isna(ratio.iloc[0])
