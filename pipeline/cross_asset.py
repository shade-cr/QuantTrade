"""Cross-asset alignment + dependency graph (Phase 2 T3).

Dependency graph (from Plan v2.3.1):

  Nivel 0 (self-contained features):
    BTCUSD, XAGUSD, EURUSD, GBPUSD, USDJPY (+ macro DXY/VIX from T2)

  Nivel 1 (depends on Nivel 0):
    ETHUSD ← BTCUSD       (btc_h4_return, btc_h4_rv24bars)
    SOLUSD ← BTCUSD       (same)
    XAUUSD ← XAGUSD       (xau_xag_ratio)

Look-ahead invariant: every cross-asset value used as a feature at bar
t comes from the source asset's value at t-1 (one full H4 bar lag).
Implemented as `.shift(1)` before forward-fill alignment so the model
sees only past information.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


LEVEL_0_ASSETS = frozenset({"BTCUSD", "XAGUSD", "EURUSD", "GBPUSD", "USDJPY"})
LEVEL_1_ASSETS = frozenset({"ETHUSD", "SOLUSD", "XAUUSD"})
ALL_ASSETS = LEVEL_0_ASSETS | LEVEL_1_ASSETS

# Per-asset dependency map (which Nivel 0 assets does this asset need?).
DEPENDENCIES: dict[str, frozenset[str]] = {
    "ETHUSD": frozenset({"BTCUSD"}),
    "SOLUSD": frozenset({"BTCUSD"}),
    "XAUUSD": frozenset({"XAGUSD"}),
}


def load_multi_asset(
    assets: list[str],
    data_dir: Path | str = Path("data/H4"),
    timeframe: str = "H4",
) -> dict[str, pd.DataFrame]:
    """Load each asset's CSV from `data_dir/{ASSET}_{timeframe}.csv`.

    Convention (matches scripts/mt5_pull_multi_h4.py output):
      - First column is 'time' — parsed to a tz-aware UTC DatetimeIndex
      - Remaining columns: open, high, low, close, volume

    Missing files raise FileNotFoundError (no silent drops — a missing
    asset corrupts cross-asset alignment downstream).
    """
    data_dir = Path(data_dir)
    out: dict[str, pd.DataFrame] = {}
    for asset in assets:
        csv_path = data_dir / f"{asset}_{timeframe}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"missing CSV for {asset}: {csv_path}. "
                f"Run scripts/mt5_pull_multi_h4.py to populate {data_dir}."
            )
        df = pd.read_csv(csv_path, parse_dates=["time"])
        df = df.set_index("time").sort_index()
        # The MT5 puller already writes UTC tz-aware timestamps, but
        # round-tripping through CSV can drop the tz — re-attach.
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        out[asset] = df
    return out


def align_to_master_index(
    source_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Reindex `source_df` to `target_index` with forward-fill.

    Target bars before `source_df.index[0]` get NaN — forward-fill cannot
    fabricate past data and we don't want to silently impute zeros.
    """
    return source_df.reindex(target_index, method="ffill")


def compute_btc_features(
    btc_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute (btc_h4_return, btc_h4_rv24bars) and align to `target_index`.

    Both columns are .shift(1) so a feature value at bar t comes from
    BTC's value at t-1 (strictly past). Crucial: without shift(1), an
    ETH model could see BTC's return at t when predicting ETH's outcome
    AT t — leak.
    """
    close = btc_df["close"]
    log_ret = np.log(close / close.shift(1))
    # Realised vol over 24 H4 bars, annualised by sqrt(252) for
    # consistency with the D1 / H4-base convention in features.py.
    rv24 = log_ret.rolling(24).std() * np.sqrt(252)

    raw = pd.DataFrame(
        {
            "btc_h4_return": log_ret.shift(1),
            "btc_h4_rv24bars": rv24.shift(1),
        },
        index=btc_df.index,
    )
    return align_to_master_index(raw, target_index)


def compute_xau_xag_ratio(
    xau_df: pd.DataFrame,
    xag_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Compute XAU/XAG close ratio with shift(1) and align to `target_index`.

    Both legs must have the same index in `target_index` for the ratio to
    make sense; we re-align each to target_index via forward-fill from
    their own histories and then divide.
    """
    xau_aligned = align_to_master_index(xau_df[["close"]].rename(columns={"close": "xau"}), target_index)
    xag_aligned = align_to_master_index(xag_df[["close"]].rename(columns={"close": "xag"}), target_index)
    ratio = xau_aligned["xau"] / xag_aligned["xag"]
    return ratio.shift(1).rename("xau_xag_ratio")


def level_of(asset: str) -> int:
    """Return the dependency level: 0 = self-contained, 1 = needs Nivel 0."""
    if asset in LEVEL_0_ASSETS:
        return 0
    if asset in LEVEL_1_ASSETS:
        return 1
    raise ValueError(f"unknown asset {asset!r}; expected one of {sorted(ALL_ASSETS)}")


def dependencies_of(asset: str) -> frozenset[str]:
    """Return the set of assets this one needs computed first."""
    if asset not in ALL_ASSETS:
        raise ValueError(f"unknown asset {asset!r}; expected one of {sorted(ALL_ASSETS)}")
    return DEPENDENCIES.get(asset, frozenset())


def topological_order(assets: list[str]) -> list[str]:
    """Sort `assets` so each asset's dependencies appear before it.

    Within a single level, input order is preserved (stable sort) — this
    gives callers a deterministic execution order for reproducibility.
    """
    return sorted(assets, key=lambda a: (level_of(a), assets.index(a)))
