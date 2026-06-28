"""Perpetual-swap funding-rate features for crypto (Tier B2).

Pulls Binance USDT-perpetual funding-rate history (public REST, no auth)
and builds H4-aligned features for BTC/ETH/SOL meta-labelers.

Source: https://fapi.binance.com/fapi/v1/fundingRate
  - Payload: [{symbol, fundingTime, fundingRate, markPrice}, ...]
  - Funding settles every 8h (00:00, 08:00, 16:00 UTC).
  - 1000-row limit per call -> caller paginates by advancing startTime.

Alignment invariant (CRITICAL — see CLAUDE.md "strict-less-than"):
  A funding stamp at fundingTime = 2024-03-15 08:00:00 UTC is published
  AT that instant (the settlement happens on-chain at 08:00). For an H4
  bar at time t, the visible funding is the most recent stamp with
  fundingTime < t. We use `searchsorted(side='left') - 1` — the same
  pattern as `macro_fetch_intraday.get_macro_value_at_bar`.

Features per crypto asset (BTC/ETH/SOL):
  funding_h4              — most recent applicable funding rate
  funding_h4_z252         — 252-bar rolling z-score
  funding_h4_chg_5bars    — change over last 5 H4 bars (20h)
  funding_cumulative_7d   — sum over last 42 H4 bars (7 days)
  funding_extreme_positive— 1 if z > +1.5
  funding_extreme_negative— 1 if z < -1.5

For non-crypto assets, returns an empty DataFrame.

Test seam: tests monkeypatch `_fetch_funding_page` to inject synthetic
payloads. Production calls `requests.get` against fapi.binance.com.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_PAGE_LIMIT = 1000  # Binance hard cap per call
FUNDING_INTERVAL_HOURS = 8

ASSET_TO_PERP_SYMBOL: dict[str, str] = {
    "BTCUSD": "BTCUSDT",
    "ETHUSD": "ETHUSDT",
    "SOLUSD": "SOLUSDT",
}

# H4-bar parameters for feature windows.
_H4_BARS_PER_DAY = 6
_H4_Z_WINDOW = 252            # ≈ 42 H4 trading days (matches the project's other z-windows)
_H4_CHG_LOOKBACK = 5          # 5 bars = 20h
_H4_CUMSUM_BARS = 42          # 7 days × 6 bars/day
_EXTREME_THRESHOLD = 1.5      # z-score threshold for extreme flags


class FundingFetchError(Exception):
    """Raised when the Binance funding endpoint fails / returns empty."""


def _fetch_funding_page(
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = BINANCE_PAGE_LIMIT,
) -> list[dict[str, Any]]:
    """Single Binance funding-rate page. Test seam — monkeypatched in tests.

    Returns up to `limit` rows in chronological order, each row a dict with
    keys {symbol, fundingTime, fundingRate, markPrice}.
    """
    try:
        import requests
    except ImportError as e:
        raise FundingFetchError("requests not installed") from e

    params = {
        "symbol": symbol,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    resp = requests.get(BINANCE_FUNDING_URL, params=params, timeout=30)
    if resp.status_code != 200:
        raise FundingFetchError(
            f"Binance returned HTTP {resp.status_code} for {symbol}: {resp.text[:200]}"
        )
    data = resp.json()
    if not isinstance(data, list):
        raise FundingFetchError(f"unexpected Binance payload for {symbol}: {data!r:.200}")
    return data


def _cache_path(cache_dir: Path, asset: str) -> Path:
    return cache_dir / f"funding_{asset}.parquet"


def _payload_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a Binance funding payload to a tidy DataFrame.

    Columns: funding_time (UTC tz-aware), funding_rate (float), mark_price (float).
    Sorted ascending by funding_time, duplicates removed.
    """
    if not rows:
        return pd.DataFrame(columns=["funding_time", "funding_rate", "mark_price"])
    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    # Binance sometimes returns empty-string for markPrice on older entries
    # (and rarely for fundingRate too). Use to_numeric with coerce so those
    # rows become NaN and get dropped below instead of crashing the pipeline.
    df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["mark_price"] = pd.to_numeric(df.get("markPrice", pd.Series(dtype=object)), errors="coerce")
    df = df[["funding_time", "funding_rate", "mark_price"]]
    df = df.dropna(subset=["funding_rate"])
    df = df.sort_values("funding_time").drop_duplicates(subset="funding_time").reset_index(drop=True)
    return df


def fetch_funding_for_asset(
    asset: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str | Path = "cache/funding",
) -> pd.DataFrame:
    """Fetch Binance USDT-perp funding history for `asset` over [start, end].

    Returns a DataFrame with columns: funding_time, funding_rate, mark_price
    (8h-cadence, chronologically sorted, deduplicated).

    Caches to parquet at `{cache_dir}/funding_{asset}.parquet`. A second
    call whose [start, end] is contained in the cached range is served
    from cache without hitting the network. If the cached range only
    partially covers the requested window, the cache is overwritten by
    the freshly-fetched superset (simple, conservative — funding history
    is small).

    Handles Binance's 1000-row pagination limit by advancing startTime
    after each page until end is reached or an empty page is returned.

    Raises FundingFetchError if no data is returned at all.
    """
    if asset not in ASSET_TO_PERP_SYMBOL:
        raise ValueError(
            f"unknown crypto asset {asset!r}; expected one of {sorted(ASSET_TO_PERP_SYMBOL)}"
        )
    symbol = ASSET_TO_PERP_SYMBOL[asset]
    cache_dir = Path(cache_dir)
    cache_file = _cache_path(cache_dir, asset)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tz is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tz is None:
        end_ts = end_ts.tz_localize("UTC")

    # Cache hit if the cached range covers the requested window.
    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
        if not cached.empty:
            if cached["funding_time"].min() <= start_ts and cached["funding_time"].max() >= end_ts:
                mask = (cached["funding_time"] >= start_ts) & (cached["funding_time"] <= end_ts + pd.Timedelta(hours=FUNDING_INTERVAL_HOURS))
                return cached.loc[mask].reset_index(drop=True)

    # Paginated fetch. Binance is exclusive on endTime in practice — we
    # request 1000 rows at a time and advance the cursor by the last seen
    # funding_time + 1 ms.
    start_ms = int(start_ts.value // 1_000_000)
    end_ms = int(end_ts.value // 1_000_000)

    all_rows: list[dict[str, Any]] = []
    cursor = start_ms
    while cursor < end_ms:
        page = _fetch_funding_page(symbol, cursor, end_ms, limit=BINANCE_PAGE_LIMIT)
        if not page:
            break
        all_rows.extend(page)
        # Advance cursor past the last fundingTime we saw.
        last_ms = int(page[-1]["fundingTime"])
        next_cursor = last_ms + 1
        if next_cursor <= cursor:
            # Defensive: avoid infinite loop on a degenerate payload.
            break
        cursor = next_cursor
        # If the page came back shorter than the limit, we've drained the range.
        if len(page) < BINANCE_PAGE_LIMIT:
            break

    df = _payload_to_frame(all_rows)
    if df.empty:
        raise FundingFetchError(
            f"no funding rows returned for {asset} ({symbol}) [{start_ts}, {end_ts}]"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file)
    return df


def _align_funding_to_index(
    funding_df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Strict-less-than alignment of the 8h-cadence funding series onto an
    H4 (or other) target index.

    Mimics `macro_fetch_intraday.get_macro_value_at_bar`: for each bar at
    time t, the visible funding is the most recent stamp with
    funding_time < t (NOT ≤). Implemented via `searchsorted(side='left') - 1`.
    """
    if funding_df.empty:
        return pd.Series(np.nan, index=target_index, name="funding_h4")
    ft = funding_df["funding_time"].to_numpy()
    rates = funding_df["funding_rate"].to_numpy()
    target_arr = target_index.to_numpy()
    idx = np.searchsorted(ft, target_arr, side="left") - 1
    out = np.where(idx >= 0, rates[idx.clip(min=0)], np.nan)
    return pd.Series(out, index=target_index, name="funding_h4")


def build_funding_features(
    asset: str,
    target_index: pd.DatetimeIndex,
    cache_dir: str | Path = "cache/funding",
    funding_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build crypto funding-rate features aligned to `target_index`.

    Args:
      asset:          asset code (e.g. "BTCUSD"). Non-crypto assets return
                      an empty DataFrame.
      target_index:   tz-aware UTC DatetimeIndex of the H4 bars to align to.
      cache_dir:      directory containing the funding parquet caches.
      funding_df:     optional pre-loaded funding DataFrame (columns
                      funding_time / funding_rate / mark_price). If None,
                      we read from cache; if cache is empty, we fetch.

    Returns:
      DataFrame with the 6 documented columns aligned to target_index,
      respecting strict-less-than no-leak. Empty DataFrame for non-crypto
      assets.
    """
    if asset not in ASSET_TO_PERP_SYMBOL:
        # Empty frame keeps the column shape consistent across asset classes
        # in the orchestrator (non-crypto callers must skip concat or accept
        # zero-width frame).
        return pd.DataFrame(index=target_index)

    if funding_df is None:
        cache_file = _cache_path(Path(cache_dir), asset)
        if cache_file.exists():
            funding_df = pd.read_parquet(cache_file)
        else:
            # Fetch a generous window covering the target.
            start = target_index.min() - pd.Timedelta(days=60)  # warm-up for z-score
            end = target_index.max() + pd.Timedelta(hours=FUNDING_INTERVAL_HOURS)
            funding_df = fetch_funding_for_asset(asset, start, end, cache_dir=cache_dir)

    # Strict-less-than alignment onto target_index.
    funding_h4 = _align_funding_to_index(funding_df, target_index)

    out = pd.DataFrame(index=target_index)
    out["funding_h4"] = funding_h4

    # Rolling stats on the already-lagged series. Because funding_h4 at
    # position k is built from rows strictly before target_index[k], any
    # rolling stat ending at k is itself strict-less-than safe.
    mean = funding_h4.rolling(_H4_Z_WINDOW).mean()
    std = funding_h4.rolling(_H4_Z_WINDOW).std()
    z = (funding_h4 - mean) / std
    out["funding_h4_z252"] = z

    out["funding_h4_chg_5bars"] = funding_h4.diff(_H4_CHG_LOOKBACK)
    out["funding_cumulative_7d"] = funding_h4.rolling(_H4_CUMSUM_BARS).sum()

    out["funding_extreme_positive"] = (z > _EXTREME_THRESHOLD).astype(float)
    out["funding_extreme_negative"] = (z < -_EXTREME_THRESHOLD).astype(float)

    return out
