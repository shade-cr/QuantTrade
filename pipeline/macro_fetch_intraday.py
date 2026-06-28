"""Intraday macro fetcher for H4 (DXY proxy + VIX) via yfinance (Phase 2 T2).

Tickers:
  - DXY proxy: `DX-Y.NYB` (US Dollar Index, NYSE Cotton Exchange).
    `^DXY` is NOT available on yfinance — DX-Y.NYB is the standard proxy.
  - VIX: `^VIX` (CBOE Volatility Index).

Yahoo's H4 intraday data is only available for the last ~730 days. For
older history we fall back to daily FRED via `pipeline.macro_fetch`,
forward-filled to H4 timestamps. The `dxy_stale` / `vix_stale` flags
mark bars where the most recent macro value is > 24h old — the
downstream model learns to discount stale macro signal.

Test seam: `_make_yf_ticker(symbol)` is monkeypatched in tests to inject
a fake Ticker. Production code calls the real `yfinance.Ticker`.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


class IntradayFetchError(Exception):
    """Raised when yfinance fetch fails after all retries."""


TICKER_DXY = "DX-Y.NYB"
TICKER_VIX = "^VIX"
STALE_THRESHOLD_HOURS = 24.0


def _make_yf_ticker(symbol: str):
    """Test seam — monkeypatch to inject fake Tickers."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise IntradayFetchError("yfinance not installed") from e
    return yf.Ticker(symbol)


def _cache_path(cache_dir: Path, ticker: str, interval: str) -> Path:
    """Sanitise ticker for filename (^/= are not portable). Return cache file path."""
    safe = ticker.replace("^", "_").replace("=", "_").replace("/", "_")
    return cache_dir / f"{safe}_{interval}.parquet"


def fetch_intraday_series(
    ticker: str,
    start: str,
    end: str,
    interval: str = "4h",
    cache_dir: Path | str = Path("cache/yahoo"),
) -> pd.Series:
    """Fetch the Close-price series for `ticker` over [start, end] at `interval`.

    Caches the full fetched range as parquet keyed by (ticker, interval).
    A subsequent call within the cached range returns from cache without
    touching yfinance.

    Raises IntradayFetchError if yfinance returns empty.
    """
    cache_dir = Path(cache_dir)
    cache_path = _cache_path(cache_dir, ticker, interval)
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)["close"]
        # The cache must span the requested range to be a hit.
        if cached.index.min() <= start_ts and cached.index.max() >= end_ts:
            return cached.loc[start_ts:end_ts]

    t = _make_yf_ticker(ticker)
    df = t.history(start=start, end=end, interval=interval, auto_adjust=False)
    if df is None or df.empty:
        raise IntradayFetchError(
            f"empty result for {ticker} [{start}, {end}, {interval}]"
        )
    close = df["Close"].rename("close")
    # Normalise index to UTC tz-aware.
    if close.index.tz is None:
        close.index = close.index.tz_localize("UTC")
    else:
        close.index = close.index.tz_convert("UTC")

    cache_dir.mkdir(parents=True, exist_ok=True)
    close.to_frame().to_parquet(cache_path)
    return close.loc[start_ts:end_ts]


def get_macro_value_at_bar(
    macro_series: pd.Series,
    bar_time_utc: pd.Timestamp,
) -> tuple[float, float]:
    """Return (last_known_value, hours_since_update) at `bar_time_utc`.

    Uses `searchsorted(side='left') - 1` to enforce STRICT less-than:
    a stamp at exactly `bar_time_utc` is treated as "published at t" and
    NOT used (would be lookahead — the macro publishes AT t, so a model
    reading the bar AT t doesn't yet have it). The most recent stamp
    strictly before `bar_time_utc` is the right answer.

    If no prior stamp exists, returns (NaN, +inf).
    """
    bar = pd.Timestamp(bar_time_utc)
    if bar.tz is None:
        bar = bar.tz_localize("UTC")
    # side='left' finds insertion index that keeps order; -1 gets the
    # last index strictly < bar. (Contrast with side='right' which would
    # include stamps == bar.)
    idx = macro_series.index.searchsorted(bar, side="left") - 1
    if idx < 0:
        return float("nan"), float("inf")
    value = float(macro_series.iloc[idx])
    hours_ago = (bar - macro_series.index[idx]).total_seconds() / 3600.0
    return value, hours_ago


def build_intraday_macro_frame_with_daily_fallback(
    bar_index: pd.DatetimeIndex,
    start: str,
    end: str,
    daily_macro_frame: pd.DataFrame,
    cache_dir: Path | str = Path("cache/yahoo"),
    stale_threshold_hours: float = STALE_THRESHOLD_HOURS,
) -> pd.DataFrame:
    """Like `build_intraday_macro_frame` but falls back to daily FRED for bars
    where yfinance returns no data (typically older than the ~730-day window
    Yahoo allows for H4 intraday).

    `daily_macro_frame` must be the output of `pipeline.macro_fetch.build_macro_frame`
    — it has DTWEXBGS (broad DXY) and VIXCLS columns at daily cadence and is
    already shifted by 1 day to model the publication lag.

    For each bar where the intraday fetch produced NaN, the function uses
    `get_macro_value_at_bar(daily_series, bar_time)` to substitute the latest
    daily value strictly before the bar. `dxy_stale` / `vix_stale` are set
    to 1.0 on every fallback row (daily macro at H4 cadence is by definition
    "older than the 24h freshness window").
    """
    # First try intraday — if Yahoo rejects the range, both columns will be NaN.
    intraday = build_intraday_macro_frame(
        bar_index=bar_index,
        start=start,
        end=end,
        cache_dir=cache_dir,
        stale_threshold_hours=stale_threshold_hours,
    )

    # Fallback per ticker. We look at the column directly: if a bar has NaN
    # in the intraday output, query daily FRED for that bar.
    dxy_daily = daily_macro_frame.get("DTWEXBGS")
    vix_daily = daily_macro_frame.get("VIXCLS")

    def _fill_from_daily(col: str, daily_series: pd.Series | None) -> None:
        if daily_series is None:
            return
        nan_mask = intraday[col].isna()
        if not nan_mask.any():
            return
        for bar_time in intraday.index[nan_mask]:
            v, _h = get_macro_value_at_bar(daily_series, bar_time)
            intraday.loc[bar_time, col] = v
            intraday.loc[bar_time, f"{col}_stale"] = 1.0  # daily-sourced → always stale at H4

    _fill_from_daily("dxy", dxy_daily)
    _fill_from_daily("vix", vix_daily)
    return intraday


def build_intraday_macro_frame(
    bar_index: pd.DatetimeIndex,
    start: str,
    end: str,
    cache_dir: Path | str = Path("cache/yahoo"),
    stale_threshold_hours: float = STALE_THRESHOLD_HOURS,
) -> pd.DataFrame:
    """Build a DataFrame with [dxy, dxy_stale, vix, vix_stale] aligned to bar_index.

    For each ticker:
      - Fetch H4 series from yfinance (cached)
      - For each bar in bar_index, find the last macro value strictly before
        the bar time and the hours since that update
      - Flag stale (1.0) when hours_since_update > stale_threshold_hours

    On IntradayFetchError, fills the ticker's value with NaN and stale=1.0
    for every bar. Caller can then choose to abort or proceed with the
    stale flag in the feature.
    """
    out = pd.DataFrame(index=bar_index)
    for col_name, ticker in (("dxy", TICKER_DXY), ("vix", TICKER_VIX)):
        try:
            series = fetch_intraday_series(
                ticker, start, end, interval="4h", cache_dir=cache_dir,
            )
        except IntradayFetchError:
            out[col_name] = np.nan
            out[f"{col_name}_stale"] = 1.0
            continue

        values = np.empty(len(bar_index), dtype=float)
        stale = np.empty(len(bar_index), dtype=float)
        for i, bar_time in enumerate(bar_index):
            v, h = get_macro_value_at_bar(series, bar_time)
            values[i] = v
            stale[i] = 1.0 if h > stale_threshold_hours else 0.0
        out[col_name] = values
        out[f"{col_name}_stale"] = stale

    return out
