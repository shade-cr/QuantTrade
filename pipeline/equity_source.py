"""Equity daily-bar data layer for QuantTrade.

EquityDataSource is the swap seam: YFinanceSource is the free first impl;
a paid vendor (Norgate/Alpaca/Sharadar) drops in behind the same method.
Equity bars MUST be split/dividend-adjusted (auto_adjust=True) — unadjusted
prices corrupt every triple-barrier label.
"""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Protocol

import pandas as pd

_RENAME = {"Open": "open", "High": "high", "Low": "low",
           "Close": "close", "Volume": "volume"}


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    missing = [k for k in _RENAME if k not in df.columns]
    if missing:
        raise ValueError(f"yfinance frame missing columns: {missing}")
    out = (df.rename(columns=_RENAME)[["open", "high", "low", "close", "volume"]]
             .apply(pd.to_numeric, errors="coerce").astype("float64"))
    idx = pd.to_datetime(df.index)
    out.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out = out.dropna().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return out


def write_contract_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "time"
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=path.parent)
    os.close(fd)
    try:
        out.reset_index().to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class EquityDataSource(Protocol):
    def fetch_daily(self, ticker: str, start: str | None = None,
                    end: str | None = None) -> pd.DataFrame: ...


class YFinanceSource:
    def fetch_daily(self, ticker: str, start: str | None = None,
                    end: str | None = None) -> pd.DataFrame:
        import yfinance as yf
        kwargs = dict(interval="1d", auto_adjust=True, progress=False)
        if start:
            df = yf.download(ticker, start=start, end=end, **kwargs)
        else:
            df = yf.download(ticker, period="max", **kwargs)
        if df is None or df.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker} — retry later.")
        return normalize_ohlcv(df)
