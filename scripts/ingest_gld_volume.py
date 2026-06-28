"""Ingest GLD ETF daily OHLCV (REAL exchange volume) into a parquet cache (B0147).

Why GLD volume: XAU spot CFD has NO real traded volume (MT5 volume = tick
count). GLD is exchange-traded with genuine share volume, published daily,
available at decision time — so it transfers to live (unlike a bar reclock,
see B0146 venue-mismatch). Expert verdict on B0147: use as a VOLATILITY /
REGIME feature, not directional alpha; cheap probe first.

Data source: Yahoo Finance daily bars via yfinance (already a project
dependency), ticker GLD, period=max (full history since 2004-11-18),
auto_adjust=False so Volume stays raw share count. (Stooq's keyless CSV
endpoint 404s as of 2026-06-11 — yfinance is the fallback chosen.)

PIT discipline: GLD's official daily volume is final after the US close. We
treat the value stamped at calendar date t as visible from t+1 onward — the
calendar-day shift is applied at LOAD time in
pipeline/alt_data/gld_volume.py::load_gld_volume_features, matching the
gld_holdings.py / macro_fetch.py convention.

Cache layout: cache/alt_data/gld_volume.parquet
  - Index: DatetimeIndex UTC tz-aware, strictly monotonic, deduped.
  - Columns: gld_close (float USD), gld_volume (float shares).

Idempotent: atomic write via temp + replace. Network failure leaves the
existing cache intact.

USAGE:
  uv run python scripts/ingest_gld_volume.py
  uv run python scripts/ingest_gld_volume.py --out cache/alt_data/gld_volume.parquet
"""
from __future__ import annotations
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

DEFAULT_CACHE_PATH = Path("cache/alt_data/gld_volume.parquet")


def fetch_yahoo_daily(ticker: str = "GLD") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, period="max", interval="1d",
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data for GLD — retry later.")
    # yfinance may return a (field, ticker) MultiIndex even for one ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns or "Volume" not in df.columns:
        raise RuntimeError(f"Unexpected yfinance columns: {list(df.columns)}")
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "gld_close": pd.to_numeric(df["Close"], errors="coerce"),
        "gld_volume": pd.to_numeric(df["Volume"], errors="coerce"),
    })
    idx = pd.to_datetime(df.index)
    out.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out = out.dropna().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out[(out["gld_close"] > 0) & (out["gld_volume"] > 0)]
    if len(out) < 1000:
        raise RuntimeError(f"Suspiciously short GLD history ({len(out)} rows); refusing to cache.")
    return out


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=path.parent)
    os.close(fd)
    try:
        df.to_parquet(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_CACHE_PATH))
    args = ap.parse_args()
    df = normalize(fetch_yahoo_daily())
    atomic_write_parquet(df, Path(args.out))
    print(f"Wrote {args.out}: {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
