"""Fetch FRED macro series with parquet cache and 1-day publication lag."""
from __future__ import annotations
import os
import time
from pathlib import Path
import pandas as pd


class MacroFetchError(RuntimeError):
    pass


FRED_SERIES = ("DTWEXBGS", "DFII5", "DGS5", "DGS2", "T5YIE", "VIXCLS", "UMCSENT")

# B0003 — benchmark-only series for the hidden-beta correlation diagnostic.
# These are NOT model features (build_macro_features never references them), so
# they are deliberately kept OUT of FRED_SERIES: a feature-pipeline run must not
# hard-fail just because a benchmark series is uncached / no FRED_API_KEY. The
# benchmark code in scripts/run_backtest.py loads these from cache/fred/ best-
# effort and reports None when absent. SP500 (S&P 500 index) joins VIXCLS
# (already a feature) as the second benchmark; cache it separately to populate
# the S&P 500 correlation.
BENCHMARK_SERIES = ("SP500",)

# UMCSENT (University of Michigan Consumer Sentiment) is the only monthly
# series in the bundle — its 3-month change is precomputed on the daily-
# aligned frame so D1 and H4 downstream consumers see the same calendar-
# anchored delta (a bar-count .diff in features.py would mean ~3 months on
# D1 but ~10 days on H4).
UMCSENT_CHG_WINDOW_DAYS = 63  # ≈ 3 trading months


def _make_fred_client():  # split for monkeypatching in tests
    try:
        from fredapi import Fred
    except ImportError as e:
        raise MacroFetchError("fredapi not installed; pip install fredapi") from e
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        raise MacroFetchError(
            "FRED_API_KEY env var not set. Get a key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and put it in .env"
        )
    return Fred(api_key=api_key)


def fetch_series(
    code: str,
    start: str,
    end: str,
    cache_dir: Path,
    max_retries: int = 3,
) -> pd.Series:
    """Fetch a single FRED series, with parquet cache. Returns tz-naive daily series."""
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{code}.parquet"
    if cache_path.exists():
        # Trust the cache fully: pipeline runs are reproducible against
        # whatever the user has cached. Coverage/freshness is managed
        # explicitly via scripts/ingest_* or by deleting the parquet. Any
        # gap on the boundaries is absorbed downstream by reindex/ffill.
        cached = pd.read_parquet(cache_path)[code]
        return cached.loc[start:end]

    fred = _make_fred_client()
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            s = fred.get_series(code, observation_start=start, observation_end=end)
            s.name = code
            s = s.dropna()
            cache_dir.mkdir(parents=True, exist_ok=True)
            s.to_frame().to_parquet(cache_path)
            return s
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep(2 ** attempt)
    raise MacroFetchError(f"Failed to fetch {code} after {max_retries} retries: {last_exc}")


def build_macro_frame(start: str, end: str, cache_dir: Path | str) -> pd.DataFrame:
    """Fetch all FRED series, forward-fill to a calendar daily index, then SHIFT(1).

    The shift models the publication lag: a series stamped at date t in FRED is in
    practice published the next day. Features at time t may only use info stamped <= t-1.
    """
    cache_dir = Path(cache_dir)
    series = {code: fetch_series(code, start, end, cache_dir) for code in FRED_SERIES}
    # Align on a daily calendar (UTC midnight).
    cal = pd.date_range(start, end, freq="D", tz="UTC")
    frame = pd.DataFrame(index=cal)
    for code, s in series.items():
        s_utc = s.copy()
        s_utc.index = pd.to_datetime(s_utc.index, utc=True)
        frame[code] = s_utc.reindex(cal, method="ffill")
    # UMCSENT is monthly; precompute its 3-month change on the daily frame so
    # the diff has a fixed calendar meaning regardless of downstream bar size.
    if "UMCSENT" in frame.columns:
        frame["UMCSENT_chg_3m"] = frame["UMCSENT"].diff(UMCSENT_CHG_WINDOW_DAYS)
    # CRITICAL: 1-day publication lag.
    frame = frame.shift(1)
    return frame
