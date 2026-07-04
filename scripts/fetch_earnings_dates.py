"""B0017 — fetch 8-K Item 2.02 (earnings announcement) history from SEC EDGAR.

Writes data/earnings/<TICKER>_8k202.parquet with the UTC acceptance timestamp
as index (the PIT knowledge moment) and filing_date/items columns. Uses the
official submissions API (data.sec.gov) including the paginated "older
files", so history reaches back to the start of electronic 8-K item tagging
(mid-2003 — complete for our 2006+ window).

SEC fair-access rules: descriptive User-Agent, <=10 req/s (we sleep 0.15s).

USAGE:
  uv run python scripts/fetch_earnings_dates.py --ticker ABT
  uv run python scripts/fetch_earnings_dates.py --universe configs/universe_equity_m3w.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from pipeline.earnings_events import EARNINGS_CACHE_DIR
from pipeline.equity_universe import load_universe

UA = {"User-Agent": "QuantTrade research lc@virtualretail.io"}
SLEEP_S = 0.15


def _get_json(url: str) -> dict:
    time.sleep(SLEEP_S)
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()


def ticker_to_cik(ticker: str, mapping: dict | None = None) -> int:
    m = mapping or _get_json("https://www.sec.gov/files/company_tickers.json")
    for v in m.values():
        if v["ticker"].upper() == ticker.upper():
            return int(v["cik_str"])
    raise KeyError(f"ticker {ticker} not found in SEC company_tickers.json")


def _extract_8k202(filings_block: dict) -> list[dict]:
    """Pull (acceptance, filing_date, items) rows for 8-Ks with Item 2.02
    from one submissions JSON block (recent or an older page)."""
    forms = filings_block["form"]
    items = filings_block["items"]
    acc = filings_block["acceptanceDateTime"]
    fdate = filings_block["filingDate"]
    rows = []
    for i in range(len(forms)):
        if forms[i] == "8-K" and items[i] and "2.02" in items[i]:
            rows.append({"acceptance": acc[i], "filing_date": fdate[i], "items": items[i]})
    return rows


def fetch_ticker(ticker: str, cik: int, out_dir: Path) -> pd.DataFrame:
    cik10 = f"{cik:010d}"
    sub = _get_json(f"https://data.sec.gov/submissions/CIK{cik10}.json")
    rows = _extract_8k202(sub["filings"]["recent"])
    for older in sub["filings"].get("files", []):
        page = _get_json(f"https://data.sec.gov/submissions/{older['name']}")
        rows.extend(_extract_8k202(page))
    if not rows:
        raise RuntimeError(f"{ticker}: no 8-K Item 2.02 filings found — refusing to write empty cache")
    df = pd.DataFrame(rows)
    df["acceptance"] = pd.to_datetime(df["acceptance"], utc=True)
    df["filing_date"] = pd.to_datetime(df["filing_date"]).dt.date
    df = (df.set_index("acceptance").sort_index())
    df = df[~df.index.duplicated(keep="first")]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{ticker}_8k202.parquet"
    df.to_parquet(out)
    print(f"Wrote {out}: {len(df)} announcements, {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker")
    ap.add_argument("--universe", help="universe yaml; fetches every stock (not ETFs — funds don't file 8-K earnings)")
    ap.add_argument("--out-dir", default=str(EARNINGS_CACHE_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    if not args.ticker and not args.universe:
        ap.error("need --ticker or --universe")

    mapping = _get_json("https://www.sec.gov/files/company_tickers.json")
    tickers = [args.ticker] if args.ticker else list(load_universe(args.universe)["stocks"])
    failures = []
    for t in tickers:
        try:
            fetch_ticker(t, ticker_to_cik(t, mapping), out_dir)
        except Exception as e:  # noqa: BLE001 — batch fetch, report at end
            failures.append((t, str(e)))
            print(f"FAILED {t}: {e}")
    if failures:
        print(f"\n{len(failures)} failures: {[t for t, _ in failures]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
