"""Fetch split/dividend-adjusted daily equity bars into the load_dataset contract.

USAGE:
  uv run python scripts/fetch_equity_daily.py --ticker NVDA
  uv run python scripts/fetch_equity_daily.py --ticker NVDA --out data/D1/NVDA_D1.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.equity_source import YFinanceSource, write_contract_csv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None,
                    help="default data/D1/<TICKER>_D1.csv")
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(f"data/D1/{args.ticker}_D1.csv")
    df = YFinanceSource().fetch_daily(args.ticker, args.start, args.end)
    if len(df) < 1000:
        raise RuntimeError(f"Suspiciously short history for {args.ticker} "
                           f"({len(df)} rows); refusing to write.")
    write_contract_csv(df, out)
    print(f"Wrote {out}: {len(df)} rows, "
          f"{df.index.min().date()} -> {df.index.max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
