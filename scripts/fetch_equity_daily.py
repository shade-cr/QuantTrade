"""Fetch split/dividend-adjusted daily equity bars into the load_dataset contract.

USAGE:
  uv run python scripts/fetch_equity_daily.py --ticker NVDA
  uv run python scripts/fetch_equity_daily.py --ticker NVDA --out data/D1/NVDA_D1.csv
  uv run python scripts/fetch_equity_daily.py --universe configs/universe_equity_m3.yaml
"""
from __future__ import annotations
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.equity_source import YFinanceSource, write_contract_csv
from pipeline.equity_universe import load_universe

# Universe mode gate: every member must cover the M3 backtest window from its
# start, or the pooled panel is unbalanced across folds (B0003).
_UNIVERSE_HISTORY_FLOOR = date(2006, 1, 3)


def _fetch_one(src: YFinanceSource, ticker: str, start, end, out: Path,
               enforce_floor: bool) -> None:
    df = src.fetch_daily(ticker, start, end)
    if len(df) < 1000:
        raise RuntimeError(f"suspiciously short history ({len(df)} rows)")
    first = df.index.min().date()
    if enforce_floor and first > _UNIVERSE_HISTORY_FLOOR:
        raise RuntimeError(f"history starts {first} > {_UNIVERSE_HISTORY_FLOOR}")
    write_contract_csv(df, out)
    print(f"Wrote {out}: {len(df)} rows, "
          f"{df.index.min().date()} -> {df.index.max().date()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--universe", default=None,
                    help="universe yaml; fetches every stock+etf into data/D1/")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None,
                    help="single-ticker mode only; default data/D1/<TICKER>_D1.csv")
    args = ap.parse_args()
    if bool(args.ticker) == bool(args.universe):
        ap.error("exactly one of --ticker / --universe is required")

    src = YFinanceSource()
    if args.ticker:
        out = Path(args.out) if args.out else Path(f"data/D1/{args.ticker}_D1.csv")
        _fetch_one(src, args.ticker, args.start, args.end, out, enforce_floor=False)
        return 0

    u = load_universe(args.universe)
    failed: list[tuple[str, str]] = []
    for ticker in list(u["stocks"]) + list(u["etfs"]):
        try:
            _fetch_one(src, ticker, args.start, args.end,
                       Path(f"data/D1/{ticker}_D1.csv"), enforce_floor=True)
        except Exception as exc:  # noqa: BLE001 — keep batch going, report at end
            failed.append((ticker, f"{type(exc).__name__}: {exc}"))
    if failed:
        print("FAILED: " + "; ".join(f"{t} ({r})" for t, r in failed))
        print("Apply the alternates rule from the universe file manually and "
              "record the substitution in B0003 history.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
