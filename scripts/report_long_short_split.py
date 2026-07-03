"""Pooled long/short split report for M3 runs (B0003 caveat 1).

Survivor-universe long-side results are inflated and short-side deflated;
the M3 claim is only ever 'meta adds value over primary WITHIN this
universe', reported per side. Pools events across all assets per
(primary, model) at the FIXED 0.55 headline threshold.

USAGE:
  uv run python scripts/report_long_short_split.py --results results/clf_equity_m3_d1 --cost-bps 10
"""
from __future__ import annotations
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HEADLINE_THRESHOLD = 0.55  # fixed by spec — never selected from the test grid
MIN_TRADES_FOR_SHARPE = 30  # pipeline invariant: NaN (not 0) below this


def _naive_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Normalize a DatetimeIndex to tz-naive UTC for alignment comparisons.

    The pooled trainer builds its oof index via np.concatenate([idx.values, ...]),
    which strips tz info, so per-asset oof_predictions.parquet ends up tz-naive
    while events_side_fwd.parquet (written separately) stays tz-aware UTC. Same
    instants, same order — just a tz-metadata mismatch. Strip tz on both sides
    before comparing so genuinely misaligned inputs (different length/order)
    still raise.
    """
    if idx.tz is not None:
        return idx.tz_convert("UTC").tz_localize(None)
    return idx


def split_metrics(pnl: pd.Series, years: float) -> dict:
    n = int(len(pnl))
    out = {"n_trades": n,
           "mean_pnl_per_trade": float(pnl.mean()) if n else None,
           "hit_ratio": float((pnl > 0).mean()) if n else None}
    sd = float(pnl.std(ddof=1)) if n > 1 else 0.0
    if n < MIN_TRADES_FOR_SHARPE or sd == 0.0 or years <= 0:
        out["sharpe_net"] = float("nan")
    else:
        trades_per_year = n / years
        out["sharpe_net"] = float(pnl.mean() / sd * np.sqrt(trades_per_year))
    return out


def long_short_split(oof: pd.DataFrame, events: pd.DataFrame, model: str,
                     threshold: float, cost_bps: float) -> dict:
    # POSITIONAL alignment, not index joins: pooling concatenates many assets
    # whose event timestamps overlap, so the index is NOT unique. Caller must
    # pass row-aligned frames (same order, same length).
    if len(oof) != len(events) or not (
        _naive_utc_index(oof.index) == _naive_utc_index(events.index)
    ).all():
        raise ValueError("oof and events must be row-aligned (same index, same order)")
    p = oof[model].to_numpy(dtype=float)
    side = events["side"].to_numpy()
    fwd = events["fwd_ret"].to_numpy(dtype=float)
    take = ~np.isnan(p) & (p >= threshold)
    span_years = max(
        (events.index.max() - events.index.min()).days / 365.25, 1e-9,
    ) if len(events) else 0.0
    result = {}
    for name, sval in (("long", 1), ("short", -1)):
        mask = take & (side == sval)
        pnl = pd.Series(side[mask] * fwd[mask] - cost_bps / 1e4)
        result[name] = split_metrics(pnl, span_years)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--cost-bps", type=float, required=True)
    args = ap.parse_args()
    root = Path(args.results)

    pooled: dict[tuple[str, str], list[tuple[pd.DataFrame, pd.DataFrame]]] = {}
    for ev_path in sorted(root.glob("*/*/events_side_fwd.parquet")):
        oof_path = ev_path.parent / "oof_predictions.parquet"
        if not oof_path.exists():
            continue
        primary = ev_path.parent.name
        oof = pd.read_parquet(oof_path)
        ev = pd.read_parquet(ev_path)
        if len(oof) != len(ev) or not (
            _naive_utc_index(oof.index) == _naive_utc_index(ev.index)
        ).all():
            print(f"  SKIP {ev_path.parent}: oof/events misaligned")
            continue
        for model in oof.columns:
            pooled.setdefault((primary, model), []).append((oof[[model]], ev))

    report: dict[str, dict] = {}
    for (primary, model), parts in sorted(pooled.items()):
        # Both lists concat in the same per-asset order -> positional alignment
        # is preserved even though timestamps repeat across assets.
        oof_all = pd.concat([o for o, _ in parts], axis=0)
        ev_all = pd.concat([e for _, e in parts], axis=0)
        report[f"{primary}/{model}"] = long_short_split(
            oof_all, ev_all, model=model,
            threshold=HEADLINE_THRESHOLD, cost_bps=args.cost_bps,
        )

    out = root / "long_short_split.json"
    out.write_text(json.dumps(
        {"threshold": HEADLINE_THRESHOLD, "cost_bps": args.cost_bps,
         "note": ("Survivor universe: long side inflated, short side deflated. "
                  "Within-universe relative claims only (B0003 caveat 1)."),
         "pools": report}, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(report)} primary/model pools)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
