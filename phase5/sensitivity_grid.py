"""Sensitivity grid for the regime taxonomy boundaries (XAU D1).

Addresses devil's advocate HIGH-severity objection #2 (signals/devils_advocate_reviews/
day1_regime_taxonomy_v1.json): the 80/70 hysteresis + 40-dwell choices are
arbitrary; without a pre-registered sensitivity sweep, the taxonomy is one
parameter sweep away from data mining once downstream metrics are visible.

This module sweeps:
  vol_enter_high_pct ∈ {0.75, 0.80, 0.85}
  vol_exit_high_pct  ∈ {0.60, 0.65, 0.70}     (each must be strictly < enter)
  min_dwell_d1       ∈ {20, 40, 60}

For each grid cell, reports: regime_counts, regime_fractions, n_episodes,
min_episode_bars, episodes_below_60_bars. Output is committed BEFORE any
downstream M3 metric is computed on regime-conditional candidates — this
ex-ante registration prevents retroactive justification.

Usage:
  uv run python -m phase5.sensitivity_grid --asset XAUUSD --frequency D1 \\
                                            --out results/phase5/

Output: results/phase5/sensitivity_grid_<asset>_<freq>.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from pipeline.data import load_dataset
from pipeline.regimes import label_regimes, sanity_report, _resolve_data_path


GRID = {
    "vol_enter_high_pct": (0.75, 0.80, 0.85),
    "vol_exit_high_pct": (0.60, 0.65, 0.70),
    "min_dwell_d1": (20, 40, 60),
}


def run_grid(asset: str, frequency: str, data_path: str | None = None) -> dict:
    path = _resolve_data_path(asset, frequency, data_path)
    df = load_dataset(path)
    cells: list[dict] = []
    for enter in GRID["vol_enter_high_pct"]:
        for exit_ in GRID["vol_exit_high_pct"]:
            if exit_ >= enter:
                continue
            for dwell in GRID["min_dwell_d1"]:
                regimes = label_regimes(
                    df["close"],
                    frequency=frequency,
                    vol_enter_high_pct=enter,
                    vol_exit_high_pct=exit_,
                    min_dwell_d1=dwell,
                )
                sr = sanity_report(regimes)
                cells.append(
                    {
                        "vol_enter_high_pct": enter,
                        "vol_exit_high_pct": exit_,
                        "min_dwell_d1": dwell,
                        "regime_counts": sr["regime_counts"],
                        "regime_fractions": {k: round(v, 4) for k, v in sr["regime_fractions"].items()},
                        "n_episodes": sr["n_episodes"],
                        "min_episode_bars": sr["min_episode_bars"],
                        "episodes_below_60_bars": sr["episodes_below_60_bars"],
                        "regimes_below_5pct": sr["regimes_below_5pct"],
                    }
                )
    return {
        "asset": asset,
        "frequency": frequency,
        "data_span": {
            "start": df.index.min().isoformat(),
            "end": df.index.max().isoformat(),
            "n_bars": int(len(df)),
        },
        "grid": GRID,
        "n_cells": len(cells),
        "cells": cells,
        "locked_choice": {
            "vol_enter_high_pct": 0.80,
            "vol_exit_high_pct": 0.70,
            "min_dwell_d1": 40,
            "rationale": (
                "Centre cell of the grid. Pre-registered choice; any later re-tune "
                "treated as a new experiment with its own sensitivity grid."
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--frequency", choices=("D1", "H4"), default="D1")
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--out", default="results/phase5/")
    args = ap.parse_args()

    out = run_grid(args.asset, args.frequency, args.data_path)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sensitivity_grid_{args.asset}_{args.frequency.lower()}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Grid cells evaluated: {out['n_cells']}")

    # Quick summary table
    print(f"\n  enter  exit   dwell  n_ep  min_ep  BEAR_STR%  BULL_STR%")
    for c in out["cells"]:
        print(
            f"  {c['vol_enter_high_pct']:.2f}   "
            f"{c['vol_exit_high_pct']:.2f}   "
            f"{c['min_dwell_d1']:3d}    "
            f"{c['n_episodes']:3d}   "
            f"{c['min_episode_bars']:4d}    "
            f"{c['regime_fractions']['BEAR_STRESSED']*100:5.2f}      "
            f"{c['regime_fractions']['BULL_STRESSED']*100:5.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
