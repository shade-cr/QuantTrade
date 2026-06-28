"""B0148 — Pooled-vs-per-asset comparison harness + pre-registered falsification verdict.

Reads per-asset `summary.json` + `psr_dsr.json` from BOTH trees (the per-asset
baseline `results/clf_multi_h4` and the pooled subdir `results/clf_multi_h4_pooled`)
and evaluates the spec's pre-registered falsification criterion:

  1. DSR clears a SIGNIFICANCE bar (DSR > 0.95) for STRICTLY MORE assets than the
     per-asset baseline. (Mechanical "DSR finite" relief is excluded — significance,
     not mere starvation relief.)
  2. No-regression band: pooled per-asset OOS median Sharpe may drop by no more than
     0.15 absolute vs that asset's baseline. Assets with a NaN/starved baseline are
     EXEMPT from the band but must satisfy criterion 1 to count as a win.
  3. Majority = ceil(2/3) of the assets that HAD a finite baseline Sharpe must
     satisfy the band. (2-asset metals pool → require both.)
  4. Dominance veto: if ANY single asset drops by more than 0.40 Sharpe → FALSIFIED
     for that pool regardless of the majority count.

CONFIRMED iff criterion 1 AND criterion 3 hold AND criterion 4 (veto) does not trip.

This is a REPORTING / DECISION artifact. It does NOT auto-promote anything; a human
+ DA/risk review gate any deployment_config change.

Spec: docs/superpowers/specs/2026-06-04-b0148-cross-asset-meta-pooling-design.md
Usage:
  uv run python scripts/compare_pooled_vs_per_asset.py \
      --baseline results/clf_multi_h4 --pooled results/clf_multi_h4_pooled \
      --out results/clf_multi_h4_pooled/pooled_vs_per_asset_verdict.json
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np


DSR_SIGNIFICANCE_BAR = 0.95
NO_REGRESSION_BAND = 0.15      # absolute Sharpe drop allowed
DOMINANCE_VETO = 0.40          # absolute Sharpe drop that hard-falsifies a pool
MAJORITY_FRAC = 2.0 / 3.0


def _best_dsr(psr_dsr: dict, summary: dict) -> float:
    """DSR of the asset's best_model (by median Sharpe), NaN if unavailable."""
    dsr_map = psr_dsr.get("dsr", {})
    best = summary.get("best_model")
    if best is not None and best in dsr_map:
        v = dsr_map[best]
    else:
        finite = [d for d in dsr_map.values() if isinstance(d, (int, float)) and np.isfinite(d)]
        v = max(finite) if finite else float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _median_sharpe_best(summary: dict) -> float:
    """The best_model's median annualized Sharpe (NaN if starved)."""
    ms = summary.get("median_sharpe", {})
    best = summary.get("best_model")
    if best is not None and best in ms:
        try:
            return float(ms[best])
        except (TypeError, ValueError):
            return float("nan")
    finite = [float(v) for v in ms.values() if isinstance(v, (int, float)) and np.isfinite(v)]
    return max(finite) if finite else float("nan")


def _load_pair(tree: Path, asset: str, primary: str) -> dict | None:
    pdir = tree / asset / primary
    sp = pdir / "summary.json"
    pp = pdir / "psr_dsr.json"
    if not sp.exists():
        return None
    summary = json.loads(sp.read_text(encoding="utf-8"))
    if "skip_reason" in summary:
        return None
    psr_dsr = json.loads(pp.read_text(encoding="utf-8")) if pp.exists() else {}
    return {"summary": summary, "psr_dsr": psr_dsr,
            "dsr": _best_dsr(psr_dsr, summary),
            "median_sharpe": _median_sharpe_best(summary)}


def _discover_pairs(tree: Path) -> list[tuple[str, str]]:
    """Find (asset, primary) dirs that have a summary.json under `tree`."""
    pairs = []
    if not tree.exists():
        return pairs
    for asset_dir in sorted(p for p in tree.iterdir() if p.is_dir()):
        for prim_dir in sorted(p for p in asset_dir.iterdir() if p.is_dir()):
            if (prim_dir / "summary.json").exists():
                pairs.append((asset_dir.name, prim_dir.name))
    return pairs


def compare_trees(baseline: Path, pooled: Path) -> dict:
    """Build the per-asset comparison + the pre-registered falsification verdict."""
    baseline, pooled = Path(baseline), Path(pooled)
    pooled_pairs = _discover_pairs(pooled)
    per_asset: list[dict] = []

    base_dsr_wins = 0    # baseline assets with DSR > bar
    pooled_dsr_wins = 0  # pooled assets with DSR > bar

    finite_base_assets = 0
    band_satisfied = 0
    worst_drop = 0.0
    veto_assets: list[str] = []

    for asset, primary in pooled_pairs:
        pooled_rec = _load_pair(pooled, asset, primary)
        base_rec = _load_pair(baseline, asset, primary)
        if pooled_rec is None:
            continue

        p_dsr = pooled_rec["dsr"]
        b_dsr = base_rec["dsr"] if base_rec else float("nan")
        p_ms = pooled_rec["median_sharpe"]
        b_ms = base_rec["median_sharpe"] if base_rec else float("nan")

        if np.isfinite(p_dsr) and p_dsr > DSR_SIGNIFICANCE_BAR:
            pooled_dsr_wins += 1
        if np.isfinite(b_dsr) and b_dsr > DSR_SIGNIFICANCE_BAR:
            base_dsr_wins += 1

        band_ok = None
        drop = None
        if np.isfinite(b_ms):
            finite_base_assets += 1
            drop = float(b_ms - p_ms) if np.isfinite(p_ms) else float("inf")
            band_ok = drop <= NO_REGRESSION_BAND
            if band_ok:
                band_satisfied += 1
            if np.isfinite(drop):
                worst_drop = max(worst_drop, drop)
            if np.isfinite(drop) and drop > DOMINANCE_VETO:
                veto_assets.append(asset)
        else:
            # starved baseline → exempt from band; counts as win only via DSR crit 1.
            band_ok = "exempt_starved_baseline"

        per_asset.append({
            "asset": asset, "primary": primary,
            "baseline_dsr": b_dsr, "pooled_dsr": p_dsr,
            "baseline_median_sharpe": b_ms, "pooled_median_sharpe": p_ms,
            "sharpe_drop": drop,
            "band_satisfied": band_ok,
            "dsr_win_pooled": bool(np.isfinite(p_dsr) and p_dsr > DSR_SIGNIFICANCE_BAR),
        })

    # Criterion 1: pooled DSR>0.95 for strictly MORE assets than baseline.
    crit1 = pooled_dsr_wins > base_dsr_wins
    # Criterion 3: majority (ceil 2/3) of finite-baseline assets satisfy the band.
    needed = math.ceil(MAJORITY_FRAC * finite_base_assets) if finite_base_assets else 0
    crit3 = (finite_base_assets > 0) and (band_satisfied >= needed)
    # Criterion 4 veto: any single asset drop > 0.40.
    veto = len(veto_assets) > 0

    confirmed = bool(crit1 and crit3 and not veto)

    return {
        "verdict": "CONFIRMED" if confirmed else "FALSIFIED",
        "criteria": {
            "crit1_dsr_significance": {
                "pass": bool(crit1),
                "pooled_dsr_gt_0p95_assets": pooled_dsr_wins,
                "baseline_dsr_gt_0p95_assets": base_dsr_wins,
                "bar": DSR_SIGNIFICANCE_BAR,
            },
            "crit3_majority_band": {
                "pass": bool(crit3),
                "finite_baseline_assets": finite_base_assets,
                "band_satisfied": band_satisfied,
                "needed_ceil_2_3": needed,
                "band_abs_sharpe": NO_REGRESSION_BAND,
            },
            "crit4_dominance_veto": {
                "tripped": bool(veto),
                "veto_assets": veto_assets,
                "worst_drop": worst_drop,
                "veto_threshold": DOMINANCE_VETO,
            },
        },
        "per_asset": per_asset,
        "note": ("REPORTING ONLY — no auto-promotion. CONFIRMED requires crit1 AND "
                 "crit3 AND not crit4-veto. A null on naive row pooling does not kill "
                 "the broader transfer hypothesis (spec 'If falsified' branch)."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/clf_multi_h4")
    ap.add_argument("--pooled", default="results/clf_multi_h4_pooled")
    ap.add_argument("--out", default=None,
                    help="Path to write the JSON verdict (default: <pooled>/pooled_vs_per_asset_verdict.json)")
    args = ap.parse_args()

    verdict = compare_trees(Path(args.baseline), Path(args.pooled))
    out = Path(args.out) if args.out else Path(args.pooled) / "pooled_vs_per_asset_verdict.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    print(f"VERDICT: {verdict['verdict']}")
    c = verdict["criteria"]
    print(f"  crit1 DSR>0.95: pooled={c['crit1_dsr_significance']['pooled_dsr_gt_0p95_assets']} "
          f"vs baseline={c['crit1_dsr_significance']['baseline_dsr_gt_0p95_assets']} "
          f"-> {'PASS' if c['crit1_dsr_significance']['pass'] else 'FAIL'}")
    print(f"  crit3 majority band: {c['crit3_majority_band']['band_satisfied']}/"
          f"{c['crit3_majority_band']['finite_baseline_assets']} "
          f"(need {c['crit3_majority_band']['needed_ceil_2_3']}) "
          f"-> {'PASS' if c['crit3_majority_band']['pass'] else 'FAIL'}")
    print(f"  crit4 veto: {'TRIPPED ' + str(c['crit4_dominance_veto']['veto_assets']) if c['crit4_dominance_veto']['tripped'] else 'clear'}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
