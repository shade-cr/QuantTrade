"""Audit-M2 fix: threshold-transferability analyzer.

The tri-auditor audit (FINAL_REVIEW M2) flagged: inner-CV selected
thresholds often don't transfer to outer test folds, producing 0-trade
or sub-30-trade outer folds. The orchestrator reports these as
Sharpe=0 (or NaN under CLAUDE.md invariant) — indistinguishable from
"no edge" vs "selector failure."

This script reads every existing `threshold_grid_metrics.json` and
`psr_dsr.json` and computes the **fixed-threshold-0.50** diagnostic:

  - Per-fold n_trades and Sharpe at threshold=0.50
  - Aggregated sr_per_trade at threshold=0.50 (across ALL folds)
  - Family DSR at threshold=0.50 against the n>=30 filtered pool
  - Regime diversity (v3): max DD and max rally on the underlying
    asset's close-price series across the OOS span (NOT on strategy
    equity — we test whether the model SAW a varied regime, not
    whether it profited in both directions).
  - Transferability classification (v3 — adds regime-diversity gate on
    top of the v2 median-Sharpe-positivity gate):
    * STABLE:           >= 30 trades per fold on >= 3/4 folds AND
                        median active-fold Sharpe > 0 AND
                        regime diversity pass
    * REGIME_LIMITED:   >= 2 active folds + positive median active-fold
                        Sharpe BUT OOS spans only one regime (max DD
                        < 15% OR max rally < 15%). Captures the
                        XAG/ETH feat_sentiment momentum_zscore/catboost
                        case: 2021-05 H4 history start with
                        train_min_bars=3000 pushes OOS into
                        2023-2026, and the candidate's OOS sees only
                        a sustained rally (XAG 2024-2025 silver
                        breakout) or only a sustained decline (ETH
                        2025-2026), so no cross-regime defense.
    * NOT_PROFITABLE:   >= 2 active folds (n>=30) BUT
                        median active-fold Sharpe <= 0 — captures
                        the USDJPY engine_cusum cusum_filter/rf
                        false-positive.
    * MARGINAL_2FOLDS:  2 active folds, positive median Sharpe,
                        regime diversity pass.
    * 1FOLD_CONCENTRATED: only 1 fold has n>=30 (concentrated)
    * NO-FIRE:          fewer than 2 folds with any trades (selector broke)

Output: results/threshold_transferability_analysis.json

Usage:
  uv run python scripts/analyze_threshold_transferability.py \\
    --aggregate-input results/aggregate_FINAL_postaudit.json \\
    --output results/threshold_transferability_analysis.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.metrics import deflated_sharpe_ratio


STABLE_FOLD_N_MIN = 30
STABLE_FOLDS_MIN = 3  # of 4 folds

# Regime-diversity gate (v3): OOS must span at least one >=15% drawdown
# AND one >=15% rally on the underlying asset's close-price series.
REGIME_MIN_MOVE = 0.15  # 15%


def _max_drawdown(close: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction.

    Computed on the running peak: ``max((peak - x) / peak)`` over all
    bars. Returns 0.0 for an empty or monotone-up series. Robust to
    NaNs (they propagate through running max, so we drop them up front).
    """
    arr = np.asarray(close, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(arr)
    dd = (running_peak - arr) / running_peak
    return float(np.nanmax(dd)) if dd.size else 0.0


def _max_rally(close: np.ndarray) -> float:
    """Maximum trough-to-peak rally as a positive fraction.

    Symmetric counterpart to ``_max_drawdown``: ``max((x - trough) / trough)``
    over the running trough. This captures the biggest sustained move
    UP from any prior low, even if a larger peak preceded it.
    """
    arr = np.asarray(close, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    running_trough = np.minimum.accumulate(arr)
    rally = (arr - running_trough) / running_trough
    return float(np.nanmax(rally)) if rally.size else 0.0


def _regime_diversity(
    close: np.ndarray,
    min_move: float = REGIME_MIN_MOVE,
) -> dict:
    """Compute regime-diversity gate decision for a close-price series.

    A candidate's OOS passes the gate iff the underlying asset's
    close-price series spans at least one drawdown of ``min_move`` AND
    one rally of ``min_move`` (both default 15%).

    Note: we measure on the asset's price, NOT on strategy equity. The
    question being asked is "did the model see a varied regime?", not
    "did the strategy profit in both directions?". A short-biased
    strategy in a sustained decline is still regime-limited by this
    definition because the OOS underlying never rallied 15%.
    """
    dd = _max_drawdown(close)
    rally = _max_rally(close)
    return {
        "max_dd": dd,
        "max_rally": rally,
        "pass": bool(dd >= min_move and rally >= min_move),
    }


def _classify_transferability(
    per_fold_n: list[int],
    per_fold_sharpe: list[float] | None = None,
    regime_pass: bool | None = None,
) -> str:
    """Classify how well the threshold transfers across folds.

    v3 (2026-05-25): adds a regime-diversity gate on top of v2. A
    candidate that would otherwise classify as STABLE or
    MARGINAL_2FOLDS is reclassified to REGIME_LIMITED if its OOS
    spans only one regime (max DD < 15% OR max rally < 15% on the
    underlying close-price series). Motivated by the XAG/ETH
    feat_sentiment finding: with H4 history starting 2021-05 and
    train_min_bars=3000, OOS only ever sees 2023-2026 — a sustained
    silver rally for XAG, a sustained ETH decline for ETH. Neither
    candidate has cross-regime evidence.

    v2 (2026-05-25): adds a median-active-fold-Sharpe positivity gate
    to STABLE and MARGINAL_2FOLDS. Without this gate, the old classifier
    flagged USDJPY engine_cusum cusum_filter/rf as STABLE — 3 of 4 folds
    had n_trades >= 30, but two of those active folds lost money
    (Sharpe -0.58, -0.61) with only one outlier-positive fold (+2.00).
    Median active-fold Sharpe = -0.58 -> NOT_PROFITABLE.

    Per CLAUDE.md invariant: Sharpe is NaN (not 0) when n_trades < 30
    or std == 0. We use np.nanmedian over the active-fold subset
    (filtered by n>=30) so NaN folds are excluded, not zero-imputed.

    When ``per_fold_sharpe`` is None (legacy call), the median gate is
    skipped and we fall back to the v1 n_trades-only taxonomy.
    When ``regime_pass`` is None (legacy call), the regime gate is
    skipped — preserves backward compatibility.

    Classification priority (highest to lowest):
        NOT_PROFITABLE  (>=2 active folds + median Sharpe <= 0)
        REGIME_LIMITED  (>=2 active folds + median Sharpe > 0 + regime fail)
        STABLE / MARGINAL_2FOLDS (regime pass + positive median)
        1FOLD_CONCENTRATED
        MULTI_FOLD_BUT_LOW_N / NO_FIRE
    """
    n_stable = sum(1 for n in per_fold_n if n >= STABLE_FOLD_N_MIN)
    n_any = sum(1 for n in per_fold_n if n > 0)

    # Compute median Sharpe across active folds (n >= STABLE_FOLD_N_MIN).
    # NaN-safe; uses np.nanmedian so any NaN-typed Sharpes are excluded
    # rather than coerced to 0. This is critical because "no trades" must
    # not be conflated with "trades but zero Sharpe."
    median_active_sharpe = float("nan")
    if per_fold_sharpe is not None and len(per_fold_sharpe) == len(per_fold_n):
        active = [
            s for s, n in zip(per_fold_sharpe, per_fold_n)
            if n >= STABLE_FOLD_N_MIN and s is not None and np.isfinite(s)
        ]
        if active:
            median_active_sharpe = float(np.nanmedian(active))

    # Median-positivity gate: STABLE/MARGINAL_2FOLDS require strictly
    # positive median active-fold Sharpe. If we have per-fold Sharpes
    # AND the median is <=0 with >=2 active folds, the candidate is
    # NOT_PROFITABLE regardless of n_trades coverage.
    has_sharpe_data = per_fold_sharpe is not None and np.isfinite(median_active_sharpe)
    # Regime gate only fires when we have both regime info and
    # ≥2 active folds; 1FOLD_CONCENTRATED is already a warning class
    # and is not further split by regime (see test
    # test_1fold_concentrated_not_downgraded_by_regime).
    regime_known = regime_pass is not None

    if n_stable >= STABLE_FOLDS_MIN:
        if has_sharpe_data and median_active_sharpe <= 0:
            return "NOT_PROFITABLE"
        if regime_known and not regime_pass:
            return "REGIME_LIMITED"
        return "STABLE"
    if n_stable == 2:
        if has_sharpe_data and median_active_sharpe <= 0:
            return "NOT_PROFITABLE"
        if regime_known and not regime_pass:
            return "REGIME_LIMITED"
        return "MARGINAL_2FOLDS"
    if n_stable == 1 and sum(per_fold_n) >= 30:
        return "1FOLD_CONCENTRATED"
    if n_any >= 2:
        return "MULTI_FOLD_BUT_LOW_N"
    return "NO_FIRE"


def _aggregate_threshold_50(
    grid_rows: list[dict],
    regime: dict | None = None,
) -> dict[str, dict]:
    """Extract threshold=0.50 metrics per model and aggregate across folds.

    Returns {model: {per_fold_sharpe, per_fold_n, total_n, ...}}. When
    ``regime`` is supplied (a dict from ``_regime_diversity``), it is
    threaded into the classifier and attached under
    ``regime_diversity`` in each model's output.
    """
    by_model: dict[str, list[dict]] = {}
    for row in grid_rows:
        if abs(row.get("threshold", 0) - 0.50) > 1e-6:
            continue
        model = row.get("model", "?")
        by_model.setdefault(model, []).append(row)

    out: dict[str, dict] = {}
    regime_pass: bool | None = regime["pass"] if regime is not None else None
    for model, rows in by_model.items():
        rows_sorted = sorted(rows, key=lambda r: r.get("fold", 0))
        per_fold_sharpe = []
        per_fold_n = []
        for r in rows_sorted:
            sh = r.get("sharpe_net")
            n = r.get("n_trades", 0)
            per_fold_sharpe.append(sh if sh is not None else float("nan"))
            per_fold_n.append(int(n))

        total_n = sum(per_fold_n)
        # Aggregate sr_per_trade = mean of per-fold sharpe weighted by n_trades
        # (approximation — true aggregation needs per-trade returns we don't have).
        # Use median of fold Sharpes among folds with n >= STABLE_FOLD_N_MIN.
        stable_sharpes = [
            s for s, n in zip(per_fold_sharpe, per_fold_n)
            if n >= STABLE_FOLD_N_MIN and s is not None and np.isfinite(s)
        ]
        median_stable_sharpe = (
            float(np.median(stable_sharpes)) if stable_sharpes else float("nan")
        )
        # Approximate sr_per_trade from median per-fold Sharpe / sqrt(bars_per_year)
        # is not directly possible without bars_per_year context here. We'll just
        # report the per-fold Sharpes and let downstream interpret.

        entry: dict = {
            "per_fold_sharpe": per_fold_sharpe,
            "per_fold_n_trades": per_fold_n,
            "total_n_trades": total_n,
            "n_stable_folds": sum(1 for n in per_fold_n if n >= STABLE_FOLD_N_MIN),
            "transferability": _classify_transferability(
                per_fold_n, per_fold_sharpe, regime_pass
            ),
            # v1 name preserved for backward compatibility; v2 alias added.
            "median_stable_sharpe": median_stable_sharpe,
            "median_active_fold_sharpe": median_stable_sharpe,
        }
        if regime is not None:
            entry["regime_diversity"] = regime
        out[model] = entry
    return out


def _detect_frequency(engine_dir: str, asset: str, primary: str) -> str:
    """Return 'D1' or 'H4' based on the candidate's summary.json bars_per_year.

    Convention: bars_per_year <= 365 -> D1, >=1560 -> H4. Falls back to
    'D1' if summary.json is missing or unreadable.
    """
    # Try multi-asset layout (engine_dir/asset/primary/summary.json)
    candidates = [
        Path(engine_dir) / asset / primary / "summary.json",
        Path(engine_dir) / primary / "summary.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                bpy = json.loads(path.read_text(encoding="utf-8")).get("bars_per_year")
                if bpy is not None and bpy <= 365:
                    return "D1"
                if bpy is not None and bpy >= 1500:
                    return "H4"
            except (json.JSONDecodeError, OSError):
                pass
    return "D1"


def _data_csv_for(asset: str, freq: str) -> Path:
    """Locate the canonical close-price CSV for an asset at a given frequency."""
    project_root = Path(__file__).resolve().parent.parent
    suffix = "_D1.csv" if freq == "D1" else "_H4.csv"
    return project_root / "data" / freq / f"{asset}{suffix}"


def _oos_span_from_oof(
    oof_path: Path, model: str
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return (start, end) of the model's non-NaN OOS span, or (None, None) on failure.

    Uses the model column's first_valid_index / last_valid_index per
    CLAUDE.md invariant that NaN means "no measurement" (here: not yet
    in OOS partition / outside the walk-forward test span).
    """
    if not oof_path.exists():
        return None, None
    try:
        df = pd.read_parquet(oof_path)
    except (OSError, ValueError):
        return None, None
    if model not in df.columns:
        return None, None
    col = df[model]
    start = col.first_valid_index()
    end = col.last_valid_index()
    return start, end


def _compute_regime_diversity_for_candidate(
    engine_dir: str,
    asset: str,
    primary: str,
    model: str,
    min_move: float = REGIME_MIN_MOVE,
) -> dict | None:
    """Compute the regime-diversity dict for one candidate.

    Returns None when OOS span or close-price CSV is unavailable
    (caller will skip the regime gate). Determines OOS span from the
    candidate's ``oof_predictions.parquet`` (non-NaN range for the
    given model) and reads the underlying close prices from
    ``data/<freq>/<ASSET>_<freq>.csv`` clipped to that span.
    """
    # Locate OOF predictions
    candidates = [
        Path(engine_dir) / asset / primary / "oof_predictions.parquet",
        Path(engine_dir) / primary / "oof_predictions.parquet",
    ]
    oof_path = next((p for p in candidates if p.exists()), None)
    if oof_path is None:
        return None

    start, end = _oos_span_from_oof(oof_path, model)
    if start is None or end is None:
        return None

    freq = _detect_frequency(engine_dir, asset, primary)
    csv_path = _data_csv_for(asset, freq)
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except (OSError, ValueError):
        return None
    if "time" not in df.columns or "close" not in df.columns:
        return None

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    # Clip to OOS span (inclusive)
    df = df.loc[start:end]
    if df.empty:
        return None

    close = df["close"].to_numpy(dtype=float)
    diversity = _regime_diversity(close, min_move=min_move)
    diversity["oos_start"] = str(start)
    diversity["oos_end"] = str(end)
    diversity["frequency"] = freq
    diversity["n_bars"] = int(len(df))
    return diversity


def analyze_candidates(
    aggregate_path: Path,
    output_path: Path,
    min_hl_dsr: float = 0.20,
    min_total_n: int = 30,
) -> dict:
    """Walk the surviving candidates from the aggregate and apply M2 diagnostic.

    Filters to candidates with HL_family_dsr >= min_hl_dsr AND total n_trades
    >= min_total_n in the inner-CV-selected configuration.
    """
    # Load the HL-corrected aggregate
    hl_path = aggregate_path.parent / "aggregate_FINAL_phase4_HL_v2.json"
    if hl_path.exists():
        agg = json.loads(hl_path.read_text(encoding="utf-8"))
        print(f"Using HL-corrected aggregate: {hl_path.name}")
    else:
        agg = json.loads(aggregate_path.read_text(encoding="utf-8"))
        print(f"WARN: HL aggregate not found, using raw: {aggregate_path.name}")

    candidates: list[dict] = []

    for asset, ablock in agg["per_asset"].items():
        for engine_dir, eblock in ablock.get("recomputed_dsr", {}).items():
            for primary, pblock in eblock.items():
                for model, m in pblock.items():
                    # Filter by HL DSR threshold AND minimum n
                    hl_dsr = m.get("hl_family_dsr", float("nan"))
                    n_inner = m.get("n", 0)
                    if not (isinstance(hl_dsr, (int, float))
                            and np.isfinite(hl_dsr)
                            and hl_dsr >= min_hl_dsr
                            and n_inner >= min_total_n):
                        continue

                    # Load threshold grid for this candidate
                    grid_path = Path(engine_dir) / asset / primary / "threshold_grid_metrics.json"
                    if not grid_path.exists():
                        # Try single-asset layout
                        grid_path = Path(engine_dir) / primary / "threshold_grid_metrics.json"
                        if not grid_path.exists():
                            continue
                    try:
                        grid = json.loads(grid_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        continue

                    # v3: compute regime diversity from OOF span + asset close
                    regime = _compute_regime_diversity_for_candidate(
                        engine_dir, asset, primary, model
                    )
                    thr_50_by_model = _aggregate_threshold_50(grid, regime=regime)
                    if model not in thr_50_by_model:
                        continue

                    candidates.append({
                        "asset": asset,
                        "engine_dir": engine_dir,
                        "primary": primary,
                        "model": model,
                        "inner_cv": {
                            "n_trades": n_inner,
                            "sr_per_trade": m.get("sr_observed"),
                            "per_engine_dsr": m.get("original_dsr"),
                            "hl_family_dsr": hl_dsr,
                        },
                        "threshold_50": thr_50_by_model[model],
                    })

    # Sort by transferability (STABLE first) then by hl_dsr.
    # NOT_PROFITABLE candidates (median active-fold Sharpe <= 0) come
    # after the genuinely-deployable classes but before 1FOLD/NO-FIRE
    # because they DID transfer in n_trades — just not in PnL.
    # REGIME_LIMITED (v3) sits between MARGINAL_2FOLDS and NOT_PROFITABLE:
    # the candidates ARE profitable, but OOS only covered one regime.
    transfer_order = {
        "STABLE": 0,
        "MARGINAL_2FOLDS": 1,
        "REGIME_LIMITED": 2,
        "NOT_PROFITABLE": 3,
        "MULTI_FOLD_BUT_LOW_N": 4,
        "1FOLD_CONCENTRATED": 5,
        "NO_FIRE": 6,
    }
    candidates.sort(key=lambda c: (
        transfer_order.get(c["threshold_50"]["transferability"], 99),
        -c["inner_cv"]["hl_family_dsr"],
    ))

    # Count by transferability class
    class_counts: dict[str, int] = {}
    for c in candidates:
        cls = c["threshold_50"]["transferability"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    out = {
        "filter": {"min_hl_dsr": min_hl_dsr, "min_total_n": min_total_n},
        "n_candidates_evaluated": len(candidates),
        "transferability_summary": class_counts,
        "candidates": candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    return out


def print_report(result: dict) -> None:
    """Human-readable summary."""
    print(f"\nM2 (threshold transferability) analysis")
    print(f"Filter: HL_DSR >= {result['filter']['min_hl_dsr']}, n_inner >= {result['filter']['min_total_n']}")
    print(f"Candidates evaluated: {result['n_candidates_evaluated']}")
    print()
    print("Transferability class counts:")
    for cls, n in sorted(result["transferability_summary"].items()):
        print(f"  {cls:24s}: {n}")
    print()

    if not result["candidates"]:
        return

    print(f"{'asset':>7}  {'engine':30s}  {'primary/model':25s}  "
          f"{'class':22s}  {'thr50_n':>7}  {'thr50_stable_folds':>18}  "
          f"{'inner_n':>7}  {'inner_hl_dsr':>12}")
    for c in result["candidates"]:
        eng_short = (
            c["engine_dir"]
            .replace("results\\variants\\", "")
            .replace("results\\", "")
            .replace("results/variants/", "")
            .replace("results/", "")[:28]
        )
        pm = f"{c['primary']}/{c['model']}"[:25]
        t50 = c["threshold_50"]
        cls = t50["transferability"]
        n50 = t50["total_n_trades"]
        stable = t50["n_stable_folds"]
        ic = c["inner_cv"]
        n_inner = ic["n_trades"]
        hl = ic["hl_family_dsr"]
        print(f"{c['asset']:>7}  {eng_short:30s}  {pm:25s}  "
              f"{cls:22s}  {n50:>7}  {stable:>18}  "
              f"{n_inner:>7}  {hl:>12.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aggregate-input", type=str, required=True)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--min-hl-dsr", type=float, default=0.20)
    ap.add_argument("--min-total-n", type=int, default=30)
    args = ap.parse_args()

    result = analyze_candidates(
        Path(args.aggregate_input),
        Path(args.output),
        min_hl_dsr=args.min_hl_dsr,
        min_total_n=args.min_total_n,
    )
    print_report(result)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
