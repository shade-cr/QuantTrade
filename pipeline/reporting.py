"""Write summary.json, metrics_per_fold.json, feature_importance.json, plots, report.md."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_benchmark_correlations(
    strategy_pnl: pd.Series,
    benchmark_levels: dict[str, pd.Series],
    min_overlap: int = 30,
) -> dict[str, float | None]:
    """Pearson corr of daily strategy PnL vs each benchmark's daily returns.

    maxdama §4.10 alpha checklist: a "market-neutral" XAU strategy with
    corr(returns, S&P) = -0.4 is actually a risk-off bet, not alpha. This
    surfaces hidden beta. Diagnostic-only — never gates a decision.

    strategy_pnl is the per-event PnL series (sparse; indexed by event
    timestamp). It is summed to a daily series before correlating against
    each benchmark's daily simple returns. Returns None for a benchmark when
    the overlapping sample is < min_overlap (correlation not meaningful).
    """
    out: dict[str, float | None] = {}
    if strategy_pnl.empty:
        return {name: None for name in benchmark_levels}
    daily_pnl = strategy_pnl.groupby(strategy_pnl.index.normalize()).sum()
    for name, level in benchmark_levels.items():
        lvl = level.sort_index()
        if lvl.index.tz is None:
            lvl.index = pd.to_datetime(lvl.index).tz_localize("UTC")
        ret = lvl.pct_change()
        ret_daily = ret.groupby(ret.index.normalize()).last()
        aligned = pd.concat([daily_pnl, ret_daily], axis=1, join="inner").dropna()
        if len(aligned) < min_overlap:
            out[name] = None
        else:
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            # NaN when either side has zero variance (e.g. constant pnl) —
            # not a meaningful correlation, report as None like low-overlap.
            out[name] = float(corr) if pd.notna(corr) else None
    return out


def compute_shutoff_status(
    per_fold_sharpe: list[float],
    rolling_window: int,
    threshold: float,
) -> dict:
    """Decommissioning rule (maxdama §4.10): retire a primary when its rolling
    median fold Sharpe decays below a threshold.

    Reports a boolean only — NO automatic action. A practitioner reads this
    during a periodic review. `current_rolling_sharpe` is the nan-median of the
    last `rolling_window` folds (nan-safe per the project's NaN-when-n_trades<30
    invariant). status:
      - "RETIRE"            current_rolling_sharpe < threshold (finite)
      - "ACTIVE"            current_rolling_sharpe >= threshold
      - "INSUFFICIENT_DATA" all folds in the window are NaN (no measurement)
    """
    window = [s for s in per_fold_sharpe[-rolling_window:]]
    finite = [s for s in window if s is not None and np.isfinite(s)]
    if not finite:
        current = float("nan")
        status = "INSUFFICIENT_DATA"
    else:
        current = float(np.nanmedian(finite))
        status = "RETIRE" if current < threshold else "ACTIVE"
    return {
        "rolling_window": rolling_window,
        "threshold": threshold,
        "current_rolling_sharpe": current,
        "n_folds_in_window": len(window),
        "n_finite_folds_in_window": len(finite),
        "status": status,
    }


def write_summary_json(out_dir: Path, payload: dict) -> Path:
    p = out_dir / "summary.json"
    p.write_text(json.dumps(payload, indent=2, default=str))
    return p


def write_oof_parquet(out_dir: Path, oof: pd.DataFrame) -> Path:
    p = out_dir / "oof_predictions.parquet"
    oof.to_parquet(p)
    return p


def plot_calibration(out_dir: Path, y_true: np.ndarray, y_prob: np.ndarray, model_name: str) -> Path:
    bins = np.linspace(0, 1, 11)
    bucket = np.digitize(y_prob, bins) - 1
    df = pd.DataFrame({"y": y_true, "b": bucket})
    grp = df.groupby("b")["y"].agg(["mean", "count"])
    p = out_dir / f"calibration_{model_name}.png"
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot((bins[:-1] + bins[1:]) / 2, grp["mean"].reindex(range(10)).values, "o-")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title(f"Calibration — {model_name}")
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_equity(out_dir: Path, equity: pd.Series, label: str) -> Path:
    p = out_dir / f"equity_{label}.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(equity.index, equity.values)
    ax.set_title(f"Equity — {label}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def write_report_md(
    out_dir: Path,
    stack_decision_text: str,
    best_model: str,
    threshold: float,
    metrics_table: pd.DataFrame,
    top_features: list[str],
    next_steps: list[str],
    threshold_grid_table: pd.DataFrame | None = None,
    psr_per_model: dict[str, float] | None = None,
    dsr_per_model: dict[str, float] | None = None,
    selected_thresholds_per_fold: dict[str, list[float]] | None = None,
    n_trades_per_fold_per_model: dict[str, list[int]] | None = None,
    benchmark_correlations: dict[str, float | None] | None = None,
    shutoff_status: dict | None = None,
) -> Path:
    lines = [
        "# XAU D1 Meta-Labeling — Run Report",
        "",
        f"**Stack decision:** {stack_decision_text}",
        f"**Best single model (by median Sharpe):** `{best_model}`",
        f"**Best model's median selected threshold (inner-CV per fold):** {threshold:.3f}",
        "",
        "## Headline metrics (each model at its per-fold inner-CV-selected threshold)",
        "",
        metrics_table.to_markdown(),
        "",
    ]
    if selected_thresholds_per_fold is not None:
        rows = []
        for m, thrs in selected_thresholds_per_fold.items():
            trades = (n_trades_per_fold_per_model or {}).get(m, [None] * len(thrs))
            for fold_k, (thr, n) in enumerate(zip(thrs, trades)):
                rows.append({"model": m, "fold": fold_k,
                             "selected_threshold": thr,
                             "n_trades": n})
        sel_df = pd.DataFrame(rows)
        lines += [
            "## Inner-CV threshold selection",
            "",
            "Each row shows the threshold the inner CV picked for that (model, fold) — applied OOS to the outer test fold.",
            "",
            sel_df.to_markdown(index=False),
            "",
        ]
    if psr_per_model is not None and dsr_per_model is not None:
        psr_dsr_df = pd.DataFrame({"PSR(vs SR=0)": psr_per_model, "DSR (deflated)": dsr_per_model}).round(3)
        lines += [
            "## Deflated statistics (AFML §14)",
            "",
            "- **PSR** = Pr(true SR > 0 | sample), per Bailey & López de Prado 2012.",
            "- **DSR** = PSR deflated by trial selection (sr_benchmark = E[max trial SR under H0]).",
            "  Trials universe = all (model × fold) Sharpes within this primary.",
            "",
            psr_dsr_df.to_markdown(),
            "",
            "_DSR < PSR is expected — that's the cost of trying multiple models. A high DSR (≥ 0.95) is the rigorous bar._",
            "",
        ]
    if threshold_grid_table is not None:
        lines += [
            "## Threshold grid diagnostic (mean across folds, NOT used for selection)",
            "",
            threshold_grid_table.to_markdown(),
            "",
            "_The threshold grid is reported as information only; per-fold inner-CV threshold selection is deferred to Paso 2._",
            "",
        ]
    if benchmark_correlations:
        def _fmt(v: float | None) -> str:
            return f"{v:+.3f}" if v is not None else "n/a (insufficient overlap)"
        lines += [
            "## Benchmark correlation (maxdama §4.10 — is this hidden beta?)",
            "",
            "Pearson corr of daily strategy PnL vs each benchmark's daily returns.",
            "A large-magnitude correlation means the 'edge' is partly a directional "
            "bet on that benchmark (e.g. a risk-off play), not standalone alpha.",
            "",
            *[f"- **{name}**: {_fmt(v)}" for name, v in benchmark_correlations.items()],
            "",
            "_Diagnostic only — does not gate the stack decision._",
            "",
        ]
    if shutoff_status:
        cur = shutoff_status.get("current_rolling_sharpe")
        cur_str = f"{cur:.3f}" if cur is not None and np.isfinite(cur) else "n/a"
        lines += [
            "## Decommissioning check (maxdama §4.10)",
            "",
            f"- **status**: `{shutoff_status.get('status')}`",
            f"- rolling median Sharpe (last {shutoff_status.get('rolling_window')} folds): {cur_str} "
            f"vs retire-below threshold {shutoff_status.get('threshold')}",
            "",
            "_Reported only — no automatic action. RETIRE flags a primary for human review._",
            "",
        ]
    lines += [
        "## Top features (MDA permutation importance, best model)",
        "",
        *[f"- {f}" for f in top_features],
        "",
        "## Next steps (Paso 2)",
        "",
        *[f"- {s}" for s in next_steps],
    ]
    p = out_dir / "report.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p
