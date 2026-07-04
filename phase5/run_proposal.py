"""Single-proposal evaluator: proposal JSON -> pipeline run -> M3 audit -> result JSON.

The full chain:
  1. Validate proposal (schema + lookahead lint + barrier_geometry_attestation).
  2. Pre-flight: cross-check n_trades_total_min against any
     `fire_rate_estimate` extra criterion (Day-2 skeptic caveat 1).
  3. Build regime mask parquet from data/regimes/<asset>_d1_regimes.parquet.
  4. Generate transient pipeline config inheriting configs/xau_d1_22y_with_cot.yaml,
     overlaying primary.candidates, output_dir, regime_mask_path, triple_barrier params.
  5. Run scripts/run_backtest.py as a subprocess.
  6. Parse threshold_grid_metrics.json + summary.json + psr_dsr.json.
  7. Compute regime_diversity on OOS span; run _classify_transferability.
  8. Evaluate falsification_criterion + extra_falsification_criteria.
  9. Write signals/audit_results/<id>.json.

Usage:
  uv run python -m phase5.run_proposal --proposal signals/proposals/<id>.json
  uv run python -m phase5.run_proposal --proposal signals/proposals/<id>.json --dry-run
  uv run python -m phase5.run_proposal --proposal signals/proposals/<id>.json --preflight-only
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from phase5.proposal import (
    Proposal, load_proposal, DEFAULT_FALSIFICATION, ProposalValidationError,
)
from pipeline.primary_contracts import normalize_primary_params
from pipeline.thresholds import ev_breakeven_grid
from pipeline.walk_forward import wf_event_floor

# Import the v3 classifier helpers via path injection
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_threshold_transferability import (
    _classify_transferability, _regime_diversity, _max_drawdown, _max_rally,
)


# B0149 — system-level DSR floor (autonomous promotion criterion). DSR here is
# the Bailey–López de Prado Deflated Sharpe Ratio: the probability, after
# deflating for the number of trials, that the true Sharpe exceeds 0 — a value
# in [0, 1] where 0.95 is the standard significance bar. B0089 made DSR a hard
# gate but only when the proposal set dsr_min, leaving enforcement to manual
# reviewer discretion for the (entire) dsr-less Loop-A corpus. This floor turns
# the gate on for EVERY audit at result-reading time. Tightening-only by
# construction: a proposal's own dsr_min can only RAISE the bar, never lower
# it — consistent with the methodology rule that falsification criteria may
# never be loosened. Frozen proposal JSONs are not edited.
SYSTEM_DSR_MIN = 0.95

SIGNALS_DIR = Path("signals")
AUDIT_RESULTS_DIR = SIGNALS_DIR / "audit_results"
RUNTIME_DIR = Path("phase5/runtime")
RESULTS_PHASE5_DIR = Path("results/phase5")
REGIMES_DIR = Path("data/regimes")
TEMPLATE_CONFIG = Path("configs/xau_d1_22y_correct_geometry.yaml")  # single-asset format with data_path


def check_n_trades_consistency(p: Proposal) -> dict:
    """Day-2 skeptic caveat 1: n_trades_total_min must not exceed 1.5x the
    proposal's own fire_rate_estimate.expected_n_trades_high (if present)."""
    headline = p.falsification_criterion.n_trades_total_min
    for crit in p.extra_falsification_criteria:
        if not isinstance(crit, dict) or crit.get("name") != "fire_rate_estimate":
            continue
        high = crit.get("expected_n_trades_high")
        if high is None:
            high = crit.get("expected_fire_rate_per_1000_bars_high")
        if high is not None and headline > 1.5 * high:
            return {
                "passed": False,
                "reason": (
                    f"n_trades_total_min={headline} exceeds 1.5x own "
                    f"fire_rate_estimate.expected_n_trades_high={high}. "
                    f"Audit will mechanically return MULTI_FOLD_BUT_LOW_N "
                    f"regardless of edge quality. Lower n_trades_total_min "
                    f"or widen the entry conjunction."
                ),
                "expected_high": high,
                "headline_min": headline,
            }
    return {"passed": True, "reason": "no fire_rate_estimate present or consistent"}


def evaluate_event_floor(n_events: int, n_trades_total_min: int, wf_floor: int) -> dict:
    """B0048 — gate the post-primary, in-regime event count against the floor.

    A proposal must clear BOTH its own committed edge-quality floor
    (`n_trades_total_min`) AND the walk-forward geometry floor (`wf_floor`, from
    pipeline.walk_forward.wf_event_floor). The n_trades-only pre-flight let
    T011D2M (35 events) and T015D2M (73 events, declared min 50) reach the audit
    subprocess, where they died on the WF refusal cliff. This makes the
    geometry floor a first-class, pre-subprocess falsification.
    """
    required = max(int(n_trades_total_min), int(wf_floor))
    passed = int(n_events) >= required
    binding = "declared n_trades_total_min" if n_trades_total_min >= wf_floor else "walk-forward geometry floor"
    return {
        "passed": passed,
        "n_events": int(n_events),
        "n_trades_total_min": int(n_trades_total_min),
        "wf_event_floor": int(wf_floor),
        "required_events": required,
        "reason": (
            f"event_floor: {n_events} in-regime events < required {required} "
            f"(binding floor: {binding}). Falsified pre-subprocess on the "
            f"walk-forward refusal cliff — relax regime scope / selectivity, "
            f"move to a denser timeframe, or use the per-episode criterion."
            if not passed else
            f"{n_events} events >= required {required}"
        ),
    }


def _infer_frequency(p: Proposal) -> str:
    """Detect H4 vs D1 from the proposal id (convention: id contains '-H4-' for H4)."""
    if "-H4-" in p.id:
        return "H4"
    return "D1"


def build_regime_mask(p: Proposal, frequency: str | None = None) -> Path:
    if frequency is None:
        frequency = _infer_frequency(p)
    """Write a boolean mask parquet aligned to the regime parquet's bar index."""
    regimes_path = Path("data/regimes") / f"{p.asset}_{frequency.lower()}_regimes.parquet"
    if not regimes_path.exists():
        raise FileNotFoundError(
            f"Regime parquet not found at {regimes_path}. "
            f"Run: uv run python -m pipeline.regimes --asset {p.asset} --frequency {frequency}"
        )
    regimes_df = pd.read_parquet(regimes_path)
    scope = set(p.regime_scope)
    mask_series = regimes_df["regime_id"].isin(scope)
    mask_df = pd.DataFrame({"mask": mask_series}, index=regimes_df.index)
    out_dir = RUNTIME_DIR / "regime_masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{p.id}.parquet"
    mask_df.to_parquet(out_path)
    n_in = int(mask_series.sum())
    n_total = int(len(mask_series))
    print(f"Wrote regime mask {out_path} ({n_in}/{n_total} bars in scope = {n_in/max(n_total,1):.2%})")
    return out_path


def build_transient_config(p: Proposal, regime_mask_path: Path) -> Path:
    """Generate a transient YAML config inheriting the template + overlays."""
    if not TEMPLATE_CONFIG.exists():
        raise FileNotFoundError(f"Template config not found at {TEMPLATE_CONFIG}")
    with TEMPLATE_CONFIG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = RESULTS_PHASE5_DIR / p.id
    cfg["output_dir"] = str(out_dir)
    cfg["primary"]["candidates"] = [p.primary]
    # supervised-direct mode: signal the subprocess that no primary signal is used.
    # _run_one_primary detects this via primary_name == "supervised_direct".
    if p.primary == "supervised_direct":
        cfg["primary"]["mode"] = "supervised_direct"
        cfg["primary"]["supervised_horizon"] = cfg["triple_barrier"]["horizon"]
    cfg["regime_mask_path"] = str(regime_mask_path.resolve())
    # B0015b: propagate primary_feature_blacklist (default empty) so the
    # orchestrator can apply apply_primary_feature_blacklist before passing
    # features to the meta-labeler. See docs/superpowers/specs/2026-05-26-
    # edge-search-scope-decision.md §Precondición.
    cfg["primary_feature_blacklist"] = list(p.primary_feature_blacklist)
    # B0079: wire feature_overrides into the audit meta feature matrix.
    # .drop is applied symmetrically to primary_feature_blacklist (subtract from X).
    # .add is validated against available tier2 features; missing ones are recorded
    # as "not_in_tier2_skipped" in the audit artifact so reviewers have evidence.
    cfg["feature_overrides_add"] = list(p.feature_overrides.add)
    cfg["feature_overrides_drop"] = list(p.feature_overrides.drop)
    cfg["triple_barrier"]["tp_atr_mult"] = p.barrier_geometry_attestation.tp_atr_mult
    cfg["triple_barrier"]["sl_atr_mult"] = p.barrier_geometry_attestation.sl_atr_mult
    # B0155 — ev_breakeven_v1: evaluate the audit at the pre-registered EV
    # breakeven p* instead of the payoff-blind 0.50. The DIAGNOSTIC grid is
    # re-centered at p* (p*, +0.05, +0.10, +0.15) so grid rows exist at
    # exactly p* — the grid stays diagnostic-only (one threshold per audit
    # drives the verdict; the DSR trials count does not widen).
    # `audit_effective_threshold` tells run_backtest to ALSO persist the
    # per-event pnl artifact at p* (strategy_pnl_effective.parquet + sidecar)
    # for the per-episode survival gate. fixed_0.50 proposals take neither
    # branch — their transient config is bit-for-bit the pre-B0155 one.
    if p.threshold_rule == "ev_breakeven_v1":
        p_star = p.effective_threshold()
        # Derived via the canonical grid builder (pipeline.thresholds). The
        # offsets here are the B0155 pre-registered 4-point set and deliberately
        # differ from multi_h4's B0161 5-point DEFAULT_GRID_OFFSETS — each was
        # frozen with its own familywise len(grid) term; do not unify post-hoc.
        cfg["metrics"]["threshold_grid"] = ev_breakeven_grid(
            p.barrier_geometry_attestation.tp_atr_mult,
            p.barrier_geometry_attestation.sl_atr_mult,
            offsets=(0.0, 0.05, 0.10, 0.15),
        )
        cfg["metrics"]["audit_effective_threshold"] = float(p_star)
    # B0147 — GLD real-volume features are GOLD-domain alt-data: the template
    # config (XAU) enables them, but transient configs serve every asset, so
    # the flag is set EXPLICITLY per asset_class here (metals only). Without
    # this override an FX/crypto audit would inherit gold features.
    cfg.setdefault("features", {})["gld_volume"] = (p.asset_class == "metal")
    # Frequency-aware data path + bars_per_year override
    freq = _infer_frequency(p)
    if freq == "H4":
        cfg["data_path"] = f"data/H4/{p.asset}_H4.csv"
        cfg["timeframe"] = "H4"
        cfg["metrics"]["bars_per_year"] = 1512  # 252 * 6
        cfg["triple_barrier"]["horizon"] = 60   # ~10 trading days on H4
        # Regime-gating + 5y H4 dataset narrows events; relax walk-forward
        # to n_folds=2 (max audit class MARGINAL_2FOLDS) so the audit
        # actually runs and produces empirical Sharpe/n_trades numbers
        # rather than rejecting at make_folds.
        cfg["walk_forward"]["n_folds"] = 2
        cfg["walk_forward"]["train_min_bars"] = 50
        cfg["walk_forward"]["purge_bars"] = 5  # H4 inner-CV: purge=40 (D1 default) too big for 165-event train slice
        cfg["hyperparam_search"]["cv_purge_bars"] = 5
        cfg["hyperparam_search"]["cv_splits"] = 2
        # B0032: H4 selective primaries land on small calibration slices.
        # Raise calib_holdout_pct to 0.40 (mirrors D1 phase5_* branch below)
        # so the tail has more class diversity for the inner StratifiedKFold.
        # The cv-cap in pipeline/train.py handles the residual minority-class
        # case; this is the defensive belt to its suspenders.
        cfg["calibration"]["calib_holdout_pct"] = 0.40
    else:
        # FX assets: always use data/D1/ (27-year MT5 pull, more history than
        # the old 5-year data/D1_22y/ stubs).  The regime gate reduces FX
        # in-regime event counts to ~400-500 even for built-in primaries —
        # the XAU 22y default (n_folds=3, train_min_bars=1500, floor=599)
        # rejects them all.  Apply the same n_folds=2 relaxation used for H4
        # and phase5_* D1 so FX audits can actually produce empirical results.
        if p.asset_class == "fx":
            cfg["data_path"] = f"data/D1/{p.asset}_D1.csv"
            cfg["walk_forward"]["n_folds"] = 2
            cfg["walk_forward"]["train_min_bars"] = 200
            cfg["walk_forward"]["purge_bars"] = 10
            cfg["hyperparam_search"]["cv_purge_bars"] = 10
            cfg["hyperparam_search"]["cv_splits"] = 2
            cfg["calibration"]["calib_holdout_pct"] = 0.40
        else:
            # Non-FX D1: prefer 22-year archive; fall back to data/D1/.
            _d1_22y = Path(f"data/D1_22y/{p.asset}_D1.csv")
            cfg["data_path"] = str(_d1_22y) if _d1_22y.exists() else f"data/D1/{p.asset}_D1.csv"
        # D1 phase5_* custom primaries are typically more selective than the
        # 4 built-in primaries (ema_cross etc.) and produce 200-700 events
        # post-regime-gate. The standard n_folds=3 + train_min=500 geometry
        # rejects that range at make_folds (needs ~1100+ events). Apply the
        # same n_folds=2 relaxation as H4 — consistent with proposals'
        # audit_class_in commitment to {STABLE, MARGINAL_2FOLDS}.
        # 4 built-in primaries on the full 22y dataset have ~5000 events and
        # are unaffected by this branch (they don't start with phase5_).
        if p.primary.startswith("phase5_"):
            cfg["walk_forward"]["n_folds"] = 2
            cfg["walk_forward"]["train_min_bars"] = 100
            cfg["walk_forward"]["purge_bars"] = 5
            cfg["hyperparam_search"]["cv_purge_bars"] = 5
            cfg["hyperparam_search"]["cv_splits"] = 2
            # phase5_* primaries on D1 produce 200-700 events; inner-CV slices
            # are ~half of outer-train, calibration tail at default 0.30 can
            # land on a single-class subset (RefittingCalibratedPipeline
            # requires both classes present). Raise calib_holdout to 0.40 to
            # give the tail more class diversity. If this still fails, the
            # proposal is operationally falsified by FOLD_CONSTRUCTION_FAILED
            # equivalent (same path as BEARQ-002 in Phase 5 spike).
            cfg["calibration"]["calib_holdout_pct"] = 0.40
    # Diagnostic-only proposals need relaxed walk-forward
    if p.diagnostic_only:
        cfg["walk_forward"]["n_folds"] = 2
        cfg["walk_forward"]["train_min_bars"] = 20
        cfg["walk_forward"]["purge_bars"] = 2
        cfg["walk_forward"]["embargo_pct"] = 0.0
    # Apply primary_params, creating the primary section if it doesn't exist
    # (template configs only include the primaries they were originally run with;
    # phase5 may invoke any of the 4 standard primaries + phase5_custom).
    #
    # B0085: the hypothesizer emits guessed param key names; _select_primary reads
    # canonical keys. normalize_primary_params resolves unambiguous true-synonyms
    # (threshold_atr_mult -> threshold_atr, ...) and raises PrimaryParamError here
    # — fail fast at build time — if a required canonical key is still missing,
    # instead of dying on an opaque KeyError inside the audit subprocess. Custom
    # phase5_* primaries pass through untouched (they own their signature).
    normalized_params = normalize_primary_params(p.primary, p.primary_params)
    cfg.setdefault("primary", {})
    cfg["primary"].setdefault(p.primary, {})
    for k, v in normalized_params.items():
        cfg["primary"][p.primary][k] = v
    runtime_dir = RUNTIME_DIR / "configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = runtime_dir / f"{p.id}.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote transient config {cfg_path}")
    return cfg_path


def run_pipeline_subprocess(cfg_path: Path, dry_run: bool = False) -> subprocess.CompletedProcess:
    """Invoke scripts/run_backtest.py with the transient config."""
    cmd = [sys.executable, "scripts/run_backtest.py", "--config", str(cfg_path)]
    if dry_run:
        cmd.append("--dry-run")
    print(f"+ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def count_events_subprocess(cfg_path: Path, primary: str) -> dict:
    """B0048 — run only the front-half of the pipeline and parse the emitted
    per-primary event count + walk-forward floor. Raises RuntimeError if the
    subprocess fails or the JSON marker is absent."""
    cmd = [sys.executable, "scripts/run_backtest.py", "--config", str(cfg_path),
           "--preflight-event-count"]
    print(f"+ {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    marker = "PREFLIGHT_EVENT_COUNT_JSON "
    for line in (proc.stdout or "").splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])[primary]
    raise RuntimeError(
        f"preflight event-count subprocess did not emit counts "
        f"(returncode {proc.returncode}): {(proc.stderr or '')[-1000:]}"
    )


def parse_pipeline_results(proposal_id: str, primary: str) -> dict:
    """Locate and parse the per-fold metrics from the pipeline's output."""
    out_dir = RESULTS_PHASE5_DIR / proposal_id / primary
    summary_path = out_dir / "summary.json"
    grid_path = out_dir / "threshold_grid_metrics.json"
    psr_path = out_dir / "psr_dsr.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Pipeline did not produce {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    grid_rows: list[dict] = []
    if grid_path.exists():
        grid_rows = json.loads(grid_path.read_text(encoding="utf-8"))
    psr = {}
    if psr_path.exists():
        psr = json.loads(psr_path.read_text(encoding="utf-8"))
    # B0079: read feature_overrides status written by run_backtest._run_one_primary
    fo_status_path = out_dir / "feature_overrides_status.json"
    fo_status = json.loads(fo_status_path.read_text(encoding="utf-8")) if fo_status_path.exists() else {}
    return {
        "summary": summary,
        "grid_rows": grid_rows,
        "psr_dsr": psr,
        "output_dir": str(out_dir),
        "feature_overrides_status": fo_status,
    }


def aggregate_at_threshold(grid_rows: list[dict], threshold: float) -> dict[str, dict]:
    """Aggregate per-(model, fold) grid rows at a single threshold — same
    conventions as M3. B0155: generalized from the 0.50-only aggregator so an
    ev_breakeven_v1 proposal is evaluated at its pre-registered p* (one
    threshold per audit, exactly as before — the DSR trials count does NOT
    widen)."""
    by_model: dict[str, list[dict]] = {}
    for row in grid_rows:
        if abs(row.get("threshold", 0) - threshold) > 1e-6:
            continue
        model = row.get("model", "?")
        by_model.setdefault(model, []).append(row)
    out: dict[str, dict] = {}
    for model, rows in by_model.items():
        rows_sorted = sorted(rows, key=lambda r: r.get("fold", 0))
        per_fold_sharpe = []
        per_fold_n = []
        for r in rows_sorted:
            sh = r.get("sharpe_net")
            n = r.get("n_trades", 0)
            per_fold_sharpe.append(sh if sh is not None else float("nan"))
            per_fold_n.append(int(n))
        out[model] = {"per_fold_sharpe": per_fold_sharpe, "per_fold_n_trades": per_fold_n,
                      "total_n_trades": sum(per_fold_n)}
    return out


def aggregate_threshold_50(grid_rows: list[dict]) -> dict[str, dict]:
    """Thin backward-compat wrapper: aggregate at the legacy fixed 0.50."""
    return aggregate_at_threshold(grid_rows, 0.50)


def compute_oos_regime_diversity(proposal_id: str, primary: str, asset: str, frequency: str = "D1") -> dict:
    """Compute regime_diversity on the underlying asset's OOS close-price span."""
    out_dir = RESULTS_PHASE5_DIR / proposal_id / primary
    oof_path = out_dir / "oof_predictions.parquet"
    asset_csv = Path("data/D1_22y") / f"{asset}_{frequency}.csv"
    if not asset_csv.exists():
        asset_csv = Path("data") / frequency / f"{asset}_{frequency}.csv"
    if not asset_csv.exists():
        return {"max_dd": 0.0, "max_rally": 0.0, "pass": False, "note": "asset CSV not found"}
    df = pd.read_csv(asset_csv)
    time_col = "time" if "time" in df.columns else "timestamps"
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col).sort_index()
    if oof_path.exists():
        oof = pd.read_parquet(oof_path)
        oos_start = oof.index.min()
        oos_end = oof.index.max()
        span = df.loc[oos_start:oos_end, "close"].values
    else:
        # Fallback: use entire dataset
        span = df["close"].values
    return _regime_diversity(np.asarray(span, dtype=float))


def evaluate_against_falsification(per_fold_n: list[int], per_fold_sharpe: list[float],
                                    regime_pass: bool, criterion: dict,
                                    dsr: float | None = None) -> tuple[str, str]:
    """Run M3 v3 classifier + compare to criterion. Returns (audit_class, verdict).

    B0089 — Deflated Sharpe Ratio HARD GATE. `dsr` is the threshold-0.50
    per-trade DSR of the model under audit (from pipeline.metrics
    .deflated_sharpe_ratio, persisted by scripts/run_backtest.py to psr_dsr.json
    keyed by model name). When `criterion["dsr_min"]` is set (not None), a
    candidate whose DSR < dsr_min — OR whose DSR is unavailable (None) / not
    finite (NaN) — is FALSIFIED and its audit_class is DOWNGRADED to
    NOT_PROFITABLE, EVEN IF its per-fold Sharpe + n_trades clear their floors.
    DSR deflates the observed Sharpe by E[max Sharpe over N trials], so an edge
    that is an artifact of selecting the best of many model/threshold/regime
    configurations cannot promote. This subsumes the external "OOS can't beat
    IS by >30%" heuristic, which is a crude proxy for the same multiple-testing
    inflation DSR measures directly. The gate only ever TIGHTENS the verdict:
    it is checked after the fold criterion passes, so it never rescues an
    already-falsified candidate (which keeps its original M3 class).

    `dsr_min=None` (default) leaves the gate off — old behavior, DSR ignored.
    """
    cls = _classify_transferability(per_fold_n, per_fold_sharpe, regime_pass)
    allowed = set(criterion.get("audit_class_in", DEFAULT_FALSIFICATION["audit_class_in"]))
    if cls not in allowed:
        return cls, "falsified"
    # Sharpe + n checks
    active = [s for s, n in zip(per_fold_sharpe, per_fold_n)
              if n >= 30 and s is not None and np.isfinite(s)]
    median = float(np.nanmedian(active)) if active else float("nan")
    n_total = sum(per_fold_n)
    if not np.isfinite(median) or median < criterion.get("median_active_fold_sharpe_min", 0.5):
        return cls, "falsified"
    if n_total < criterion.get("n_trades_total_min", 50):
        return cls, "falsified"
    # B0089 — DSR hard gate. Applied last so it only tightens a candidate that
    # has already cleared class + Sharpe + n_trades. A missing (None) or
    # non-finite (NaN) DSR is "no measurement" and must NOT promote when the
    # gate is active. Downgrade the class to NOT_PROFITABLE so the failure mode
    # is legible (the raw folds looked profitable; the deflated edge is not).
    dsr_min = criterion.get("dsr_min")
    if dsr_min is not None:
        if dsr is None or not np.isfinite(dsr) or dsr < dsr_min:
            return "NOT_PROFITABLE", "falsified"
    return cls, "survives"


def _proposal_criterion_as_dict(p: Proposal) -> dict:
    # B0149 — the DSR gate is always on. A proposal that omits dsr_min gets the
    # system floor; a proposal that sets one can only raise the bar above it.
    proposal_dsr_min = p.falsification_criterion.dsr_min
    if proposal_dsr_min is None:
        dsr_min = SYSTEM_DSR_MIN
        dsr_gate_source = "system_floor"
    elif float(proposal_dsr_min) >= SYSTEM_DSR_MIN:
        dsr_min = float(proposal_dsr_min)
        dsr_gate_source = "proposal"
    else:
        dsr_min = SYSTEM_DSR_MIN
        dsr_gate_source = "proposal_raised_to_system_floor"
    return {
        "audit_class_in": list(p.falsification_criterion.audit_class_in),
        "median_active_fold_sharpe_min": p.falsification_criterion.median_active_fold_sharpe_min,
        "n_trades_total_min": p.falsification_criterion.n_trades_total_min,
        "per_episode_survival_fraction": p.falsification_criterion.per_episode_survival_fraction,
        "per_episode_min_trades": p.falsification_criterion.per_episode_min_trades,
        "per_episode_net_pnl_margin": p.falsification_criterion.per_episode_net_pnl_margin,
        # B0089 — DSR hard gate floor; B0149 — never None at evaluation time.
        "dsr_min": dsr_min,
        "dsr_gate_source": dsr_gate_source,
    }


def evaluate_per_episode(
    proposal_id: str, primary: str, asset: str, model: str,
    criterion: dict, regime_scope: list[str], frequency: str = "D1",
    effective_threshold: float = 0.50,
) -> dict:
    """B0035 — cross-episode survival sign test.

    An episode "survives" if its net PnL at the audit's EFFECTIVE threshold
    > margin AND it has at least `per_episode_min_trades` trades (active).
    The proposal passes the gate iff the number of surviving episodes >=
    ceil(fraction * n_active), with n_active >= 2 (you cannot assess
    cross-episode robustness on a single active episode).

    B0155 — `effective_threshold` parametrizes which per-event pnl artifact
    is read:
      0.50 (default, fixed_0.50 proposals): the legacy
        strategy_pnl_threshold50.parquet — existing artifacts stay readable.
      otherwise (ev_breakeven_v1 p*): strategy_pnl_effective.parquet, whose
        strategy_pnl_effective.json sidecar must record the SAME threshold
        (tolerance 1e-9) — a stale artifact from a different threshold cannot
        silently drive the gate.
    """
    import math
    frac = criterion.get("per_episode_survival_fraction")
    if frac is None:
        return {"applicable": False}
    min_trades = int(criterion.get("per_episode_min_trades", 5))
    margin = float(criterion.get("per_episode_net_pnl_margin", 0.0))

    out_dir = RESULTS_PHASE5_DIR / proposal_id / primary
    if abs(effective_threshold - 0.50) <= 1e-9:
        pnl_path = out_dir / "strategy_pnl_threshold50.parquet"
        if not pnl_path.exists():
            return {"applicable": True, "passed": False,
                    "reason": f"per-event pnl artifact missing at {pnl_path}"}
    else:
        pnl_path = out_dir / "strategy_pnl_effective.parquet"
        sidecar_path = out_dir / "strategy_pnl_effective.json"
        if not pnl_path.exists() or not sidecar_path.exists():
            return {"applicable": True, "passed": False,
                    "reason": f"per-event pnl artifact missing at {pnl_path} "
                              f"(or its threshold sidecar {sidecar_path.name})"}
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        recorded_thr = float(sidecar.get("threshold", float("nan")))
        if not (abs(recorded_thr - effective_threshold) <= 1e-9):
            return {"applicable": True, "passed": False,
                    "reason": (
                        f"effective-threshold mismatch: sidecar records "
                        f"{recorded_thr!r} but the audit expects "
                        f"{effective_threshold!r} — stale artifact, re-run the pipeline"
                    )}
    pnl_df = pd.read_parquet(pnl_path)
    if model not in pnl_df.columns:
        return {"applicable": True, "passed": False,
                "reason": f"model {model!r} not in {pnl_path.name}"}
    pnl = pnl_df[model]

    regimes_path = REGIMES_DIR / f"{asset}_{frequency.lower()}_regimes.parquet"
    if not regimes_path.exists():
        return {"applicable": True, "passed": False,
                "reason": f"regimes parquet missing at {regimes_path}"}
    from pipeline.regimes import regime_episodes
    regimes_df = pd.read_parquet(regimes_path)
    episodes = regime_episodes(regimes_df["regime_id"])
    episodes = episodes[episodes["regime_id"].isin(set(regime_scope))].reset_index(drop=True)

    per_ep: list[dict] = []
    n_active = 0
    n_survivors = 0
    for _, ep in episodes.iterrows():
        seg = pnl.loc[ep["start_ts"]:ep["end_ts"]]
        n = int(seg.notna().sum())
        cum = float(seg.sum())  # NaN-skipping; 0.0 when the episode has no OOS events
        active = n >= min_trades
        survives = bool(active and cum > margin)
        if active:
            n_active += 1
        if survives:
            n_survivors += 1
        per_ep.append({
            "regime_id": ep["regime_id"], "start_ts": str(ep["start_ts"]),
            "n_trades": n, "net_pnl": cum, "active": active, "survives": survives,
        })

    if n_active < 2:
        return {"applicable": True, "passed": False, "n_active": n_active,
                "n_survivors": n_survivors, "required_survivors": None,
                "reason": "fewer than 2 active episodes — cannot assess cross-episode robustness",
                "per_episode": per_ep}
    # ceil with a float-dust guard so e.g. frac=0.6, n_active=5 → 3 (not 4)
    # and exact-ratio boundaries land where intended.
    required = max(1, math.ceil(frac * n_active - 1e-9))
    return {"applicable": True, "passed": bool(n_survivors >= required),
            "n_active": n_active, "n_survivors": n_survivors,
            "required_survivors": required, "survival_fraction": frac,
            "per_episode": per_ep}


def _is_unmaterialized_custom_primary(p: Proposal) -> bool:
    """True iff `p.primary` is a phase5_* custom primary whose backing signal
    module is not yet on disk (B0040 Option B materialization gate).

    The four built-in primaries (ema_cross, momentum_zscore, cusum_filter,
    bollinger_meanrev) and any phase5_* primary that already has a module at
    pipeline/primaries_phase5/<primary>.py return False — they run normally.
    """
    if not p.primary.startswith("phase5_"):
        return False
    module_file = _REPO_ROOT / "pipeline" / "primaries_phase5" / f"{p.primary}.py"
    return not module_file.exists()


def _infer_asset_from_path(path: Path) -> str | None:
    """Infer the asset symbol from a proposal filename (B0107).

    Expected format: YYYYMMDD-ASSET-...-<id>.json  (asset = second '-' component).
    Returns None when the filename does not match the expected pattern, e.g.
    when a hand-named file is used or the second component is not uppercase letters.
    """
    parts = Path(path).stem.split("-")
    if len(parts) < 2:
        return None
    candidate = parts[1]
    if re.fullmatch(r"[A-Z]{3,8}", candidate):
        return candidate
    return None


def run(proposal_path: Path, dry_run: bool = False, preflight_only: bool = False,
        skip_subprocess: bool = False, asset_override: str | None = None) -> dict:
    """Full chain. Returns the audit_result record.

    `asset_override` (or the value inferred from the proposal filename) is
    injected into the Proposal when the JSON omits `asset` — this allows
    direct re-auditing of asset-blind Loop-A proposals without editing the
    frozen JSON files (B0107).
    """
    _asset = asset_override or _infer_asset_from_path(proposal_path)
    p = load_proposal(proposal_path, asset_override=_asset)
    try:
        p.validate()
    except ProposalValidationError as e:
        return _persist_record(p, status="failed_validation", errors=[str(e)])
    lint_ok, lint_summary = p.run_lookahead_lint()
    if not lint_ok:
        return _persist_record(p, status="failed_lookahead_lint", errors=[lint_summary])

    preflight = check_n_trades_consistency(p)
    if not preflight["passed"]:
        return _persist_record(p, status="preflight_skipped",
                               errors=[preflight["reason"]],
                               extras={"preflight": preflight})
    if preflight_only:
        return _persist_record(p, status="preflight_passed",
                               extras={"preflight": preflight})

    # B0040 (Option B) — custom-primary materialization gate. A phase5_*
    # custom primary needs an importable signal module at
    # pipeline/primaries_phase5/<primary>.py. The hypothesizer emits only
    # `custom_primary_pseudocode` (a STRING), so the module does not exist
    # until a human materializes it. Rather than dispatch a subprocess that
    # crashes on ImportError, park the proposal at `pending_materialization`
    # and surface the pseudocode + the exact module path the human must
    # create. Re-running this command after the module exists IS the
    # promotion step: the gate returns False once the file is on disk and
    # the full audit proceeds.
    if _is_unmaterialized_custom_primary(p):
        return _persist_record(
            p, status="pending_materialization",
            extras={
                "preflight": preflight,
                "materialization": {
                    "expected_module": f"pipeline/primaries_phase5/{p.primary}.py",
                    "entry_point": "signal(ohlcv, features, cfg) -> pd.Series in {-1, 0, +1}",
                    "custom_primary_pseudocode": p.custom_primary_pseudocode,
                    "instructions": (
                        "A human must (1) write the signal module from the pseudocode, "
                        "(2) get it adversarially reviewed for lookahead (custom primary "
                        "code is not auto-trusted), then (3) re-run this same command — "
                        "the audit proceeds automatically once the module exists."
                    ),
                    "promote_command": (
                        f"uv run python -m phase5.run_proposal --proposal {proposal_path}"
                    ),
                },
            },
        )

    # Build mask + config
    try:
        mask_path = build_regime_mask(p)
        cfg_path = build_transient_config(p, mask_path)
    except FileNotFoundError as e:
        return _persist_record(p, status="setup_failed", errors=[str(e)])

    if skip_subprocess:
        return _persist_record(p, status="setup_complete_subprocess_skipped",
                               extras={"regime_mask_path": str(mask_path),
                                       "config_path": str(cfg_path)})

    # B0048 — event-floor gate. Count post-primary in-regime events cheaply
    # (front-half only) and falsify on the walk-forward refusal cliff BEFORE
    # the heavy training subprocess. Converts the opaque subprocess_failed that
    # killed T011D2M (35 events) and T015D2M (73 events, declared min 50) into
    # an honest, pre-subprocess event_floor falsification.
    counts = count_events_subprocess(cfg_path, p.primary)
    floor = evaluate_event_floor(
        n_events=counts["n_events"],
        n_trades_total_min=p.falsification_criterion.n_trades_total_min,
        wf_floor=counts["wf_event_floor"],
    )
    if not floor["passed"]:
        return _persist_record(p, status="event_floor",
                               errors=[floor["reason"]],
                               extras={"preflight": preflight,
                                       "event_floor": {**floor, **counts}})

    # Subprocess (heavy)
    proc = run_pipeline_subprocess(cfg_path, dry_run=dry_run)
    if proc.returncode != 0:
        # B0156: keep head AND tail — the terminal exception is at the END of
        # stderr; the old [:2000] head-only cap stored warnings and hid it.
        err_txt = proc.stderr or ""
        if len(err_txt) > 10000:
            err_txt = (err_txt[:2000] + "\n...[stderr truncated]...\n"
                       + err_txt[-8000:])
        return _persist_record(p, status="subprocess_failed",
                               errors=[f"return code {proc.returncode}",
                                       err_txt])

    # Parse + audit
    try:
        results = parse_pipeline_results(p.id, p.primary)
    except FileNotFoundError as e:
        return _persist_record(p, status="result_parse_failed", errors=[str(e)])
    # B0155 — evaluate at the proposal's pre-registered effective threshold:
    # 0.50 for fixed_0.50 (legacy behavior), p* for ev_breakeven_v1.
    effective_threshold = p.effective_threshold()
    per_model = aggregate_at_threshold(results["grid_rows"], effective_threshold)
    regime = compute_oos_regime_diversity(p.id, p.primary, p.asset)

    # B0089 — per-model DSR keyed by model name, as persisted by
    # scripts/run_backtest.py to psr_dsr.json ({"dsr": {model: float}}). Threaded
    # into evaluate_against_falsification so the DSR hard gate is load-bearing
    # in the promotion decision, not just in deployment-tier sizing.
    dsr_by_model: dict[str, float] = (results.get("psr_dsr") or {}).get("dsr", {}) or {}

    per_model_audit: dict[str, dict] = {}
    survives_any = False
    criterion_dict = _proposal_criterion_as_dict(p)
    for model, agg in per_model.items():
        model_dsr = dsr_by_model.get(model)
        audit_class, verdict = evaluate_against_falsification(
            agg["per_fold_n_trades"],
            agg["per_fold_sharpe"],
            regime.get("pass"),
            criterion_dict,
            dsr=model_dsr,
        )
        # B0089 — record the DSR + the gate decision for this model.
        # B0149 — the gate is ALWAYS active now (system floor when the proposal
        # omitted dsr_min); the None branch survives only as a defensive guard.
        dsr_min = criterion_dict.get("dsr_min")
        if dsr_min is None:
            dsr_gate_passed = None
        else:
            dsr_gate_passed = bool(
                model_dsr is not None and np.isfinite(model_dsr) and model_dsr >= dsr_min
            )
        # B0035 — cross-episode survival gate. Only applies to models that
        # already clear the per-fold criterion (no point checking episodes for
        # an already-falsified model) and only when the proposal opted in.
        per_episode_result = None
        if verdict == "survives" and p.falsification_criterion.per_episode_survival_fraction is not None:
            per_episode_result = evaluate_per_episode(
                p.id, p.primary, p.asset, model, criterion_dict,
                list(p.regime_scope), _infer_frequency(p),
                effective_threshold=effective_threshold,
            )
            if not per_episode_result.get("passed"):
                verdict = "falsified_per_episode"
        per_model_audit[model] = {
            **agg,
            "audit_class": audit_class,
            "falsification_verdict": verdict,
            "per_episode": per_episode_result,
            # B0089 — DSR + gate decision recorded per model.
            "dsr": (float(model_dsr) if model_dsr is not None else None),
            "dsr_min": dsr_min,
            "dsr_gate_passed": dsr_gate_passed,
            # B0149 — where the floor came from: proposal | system_floor |
            # proposal_raised_to_system_floor.
            "dsr_gate_source": criterion_dict.get("dsr_gate_source"),
        }
        if verdict == "survives":
            survives_any = True

    # Diagnostic-only: even if any model survives, never PROMOTE
    overall_verdict = "survives" if (survives_any and not p.diagnostic_only) else (
        "diagnostic_pass" if (survives_any and p.diagnostic_only) else "falsified"
    )

    return _persist_record(
        p, status="completed",
        extras={
            "per_model_audit": per_model_audit,
            "regime_diversity": regime,
            "overall_verdict": overall_verdict,
            "diagnostic_only": p.diagnostic_only,
            "preflight": preflight,
            "regime_mask_path": str(mask_path),
            "config_path": str(cfg_path),
            "results_dir": results["output_dir"],
            # B0079: evidence that feature_overrides.add/drop were applied.
            "feature_overrides_status": results.get("feature_overrides_status", {}),
        }
    )


def _persist_record(p: Proposal, status: str, errors: list[str] | None = None,
                    extras: dict | None = None) -> dict:
    AUDIT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # B0155 — every audit record carries the threshold policy: the rule, the
    # effective threshold the verdict was (or would have been) evaluated at,
    # and — for ev_breakeven_v1 — the inputs p* was derived from. The global
    # constants are recorded so a future constant amendment cannot silently
    # reinterpret an old record.
    from phase5.proposal import C_ATR, LAMBDA_MARGIN
    try:
        _eff_thr = float(p.effective_threshold())
    except Exception:  # defensive: never let a malformed attestation block persistence
        _eff_thr = None
    if p.threshold_rule == "ev_breakeven_v1":
        _thr_inputs = {
            "tp_atr_mult": float(p.barrier_geometry_attestation.tp_atr_mult),
            "sl_atr_mult": float(p.barrier_geometry_attestation.sl_atr_mult),
            "C_ATR": C_ATR,
            "LAMBDA_MARGIN": LAMBDA_MARGIN,
        }
    else:
        _thr_inputs = None
    record = {
        "proposal_id": p.id,
        "asset": p.asset,
        "regime_scope": list(p.regime_scope),
        "primary": p.primary,
        "status": status,
        "errors": errors or [],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "threshold_rule": p.threshold_rule,
        "effective_threshold": _eff_thr,
        "threshold_inputs": _thr_inputs,
        **(extras or {}),
    }
    out_path = AUDIT_RESULTS_DIR / f"{p.id}.json"
    out_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path} (status={status})")
    return record


# --------------------------------------------------------------------------- #
# B0010 — pooled-universe audit mode (Task 4). The single-name path above
# regime-masks and audits ONE asset. This path audits a proposal against the
# M3 many-ticker cross-sectional pool (configs/equity_m3_d1.yaml +
# scripts/run_pooled_equity_d1.py -> scripts/run_multi_h4.py::_run_one_pool)
# instead. grading_version "pooled_v1" NEVER auto-promotes: the status is
# always "event_floor" or "completed_pending_human_read" — a human (or the
# skeptic agent) reads the artifacts before any promotion decision.
# --------------------------------------------------------------------------- #

POOLED_TEMPLATE_CONFIG = Path("configs/equity_m3_d1.yaml")
POOLED_RESULTS_DIR = Path("results/phase5_pooled")
# B0014 horizon curve (2026-07-04): pooled rho=1 effective-N is bounded by
# calendar-span/event-duration — h40→654, h20→716, h10→905+ vs floor 799.
# 10 is the only D1 horizon measured to clear the pre-flight floor.
POOLED_AUDIT_HORIZON = 10


def build_transient_pooled_config(p: Proposal) -> Path:
    """Overlay a Proposal onto the M3 pooled-universe template.

    Mirrors build_transient_config's overlay pattern but targets
    scripts/run_pooled_equity_d1.py's config contract (top-level
    `regime_scope` / `feature_overrides_add` / `feature_overrides_drop`,
    `features.cross_sectional`, `primary.candidates` + per-primary param
    block) rather than the single-name regime-mask-parquet path.

    `horizon` is overlaid to POOLED_AUDIT_HORIZON (10). The template's 40
    caps pooled effective-N at ~650 < floor 799 regardless of breadth or
    regime dispersion (B0014 horizon curve: h40→654, h20→716, h10→905+;
    T005 died at 658.3 under h40 after attesting h10-compatible geometry).
    Every pooled audit therefore runs on the one horizon empirically able
    to clear the pre-flight floor — proposals attest tp/sl only.
    """
    if not POOLED_TEMPLATE_CONFIG.exists():
        raise FileNotFoundError(f"Pooled template config not found at {POOLED_TEMPLATE_CONFIG}")
    with POOLED_TEMPLATE_CONFIG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["primary"]["candidates"] = [p.primary]
    # B0085 parity with the single-name path: normalize aliases / fail fast on
    # a missing canonical key at build time rather than an opaque KeyError
    # inside the pooled subprocess.
    normalized_params = normalize_primary_params(p.primary, p.primary_params)
    cfg["primary"][p.primary] = dict(normalized_params)

    cfg["triple_barrier"]["tp_atr_mult"] = p.barrier_geometry_attestation.tp_atr_mult
    cfg["triple_barrier"]["sl_atr_mult"] = p.barrier_geometry_attestation.sl_atr_mult
    cfg["triple_barrier"]["horizon"] = POOLED_AUDIT_HORIZON

    cfg["regime_scope"] = list(p.regime_scope)
    # B0014: carry the proposal's gate mode into the pooled runner (which
    # fail-louds on modes it does not implement, e.g. weight_events).
    cfg["regime_gate_mode"] = p.regime_gate.mode
    cfg["feature_overrides_add"] = list(p.feature_overrides.add)
    cfg["feature_overrides_drop"] = list(p.feature_overrides.drop)

    cfg.setdefault("features", {})
    cfg["features"]["cross_sectional"] = True
    cfg["features"]["gld_volume"] = False
    # B0017: switch the PIT earnings-calendar join on when the proposal needs
    # it (the earnings primary, or any requested calendar meta-feature).
    from pipeline.earnings_events import EARNINGS_CALENDAR_FEATURES
    cfg["features"]["earnings_calendar"] = (
        p.primary == "phase5_earnings_premium"
        or any(f in EARNINGS_CALENDAR_FEATURES for f in p.feature_overrides.add)
    )

    # Evaluate every (asset, fold, model) cell at the SAME pre-registered
    # audit threshold (fixed_0.50 -> 0.50, ev_breakeven_v1 -> p*) instead of
    # per-fold inner-CV selection, so metrics_per_fold.json rows are directly
    # comparable across the pool for the pooled_v1 grading below. Reuses the
    # same effective_threshold() helper the single-name path uses.
    cfg.setdefault("threshold_selection", {})["method"] = "fixed_ev"
    cfg.setdefault("metrics", {})
    cfg["metrics"]["audit_effective_threshold"] = float(p.effective_threshold())

    out_dir = POOLED_RESULTS_DIR / p.id
    cfg["output_dir"] = str(out_dir)

    runtime_dir = RUNTIME_DIR / "configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = runtime_dir / f"{p.id}_pooled.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote transient pooled config {cfg_path}")
    return cfg_path


def count_pooled_events_subprocess(cfg_path: Path) -> list[dict]:
    """Run scripts/run_pooled_equity_d1.py --count-events-only and parse the
    member_event_counts.json it writes at <output_dir>/member_event_counts.json.

    Raises RuntimeError if the subprocess fails or the file is absent — the
    caller (run_pooled_audit) turns that into a `subprocess_failed` record.
    """
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["output_dir"])
    cmd = [sys.executable, "scripts/run_pooled_equity_d1.py", "--config", str(cfg_path),
           "--count-events-only"]
    print(f"+ {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    counts_path = out_dir / "member_event_counts.json"
    if proc.returncode != 0 or not counts_path.exists():
        raise RuntimeError(
            f"pooled count-events-only subprocess did not produce {counts_path} "
            f"(returncode {proc.returncode}): {(proc.stderr or '')[-1000:]}"
        )
    return json.loads(counts_path.read_text(encoding="utf-8"))


def effective_n_pooled_subprocess(cfg_path: Path, primary: str) -> dict:
    """Run scripts/run_pooled_equity_d1.py --effective-n-only and parse the
    effective_n_<primary>.json it writes at the config's output_dir (B0011
    persistence). Cheap (~2 min: panel load + weights, no model fitting).

    B0013: the binding pooled floor quantity is the concurrency-adjusted
    effective-N, not the raw event count — regime-gated pools concentrate
    events onto the same stress days and collapse an order of magnitude
    below raw (T004D1: raw 2,852 vs effective 521.7 < floor 799).

    Raises RuntimeError on subprocess failure or a missing/inconsistent
    artifact — the caller turns that into `subprocess_failed` rather than
    silently degrading to the raw gate.
    """
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["output_dir"])
    cmd = [sys.executable, "scripts/run_pooled_equity_d1.py", "--config", str(cfg_path),
           "--effective-n-only"]
    print(f"+ {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    eff_path = out_dir / f"effective_n_{primary}.json"
    if proc.returncode != 0 or not eff_path.exists():
        raise RuntimeError(
            f"pooled effective-n-only subprocess did not produce {eff_path} "
            f"(returncode {proc.returncode}): {(proc.stderr or '')[-1000:]}"
        )
    diag = json.loads(eff_path.read_text(encoding="utf-8"))
    if "effective_n_rho1" not in diag:
        raise RuntimeError(f"{eff_path} lacks 'effective_n_rho1' — refusing to gate on raw N")
    return diag


def run_pooled_pipeline_subprocess(cfg_path: Path, dry_run: bool = False) -> subprocess.CompletedProcess:
    """Invoke scripts/run_pooled_equity_d1.py for the full pooled run."""
    cmd = [sys.executable, "scripts/run_pooled_equity_d1.py", "--config", str(cfg_path)]
    if dry_run:
        cmd.append("--dry-run")
    print(f"+ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def run_long_short_split_subprocess(out_dir: Path, cost_bps: float) -> subprocess.CompletedProcess:
    """Invoke scripts/report_long_short_split.py against the pooled output dir."""
    cmd = [sys.executable, "scripts/report_long_short_split.py",
           "--results", str(out_dir), "--cost-bps", str(cost_bps)]
    print(f"+ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _read_asset_best_model(asset_dir: Path) -> str | None:
    summary_path = asset_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary.get("best_model")


def grade_pooled_v1(assets: list[str], primary: str, output_dir: Path) -> dict:
    """v1 pooled grading (B0010).

    For each asset, reads `<output_dir>/<asset>/<primary>/summary.json` for
    the per-asset `best_model` (the existing T13 selector, already computed by
    _run_one_pool) and that same directory's `metrics_per_fold.json` for the
    best model's per-fold `sharpe_net` / `n_trades` — evaluated at the audit's
    fixed effective threshold because build_transient_pooled_config sets
    `threshold_selection.method="fixed_ev"`.

    - `n_trades_total`: sum of `n_trades` across every (asset, fold)
      best-model cell.
    - `median_active_fold_sharpe`: `np.nanmedian` over per-(asset, fold)
      `sharpe_net` where `n_trades >= 30` — the project's NaN-not-zero Sharpe
      floor. Cells below 30 trades are EXCLUDED from the input list, never
      coerced to 0 and never NaN-filled into the median.
    - `breadth`: count of assets whose best-model total `n_trades >= 30` AND
      whose own aggregate sharpe (`nanmedian` over that asset's active folds)
      is finite and positive.

    An asset missing `summary.json` / `metrics_per_fold.json` is recorded with
    `n_trades_total=0`, `breadth_pass=False` rather than raising — a partial
    pooled run still grades on what is present, and the per-asset breakdown
    surfaces the missing member for the reviewer.
    """
    per_asset: dict[str, dict] = {}
    all_active_sharpes: list[float] = []
    n_trades_total = 0
    breadth = 0

    for asset in assets:
        asset_dir = Path(output_dir) / asset / primary
        best_model = _read_asset_best_model(asset_dir)
        mpf_path = asset_dir / "metrics_per_fold.json"
        if best_model is None or not mpf_path.exists():
            per_asset[asset] = {
                "best_model": best_model, "n_trades_total": 0,
                "aggregate_sharpe": None, "breadth_pass": False,
                "note": "missing summary.json or metrics_per_fold.json",
            }
            continue

        rows = json.loads(mpf_path.read_text(encoding="utf-8"))
        model_rows = [r for r in rows if r.get("model") == best_model]
        fold_n = [int(r.get("n_trades", 0) or 0) for r in model_rows]
        fold_sharpe = [r.get("sharpe_net") for r in model_rows]

        asset_n_total = sum(fold_n)
        n_trades_total += asset_n_total

        asset_active = [
            float(s) for s, n in zip(fold_sharpe, fold_n)
            if n >= 30 and s is not None and np.isfinite(s)
        ]
        all_active_sharpes.extend(asset_active)
        asset_agg_sharpe = float(np.nanmedian(asset_active)) if asset_active else float("nan")
        breadth_pass = bool(
            asset_n_total >= 30 and np.isfinite(asset_agg_sharpe) and asset_agg_sharpe > 0
        )
        if breadth_pass:
            breadth += 1
        per_asset[asset] = {
            "best_model": best_model,
            "n_trades_total": int(asset_n_total),
            "aggregate_sharpe": asset_agg_sharpe if np.isfinite(asset_agg_sharpe) else None,
            "breadth_pass": breadth_pass,
        }

    median_active_fold_sharpe = (
        float(np.nanmedian(all_active_sharpes)) if all_active_sharpes else float("nan")
    )
    return {
        "n_assets": len(assets),
        "n_trades_total": int(n_trades_total),
        "median_active_fold_sharpe": (
            median_active_fold_sharpe if np.isfinite(median_active_fold_sharpe) else None
        ),
        "breadth": breadth,
        "per_asset": per_asset,
    }


def _criterion_eval_pooled_v1(p: Proposal, grading: dict) -> dict:
    """Evaluate falsification_criterion keys against pooled_v1 grading output.

    `audit_class_in` / `per_episode_survival_fraction` / `dsr_min` have no
    pooled-universe analog (they are per-single-name classifier / episode /
    DSR concepts) and are recorded as the string "not_applicable_pooled_v1"
    rather than silently skipped or forced to pass — a reviewer must read
    them explicitly. This function only annotates the record; it never
    changes run_pooled_audit's status (always event_floor or
    completed_pending_human_read — no auto-promotion).
    """
    fc = p.falsification_criterion
    med = grading["median_active_fold_sharpe"]
    out: dict[str, Any] = {}
    if med is None:
        out["median_active_fold_sharpe_min"] = {
            "computed": None, "threshold": fc.median_active_fold_sharpe_min,
            "passed": False,
            "note": "no active (n_trades>=30) fold anywhere in the pool",
        }
    else:
        out["median_active_fold_sharpe_min"] = {
            "computed": med, "threshold": fc.median_active_fold_sharpe_min,
            "passed": bool(med >= fc.median_active_fold_sharpe_min),
        }
    out["n_trades_total_min"] = {
        "computed": grading["n_trades_total"], "threshold": fc.n_trades_total_min,
        "passed": bool(grading["n_trades_total"] >= fc.n_trades_total_min),
    }
    for key in ("audit_class_in", "per_episode_survival_fraction", "dsr_min"):
        out[key] = "not_applicable_pooled_v1"
    return out


def run_pooled_audit(p: Proposal, dry_run: bool = False) -> dict:
    """Full pooled-universe audit chain (B0010).

    (0) Scope guard — pooled_universe mode v1 supports built-in primaries
        with an empty primary_feature_blacklist only. The pooled runner
        (scripts/run_pooled_equity_d1.py -> _run_one_pool) does not call
        apply_primary_feature_blacklist and has no B0015b input-disjointness
        check, so a phase5_* custom primary or a non-empty blacklist would
        silently skip enforcement the single-name path guarantees. Refused
        fail-loud, before any subprocess, as a persisted "failed_validation"
        record — consistent with how p.validate() failures are reported
        elsewhere in this module.
    (1) count-events-only subprocess -> member_event_counts.json -> pooled
        event-floor check (sum of in-scope events across the pool vs
        wf_event_floor(n_folds, train_min_bars)); starved -> status
        "event_floor" (same record shape as the single-name path, plus
        "mode": "pooled_universe" and per-member counts), no full run
        attempted.
    (2) Otherwise: full pooled run, then report_long_short_split, then the
        pooled_v1 grading. Status is always "completed_pending_human_read" —
        this NEVER auto-promotes.
    """
    blocked_reason = None
    if getattr(p, "primary_feature_blacklist", []):
        blocked_reason = (
            "pooled_universe mode does not apply primary_feature_blacklist — "
            "route blacklisted proposals through the single-name path."
        )
    elif p.primary.startswith("phase5_"):
        # B0017: a phase5_* custom primary is pooled-safe iff it declares
        # INPUT_COLUMNS == () (reads only ohlcv / its own event cache) — then
        # the B0015b input-disjointness concern is vacuous by construction.
        # Anything else keeps the original fail-loud refusal.
        try:
            import importlib
            mod = importlib.import_module(f"pipeline.primaries_phase5.{p.primary}")
            input_cols = tuple(getattr(mod, "INPUT_COLUMNS", ("<undeclared>",)))
        except ImportError as e:
            blocked_reason = f"custom primary module not importable: {e}"
        else:
            if input_cols != ():
                blocked_reason = (
                    f"pooled_universe supports phase5_* primaries only with "
                    f"INPUT_COLUMNS == () (feature-independent); {p.primary} "
                    f"declares {input_cols!r} and the pooled runner performs "
                    f"no input-disjointness check (B0015b)."
                )
    if blocked_reason:
        return _persist_record(
            p, status="failed_validation",
            errors=[blocked_reason],
            extras={"mode": "pooled_universe"},
        )

    cfg_path = build_transient_pooled_config(p)
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["output_dir"])

    try:
        counts = count_pooled_events_subprocess(cfg_path)
    except RuntimeError as e:
        return _persist_record(
            p, status="subprocess_failed", errors=[str(e)],
            extras={"mode": "pooled_universe", "config_path": str(cfg_path)},
        )

    primary_counts = [c for c in counts if c.get("primary") == p.primary]
    total_events = sum(int(c.get("n_events", 0)) for c in primary_counts)
    floor = wf_event_floor(cfg["walk_forward"]["n_folds"], cfg["walk_forward"]["train_min_bars"])
    if total_events < floor:
        return _persist_record(
            p, status="event_floor",
            errors=[
                f"pooled event_floor: {total_events} pooled in-scope events < "
                f"wf_event_floor {floor} (n_folds={cfg['walk_forward']['n_folds']}, "
                f"train_min_bars={cfg['walk_forward']['train_min_bars']})"
            ],
            extras={
                "mode": "pooled_universe",
                "config_path": str(cfg_path),
                "member_event_counts": primary_counts,
                "pooled_event_floor": {"total_events": total_events, "wf_event_floor": floor},
            },
        )

    # B0013: raw counts clearing the floor is necessary but NOT sufficient —
    # regime-gated pools pile events onto the same days and the rho=1
    # effective-N (the quantity make_folds actually starves on) can sit far
    # below raw. Measure it cheaply (no fitting) and gate BEFORE the full run.
    try:
        eff = effective_n_pooled_subprocess(cfg_path, p.primary)
    except RuntimeError as e:
        return _persist_record(
            p, status="subprocess_failed", errors=[str(e)],
            extras={"mode": "pooled_universe", "config_path": str(cfg_path),
                    "member_event_counts": primary_counts},
        )
    effective_n = float(eff["effective_n_rho1"])
    if effective_n < floor:
        return _persist_record(
            p, status="event_floor",
            errors=[
                f"pooled effective-N floor: effective_n_rho1 {effective_n:.1f} < "
                f"wf_event_floor {floor} despite {total_events} raw in-scope events "
                f"(concurrency collapse — events concentrated on few distinct days). "
                f"Relax the regime gate (e.g. add_feature mode), widen regime scope, "
                f"or shorten barrier durations."
            ],
            extras={
                "mode": "pooled_universe",
                "config_path": str(cfg_path),
                "member_event_counts": primary_counts,
                "pooled_effective_n": {**eff, "wf_event_floor": floor},
            },
        )

    proc = run_pooled_pipeline_subprocess(cfg_path, dry_run=dry_run)
    if proc.returncode != 0:
        err_txt = proc.stderr or ""
        if len(err_txt) > 10000:
            err_txt = err_txt[:2000] + "\n...[stderr truncated]...\n" + err_txt[-8000:]
        return _persist_record(
            p, status="subprocess_failed",
            errors=[f"pooled run return code {proc.returncode}", err_txt],
            extras={"mode": "pooled_universe", "config_path": str(cfg_path),
                    "member_event_counts": primary_counts},
        )

    ls_proc = run_long_short_split_subprocess(out_dir, float(cfg["metrics"]["cost_per_trade_bps"]))
    ls_path = out_dir / "long_short_split.json"
    if ls_proc.returncode == 0 and ls_path.exists():
        long_short = json.loads(ls_path.read_text(encoding="utf-8"))
    else:
        long_short = {
            "error": f"report_long_short_split.py failed (returncode {ls_proc.returncode})",
            "stderr": (ls_proc.stderr or "")[-2000:],
        }

    assets = sorted({c["asset"] for c in primary_counts})
    grading = grade_pooled_v1(assets, p.primary, out_dir)
    criterion_eval = _criterion_eval_pooled_v1(p, grading)

    return _persist_record(
        p, status="completed_pending_human_read",
        extras={
            "mode": "pooled_universe",
            "grading_version": "pooled_v1",
            "config_path": str(cfg_path),
            "results_dir": str(out_dir),
            "member_event_counts": primary_counts,
            "pooled_effective_n": {**eff, "wf_event_floor": floor},
            "grading": grading,
            "long_short": long_short,
            "criterion_eval": criterion_eval,
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposal", required=True)
    ap.add_argument("--dry-run", action="store_true", help="pass --dry-run to the pipeline subprocess")
    ap.add_argument("--preflight-only", action="store_true",
                    help="run validation + n_trades pre-flight check only; don't run pipeline")
    ap.add_argument("--skip-subprocess", action="store_true",
                    help="build mask + config but don't actually run the pipeline (smoke test)")
    ap.add_argument("--asset", default=None,
                    help="asset symbol to inject when the proposal JSON omits 'asset' "
                         "(e.g. XAUUSD for early T-series asset-blind proposals). "
                         "Inferred from the filename automatically when not provided.")
    ap.add_argument("--pooled-universe", action="store_true",
                    help="B0010: audit against the M3 pooled cross-sectional universe "
                         "(scripts/run_pooled_equity_d1.py) instead of the single-name "
                         "regime-mask path. Never auto-promotes (grading_version "
                         "pooled_v1) — status stops at event_floor or "
                         "completed_pending_human_read for human/skeptic review.")
    args = ap.parse_args()

    if args.pooled_universe:
        proposal_path = Path(args.proposal)
        _asset = args.asset or _infer_asset_from_path(proposal_path)
        p = load_proposal(proposal_path, asset_override=_asset)
        try:
            p.validate()
        except ProposalValidationError as e:
            record = _persist_record(p, status="failed_validation", errors=[str(e)],
                                     extras={"mode": "pooled_universe"})
            return 1
        lint_ok, lint_summary = p.run_lookahead_lint()
        if not lint_ok:
            record = _persist_record(p, status="failed_lookahead_lint", errors=[lint_summary],
                                     extras={"mode": "pooled_universe"})
            return 1
        record = run_pooled_audit(p, dry_run=args.dry_run)
        return 0 if record["status"] in ("completed_pending_human_read", "event_floor") else 1

    record = run(Path(args.proposal), dry_run=args.dry_run,
                 preflight_only=args.preflight_only,
                 skip_subprocess=args.skip_subprocess,
                 asset_override=args.asset)
    # pending_materialization (B0040) is a valid waiting state, not an error —
    # the proposal passed validation + lint + pre-flight and is awaiting a
    # human-authored signal module. Exit 0 so callers don't treat it as a crash.
    # event_floor (B0048) is an honest falsification — the system correctly
    # determined the primary is walk-forward-infeasible — not a crash; exit 0
    # just like a `completed` audit that returns a non-survival verdict.
    return 0 if record["status"] in (
        "completed", "preflight_passed", "setup_complete_subprocess_skipped",
        "pending_materialization", "event_floor",
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
