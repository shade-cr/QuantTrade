"""Run the XAU D1 meta-labeling pipeline end-to-end.

Two modes:
  --dry-run    Hyperparam search uses n_iter=5 on one fold; reports timing only.
  (default)    Full run as specified in the YAML config.
"""
from __future__ import annotations
import sys
from pathlib import Path as _Path

# Ensure the project root (parent of scripts/) is on sys.path so that
# `pipeline.*` imports work regardless of how the script is invoked.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import RandomizedSearchCV

from pipeline.data import load_dataset
from pipeline.macro_fetch import build_macro_frame
from pipeline.features import (
    build_tier2_features,
    build_technical_features,
    apply_primary_feature_blacklist,
    feature_add_status,
)
from pipeline.primaries_phase5 import assert_primary_inputs_disjoint
from pipeline.labels import (
    bollinger_meanrev_signal,
    cusum_filter_signal,
    ema_crossover_signal,
    momentum_zscore_signal,
    triple_barrier_labels,
    compute_primary_state,
)
from pipeline.sample_weights import avg_uniqueness
from pipeline.walk_forward import make_folds, PurgedTimeSeriesSplit, inner_oof_predict_proba, resolve_train_min, wf_event_floor
from pipeline.train import MODEL_FACTORIES, fit_calibrated, RefittingCalibratedPipeline
from pipeline.metrics import (
    classification_metrics,
    strategy_metrics,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    aggregate_per_trade_pnl_metrics,
)
from pipeline.feature_importance import (
    mda_importance,
    aggregate_mda_across_folds,
    neg_log_loss_scorer,
    neg_log_loss_estimator_scorer,
    cluster_features,
    clustered_mda_importance,
    aggregate_clustered_mda_across_folds,
)
from pipeline.threshold_selection import select_threshold_inner_cv
from pipeline.friction import resolve_cost_bps
from sklearn.metrics import accuracy_score
from pipeline.stack import should_stack, fit_meta_nested_wf
from pipeline.reporting import (
    write_summary_json,
    write_oof_parquet,
    plot_calibration,
    plot_equity,
    write_report_md,
    compute_benchmark_correlations,
    compute_shutoff_status,
)


HP_SPACES = {
    "xgb": {"max_depth": [3, 5, 7], "learning_rate": [0.03, 0.1],
            "n_estimators": [200, 500]},
    "catboost": {"depth": [4, 6, 8], "learning_rate": [0.03, 0.1],
                 "iterations": [300, 600], "l2_leaf_reg": [3, 7]},
    # B0053: lgbm replaces catboost in active config. leaf-wise growth gives
    # genuine diversity vs xgb. num_leaves controls model capacity (31=default,
    # 63=richer); min_child_samples regularizes sparse-event folds.
    "lgbm": {"num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.05, 0.1],
             "n_estimators": [200, 400], "min_child_samples": [10, 20, 40],
             "reg_lambda": [0.5, 1.0, 5.0]},
    "rf": {"n_estimators": [300, 600], "max_depth": [5, 10, None],
           "min_samples_leaf": [5, 20], "max_features": ["sqrt", 0.5]},
    # B0053: linear base learner — C controls L2 strength; stronger reg (lower C)
    # appropriate for high-dimensional tier2 features with sparse events.
    "lr": {"C": [0.01, 0.1, 1.0], "max_iter": [500, 1000]},
}


def _select_primary(name: str, ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: dict) -> pd.Series:
    # Phase 5 custom primaries: name starts with "phase5_<id>" and the
    # corresponding module lives at pipeline/primaries_phase5/<id>.py with
    # a `signal(ohlcv, features, cfg) -> pd.Series` entry point. Backward
    # compatible — no change to existing primaries.
    if name.startswith("phase5_"):
        import importlib
        mod_path = f"pipeline.primaries_phase5.{name}"
        try:
            mod = importlib.import_module(mod_path)
        except ImportError as e:
            raise ValueError(f"Phase 5 custom primary {name!r} not found at {mod_path}: {e}")
        if not hasattr(mod, "signal"):
            raise ValueError(f"{mod_path} missing 'signal(ohlcv, features, cfg) -> pd.Series'")
        return mod.signal(ohlcv, features, cfg)
    p = cfg["primary"]
    if name == "ema_cross":
        return ema_crossover_signal(
            ohlcv["close"], features["_atr_14"],
            fast=p["ema_cross"]["fast"], slow=p["ema_cross"]["slow"],
            dead_zone_atr=p["ema_cross"]["dead_zone_atr"],
        )
    if name == "momentum_zscore":
        return momentum_zscore_signal(
            ohlcv["close"], lookback=p["momentum_zscore"]["lookback"],
            threshold=p["momentum_zscore"]["threshold"],
        )
    if name == "cusum_filter":
        return cusum_filter_signal(
            ohlcv["close"], features["_atr_14"],
            threshold_atr=p["cusum_filter"]["threshold_atr"],
        )
    if name == "bollinger_meanrev":
        return bollinger_meanrev_signal(
            ohlcv["close"],
            period=p["bollinger_meanrev"]["period"],
            k_stdev=p["bollinger_meanrev"]["k_stdev"],
        )
    raise ValueError(f"Unknown primary: {name}")


def _run_supervised_direct(
    cfg: dict,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    dry_run: bool,
    count_events_only: bool = False,
):
    """Supervised-direct mode: bypass the primary signal entirely.

    Events = ALL in-scope (regime-gated) bars with side=+1 (long-only).
    The X matrix is built from Tier 2 features (minus blacklist / overrides)
    with NO primary_side / primary_strength / bars_since_signal columns added —
    those columns encode primary signal state which does not exist here.

    walk_forward purge/embargo is applied normally (same folds logic). This is
    inherently causal: the labeling horizon is forward in time and the feature
    window uses only closed bars. The purge/embargo bypass that applies to the
    live loop (see live_loop.py docstring) does NOT apply here — this function
    is part of the backtest path and must fully honour the WF invariants.

    The non-count path calls _run_folds_and_report (Task 2 wiring complete).
    """
    primary_name = "supervised_direct"
    out_dir = Path(cfg["output_dir"]) / primary_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the post-blacklist meta feature matrix (drop _atr_14 internal col).
    blacklist = cfg.get("primary_feature_blacklist", []) or []
    fo_drop = cfg.get("feature_overrides_drop", []) or []
    fo_add  = cfg.get("feature_overrides_add",  []) or []
    combined_drop = list(dict.fromkeys(list(blacklist) + list(fo_drop)))
    meta_features_full = apply_primary_feature_blacklist(
        features.drop(columns=["_atr_14"], errors="ignore"), combined_drop,
    )

    # Validate add requests against the post-drop column set; record for audit
    # (B0079 convention — same as labeled mode so run_proposal sees consistent
    # evidence). B0149: conceptual requests ("volume") may be satisfied by
    # derived tier2 columns via pipeline.features.FEATURE_ALIASES.
    fo_add_status = feature_add_status(fo_add, set(meta_features_full.columns))
    (out_dir / "feature_overrides_status.json").write_text(
        json.dumps({
            "add_requested": list(fo_add),
            "add_status": fo_add_status,
            "drop_applied": list(fo_drop),
            "meta_feature_count": len(meta_features_full.columns),
        }, indent=2),
        encoding="utf-8",
    )

    # All in-scope bars are events; side=+1 (long-only, forward returns).
    events = pd.DataFrame({"side": pd.Series(1, index=meta_features_full.index, dtype=int)})

    # Phase 5 regime gate: filter events to in-scope bars when a mask is set.
    regime_mask_path = cfg.get("regime_mask_path")
    if regime_mask_path:
        mask_df = pd.read_parquet(regime_mask_path)
        if "mask" not in mask_df.columns:
            raise ValueError(
                f"regime_mask parquet at {regime_mask_path} must have a 'mask' boolean column"
            )
        in_scope = mask_df["mask"].astype(bool)
        in_scope_ts = in_scope.index[in_scope]
        n_before = len(events)
        events = events[events.index.isin(in_scope_ts)]
        print(
            f"[{primary_name}] regime gate: {len(events)}/{n_before} events kept "
            f"(scope from {regime_mask_path})",
            flush=True,
        )

    # Zero-event guard: return gracefully rather than crashing in triple-barrier.
    if events.empty:
        print(f"[{primary_name}] 0 events after regime gate — skipping triple barrier", flush=True)
        if count_events_only:
            _wf_floor = wf_event_floor(
                cfg["walk_forward"]["n_folds"],
                cfg["walk_forward"]["train_min_bars"],
            )
            return {
                "primary": primary_name,
                "n_events": 0,
                "wf_event_floor": int(_wf_floor),
                "n_folds": cfg["walk_forward"]["n_folds"],
                "train_min_bars": cfg["walk_forward"]["train_min_bars"],
                "train_min_resolved": resolve_train_min(cfg["walk_forward"]["train_min_bars"], 0),
            }
        return None  # non-count path: nothing to write

    atr = features["_atr_14"]   # used by _run_folds_and_report for metrics/plots

    # TARGET: sign of N-bar forward log-return.  This is more principled than
    # triple barrier for supervised-direct: it is ALWAYS ~50/50 positive vs
    # negative regardless of regime trend strength, so calibration slices never
    # have a single class.  Triple barrier with symmetric barriers still produced
    # 95%+ label=1 in strongly trending regimes (e.g. USDJPY 2021-2024 bull run),
    # breaking RefittingCalibratedPipeline.  The N-bar sign target correctly asks
    # "is the price higher at t+H?" which is the right formulation for a direction
    # predictor.  exit_price = close[t+H] (no barrier, just the realised close).
    horizon = cfg["triple_barrier"]["horizon"]
    idx_pos = {ts: i for i, ts in enumerate(ohlcv.index)}
    close_arr = ohlcv["close"].values

    rows = []
    for ts in events.index:
        t0 = idx_pos[ts]
        t_end = t0 + horizon
        if t_end >= len(ohlcv):
            continue
        fwd_ret = float(np.log(close_arr[t_end] / close_arr[t0]))
        rows.append({
            "t_end_idx": t_end,
            "exit_price": float(close_arr[t_end]),
            "label": int(fwd_ret > 0),   # 1 = up, 0 = down/flat
        })
    valid = pd.DataFrame(rows, index=events.index[:len(rows)])

    # Sample weights: uniform (avg_uniqueness with one-bar-apart events and
    # non-overlapping windows is ~1 anyway; using simple weights avoids
    # the O(n^2) uniqueness scan on 2000+ events).
    w_all = np.ones(len(valid), dtype=float) / len(valid) if len(valid) > 0 else np.array([])

    # Build X: meta_features at event times; NO primary_side / primary_strength /
    # bars_since_signal — those columns encode primary signal state that does not
    # exist in supervised-direct mode. Adding them would be a silent look-ahead
    # on signal direction.
    X = meta_features_full.loc[valid.index].copy()

    # Drop rows with NaN (rolling-window warmup) and align weights/labels.
    pre_drop_index = X.index
    X = X.dropna()
    # Labels: 1 = win (TP hit), 0 = loss or timeout. Triple-barrier returns
    # {-1, 0, 1}; for long-only events side=+1 so label=1 means TP, label∈{-1,0}
    # means SL/timeout. We binarize: 1 → keep as 1, else → 0.
    raw_label = valid["label"].loc[X.index]
    y = (raw_label == 1).astype(int)
    keep_mask = pre_drop_index.isin(X.index)
    w = w_all[keep_mask]

    assert len(w) == len(X) == len(y), (
        f"Misalignment after NaN drop: |w|={len(w)}, |X|={len(X)}, |y|={len(y)}"
    )
    assert X.index.equals(y.index), "X.index != y.index after NaN drop"
    assert not X.isnull().any().any(), "X still contains NaN after dropna"

    n_events = len(X)

    # Class-balance guard: if one class is rarer than 3% of all events, the WF
    # folds will see single-class training slices and XGBoost/sklearn will crash.
    # Return gracefully with a class-imbalance event_floor.
    if n_events > 0:
        minority_frac = float(y.value_counts(normalize=True).min())
        if minority_frac < 0.03:
            print(f"[supervised_direct] class imbalance: minority_frac="
                  f"{minority_frac:.1%} < 3% — returning event_floor.", flush=True)
            _wf_floor = wf_event_floor(
                cfg["walk_forward"]["n_folds"],
                cfg["walk_forward"]["train_min_bars"],
            )
            return {
                "primary": primary_name, "n_events": n_events,
                "wf_event_floor": int(_wf_floor),
                "n_folds": cfg["walk_forward"]["n_folds"],
                "train_min_bars": cfg["walk_forward"]["train_min_bars"],
                "train_min_resolved": resolve_train_min(
                    cfg["walk_forward"]["train_min_bars"], n_events),
            }

    # B0048 pre-flight: return event count before make_folds and training.
    if count_events_only:
        n_folds = cfg["walk_forward"]["n_folds"]
        train_min_bars = cfg["walk_forward"]["train_min_bars"]
        return {
            "primary": primary_name,
            "n_events": int(n_events),
            "n_folds": int(n_folds),
            "train_min_bars": int(train_min_bars),
            "train_min_resolved": resolve_train_min(train_min_bars, n_events),
            "wf_event_floor": wf_event_floor(n_folds, train_min_bars),
        }

    cost_bps = resolve_cost_bps("XAUUSD", cfg)
    # B0129 (AFML §7.4.1): supervised-direct uses ~1-bar-apart events (density ~1),
    # which is exactly the regime where a fixed event-count purge UNDER-purges and
    # leaks. Purge outer folds by each event's label-end BAR (t_end_idx) instead.
    _idx_pos_sd = {ts: i for i, ts in enumerate(ohlcv.index)}
    _t_starts_sd = np.array([_idx_pos_sd[ts] for ts in valid.index])
    _t_ends_sd = valid["t_end_idx"].values
    event_start_bar = _t_starts_sd[keep_mask]
    event_end_bar = _t_ends_sd[keep_mask]
    folds = make_folds(
        n=n_events,
        n_folds=cfg["walk_forward"]["n_folds"],
        train_min=resolve_train_min(cfg["walk_forward"]["train_min_bars"], n_events),
        purge=cfg["walk_forward"]["purge_bars"],
        embargo_pct=cfg["walk_forward"]["embargo_pct"],
        event_start_bar=event_start_bar,
        event_end_bar=event_end_bar,
    )
    return _run_folds_and_report(
        primary_name=primary_name,
        cfg=cfg,
        ohlcv=ohlcv,
        features=meta_features_full,
        X=X,
        y=y,
        w=w,
        valid=valid,
        folds=folds,
        out_dir=out_dir,
        atr=atr,
        cost_bps=cost_bps,
        dry_run=dry_run,
    )


def persist_audit_pnl(
    out_dir: Path,
    oof_probs: pd.DataFrame,
    models: list[str],
    per_trade_pnl: np.ndarray,
    audit_threshold: float = 0.50,
) -> None:
    """B0035 / B0155 — persist per-event net PnL artifacts for the audit's
    cross-episode survival gate (phase5.run_proposal.evaluate_per_episode).

    Always writes `strategy_pnl_threshold50.parquet` (NaN where the model's
    OOF prob < 0.50) — the legacy artifact every pre-B0155 fixed_0.50 audit
    reads; its bytes/semantics are unchanged.

    B0155: when `audit_threshold` differs from 0.50 (an ev_breakeven_v1
    proposal evaluated at its pre-registered p*), ADDITIONALLY writes
    `strategy_pnl_effective.parquet` (NaN where prob < audit_threshold) plus a
    `strategy_pnl_effective.json` sidecar recording the threshold, so the
    audit can verify it is reading the artifact for the threshold it expects.

    `per_trade_pnl` is the threshold-independent per-event net pnl
    (side * fwd_ret - cost); thresholds only decide which events are taken.
    """
    def _masked(threshold: float) -> dict[str, np.ndarray]:
        return {
            m: np.where((oof_probs[m] >= threshold).values, per_trade_pnl, np.nan)
            for m in models
        }

    pd.DataFrame(_masked(0.50), index=oof_probs.index).to_parquet(
        out_dir / "strategy_pnl_threshold50.parquet"
    )
    if abs(float(audit_threshold) - 0.50) > 1e-9:
        pd.DataFrame(_masked(float(audit_threshold)), index=oof_probs.index).to_parquet(
            out_dir / "strategy_pnl_effective.parquet"
        )
        (out_dir / "strategy_pnl_effective.json").write_text(
            json.dumps({"threshold": float(audit_threshold)}, indent=2),
            encoding="utf-8",
        )


def _run_folds_and_report(
    *,
    primary_name: str,
    cfg: dict,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    w: np.ndarray,
    valid: pd.DataFrame,
    folds: list,
    out_dir: Path,
    atr: pd.Series,
    cost_bps: float,
    dry_run: bool,
) -> "dict | None":
    """Shared fold-training + reporting for both labeled and supervised modes.

    Both ``_run_one_primary`` (labeled path) and ``_run_supervised_direct``
    (supervised-direct path) call this function after building X/y/w and
    computing walk-forward folds.  All business logic for training, OOF
    prediction, metrics, plotting, model persistence, threshold selection,
    summary JSON, and report.md lives here exactly once.

    walk_forward purge/embargo is backtest-only and is applied by the caller
    via ``make_folds``; the live loop bypasses it (see live_loop.py docstring).

    Sharpe annualization uses ``sqrt(trades_per_year)`` (NOT ``sqrt(252)``);
    see pipeline/metrics.py ``strategy_metrics`` lines 62-78 for the formula.
    NaN when n_trades < 30 or std == 0.  Downstream uses np.nanmedian / nanmean.

    Sample weights (``w``) MUST be passed to every fit call (base estimator,
    hyperparam search, and calibration) — AFML §4 invariant; removing any one
    of these reintroduces concurrent-label overfit.
    """
    # side: +1/-1 per event (used in forward-return sign + baseline strategy).
    # Labeled mode: always present as X["primary_side"].
    # Supervised-direct mode: all events are long-only (side=+1), so the column
    # is absent — derive it as all-ones in that case.
    if "primary_side" in X.columns:
        side = X["primary_side"]
    else:
        side = pd.Series(np.ones(len(X), dtype=int), index=X.index)

    # Forward returns: entry close → realized exit_price (gap-aware from triple_barrier).
    idx_pos = {ts: i for i, ts in enumerate(ohlcv.index)}
    close = ohlcv["close"].values
    valid_kept = valid.loc[X.index]
    entry_close = close[[idx_pos[ts] for ts in X.index]]
    exit_price = valid_kept["exit_price"].values
    fwd_ret = pd.Series(np.log(exit_price / entry_close), index=X.index)
    assert not fwd_ret.isnull().any(), "fwd_ret contains NaN — check exit_price alignment"

    # B0129 (AFML §7.4.1): per-event bar positions aligned to X, used to purge
    # the INNER CV by label-end (t_end_idx) too — §7.4.3 requires purging on
    # EVERY split, including hyper-parameter fitting, not just the outer folds.
    event_start_bar_X = np.array([idx_pos[ts] for ts in X.index])
    event_end_bar_X = valid_kept["t_end_idx"].values
    inner_embargo = int(np.ceil(cfg["walk_forward"]["embargo_pct"] * len(X)))

    # Training loop.
    n_iter = cfg["dry_run"]["n_iter"] if dry_run else cfg["hyperparam_search"]["n_iter"]
    timings = {}
    oof_probs = pd.DataFrame(index=X.index, columns=cfg["models"], dtype=float)
    mda_per_fold: dict[str, list[dict[str, float]]] = {m: [] for m in cfg["models"]}
    # B0131: Clustered Feature Importance (MLfAM Ch6 §6.5.2). Per-feature MDA is
    # substitution-biased on correlated tier2 features; CFI clusters features
    # (on the TRAIN fold — no lookahead) and permutes whole clusters on the test
    # fold, re-attributing diluted importance to the cluster. Per fold we store
    # {cluster_key -> importance} and the cluster membership for aggregation.
    clustered_mda_per_fold: dict[str, list[dict[str, float]]] = {m: [] for m in cfg["models"]}
    # Per-(model, fold) selected threshold from inner-CV. Replaces the v1/v2
    # fixed headline=0.55. Fold-specific so each model can pick the operating
    # point that suits its calibrated probability distribution. Diagnostic
    # (grid scores, fallback reason) is collected for the report.
    selected_threshold_per_model_per_fold: dict[str, list[float]] = {m: [] for m in cfg["models"]}
    threshold_selection_diag: dict[str, list[dict]] = {m: [] for m in cfg["models"]}
    threshold_grid_cfg = cfg["metrics"]["threshold_grid"]
    ts_cfg = cfg.get("threshold_selection", {"method": "inner_cv", "inner_splits": 3,
                                              "min_trades_per_inner_fold": 20})

    # B0130: MDA scored by negative log-loss (AFML §8.3 / MLfAM Snippet 6.3),
    # NOT accuracy@0.5. Accuracy at a hard 0.5 cut misleads on imbalanced
    # meta-labels (the <30-trade fold-starvation regime) and ignores the
    # per-fold selected threshold; neg-log-loss is proper and continuous.
    _mda_scorer = neg_log_loss_scorer

    for model_name in cfg["models"]:
        t0 = time.time()
        for fold in (folds[:1] if dry_run else folds):
            X_tr_full = X.iloc[fold.train_idx]
            y_tr_full = y.iloc[fold.train_idx]
            w_tr_full = w[fold.train_idx]
            hold = int(len(X_tr_full) * cfg["calibration"]["calib_holdout_pct"])
            X_tr, X_ca = X_tr_full.iloc[:-hold], X_tr_full.iloc[-hold:]
            y_tr, y_ca = y_tr_full.iloc[:-hold], y_tr_full.iloc[-hold:]
            w_tr, w_ca = w_tr_full[:-hold], w_tr_full[-hold:]

            # B0129: per-event label-end bars for this fold's train slice, so the
            # inner CV purges by t_end_idx (not a fixed scalar) per AFML §7.4.3.
            esb_tr_full = event_start_bar_X[fold.train_idx]
            eeb_tr_full = event_end_bar_X[fold.train_idx]
            esb_tr, eeb_tr = esb_tr_full[:-hold], eeb_tr_full[:-hold]

            inner_cv = PurgedTimeSeriesSplit(
                n_splits=cfg["hyperparam_search"]["cv_splits"],
                purge=cfg["hyperparam_search"]["cv_purge_bars"],
                event_start_bar=esb_tr,
                event_end_bar=eeb_tr,
                embargo=inner_embargo,
            )
            base = MODEL_FACTORIES[model_name](random_state=cfg["random_seed"])
            # B0156 (fit side): a purged inner split whose TRAIN side is
            # single-class fails every candidate fit (CatBoost: "Target
            # contains only one unique value") and RandomizedSearchCV raises
            # when all fits fail. Materialize the splits, keep only fittable
            # ones, and skip the (model, fold) when none remain — honest
            # degradation (NaN OOF probs for this fold), mirroring the
            # single-class calibration skip in pipeline/train.py (B0032).
            _y_tr_arr = np.asarray(y_tr)
            inner_splits = [(tr, va) for tr, va in inner_cv.split(X_tr)
                            if len(np.unique(_y_tr_arr[tr])) >= 2]
            if not inner_splits:
                warnings.warn(
                    f"[{model_name}] all inner-CV train slices single-class on "
                    f"this fold ({len(y_tr)} events); model skipped for fold.",
                    RuntimeWarning)
                # Keep per-fold lists fold_k-aligned (read positionally at
                # metric time): sentinel threshold + diag, OOF stays NaN.
                selected_threshold_per_model_per_fold[model_name].append(0.55)
                threshold_selection_diag[model_name].append(
                    {"selected_threshold": 0.55,
                     "fallback_reason": "b0156_degenerate_inner_cv_model_skipped"})
                continue
            search = RandomizedSearchCV(
                base, HP_SPACES[model_name], n_iter=n_iter, cv=inner_splits,
                # B0156 (scoring side): NOT the string "neg_log_loss" — that
                # scorer omits labels= and raises on single-class inner-CV
                # validation slices (dense H4 regime cells, purged splits).
                scoring=neg_log_loss_estimator_scorer,
                n_jobs=1, random_state=cfg["random_seed"],
            )
            search.fit(X_tr, y_tr, sample_weight=w_tr)
            best_kwargs = search.best_params_

            # Inner-CV threshold selection (Phase 2 Option B architecture).
            # OOF probs are generated by `inner_oof_predict_proba` driving a
            # `RefittingCalibratedPipeline` over a purged 3-split inner CV;
            # `select_threshold_inner_cv` then scores thresholds over those
            # OOF probs with NaN-masked sub-blocks.
            #
            # The helper is needed (instead of sklearn.cross_val_predict)
            # because PurgedTimeSeriesSplit is not a partition (head gap
            # `[0, step+purge)` is never in any val fold). The wrapper is
            # needed (instead of passing the existing CalibratedClassifierCV)
            # because clone() preserves FrozenEstimator's fitted base, which
            # would leak outer_train into inner_val.
            #
            # Skipped on dry-run (we only validate the timing extrapolation).
            if dry_run or ts_cfg.get("method") != "inner_cv":
                sel_threshold = 0.55
                sel_diag = {"selected_threshold": 0.55,
                            "fallback_reason": "dry_run_or_grid_report_mode"}
            else:
                inner_cv_ts = PurgedTimeSeriesSplit(
                    n_splits=ts_cfg.get("inner_splits", 3),
                    purge=cfg["walk_forward"]["purge_bars"],
                    event_start_bar=esb_tr_full,  # full train slice (inner_oof uses X_tr_full)
                    event_end_bar=eeb_tr_full,
                    embargo=inner_embargo,
                )
                rcp = RefittingCalibratedPipeline(
                    model_name=model_name,
                    base_kwargs=best_kwargs,
                    calib_holdout_pct=cfg["calibration"]["calib_holdout_pct"],
                    method=cfg["calibration"].get("method", "sigmoid"),
                    random_state=cfg["random_seed"],
                )
                inner_oof_full, inner_val_indices = inner_oof_predict_proba(
                    rcp, X_tr_full, y_tr_full, inner_cv_ts,
                    sample_weight=w_tr_full,
                    return_val_indices=True,
                )
                inner_oof = pd.Series(inner_oof_full[:, 1], index=X_tr_full.index)
                # Pass the CV val_indices as sub_block_indices so the threshold
                # selector averages Sharpe over exactly the regions where OOF
                # probs were produced (matches Phase 1 v3+ per-CV-fold semantics
                # exactly, including the calendar-day-based annualisation when
                # the prediction Series carries a DatetimeIndex).
                sel_threshold, sel_diag = select_threshold_inner_cv(
                    side.loc[X_tr_full.index],
                    inner_oof,
                    fwd_ret.loc[X_tr_full.index],
                    bars_per_year=cfg["metrics"].get("bars_per_year", 252),
                    threshold_grid=np.asarray(threshold_grid_cfg, dtype=float),
                    cost_bps=cost_bps,
                    min_trades_per_inner_fold=ts_cfg.get("min_trades_per_inner_fold", 20),
                    sub_block_indices=inner_val_indices,
                )
            selected_threshold_per_model_per_fold[model_name].append(float(sel_threshold))
            threshold_selection_diag[model_name].append(sel_diag)

            clf = fit_calibrated(
                model_name, X_tr, y_tr, w_tr, X_ca, y_ca, w_ca,
                random_state=cfg["random_seed"], base_kwargs=best_kwargs,
                isotonic_min_minority=cfg["calibration"]["min_minority_for_isotonic"],
                method=cfg["calibration"].get("method", "sigmoid"),
            )
            test_idx = fold.test_idx
            oof_probs.iloc[test_idx, oof_probs.columns.get_loc(model_name)] = clf.predict_proba(X.iloc[test_idx])[:, 1]

            # MDA permutation importance on the test fold (skipped on dry-run).
            if not dry_run:
                fold_mda = mda_importance(
                    clf, X.iloc[test_idx], y.iloc[test_idx], _mda_scorer,
                    sample_weight=w[test_idx], n_repeats=5,
                    random_state=cfg["random_seed"],
                )
                mda_per_fold[model_name].append(fold_mda)

                # B0131: Clustered Feature Importance. Cluster on the TRAIN fold
                # (X_tr_full) so the grouping never sees test data; permute whole
                # clusters on the test fold. Key per-fold results by canonical
                # sorted-member tuple so clusters align across folds in aggregation.
                clusters = cluster_features(X_tr_full, random_state=cfg["random_seed"])
                cl_imp = clustered_mda_importance(
                    clf, X.iloc[test_idx], y.iloc[test_idx], _mda_scorer, clusters,
                    sample_weight=w[test_idx], n_repeats=5,
                    random_state=cfg["random_seed"],
                )
                cl_keyed = {
                    "|".join(sorted(clusters[cid])): float(imp)
                    for cid, imp in cl_imp.items()
                }
                clustered_mda_per_fold[model_name].append(cl_keyed)
        timings[model_name] = (time.time() - t0) / 60.0
        print(f"[{primary_name}] {model_name}: {timings[model_name]:.1f} min over {1 if dry_run else len(folds)} fold(s)")

    if dry_run:
        # Extrapolate.
        budget = {m: t * cfg["walk_forward"]["n_folds"] * (cfg["hyperparam_search"]["n_iter"] / cfg["dry_run"]["n_iter"])
                  for m, t in timings.items()}
        print("[dry-run] projected full-run minutes per model:", {k: f"{v:.1f}" for k, v in budget.items()})
        warn = cfg["dry_run"]["max_minutes_per_model_warn"]
        for m, v in budget.items():
            if v >= warn:
                print(f"WARNING: {m} projected at {v:.1f} min (>= {warn}). Reduce n_iter or cv_splits.")
        write_summary_json(out_dir, {"dry_run": True, "timings_min": timings, "projected_min": budget})
        return

    # Compute metrics per fold. Each (model, fold) gets its OWN threshold from
    # the inner-CV selection done above. The threshold grid is also evaluated
    # as a diagnostic so we can show what each operating point would have
    # produced — but the "headline" Sharpe driving the stack decision uses the
    # selected threshold, not a hard-coded 0.55.
    threshold_grid = cfg["metrics"]["threshold_grid"]
    fold_metrics = []
    grid_metrics = []  # one row per (fold, model, threshold)
    sharpe_per_fold_per_model = {m: [] for m in cfg["models"]}
    n_trades_per_fold_per_model = {m: [] for m in cfg["models"]}
    # Per-trade pnl per (model, fold) at the selected threshold — input to the
    # AFML §14 PSR/DSR computation on real return moments.
    pnl_per_model_per_fold: dict[str, list[np.ndarray]] = {m: [] for m in cfg["models"]}
    baseline_sharpe = []
    for fold_k, fold in enumerate(folds):
        slc = X.iloc[fold.test_idx].index
        side_f = side.loc[slc]
        fwd_f = fwd_ret.loc[slc]
        y_f = y.loc[slc]
        # Annualization basis: chronological span of this test fold. Events are
        # filtered signals, so len(slc) is # events not # bars — we need the
        # real time span to compute trades_per_year.
        if len(slc) > 1:
            span_days = (slc[-1] - slc[0]).days
            years_in_window = max(span_days / 365.25, 1e-9)
        else:
            years_in_window = 1e-9
        # Baseline: trade every primary signal (no filter).
        base_m = strategy_metrics(side_f, pd.Series(np.ones(len(slc), dtype=float), index=slc),
                                  fwd_f, cost_bps=cost_bps,
                                  threshold=0.5, years_in_window=years_in_window)
        base_m.pop("per_trade_pnl", None)
        baseline_sharpe.append(base_m["sharpe_net"])
        for m_name in cfg["models"]:
            p = oof_probs.iloc[fold.test_idx][m_name]
            if p.isna().all():
                # B0156: model skipped this fold (degenerate inner CV) — no
                # measurement. NaN metrics, zero trades downstream (NaN probs
                # never cross any threshold). Partial NaN still crashes loudly
                # in classification_metrics: that would be a real bug.
                cm = {"mcc": float("nan"), "pr_auc": float("nan"),
                      "brier": float("nan"),
                      "precision_at_recall_0.3": float("nan"),
                      "precision_at_recall_0.5": float("nan")}
            else:
                cm = classification_metrics(y_f.values, p.values)
            sel_thr = selected_threshold_per_model_per_fold[m_name][fold_k]
            sm_selected = strategy_metrics(side_f, p, fwd_f,
                                           cost_bps=cost_bps,
                                           threshold=sel_thr,
                                           years_in_window=years_in_window)
            # Separate the per-trade pnl array from the JSON-serializable metrics.
            pnl_per_model_per_fold[m_name].append(sm_selected.pop("per_trade_pnl"))
            fold_metrics.append({
                "fold": fold_k, "primary": primary_name, "model": m_name,
                "threshold": sel_thr, **cm, **sm_selected,
            })
            sharpe_per_fold_per_model[m_name].append(sm_selected["sharpe_net"])
            n_trades_per_fold_per_model[m_name].append(int(sm_selected["n_trades"]))
            # Diagnostic: same metrics over the full grid (not used for selection).
            for thr in threshold_grid:
                sm_grid = strategy_metrics(side_f, p, fwd_f,
                                           cost_bps=cost_bps,
                                           threshold=thr, years_in_window=years_in_window)
                sm_grid.pop("per_trade_pnl", None)
                grid_metrics.append({
                    "fold": fold_k, "primary": primary_name, "model": m_name,
                    "threshold": thr, **sm_grid,
                })

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_json(out_dir / "metrics_per_fold.json", orient="records", indent=2)
    pd.DataFrame(grid_metrics).to_json(out_dir / "threshold_grid_metrics.json", orient="records", indent=2)
    write_oof_parquet(out_dir, oof_probs)

    # Stack decision.
    oof_clean = oof_probs.dropna()
    decision = should_stack(
        sharpe_per_fold_per_model, baseline_sharpe,
        oof_clean.corr().values,
        n_trades_per_fold_per_model=n_trades_per_fold_per_model,
        min_models=cfg["stacking"]["min_models_beating_baseline"],
        min_folds=cfg["stacking"]["min_folds_beating_baseline"],
        max_corr=cfg["stacking"]["max_oof_corr"],
        min_trades_per_fold=cfg["stacking"].get("min_trades_per_fold", 30),
    )

    # If stacking, run nested-WF meta.
    if decision.stack:
        y_clean = y.loc[oof_clean.index]
        extra = features[["rv_regime"]].loc[oof_clean.index]
        meta_oof, _ = fit_meta_nested_wf(oof_clean, extra, y_clean,
                                         n_folds=cfg["stacking"]["meta_n_folds"],
                                         purge=cfg["walk_forward"]["purge_bars"],
                                         embargo_pct=cfg["walk_forward"]["embargo_pct"],
                                         C=cfg["stacking"]["meta_C"])
        oof_probs["meta"] = meta_oof

    # Best single model = highest nan-safe median Sharpe (NaN folds are excluded).
    # If a model is all-NaN (no fold ever produced a Sharpe), median = NaN and the
    # model cannot be best.
    median_sharpes = {m: float(np.nanmedian(sharpe_per_fold_per_model[m]))
                      if any(np.isfinite(s) for s in sharpe_per_fold_per_model[m])
                      else float("nan")
                      for m in cfg["models"]}
    finite_medians = {m: v for m, v in median_sharpes.items() if np.isfinite(v)}
    best_model = max(finite_medians, key=finite_medians.get) if finite_medians else cfg["models"][0]

    # Threshold-grid pivot for the report.
    grid_df = pd.DataFrame(grid_metrics)
    threshold_pivot = (
        grid_df.groupby(["model", "threshold"])
        .agg(sharpe_net=("sharpe_net", "mean"),
             n_trades=("n_trades", "mean"),
             hit_ratio=("hit_ratio", "mean"))
        .round(3)
    )

    # PSR + DSR computed on REAL per-trade pnl moments (AFML §14 standard).
    # Refactor from earlier proxy: we no longer estimate skew/kurt from the
    # per-fold Sharpe distribution (degenerate when n_folds is small or many
    # folds have 0 trades). Instead we concat per-trade pnl across folds per
    # model and use its mean/std/skew/kurt directly. Trial pool for DSR is
    # the set of per-fold per-trade Sharpes across all models in this primary.
    pnl_agg = aggregate_per_trade_pnl_metrics(pnl_per_model_per_fold)
    sr_trials = np.array(
        [sr for d in pnl_agg.values() for sr in d["per_fold_sr_per_trade"]],
        dtype=float,
    )
    # B0132 (AFML §11/§13 — deflate by the FULL number of trials): the empirical
    # `sr_trials` pool (per-fold per-trade Sharpes) only estimates the trial
    # VARIANCE. The trial COUNT N fed to the E[max] threshold must reflect the
    # whole within-run search space — models × folds × threshold-grid points —
    # NOT just the handful of Sharpes collected. This is still a per-primary
    # LOWER BOUND on family-wise trials: the other primary and any cross-config
    # search further inflate N (handled by the family DSR aggregator).
    n_trials_familywise = (
        len(cfg["models"]) * len(folds) * len(threshold_grid_cfg)
    )
    psr_per_model: dict[str, float] = {}
    dsr_per_model: dict[str, float] = {}
    for m_name in cfg["models"]:
        d = pnl_agg[m_name]
        if (not np.isfinite(d["sr_per_trade"])
                or d["n_trades"] < 2
                or len(sr_trials) < 2):
            psr_per_model[m_name] = float("nan")
            dsr_per_model[m_name] = float("nan")
            continue
        psr_per_model[m_name] = probabilistic_sharpe_ratio(
            sr_observed=d["sr_per_trade"], sr_benchmark=0.0,
            n=d["n_trades"], skew=d["skew"], kurt=d["kurt"],
        )
        dsr_per_model[m_name] = deflated_sharpe_ratio(
            sr_observed=d["sr_per_trade"], sr_trials=sr_trials,
            n=d["n_trades"], skew=d["skew"], kurt=d["kurt"],
            n_trials=n_trials_familywise,
        )

    # Aggregate MDA across folds: {model -> {feature -> {mean, std}}}.
    mda_aggregated = {m: aggregate_mda_across_folds(mda_per_fold[m]) for m in cfg["models"]}
    # B0131: aggregate clustered MDA the same way (keyed by member-string).
    clustered_mda_aggregated = {
        m: aggregate_mda_across_folds(clustered_mda_per_fold[m]) for m in cfg["models"]
    }
    # Top-10 by mean MDA for the best model (used in the report header).
    best_mda = mda_aggregated[best_model]
    top10_mda = sorted(best_mda.items(), key=lambda kv: kv[1]["mean"], reverse=True)[:10]
    top_features = [f"{name} (MDA={stats['mean']:.4f}±{stats['std']:.4f})" for name, stats in top10_mda]

    # Persist full MDA + PSR/DSR for downstream inspection.
    (out_dir / "mda_per_fold.json").write_text(
        json.dumps({m: mda_per_fold[m] for m in cfg["models"]}, indent=2, default=str)
    )
    # B0131: clustered feature importance (MLfAM Ch6) — re-attributes the
    # substitution-diluted per-feature MDA to mutually-dissimilar clusters.
    (out_dir / "clustered_mda.json").write_text(
        json.dumps({
            "per_fold": {m: clustered_mda_per_fold[m] for m in cfg["models"]},
            "aggregated": clustered_mda_aggregated,
        }, indent=2, default=str)
    )
    (out_dir / "psr_dsr.json").write_text(
        json.dumps({
            "psr": psr_per_model,
            "dsr": dsr_per_model,
            # B0132: empirical trial-Sharpe sample size (variance estimator) vs
            # the family-wise trial COUNT used in the E[max] threshold. The DSR
            # above is deflated by n_trials_familywise (a per-primary lower bound).
            "n_trial_sharpes_sampled": int(len(sr_trials)),
            "n_trials_familywise": int(len(cfg["models"]) * len(folds) * len(threshold_grid_cfg)),
            "n_trials_familywise_note": "per-primary lower bound: models × folds × threshold-grid points; cross-primary trials inflate this further",
            "per_model_aggregate": {
                m: {k: v for k, v in d.items() if k != "per_fold_sr_per_trade"}
                for m, d in pnl_agg.items()
            },
            "trial_pool_sr_per_trade": [float(s) for s in sr_trials],
        }, indent=2)
    )

    # Persist the inner-CV threshold selection trail for inspection / debugging.
    (out_dir / "threshold_selection.json").write_text(
        json.dumps({
            "selected_per_fold": selected_threshold_per_model_per_fold,
            "diagnostics": threshold_selection_diag,
        }, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
    )

    # Calibration plot per model (uses OOF labels + probs across all folds).
    y_oof = y.loc[oof_probs.index]
    for m_name in cfg["models"]:
        mask = oof_probs[m_name].notna()
        if mask.sum() > 0:
            plot_calibration(out_dir,
                             y_true=y_oof[mask].values,
                             y_prob=oof_probs.loc[mask, m_name].values,
                             model_name=m_name)

    # Equity curve for the best model over the OOF window, using each fold's
    # OWN selected threshold (so the curve matches what the stack-decision saw).
    best_thresholds = selected_threshold_per_model_per_fold[best_model]
    take_best = np.zeros(len(oof_probs), dtype=bool)
    for fold_k, fold in enumerate(folds):
        thr_k = best_thresholds[fold_k]
        for j in fold.test_idx:
            p = oof_probs.iloc[j][best_model]
            if pd.notna(p) and p >= thr_k:
                take_best[j] = True
    pnl_best = side.loc[oof_probs.index].values * fwd_ret.loc[oof_probs.index].values * take_best \
        - (cost_bps / 1e4) * take_best
    equity_best = pd.Series(pnl_best.cumsum(), index=oof_probs.index)
    median_best_thr = float(np.median(best_thresholds))
    plot_equity(out_dir, equity_best, label=f"{best_model}_per_fold_thr_med{int(median_best_thr*100)}")

    # Model persistence: re-fit each model on the FULL event set (no calibration split)
    # so Phase 3 / live can load a single deployable artifact per model.
    import joblib
    fitted_models_dir = out_dir / "models"
    fitted_models_dir.mkdir(exist_ok=True)
    feature_names = list(X.columns)
    (fitted_models_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    holdout_pct = cfg["calibration"]["calib_holdout_pct"]
    hold_full = int(len(X) * holdout_pct)
    X_tr_all, X_ca_all = X.iloc[:-hold_full], X.iloc[-hold_full:]
    y_tr_all, y_ca_all = y.iloc[:-hold_full], y.iloc[-hold_full:]
    # w is post-NaN-drop weights aligned to X (same as w_all in labeled path).
    w_tr_all, w_ca_all = w[:-hold_full], w[-hold_full:]
    for m_name in cfg["models"]:
        clf_full = fit_calibrated(
            m_name, X_tr_all, y_tr_all, w_tr_all, X_ca_all, y_ca_all, w_ca_all,
            random_state=cfg["random_seed"],
            isotonic_min_minority=cfg["calibration"]["min_minority_for_isotonic"],
            method=cfg["calibration"].get("method", "sigmoid"),
        )
        joblib.dump(clf_full, fitted_models_dir / f"{m_name}.joblib")

    median_selected_threshold_per_model = {
        m: float(np.median(selected_threshold_per_model_per_fold[m]))
        for m in cfg["models"]
    }

    # B0015b / B0016a: cost_sensitivity block — net Sharpe at 5 cost levels for
    # the best model + per-fold median threshold. Reporting-only per the
    # B0000 v2 anti-smuggling commitment (NO auto-flag, NO auto-reject; humans
    # read it during post-audit review). Reuses pipeline.metrics.strategy_metrics
    # to preserve sqrt(trades_per_year) annualization + NaN-when-n_trades<30
    # invariant from CLAUDE.md.
    COST_BPS_GRID = (0.5, 1.0, 5.0, 10.0, 15.0)
    cost_sensitivity_block = {
        "cost_bps_grid": list(COST_BPS_GRID),
        "net_sharpe_at_grid": [],
        "net_trades_per_year_at_grid": [],
        "best_model": best_model,
    }
    # Compute the OOF span for trades_per_year normalization (same convention
    # as per-fold years_in_window above).
    if len(oof_probs.index) > 1:
        oof_span_days = (oof_probs.index[-1] - oof_probs.index[0]).days
        oof_years = max(oof_span_days / 365.25, 1e-9)
    else:
        oof_years = 1e-9
    # Predictions = best-model OOF probs. Threshold = median per-fold threshold
    # of the best model. side + fwd_ret aligned to oof_probs.index.
    _best_pred = oof_probs[best_model]
    _best_thr = median_selected_threshold_per_model.get(best_model, 0.55)
    for cb in COST_BPS_GRID:
        sm = strategy_metrics(
            side.loc[oof_probs.index],
            _best_pred,
            fwd_ret.loc[oof_probs.index],
            cost_bps=cb,
            threshold=_best_thr,
            years_in_window=oof_years,
        )
        sm.pop("per_trade_pnl", None)
        cost_sensitivity_block["net_sharpe_at_grid"].append(sm["sharpe_net"])
        cost_sensitivity_block["net_trades_per_year_at_grid"].append(
            float(sm["n_trades"]) / oof_years if oof_years > 1e-6 else float("nan")
        )

    # B0035 — persist per-event net PnL per model so the audit
    # (phase5.run_proposal) can compute cross-episode survival without
    # re-running the pipeline. NaN where the trade is not taken; non-NaN count
    # per episode = n_trades, sum = net PnL. Every row in oof_probs.index is a
    # real event (side != 0 — the meta only sees fired events).
    # B0155 — `audit_effective_threshold` (injected into the transient config
    # by phase5.run_proposal.build_transient_config for ev_breakeven_v1
    # proposals) additionally persists the pnl at the pre-registered p*;
    # absent/0.50 keeps the legacy threshold-50-only behavior bit-for-bit.
    _idx50 = oof_probs.index
    _side50 = side.loc[_idx50].values
    _fwd50 = fwd_ret.loc[_idx50].values
    _per_trade50 = _side50 * _fwd50 - (cost_bps / 1e4)
    persist_audit_pnl(
        out_dir, oof_probs, list(cfg["models"]), _per_trade50,
        audit_threshold=float(cfg["metrics"].get("audit_effective_threshold", 0.50)),
    )

    # B0003 — benchmark correlations (maxdama §4.10): is this hidden beta?
    # Load VIX + S&P 500 levels from the FRED cache (VIXCLS always present;
    # SP500 present once build_macro_frame has fetched it). Degrades to None
    # per-benchmark when the cache is missing — never crashes the run.
    benchmark_levels: dict[str, pd.Series] = {}
    for label, code in (("VIX", "VIXCLS"), ("SP500", "SP500")):
        cache_path = Path("cache/fred") / f"{code}.parquet"
        if cache_path.exists():
            benchmark_levels[label] = pd.read_parquet(cache_path)[code]
    strategy_pnl_series = pd.Series(pnl_best, index=oof_probs.index)
    benchmark_correlations = compute_benchmark_correlations(
        strategy_pnl_series, benchmark_levels
    )

    # B0001 — position-sizing diagnostics (Gaussian vs empirical Kelly) on the
    # best model's realized per-trade PnL (taken trades only). Informational;
    # not wired into live sizing.
    from pipeline.sizing import kelly_sizing, bet_sizing_diagnostics
    kelly_block = kelly_sizing(pnl_best[take_best])
    # B0120 — AFML ch.10 2N(z)-1 size chain on the best model's OOF probs
    # (raw -> concurrency-averaged -> discretized). Informational pre-check;
    # not wired into live sizing.
    bet_sizing_block = bet_sizing_diagnostics(
        oof_probs[best_model].values, take_best,
        event_start_bar_X, event_end_bar_X,
    )

    # B0002 — decommissioning rule: rolling median fold Sharpe (best model) vs
    # a configurable threshold. Reported only; no automatic action.
    _shutoff_cfg = cfg.get("shutoff", {})
    shutoff_status = compute_shutoff_status(
        sharpe_per_fold_per_model[best_model],
        rolling_window=int(_shutoff_cfg.get("rolling_window", 6)),
        threshold=float(_shutoff_cfg.get("threshold", 0.0)),
    )

    write_summary_json(out_dir, {
        "primary": primary_name,
        "n_events": int(len(X)),
        "n_folds": len(folds),
        "baseline_sharpe_per_fold": baseline_sharpe,
        "sharpe_per_fold_per_model": sharpe_per_fold_per_model,
        "n_trades_per_fold_per_model": n_trades_per_fold_per_model,
        "selected_threshold_per_fold_per_model": selected_threshold_per_model_per_fold,
        "median_selected_threshold_per_model": median_selected_threshold_per_model,
        "median_sharpe": median_sharpes,
        "best_model": best_model,
        "stack_decision": {"stack": decision.stack, "reason": decision.reason,
                           "n_models_passing": decision.n_models_passing,
                           "max_pair_corr": decision.max_pair_corr},
        "cost_sensitivity": cost_sensitivity_block,
        "benchmark_correlations": benchmark_correlations,
        "shutoff": shutoff_status,
        "kelly_sizing": kelly_block,
        "bet_sizing": bet_sizing_block,
    })

    write_report_md(
        out_dir,
        stack_decision_text=f"{'STACK' if decision.stack else 'NO STACK'} — {decision.reason}",
        best_model=best_model,
        threshold=median_selected_threshold_per_model.get(best_model, 0.55),
        metrics_table=metrics_df.groupby("model").mean(numeric_only=True).round(3),
        top_features=top_features,
        next_steps=[
            "Replicate pipeline on H4 multi-asset (8 assets)",
            "Pipe per-trade PnL through strategy_metrics so PSR/DSR use realized skew/kurt instead of Sharpe-distribution proxy",
            "Add COT positioning only if D1 results justify it",
            "Hyperparam re-search per asset (not transferable across instruments)",
        ],
        threshold_grid_table=threshold_pivot,
        psr_per_model=psr_per_model,
        dsr_per_model=dsr_per_model,
        selected_thresholds_per_fold=selected_threshold_per_model_per_fold,
        n_trades_per_fold_per_model=n_trades_per_fold_per_model,
        benchmark_correlations=benchmark_correlations,
        shutoff_status=shutoff_status,
    )


def _run_one_primary(primary_name: str, cfg: dict, ohlcv: pd.DataFrame, features: pd.DataFrame, dry_run: bool, count_events_only: bool = False):
    out_dir = Path(cfg["output_dir"]) / primary_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dispatch supervised-direct mode before touching the primary signal path.
    # Checks both the primary_name argument AND the cfg.primary.mode key so that
    # callers can use either convention interchangeably.
    primary_mode = (cfg.get("primary") or {}).get("mode", "labeled")
    if primary_name == "supervised_direct" or primary_mode == "supervised_direct":
        return _run_supervised_direct(
            cfg, ohlcv, features, dry_run=dry_run, count_events_only=count_events_only,
        )

    cost_bps = resolve_cost_bps("XAUUSD", cfg)

    # B0015b: the primary sees the FULL features (so phase5_cot_extremes can
    # read dtwexbgs_close). The meta-labeler will see features-MINUS-blacklist
    # — the filter is applied below (see "Apply primary_feature_blacklist"),
    # AFTER signal() returns but BEFORE the meta uses features.
    sig = _select_primary(primary_name, ohlcv, features, cfg)
    events = pd.DataFrame({"side": sig[sig != 0].astype(int)})

    # B0015b Layer (c) runtime intersection check: enforce that the primary's
    # declared INPUT_COLUMNS are disjoint from the columns the meta will see
    # (post-blacklist). Custom phase5_* primaries declare INPUT_COLUMNS as a
    # module-level tuple; built-in primaries (ema_cross, etc.) are exempt.
    blacklist = cfg.get("primary_feature_blacklist", []) or []
    if primary_name.startswith("phase5_"):
        import importlib
        primary_mod = importlib.import_module(f"pipeline.primaries_phase5.{primary_name}")
        primary_input_cols = getattr(primary_mod, "INPUT_COLUMNS", ())
        # Build the meta-visible feature set: everything in features minus the
        # blacklist minus _atr_14 (internal artifact dropped just before fit).
        meta_features_preview = apply_primary_feature_blacklist(
            features.drop(columns=["_atr_14"], errors="ignore"), blacklist,
        )
        assert_primary_inputs_disjoint(
            primary_inputs=primary_input_cols,
            meta_features_columns=set(meta_features_preview.columns),
        )

    # Phase 5 regime-gating (additive). When the config carries a
    # `regime_mask_path`, load the boolean mask and drop events outside
    # the scope. This implements the SKILL.md `regime_gate.mode = "filter_events"`
    # mechanism without modifying the sample-weight invariant — events
    # are filtered BEFORE triple-barrier labeling, so weights/labels/folds
    # are computed only on in-scope events.
    regime_mask_path = cfg.get("regime_mask_path")
    if regime_mask_path:
        mask_df = pd.read_parquet(regime_mask_path)
        if "mask" not in mask_df.columns:
            raise ValueError(f"regime_mask parquet at {regime_mask_path} must have a 'mask' boolean column")
        in_scope = mask_df["mask"].astype(bool)
        in_scope_ts = in_scope.index[in_scope]
        events_in_scope = events.index.isin(in_scope_ts)
        n_before = len(events)
        events = events[events_in_scope]
        print(f"[{primary_name}] regime gate: {len(events)}/{n_before} events kept (scope from {regime_mask_path})", flush=True)

    # Guard: if the regime gate reduced events to zero (very selective
    # custom primaries), skip triple-barrier and return an empty result so
    # that count_events_only and the event-floor gate both see 0 events
    # rather than crashing with KeyError on an empty DataFrame.
    if events.empty:
        print(f"[{primary_name}] 0 events after regime gate — skipping triple barrier", flush=True)
        if count_events_only:
            _wf_floor = wf_event_floor(
                cfg["walk_forward"]["n_folds"],
                cfg["walk_forward"]["train_min_bars"],
            )
            return {
                "n_events": 0,
                "wf_event_floor": int(_wf_floor),
                "n_folds": cfg["walk_forward"]["n_folds"],
                "train_min_bars": cfg["walk_forward"]["train_min_bars"],
                "train_min_resolved": resolve_train_min(cfg["walk_forward"]["train_min_bars"], 0),
            }
        return  # non-count path: nothing to write

    # Triple-barrier labels.
    atr = features["_atr_14"]
    labels = triple_barrier_labels(
        ohlcv, events, atr,
        horizon=cfg["triple_barrier"]["horizon"],
        tp_mult=cfg["triple_barrier"]["tp_atr_mult"],
        sl_mult=cfg["triple_barrier"]["sl_atr_mult"],
    )
    # Drop events whose label resolution would extend past the data.
    valid = labels[labels["t_end_idx"] < len(ohlcv)]

    # Sample weights via avgUniqueness.
    idx_pos = {ts: i for i, ts in enumerate(ohlcv.index)}
    t_starts_all = np.array([idx_pos[ts] for ts in valid.index])
    t_ends_all = valid["t_end_idx"].values
    w_all = avg_uniqueness(t_starts_all, t_ends_all, n_bars=len(ohlcv))

    # Build X matrix: Tier 2 features at event times + primary state.
    # B0015b: apply the proposal's primary_feature_blacklist HERE so the meta
    # never sees columns the primary keyed on (cot_*, dxy_*, dtwexbgs_*, etc.).
    # The primary already received the full features above; this drop only
    # affects the meta's view from this point onward.
    state = compute_primary_state(valid["side"], cap=60)
    # B0079: apply feature_overrides.drop (on top of primary_feature_blacklist)
    # and validate feature_overrides.add against available columns.  This makes
    # feature_overrides symmetric: .drop subtracts from X (like blacklist);
    # .add checks presence and records evidence in cfg so the audit artifact can
    # prove which features the meta actually saw.
    fo_drop = cfg.get("feature_overrides_drop", []) or []
    fo_add  = cfg.get("feature_overrides_add",  []) or []
    combined_drop = list(dict.fromkeys(list(blacklist) + list(fo_drop)))  # deduped, order-stable
    meta_features_full = apply_primary_feature_blacklist(
        features.drop(columns=["_atr_14"]), combined_drop,
    )
    # Validate add requests against the post-drop column set; record for audit.
    # B0149: conceptual requests ("volume") may be satisfied by derived tier2
    # columns via pipeline.features.FEATURE_ALIASES.
    fo_add_status = feature_add_status(fo_add, set(meta_features_full.columns))
    # Write evidence to disk so run_proposal.py can surface it in the audit artifact.
    (out_dir / "feature_overrides_status.json").write_text(
        json.dumps({
            "add_requested": list(fo_add),
            "add_status": fo_add_status,
            "drop_applied": list(fo_drop),
            "meta_feature_count": len(meta_features_full.columns),
        }, indent=2),
        encoding="utf-8",
    )
    X = meta_features_full.loc[valid.index].copy()
    X["primary_side"] = state["primary_side"].values
    if primary_name == "ema_cross":
        spread = (ohlcv["close"].ewm(span=cfg["primary"]["ema_cross"]["fast"]).mean()
                  - ohlcv["close"].ewm(span=cfg["primary"]["ema_cross"]["slow"]).mean())
        X["primary_strength"] = (spread / atr).loc[valid.index].values
    else:
        X["primary_strength"] = features["z_r20"].loc[valid.index].values
    X["bars_since_signal"] = state["bars_since_signal"].values

    # Drop rows with NaN in any feature (rolling-window warmup, etc.) and align weights/labels.
    pre_drop_index = X.index
    X = X.dropna()
    y = valid["label"].loc[X.index]
    keep_mask = pre_drop_index.isin(X.index)
    w = w_all[keep_mask]

    # Defensive asserts: alignment is invariant downstream of this point.
    assert len(w) == len(X) == len(y), (
        f"Misalignment after NaN drop: |w|={len(w)}, |X|={len(X)}, |y|={len(y)}"
    )
    assert X.index.equals(y.index), "X.index != y.index after NaN drop"
    assert not X.isnull().any().any(), "X still contains NaN after dropna"

    # Walk-forward folds.
    n_events = len(X)

    # B0048 — pre-flight event-floor mode. Stop here, after the post-primary,
    # in-regime, post-NaN-drop event count is known but BEFORE make_folds (which
    # would raise WalkForwardGeometryError) and BEFORE any training. Lets the
    # caller gate on the walk-forward geometry floor cheaply and honestly.
    if count_events_only:
        n_folds = cfg["walk_forward"]["n_folds"]
        train_min_bars = cfg["walk_forward"]["train_min_bars"]
        return {
            "primary": primary_name,
            "n_events": int(n_events),
            "n_folds": int(n_folds),
            "train_min_bars": int(train_min_bars),
            "train_min_resolved": resolve_train_min(train_min_bars, n_events),
            "wf_event_floor": wf_event_floor(n_folds, train_min_bars),
        }

    # B0129 (AFML §7.4.1): purge outer folds by each event's label-end BAR
    # (t_end_idx), not by a fixed event-count scalar. event_start_bar/end_bar
    # are aligned to X (post-NaN-drop) via keep_mask.
    event_start_bar = t_starts_all[keep_mask]
    event_end_bar = t_ends_all[keep_mask]
    folds = make_folds(
        n=n_events,
        n_folds=cfg["walk_forward"]["n_folds"],
        train_min=resolve_train_min(cfg["walk_forward"]["train_min_bars"], n_events),
        purge=cfg["walk_forward"]["purge_bars"],
        embargo_pct=cfg["walk_forward"]["embargo_pct"],
        event_start_bar=event_start_bar,
        event_end_bar=event_end_bar,
    )
    return _run_folds_and_report(
        primary_name=primary_name,
        cfg=cfg,
        ohlcv=ohlcv,
        features=meta_features_full,
        X=X,
        y=y,
        w=w,
        valid=valid,
        folds=folds,
        out_dir=out_dir,
        atr=atr,
        cost_bps=cost_bps,
        dry_run=dry_run,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--preflight-event-count",
        action="store_true",
        help=(
            "B0048: run only the front-half (data -> features -> primary -> "
            "triple-barrier -> regime mask), emit the post-primary in-regime "
            "event count + walk-forward floor as JSON on stdout, and exit "
            "WITHOUT training. Used by phase5.run_proposal to gate the WF "
            "refusal cliff before the heavy audit."
        ),
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ohlcv = load_dataset(cfg["data_path"])
    # Trim to date range.
    if cfg.get("date_range"):
        s, e = cfg["date_range"]["start"], cfg["date_range"]["end"]
        ohlcv = ohlcv.loc[s:e]

    macro = build_macro_frame(cfg["date_range"]["start"], cfg["date_range"]["end"], Path("cache/fred"))
    features = build_tier2_features(ohlcv, macro)
    # B0147 — GLD real-volume features (exogenous, PIT calendar-shifted).
    # Config-gated: base XAU configs set features.gld_volume=true; phase5
    # transient configs override it per asset_class (metals only) so FX/crypto
    # audits never grow gold columns. Missing cache degrades with a warning
    # (the audit must not hard-depend on an optional alt-data pull).
    if (cfg.get("features") or {}).get("gld_volume"):
        from pipeline.alt_data.gld_volume import (
            GldVolumeCacheMissing,
            load_gld_volume_features,
        )
        try:
            features = features.join(load_gld_volume_features(features.index))
        except GldVolumeCacheMissing as e:
            warnings.warn(f"features.gld_volume=true but cache missing — "
                          f"skipped: {e}", RuntimeWarning)
    features = features.dropna()
    ohlcv = ohlcv.loc[features.index]

    if args.preflight_event_count:
        # B0048: emit per-primary event counts as a single JSON object on
        # stdout (machine-parsed by phase5.run_proposal). No output_dir writes.
        counts = {
            primary: _run_one_primary(
                primary, cfg, ohlcv, features, dry_run=args.dry_run,
                count_events_only=True,
            )
            for primary in cfg["primary"]["candidates"]
        }
        print("PREFLIGHT_EVENT_COUNT_JSON " + json.dumps(counts))
        return

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for primary in cfg["primary"]["candidates"]:
        _run_one_primary(primary, cfg, ohlcv, features, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
