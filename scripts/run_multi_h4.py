"""Multi-asset H4 meta-labeling orchestrator (Phase 2 T9.B).

Loads `configs/multi_h4.yaml`, pulls per-asset data already on disk
(via `scripts/mt5_pull_multi_h4.py`), runs primary pre-screening via
Hurst (T7), then trains one meta-learner per (asset, primary) using
the Option B threshold pipeline (T9.A) on H4 features (T5).

This first commit (T9.B.1) implements the PLANNING phase: load config
+ data + intraday macro + pre-screen primaries + decide what to train.
Training itself is added in T9.B.2.

Outputs:
  results/clf_multi_h4/primary_screening.json    — per-asset Hurst decisions
  results/clf_multi_h4/training_plan.json        — (asset, primary) pairs to train
  results/clf_multi_h4/{asset}/{primary}/...     — populated by T9.B.2
  results/clf_multi_h4/deployment_config.yaml    — populated after training (T9.B.3)
"""
from __future__ import annotations
import sys
from pathlib import Path as _Path

# Ensure repo root is on sys.path for `from pipeline.* import ...`.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

# Console-encoding hardening: several progress prints use em-dashes / arrows,
# which crash a cp1252 Windows console (UnicodeEncodeError). Output-encoding
# only — no pipeline behavior change. `errors="replace"` keeps the run alive
# even on exotic consoles.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

from pipeline.cross_asset import (
    LEVEL_0_ASSETS,
    LEVEL_1_ASSETS,
    compute_xau_xag_ratio,
    load_multi_asset,
    topological_order,
)
from pipeline.features import (
    build_tier2_h4_features,
    build_crossasset_features,
)
from pipeline.feature_importance import (
    aggregate_mda_across_folds,
    mda_importance,
    neg_log_loss_scorer,
    neg_log_loss_estimator_scorer,
    cluster_features,
    clustered_mda_importance,
)
from pipeline.cointegration import (
    coint_spread_signal,
    load_pairs_config,
    lookup_pair_for_asset,
)
from pipeline.labels import (
    bollinger_meanrev_signal,
    compute_primary_state,
    cusum_filter_signal,
    ema_crossover_signal,
    momentum_zscore_signal,
    triple_barrier_labels,
    vix_regime_riskflow_signal,
)
from pipeline.macro_fetch import build_macro_frame
from pipeline.macro_fetch_intraday import (
    build_intraday_macro_frame,
    build_intraday_macro_frame_with_daily_fallback,
)
from pipeline.metrics import (
    aggregate_per_trade_pnl_metrics,
    classification_metrics,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    strategy_metrics,
)
from pipeline.primary_screening import screen_primaries_for_asset
from pipeline.reporting import (
    plot_calibration,
    write_oof_parquet,
    write_summary_json,
)
from pipeline.sample_weights import avg_uniqueness
from pipeline.stack import should_stack
from pipeline.threshold_selection import select_threshold_inner_cv
from pipeline.thresholds import resolve_threshold_grid
from pipeline.friction import resolve_cost_bps
from pipeline.train import (
    MODEL_FACTORIES,
    RefittingCalibratedPipeline,
    fit_calibrated,
)
from pipeline.walk_forward import (
    PurgedTimeSeriesSplit,
    inner_oof_predict_proba,
    make_folds,
    wf_event_floor,
)
from pipeline.pooled_walk_forward import (
    make_pooled_time_folds,
    PurgedTimeGroupSplit,
)
from pipeline.sample_weights import pooled_avg_uniqueness


# Same hyperparameter grids as scripts/run_xau_d1.py — H4 doesn't need
# fundamentally different sweeps; the asset-specific signal differences
# are captured in walk_forward and bars_per_year, not hyperparams.
HP_SPACES = {
    "xgb": {"max_depth": [3, 5, 7], "learning_rate": [0.03, 0.1],
            "n_estimators": [200, 500]},
    "catboost": {"depth": [4, 6, 8], "learning_rate": [0.03, 0.1],
                 "iterations": [300, 600], "l2_leaf_reg": [3, 7]},
    "rf": {"n_estimators": [300, 600], "max_depth": [5, 10, None],
           "min_samples_leaf": [5, 20], "max_features": ["sqrt", 0.5]},
}


# Asset class lookup used to pick the bars_per_year annualisation factor.
ASSET_CLASS_BY_NAME = {
    "EURUSD": "fx", "GBPUSD": "fx", "USDJPY": "fx",
    "XAUUSD": "metal", "XAGUSD": "metal",
    "BTCUSD": "crypto", "ETHUSD": "crypto", "SOLUSD": "crypto",
}


def load_config(config_path: Path | str) -> dict:
    """Read multi_h4.yaml and return the parsed dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_asset_overrides(cfg: dict, asset: str) -> dict:
    """Return a per-asset effective config: cfg overridden by cfg['asset_overrides'][asset].

    Used so SOLUSD (with only 1.4 years of data) can have a smaller
    train_min_bars without polluting the global defaults.
    """
    if "asset_overrides" not in cfg or asset not in cfg["asset_overrides"]:
        return cfg
    out = json.loads(json.dumps(cfg))  # deep copy via JSON round-trip
    overrides = cfg["asset_overrides"][asset]
    for section, section_overrides in overrides.items():
        out.setdefault(section, {})
        out[section].update(section_overrides)
    return out


def run_pre_screening(asset_dfs: dict, cfg: dict) -> dict:
    """Apply Hurst pre-screening (T7) to each asset.

    Returns a dict {asset: {hurst, regime, screened, ...}}.
    """
    screening_cfg = cfg.get("primary_screening", {})
    if not screening_cfg.get("enabled", True):
        return {
            asset: {
                "screened": list(cfg["primary"]["candidates"]),
                "regime": "screening_disabled",
                "hurst": None,
            }
            for asset in asset_dfs
        }

    force_all = screening_cfg.get("force_all_primaries", False)
    if force_all:
        return {
            asset: {
                "screened": list(apply_asset_overrides(cfg, asset)["primary"]["candidates"]),
                "regime": "force_all_primaries",
                "hurst": None,
            }
            for asset in asset_dfs
        }

    out: dict = {}
    for asset, df in asset_dfs.items():
        asset_cfg = apply_asset_overrides(cfg, asset)
        train_min_bars = asset_cfg["walk_forward"]["train_min_bars"]
        screened, diag = screen_primaries_for_asset(
            df,
            asset=asset,
            train_min_bars=train_min_bars,
            candidates=cfg["primary"]["candidates"],
            h_trending=screening_cfg.get("hurst_threshold_trending", 0.52),
            h_mr=screening_cfg.get("hurst_threshold_mr", 0.48),
            force_both_primaries=screening_cfg.get("force_both_primaries", []),
        )
        diag["screened"] = screened
        out[asset] = diag
    return out


def build_training_plan(screening: dict, cfg: dict) -> list[dict]:
    """Convert pre-screening output into a list of (asset, primary) pairs to train.

    Each entry includes the asset class (for bars_per_year) and the
    effective config for that asset (after asset_overrides).
    """
    plan: list[dict] = []
    for asset, screen_info in screening.items():
        for primary in screen_info["screened"]:
            asset_cfg = apply_asset_overrides(cfg, asset)
            plan.append({
                "asset": asset,
                "primary": primary,
                "asset_class": ASSET_CLASS_BY_NAME.get(asset, "unknown"),
                "bars_per_year": cfg["bars_per_year_by_class"][
                    ASSET_CLASS_BY_NAME.get(asset, "fx")
                ],
                "train_min_bars": asset_cfg["walk_forward"]["train_min_bars"],
                "n_folds": asset_cfg["walk_forward"]["n_folds"],
                "screen_regime": screen_info["regime"],
                "hurst": screen_info.get("hurst"),
            })
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/multi_h4.yaml")
    ap.add_argument("--data-dir", default="data/H4")
    ap.add_argument(
        "--plan-only", action="store_true",
        help="Stop after pre-screening and write training_plan.json (no training).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Per-asset: n_iter=5 on 1 fold; reports timing only (~1 min/asset).",
    )
    ap.add_argument(
        "--assets", default=None,
        help="Comma-separated subset of assets to train (default: all in config).",
    )
    args = ap.parse_args()
    if args.assets:
        cli_assets = {a.strip() for a in args.assets.split(",") if a.strip()}
    else:
        cli_assets = None

    cfg = load_config(args.config)

    # B0161: resolve the effective threshold grid ONCE, before any consumer
    # reads cfg["metrics"]["threshold_grid"] (grid metrics, inner-CV selection,
    # familywise N). Under `threshold_rule: ev_breakeven_v1` the grid is derived
    # from barrier geometry + global cost constants; p* also becomes the
    # fixed-fallback threshold (audit_effective_threshold), replacing the
    # payoff-blind 0.55.
    _grid, _p_star = resolve_threshold_grid(cfg["metrics"], cfg["triple_barrier"])
    cfg["metrics"]["threshold_grid"] = _grid
    if _p_star is not None:
        # The grid is derived ONCE from the GLOBAL geometry; a per-asset
        # triple_barrier/metrics override would silently trade at a p* computed
        # from stale geometry — refuse loudly instead.
        for _ov_asset, _ov in (cfg.get("asset_overrides") or {}).items():
            if "triple_barrier" in _ov or "metrics" in _ov:
                raise ValueError(
                    f"threshold_rule=ev_breakeven_v1 is incompatible with asset_overrides."
                    f"{_ov_asset}.{{triple_barrier,metrics}}: p* is resolved once from the "
                    f"global geometry and would not match the override"
                )
        cfg["metrics"]["audit_effective_threshold"] = float(_p_star)
        print(f"threshold_rule=ev_breakeven_v1 -> p*={_p_star:.4f} "
              f"(tp={cfg['triple_barrier']['tp_atr_mult']}, sl={cfg['triple_barrier']['sl_atr_mult']}) "
              f"grid={[round(t, 4) for t in _grid]}")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    assets = list(cfg["assets"])
    print(f"Configured assets: {assets}")

    # Topological order: Level 0 (no deps) → Level 1 (depends on Level 0).
    ordered_assets = topological_order(assets)
    print(f"Execution order: {ordered_assets}")

    # Load OHLCV per asset from data/H4/.
    # Derive timeframe suffix from data_dir's last segment (e.g. "data/H4" → "H4",
    # "data/D1" → "D1"). Falls back to "H4" if the directory name is unrecognized
    # so existing H4 callers are unaffected.
    _data_dir_path = Path(args.data_dir)
    _data_tf = _data_dir_path.name.upper() if _data_dir_path.name.upper() in ("H4", "D1") else "H4"
    asset_dfs = load_multi_asset(ordered_assets, data_dir=_data_dir_path, timeframe=_data_tf)
    # Stash data_dir in cfg so the lazy-loader inside _build_features_for_asset
    # picks up the same timeframe (H4 vs D1).
    cfg["_data_dir"] = str(_data_dir_path)
    for asset, df in asset_dfs.items():
        print(f"  {asset:8s} {len(df):5d} bars")

    # Hurst pre-screening per asset.
    screening = run_pre_screening(asset_dfs, cfg)
    screening_path = output_dir / "primary_screening.json"
    screening_path.write_text(
        json.dumps(screening, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nPre-screening -> {screening_path}")
    for asset, info in screening.items():
        h_str = f"{info['hurst']:.3f}" if info.get("hurst") is not None else "n/a"
        print(f"  {asset:8s} H={h_str:>6s} regime={info['regime']:<15s} train={info['screened']}")

    # Build training plan.
    plan = build_training_plan(screening, cfg)
    plan_meta = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "config_path": str(args.config),
        "n_pairs": len(plan),
        "plan": plan,
    }
    plan_path = output_dir / "training_plan.json"
    plan_path.write_text(
        json.dumps(plan_meta, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nTraining plan -> {plan_path} ({len(plan)} pairs)")
    for entry in plan:
        print(
            f"  [{entry['asset']:8s}/{entry['primary']:16s}] "
            f"class={entry['asset_class']:<6s} bars/yr={entry['bars_per_year']} "
            f"train_min={entry['train_min_bars']} n_folds={entry['n_folds']}"
        )

    if args.plan_only:
        print("\n--plan-only: stopping before training.")
        return

    # Pre-compute macro frames shared across assets (daily FRED + intraday Yahoo).
    # The daily frame is asset-agnostic; the intraday frame's macro values are
    # fetched once and then aligned per asset to that asset's H4 index.
    if not cfg.get("date_range") or not cfg["date_range"].get("start"):
        raise ValueError("config must specify date_range.start")
    overall_start = cfg["date_range"]["start"]
    overall_end = max(df.index.max() for df in asset_dfs.values()).date().isoformat()
    print(f"\nMacro fetch window: {overall_start} - {overall_end}")
    daily_macro = build_macro_frame(overall_start, overall_end, Path("cache/fred"))
    print(f"  Daily FRED frame: {len(daily_macro)} days")

    # Optional CLI filter for asset subset.
    if cli_assets is not None:
        plan = [p for p in plan if p["asset"] in cli_assets]
        print(f"\nFiltered to {len(plan)} pair(s) via --assets={sorted(cli_assets)}")

    # B0148: pooled cross-asset meta-learner path (default OFF → never taken).
    if cfg.get("meta_pooling", {}).get("enabled", False):
        print("\nmeta_pooling.enabled=true → POOLED path "
              f"(per-asset outputs under {cfg['meta_pooling'].get('output_subdir')}).")
        _run_pooled(
            plan=plan,
            cfg=cfg,
            asset_dfs=asset_dfs,
            daily_macro=daily_macro,
            dry_run=args.dry_run,
        )
        print("\nPooled path complete. Per-asset baseline tree untouched; "
              "run scripts/compare_pooled_vs_per_asset.py for the A/B verdict.")
        return

    # Per-asset training loop.
    n_pairs = len(plan)
    for i, entry in enumerate(plan, start=1):
        asset = entry["asset"]
        primary = entry["primary"]
        print(f"\n[{i}/{n_pairs}] {asset} / {primary} (class={entry['asset_class']})")
        try:
            _run_one_asset_primary(
                entry=entry,
                cfg=apply_asset_overrides(cfg, asset),
                asset_dfs=asset_dfs,
                daily_macro=daily_macro,
                dry_run=args.dry_run,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            print(f"  (continuing with remaining plan entries)")
            continue

    if args.dry_run:
        print("\nTraining loop complete (dry-run — skipping deployment_config aggregation).")
        return

    print("\nTraining loop complete. Aggregating -> deployment_config.yaml ...")
    deployment = aggregate_to_deployment_config(
        output_dir=output_dir,
        plan=plan,
        cfg=cfg,
    )
    deployable = {a: b for a, b in deployment["assets"].items() if b["enabled"]}
    print(
        f"deployment_config.yaml written; "
        f"{len(deployable)}/{len(deployment['assets'])} assets enabled; "
        f"skipped: {deployment['skipped']}"
    )
    for asset, block in deployment["assets"].items():
        print(
            f"  {asset:8s} tier={block['deployment_tier']:<11s} "
            f"kelly={block['kelly_fraction']:.2f} "
            f"DSR={block['metrics_oos']['dsr']!s:.7s} "
            f"best={block['best_model']}"
        )


# ===========================================================================
# Per-asset/primary training loop (T9.B.2)
#
# Adapted from scripts/run_xau_d1.py::_run_one_primary. Key differences for
# multi-asset H4:
#   - Features come from build_tier2_h4_features + build_crossasset_features
#     (T5 Phase A + B), NOT build_tier2_features (D1).
#   - bars_per_year comes from the plan entry (1560 FX/metal, 2190 crypto)
#     rather than the global config.
#   - Output goes under results/clf_multi_h4/{asset}/{primary}/.
#   - asset_overrides (e.g. SOLUSD train_min_bars=1200) are pre-applied to
#     the entry's effective config in build_training_plan + apply_asset_overrides.
# ===========================================================================

def _select_primary_signal(
    name: str,
    ohlcv: pd.DataFrame,
    atr: pd.Series,
    cfg: dict,
    daily_macro: pd.DataFrame | None = None,
    asset: str | None = None,
    asset_dfs: dict | None = None,
    features: pd.DataFrame | None = None,
) -> pd.Series:
    """Primary signal dispatcher.

    daily_macro and asset are only required by macro-driven primaries (e.g.
    vix_regime_riskflow). Price-only primaries ignore them; backward-compat
    preserved.
    """
    p = cfg["primary"]
    if name == "ema_cross":
        return ema_crossover_signal(
            ohlcv["close"], atr,
            fast=p["ema_cross"]["fast"],
            slow=p["ema_cross"]["slow"],
            dead_zone_atr=p["ema_cross"]["dead_zone_atr"],
        )
    if name == "momentum_zscore":
        return momentum_zscore_signal(
            ohlcv["close"],
            lookback=p["momentum_zscore"]["lookback"],
            threshold=p["momentum_zscore"]["threshold"],
        )
    if name == "cusum_filter":
        return cusum_filter_signal(
            ohlcv["close"], atr,
            threshold_atr=p["cusum_filter"]["threshold_atr"],
        )
    if name == "bollinger_meanrev":
        return bollinger_meanrev_signal(
            ohlcv["close"],
            period=p["bollinger_meanrev"]["period"],
            k_stdev=p["bollinger_meanrev"]["k_stdev"],
        )
    if name == "vix_regime_riskflow":
        if daily_macro is None or asset is None:
            raise ValueError(
                "vix_regime_riskflow needs daily_macro and asset args — "
                "wire them through _run_one_asset_primary"
            )
        params = p["vix_regime_riskflow"]
        return vix_regime_riskflow_signal(
            target_close=ohlcv["close"],
            vix_daily=daily_macro["VIXCLS"],
            dxy_daily=daily_macro["DTWEXBGS"],
            target_symbol=asset,
            vix_lookback=params.get("vix_lookback", 252),
            vix_low_pct=params.get("vix_low_pct", 0.25),
            vix_high_pct=params.get("vix_high_pct", 0.75),
            dxy_ema_fast=params.get("dxy_ema_fast", 5),
            dxy_ema_slow=params.get("dxy_ema_slow", 20),
        )
    if name == "llm_distilled":
        # Phase 4 primary: load a frozen distilled classifier and apply it
        # to the asset's already-built feature matrix.
        from pipeline.llm_distillation import distilled_signal
        import joblib
        if features is None:
            raise ValueError(
                "llm_distilled primary needs `features` arg — wire through _run_one_asset_primary"
            )
        params = p["llm_distilled"]
        model_path = params["model_path"]   # e.g. results/phase4_distilled/{asset}_ollama_logreg.joblib
        if "{asset}" in model_path:
            model_path = model_path.format(asset=asset)
        try:
            classifier = joblib.load(model_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"llm_distilled primary requires a fitted classifier at {model_path}. "
                f"Run scripts/run_llm_distillation.py first."
            )
        # The classifier was trained on a feature subset; need to match column names.
        feature_names = params.get("feature_names")
        if feature_names is None:
            # Use all numeric feature columns from `features`
            X = features.select_dtypes(include=["float64", "float32", "int64"]).copy()
        else:
            missing = [c for c in feature_names if c not in features.columns]
            if missing:
                raise ValueError(
                    f"llm_distilled classifier expects features {missing} but they are not "
                    f"in the orchestrator's feature set. Re-train with build_tier2_h4_features "
                    f"to match the orchestrator's columns."
                )
            X = features[feature_names]
        return distilled_signal(
            X, classifier,
            confidence_threshold=params.get("confidence_threshold", 0.5),
        )
    if name == "cointegration_spread":
        if asset is None or asset_dfs is None:
            raise ValueError(
                "cointegration_spread needs asset and asset_dfs args"
            )
        params = p["cointegration_spread"]
        pairs_path = params.get("pairs_config", "configs/cointegration_pairs.yaml")
        pairs = load_pairs_config(pairs_path)
        pair = lookup_pair_for_asset(asset, pairs)
        if pair is None:
            # Asset not in any pair → return all-zero signal
            return pd.Series(0.0, index=ohlcv.index)
        other_asset = pair["other"]
        if other_asset not in asset_dfs:
            # Other asset not loaded → cannot compute spread
            return pd.Series(0.0, index=ohlcv.index)
        # Align other's close to primary's bar index (ffill for sparse alt)
        other_close = asset_dfs[other_asset]["close"].reindex(ohlcv.index, method="ffill")

        # Compute bar-space fold indices for the p-value gate
        from pipeline.walk_forward import make_folds
        wf_cfg = cfg.get("walk_forward", {})
        n_bars = len(ohlcv)
        try:
            bar_folds = make_folds(
                n=n_bars,
                n_folds=wf_cfg.get("n_folds", 4),
                train_min=min(wf_cfg.get("train_min_bars", 3000), n_bars // 2),
                purge=wf_cfg.get("purge_bars", 24),
                embargo_pct=wf_cfg.get("embargo_pct", 0.01),
            )
            fold_train_indices = [f.train_idx for f in bar_folds]
        except Exception:
            fold_train_indices = None

        return coint_spread_signal(
            primary_close=ohlcv["close"],
            other_close=other_close,
            fold_train_indices=fold_train_indices,
            zscore_lookback=params.get("zscore_lookback", 60),
            entry_threshold=params.get("entry_threshold", 2.0),
            exit_threshold=params.get("exit_threshold", 0.5),
            coint_pvalue_cutoff=params.get("coint_pvalue_cutoff", 0.10),
        )
    raise ValueError(f"Unknown primary: {name}")


def _build_features_for_asset(
    asset: str,
    asset_class: str,
    ohlcv: pd.DataFrame,
    daily_macro: pd.DataFrame,
    asset_dfs: dict[str, pd.DataFrame],
    cache_dir_yahoo: Path,
    sentiment_cache_dir: Path | None = None,
    cot_cache_dir: Path | None = None,
    funding_enabled: bool = False,
    funding_cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Compose Tier 2 H4 base + cross-asset features for one asset.

    Returns a single DataFrame ready for training (NaN rows still present;
    caller .dropna() after merging with labels/weights).
    """
    # Intraday DXY/VIX aligned to this asset's H4 index. Yahoo Finance only
    # provides H4 intraday for ~730 days; for older bars we fall back to
    # daily FRED forward-filled (with stale=1 flagged on each fallback row).
    intraday_macro = build_intraday_macro_frame_with_daily_fallback(
        bar_index=ohlcv.index,
        start=ohlcv.index.min().date().isoformat(),
        end=ohlcv.index.max().date().isoformat(),
        daily_macro_frame=daily_macro,
        cache_dir=cache_dir_yahoo,
    )

    # 24 base features (10 tech + 3 vol + 8 macro + 3 session) +
    # optional Tier 1 sentiment/COT augmentations when cache dirs provided.
    # Use the asset-aware wrapper so the COT branch can resolve the asset's
    # CFTC contract. Falls through to the same code path for non-COT runs.
    from pipeline.features import build_tier2_h4_features_for_asset
    base = build_tier2_h4_features_for_asset(
        asset=asset,
        ohlcv=ohlcv,
        macro_frame=daily_macro,
        sentiment_cache_dir=sentiment_cache_dir,
        cot_cache_dir=cot_cache_dir,
        asset_class=asset_class,
    )

    # 3-4 cross-asset features per class.
    # Cross-asset dependencies need their source DataFrames loaded from
    # data/H4/ even if those assets aren't in the current run's asset list
    # (e.g. running on subset [EURUSD, BTCUSD, XAUUSD] still needs XAGUSD
    # data for the xau_xag_ratio feature). Load on demand from disk.
    def _get_or_load(asset_name: str) -> pd.DataFrame:
        if asset_name in asset_dfs:
            return asset_dfs[asset_name]
        # Lazy-load from the same data_dir/timeframe as the current run.
        # Inferred from cfg fallback chain; defaults to H4 for backward compat.
        from pipeline.cross_asset import load_multi_asset as _lma
        _data_dir = cfg.get("_data_dir", "data/H4")
        _tf = Path(_data_dir).name.upper() if Path(_data_dir).name.upper() in ("H4", "D1") else "H4"
        return _lma([asset_name], data_dir=Path(_data_dir), timeframe=_tf)[asset_name]

    cross_kwargs: dict = {"intraday_macro": intraday_macro}
    if asset_class == "crypto" and asset != "BTCUSD":
        cross_kwargs["btc_df"] = _get_or_load("BTCUSD")
    elif asset_class == "crypto" and asset == "BTCUSD":
        cross_kwargs["btc_df"] = ohlcv  # self-ref, will yield 0 in _crypto_cross
    elif asset_class == "metal":
        cross_kwargs["daily_macro_frame"] = daily_macro
        if asset == "XAUUSD":
            cross_kwargs["xag_df"] = _get_or_load("XAGUSD")
    if asset_class == "crypto" and funding_enabled:
        cross_kwargs["funding_features_enabled"] = True
        if funding_cache_dir is not None:
            cross_kwargs["funding_cache_dir"] = str(funding_cache_dir)
    cross = build_crossasset_features(asset, ohlcv.index, **cross_kwargs)

    # For XAUUSD the metal helper leaves xau_xag_ratio as NaN — fill it now
    # via compute_xau_xag_ratio against the actual XAU OHLCV.
    if asset == "XAUUSD":
        cross["xau_xag_ratio"] = compute_xau_xag_ratio(
            ohlcv, _get_or_load("XAGUSD"), ohlcv.index,
        )

    return pd.concat([base, cross], axis=1)


def _build_asset_primary_inputs(
    entry: dict,
    cfg: dict,
    asset_dfs: dict[str, pd.DataFrame],
    daily_macro: pd.DataFrame,
) -> dict | None:
    """Phase A — build the per-(asset, primary) training inputs.

    Extracted VERBATIM from `_run_one_asset_primary` steps 1-8 (B0148 Slice 2) so
    both the per-asset path and the pooled path build identical X/y/w/fwd_ret from
    the same code — the OFF=parity guarantee. Returns a dict of everything the
    downstream fold/train block needs, OR None when the pair must be skipped (in
    which case the skip `summary.json` has already been written, mirroring the
    pre-refactor early returns).

    Returned keys: asset, primary_name, asset_class, bars_per_year, train_min_bars,
    n_folds_cfg, out_dir, cost_bps, ohlcv, X, y, w, side, fwd_ret, valid_kept,
    event_start_bar, event_end_bar (aligned to X), event_start_bar_X,
    event_end_bar_X, inner_embargo, event_time (DatetimeIndex of X.index),
    label_end_time (timestamps of t_end_idx bars on this asset).
    """
    asset = entry["asset"]
    primary_name = entry["primary"]
    asset_class = entry["asset_class"]
    bars_per_year = entry["bars_per_year"]
    train_min_bars = entry["train_min_bars"]
    n_folds_cfg = entry["n_folds"]

    out_dir = Path(cfg["output_dir"]) / asset / primary_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cost_bps = resolve_cost_bps(asset, cfg)

    ohlcv = asset_dfs[asset]

    # 1. Feature composition (base + cross-asset).
    # Tier 1 feature augmentations: enabled when cfg has explicit cache dirs.
    sentiment_dir = None
    if cfg.get("sentiment", {}).get("enabled"):
        sentiment_dir = Path(cfg["sentiment"].get("cache_dir", "cache/sentiment"))
    cot_dir = None
    if cfg.get("cot", {}).get("enabled"):
        cot_dir = Path(cfg["cot"].get("cache_dir", "cache/cot"))
    funding_enabled = bool(cfg.get("funding_features", {}).get("enabled"))
    funding_dir = Path(cfg["funding_features"].get("cache_dir", "cache/funding")) if funding_enabled else None

    features = _build_features_for_asset(
        asset=asset,
        asset_class=asset_class,
        ohlcv=ohlcv,
        daily_macro=daily_macro,
        asset_dfs=asset_dfs,
        cache_dir_yahoo=Path("cache/yahoo"),
        sentiment_cache_dir=sentiment_dir,
        cot_cache_dir=cot_dir,
        funding_enabled=funding_enabled,
        funding_cache_dir=funding_dir,
    )
    # We need `_atr_14` for triple-barrier + ema_cross dead-zone; the H4
    # builder writes it as an internal artifact in features._atr_14.
    atr = features["_atr_14"]

    # 2. Primary signal → events.
    sig = _select_primary_signal(
        primary_name, ohlcv, atr, cfg,
        daily_macro=daily_macro, asset=asset, asset_dfs=asset_dfs,
        features=features,
    )
    events = pd.DataFrame({"side": sig[sig != 0].astype(int)})

    if len(events) == 0:
        print(f"  WARNING: {asset}/{primary_name} produced 0 primary signals — skipping.")
        write_summary_json(out_dir, {
            "primary": primary_name, "asset": asset,
            "n_events": 0, "skip_reason": "no primary signals",
        })
        return None

    # 3. Triple-barrier labels.
    labels = triple_barrier_labels(
        ohlcv, events, atr,
        horizon=cfg["triple_barrier"]["horizon"],
        tp_mult=cfg["triple_barrier"]["tp_atr_mult"],
        sl_mult=cfg["triple_barrier"]["sl_atr_mult"],
    )
    valid = labels[labels["t_end_idx"] < len(ohlcv)]

    if len(valid) == 0:
        print(f"  WARNING: {asset}/{primary_name} produced 0 valid events after triple-barrier — skipping.")
        write_summary_json(out_dir, {
            "primary": primary_name, "asset": asset,
            "n_events": 0, "skip_reason": "no events survive triple-barrier",
        })
        return None

    # 4. Sample weights via avgUniqueness.
    idx_pos = {ts: i for i, ts in enumerate(ohlcv.index)}
    t_starts_all = np.array([idx_pos[ts] for ts in valid.index])
    t_ends_all = valid["t_end_idx"].values
    w_all = avg_uniqueness(t_starts_all, t_ends_all, n_bars=len(ohlcv))

    # 5. Build X: features at event times + primary state.
    state = compute_primary_state(valid["side"], cap=240)  # H4 cap = 40 days
    X = features.drop(columns=["_atr_14"]).loc[valid.index].copy()
    X["primary_side"] = state["primary_side"].values
    if primary_name == "ema_cross":
        spread = (
            ohlcv["close"].ewm(span=cfg["primary"]["ema_cross"]["fast"]).mean()
            - ohlcv["close"].ewm(span=cfg["primary"]["ema_cross"]["slow"]).mean()
        )
        X["primary_strength"] = (spread / atr).loc[valid.index].values
    else:  # momentum_zscore
        X["primary_strength"] = features["z_r24bars"].loc[valid.index].values
    X["bars_since_signal"] = state["bars_since_signal"].values

    # 6. Drop NaN rows (rolling-window warmup) and align y, w.
    # First drop columns that are 100% NaN for this asset — e.g. sent__fgi_*
    # for non-crypto assets is emitted as NaN by design (schema consistency).
    # Without this guard, the row-wise dropna() below would wipe every event
    # for FX/metal assets.
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        print(f"  Dropping all-NaN columns for {asset}/{primary_name}: {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)
    pre_drop_index = X.index
    X = X.dropna()
    if len(X) < 50:
        print(f"  WARNING: {asset}/{primary_name} has only {len(X)} rows after NaN drop — skipping.")
        write_summary_json(out_dir, {
            "primary": primary_name, "asset": asset,
            "n_events": int(len(X)), "skip_reason": "insufficient events after NaN drop",
        })
        return None
    y = valid["label"].loc[X.index]
    keep_mask = pre_drop_index.isin(X.index)
    w = w_all[keep_mask]

    assert len(w) == len(X) == len(y), (
        f"Misalignment: |w|={len(w)}, |X|={len(X)}, |y|={len(y)}"
    )
    assert X.index.equals(y.index), "X.index != y.index after NaN drop"
    assert not X.isnull().any().any(), "X still contains NaN after dropna"

    # 7. Walk-forward purge bar arrays (asset_overrides honored via entry).
    # B0129 (AFML §7.4.1): purge outer folds by each event's label-end BAR
    # (t_end_idx), aligned to X via keep_mask — not by a fixed event-count scalar.
    event_start_bar = t_starts_all[keep_mask]
    event_end_bar = t_ends_all[keep_mask]

    # 8. Forward returns from triple-barrier exit_price (gap-aware).
    close = ohlcv["close"].values
    valid_kept = valid.loc[X.index]
    entry_close = close[[idx_pos[ts] for ts in X.index]]
    exit_price = valid_kept["exit_price"].values
    fwd_ret = pd.Series(np.log(exit_price / entry_close), index=X.index)
    side = X["primary_side"]
    assert not fwd_ret.isnull().any(), "fwd_ret contains NaN — check exit_price alignment"

    # B0129 (AFML §7.4.3): per-event bar positions aligned to X for label-end
    # purging of the INNER CV (purge on every split, not just the outer folds).
    event_start_bar_X = np.array([idx_pos[ts] for ts in X.index])
    event_end_bar_X = valid_kept["t_end_idx"].values
    inner_embargo = int(np.ceil(cfg["walk_forward"]["embargo_pct"] * len(X)))

    # B0148: wall-clock timestamps for the POOLED time-purged splitters.
    # event_time = each event's entry-bar time (X.index); label_end_time =
    # the timestamp of its triple-barrier resolution bar on THIS asset.
    event_time = pd.DatetimeIndex(X.index)
    label_end_time = pd.DatetimeIndex(ohlcv.index[valid_kept["t_end_idx"].values])

    return {
        "asset": asset,
        "primary_name": primary_name,
        "asset_class": asset_class,
        "bars_per_year": bars_per_year,
        "train_min_bars": train_min_bars,
        "n_folds_cfg": n_folds_cfg,
        "out_dir": out_dir,
        "cost_bps": cost_bps,
        "ohlcv": ohlcv,
        "X": X,
        "y": y,
        "w": w,
        "side": side,
        "fwd_ret": fwd_ret,
        "valid_kept": valid_kept,
        "event_start_bar": event_start_bar,
        "event_end_bar": event_end_bar,
        "event_start_bar_X": event_start_bar_X,
        "event_end_bar_X": event_end_bar_X,
        "inner_embargo": inner_embargo,
        "event_time": event_time,
        "label_end_time": label_end_time,
    }


def _run_one_asset_primary(
    entry: dict,
    cfg: dict,
    asset_dfs: dict[str, pd.DataFrame],
    daily_macro: pd.DataFrame,
    dry_run: bool,
) -> None:
    """Train one (asset, primary) pair through the Option B threshold pipeline."""
    built = _build_asset_primary_inputs(entry, cfg, asset_dfs, daily_macro)
    if built is None:
        return  # skip summary already written by the builder

    asset = built["asset"]
    primary_name = built["primary_name"]
    asset_class = built["asset_class"]
    bars_per_year = built["bars_per_year"]
    train_min_bars = built["train_min_bars"]
    n_folds_cfg = built["n_folds_cfg"]
    out_dir = built["out_dir"]
    cost_bps = built["cost_bps"]
    X = built["X"]
    y = built["y"]
    w = built["w"]
    side = built["side"]
    fwd_ret = built["fwd_ret"]
    event_start_bar = built["event_start_bar"]
    event_end_bar = built["event_end_bar"]
    event_start_bar_X = built["event_start_bar_X"]
    event_end_bar_X = built["event_end_bar_X"]
    inner_embargo = built["inner_embargo"]

    # 7. Walk-forward folds (asset_overrides honored via entry).
    n_events = len(X)
    folds = make_folds(
        n=n_events,
        n_folds=n_folds_cfg,
        train_min=min(train_min_bars, n_events // 2),
        purge=cfg["walk_forward"]["purge_bars"],
        embargo_pct=cfg["walk_forward"]["embargo_pct"],
        event_start_bar=event_start_bar,
        event_end_bar=event_end_bar,
    )

    # 9. Training loop — same Option B pipeline as run_xau_d1.py.
    n_iter = cfg["dry_run"]["n_iter"] if dry_run else cfg["hyperparam_search"]["n_iter"]
    n_folds_eff = 1 if dry_run else len(folds)
    oof_probs = pd.DataFrame(index=X.index, columns=cfg["models"], dtype=float)
    mda_per_fold: dict[str, list[dict[str, float]]] = {m: [] for m in cfg["models"]}
    selected_threshold_per_model_per_fold: dict[str, list[float]] = {m: [] for m in cfg["models"]}
    threshold_selection_diag: dict[str, list[dict]] = {m: [] for m in cfg["models"]}
    threshold_grid_cfg = cfg["metrics"]["threshold_grid"]
    ts_cfg = cfg.get("threshold_selection", {"method": "inner_cv", "inner_splits": 3,
                                              "min_trades_per_inner_fold": 20})
    timings: dict[str, float] = {}

    # B0130: MDA scored by neg-log-loss (AFML §8.3 / MLfAM 6.3), not accuracy@0.5.
    _mda_scorer = neg_log_loss_scorer
    clustered_mda_per_fold: dict[str, list[dict[str, float]]] = {m: [] for m in cfg["models"]}

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

            # B0129: label-end bars for this fold's train slice (inner-CV purge).
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
            # XGBoost can need both classes per inner split; if degenerate skip.
            if len(np.unique(y_tr)) < 2:
                continue
            # B0156 (fit side): drop purged inner splits whose TRAIN side is
            # single-class (all candidate fits fail and the search raises);
            # skip the (model, fold) when nothing fittable remains.
            _y_tr_arr = np.asarray(y_tr)
            inner_splits = [(tr, va) for tr, va in inner_cv.split(X_tr)
                            if len(np.unique(_y_tr_arr[tr])) >= 2]
            if not inner_splits:
                continue
            search = RandomizedSearchCV(
                base, HP_SPACES[model_name], n_iter=n_iter, cv=inner_splits,
                # B0156: callable scorer, robust to single-class inner-CV slices.
                scoring=neg_log_loss_estimator_scorer,
                n_jobs=1, random_state=cfg["random_seed"],
            )
            search.fit(X_tr, y_tr, sample_weight=w_tr)
            best_kwargs = search.best_params_

            # Option B threshold selection (same as run_xau_d1.py post-T9.A.3).
            if dry_run or ts_cfg.get("method") != "inner_cv":
                # B0161: under ev_breakeven the fixed fallback is p*, not the
                # payoff-blind 0.55 (which is unreachable for tp=3/sl=1 calibrated p).
                sel_threshold = float(cfg["metrics"].get("audit_effective_threshold", 0.55))
                sel_diag = {"selected_threshold": sel_threshold,
                            "fallback_reason": "dry_run_or_grid_report_mode"}
            else:
                inner_cv_ts = PurgedTimeSeriesSplit(
                    n_splits=ts_cfg.get("inner_splits", 3),
                    purge=cfg["walk_forward"]["purge_bars"],
                    event_start_bar=esb_tr_full,
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
                sel_threshold, sel_diag = select_threshold_inner_cv(
                    side.loc[X_tr_full.index],
                    inner_oof,
                    fwd_ret.loc[X_tr_full.index],
                    bars_per_year=bars_per_year,
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

            if not dry_run:
                fold_mda = mda_importance(
                    clf, X.iloc[test_idx], y.iloc[test_idx], _mda_scorer,
                    sample_weight=w[test_idx], n_repeats=5,
                    random_state=cfg["random_seed"],
                )
                mda_per_fold[model_name].append(fold_mda)

                # B0131: Clustered Feature Importance (cluster on train, permute
                # whole clusters on test). Keyed by sorted-member string.
                clusters = cluster_features(X_tr_full, random_state=cfg["random_seed"])
                cl_imp = clustered_mda_importance(
                    clf, X.iloc[test_idx], y.iloc[test_idx], _mda_scorer, clusters,
                    sample_weight=w[test_idx], n_repeats=5,
                    random_state=cfg["random_seed"],
                )
                clustered_mda_per_fold[model_name].append({
                    "|".join(sorted(clusters[cid])): float(imp)
                    for cid, imp in cl_imp.items()
                })
        timings[model_name] = (time.time() - t0) / 60.0
        print(f"  [{primary_name}] {model_name}: {timings[model_name]:.1f} min over {n_folds_eff} fold(s)")

    if dry_run:
        budget = {m: t * n_folds_cfg * (cfg["hyperparam_search"]["n_iter"] / cfg["dry_run"]["n_iter"])
                  for m, t in timings.items()}
        print(f"  [dry-run] projected min/model: {budget}")
        write_summary_json(out_dir, {
            "primary": primary_name, "asset": asset,
            "dry_run": True, "timings_min": timings, "projected_min": budget,
        })
        return

    # 10-14. Per-asset OOS metrics / stack / PSR / DSR / summary (factored out
    # so the pooled Phase D reuses the IDENTICAL machinery on sliced OOF).
    _write_per_asset_oos(
        out_dir=out_dir,
        cfg=cfg,
        primary_name=primary_name,
        asset=asset,
        asset_class=asset_class,
        bars_per_year=bars_per_year,
        cost_bps=cost_bps,
        X=X, y=y, side=side, fwd_ret=fwd_ret, w=w,
        oof_probs=oof_probs,
        folds_test_idx=[f.test_idx for f in folds],
        selected_threshold_per_model_per_fold=selected_threshold_per_model_per_fold,
        threshold_selection_diag=threshold_selection_diag,
        mda_per_fold=mda_per_fold,
        clustered_mda_per_fold=clustered_mda_per_fold,
        n_trials_familywise=len(cfg["models"]) * len(folds) * len(threshold_grid_cfg),
    )


def _write_per_asset_oos(
    out_dir: Path,
    cfg: dict,
    primary_name: str,
    asset: str,
    asset_class: str,
    bars_per_year: int,
    cost_bps: float,
    X: pd.DataFrame,
    y: pd.Series,
    side: pd.Series,
    fwd_ret: pd.Series,
    w: np.ndarray,
    oof_probs: pd.DataFrame,
    folds_test_idx: list[np.ndarray],
    selected_threshold_per_model_per_fold: dict[str, list[float]],
    threshold_selection_diag: dict[str, list[dict]],
    mda_per_fold: dict[str, list[dict[str, float]]],
    clustered_mda_per_fold: dict[str, list[dict[str, float]]],
    n_trials_familywise: int,
    dsr_cluster_n: int | None = None,
    dsr_cluster_note: str | None = None,
) -> dict:
    """Phase D — per-asset OOS metrics, stack decision, PSR/DSR, summary.json.

    Extracted VERBATIM from `_run_one_asset_primary` steps 10-14 (B0148 Slice 2).
    Both the per-asset path AND the pooled path call this on their own OOF probs +
    test-index blocks, so the per-asset OUTPUT schema is identical regardless of
    which path produced the probabilities. Returns the summary dict (also written).

    `folds_test_idx` are positional index arrays into X (one per fold/block). For
    the per-asset path these are `make_folds` test_idx; for the pooled path they
    are the pooled OOF rows sliced to this asset, re-indexed into the asset's X.
    """
    models = cfg["models"]
    threshold_grid_cfg = cfg["metrics"]["threshold_grid"]
    # B0161 provenance guard: a cfg carrying threshold_rule WITHOUT the resolved
    # p* means main()'s resolution was bypassed (direct function call) — the run
    # would use the legacy grid while summary.json claims the EV rule. Refuse.
    if cfg["metrics"].get("threshold_rule") is not None \
            and cfg["metrics"].get("audit_effective_threshold") is None:
        raise RuntimeError(
            "metrics.threshold_rule is set but audit_effective_threshold is missing — "
            "call resolve_threshold_grid(cfg['metrics'], cfg['triple_barrier']) first "
            "(main() does this; direct callers must too)"
        )

    # 10. Compute per-fold metrics + baseline.
    fold_metrics: list[dict] = []
    grid_metrics: list[dict] = []
    sharpe_per_fold_per_model = {m: [] for m in models}
    n_trades_per_fold_per_model = {m: [] for m in models}
    max_dd_per_fold_per_model = {m: [] for m in models}
    pnl_per_model_per_fold: dict[str, list[np.ndarray]] = {m: [] for m in models}
    baseline_sharpe: list[float] = []

    for fold_k, test_idx in enumerate(folds_test_idx):
        slc = X.iloc[test_idx].index
        side_f = side.loc[slc]
        fwd_f = fwd_ret.loc[slc]
        y_f = y.loc[slc]
        if len(slc) > 1:
            span_days = (slc[-1] - slc[0]).days
            years_in_window = max(span_days / 365.25, 1e-9)
        else:
            years_in_window = 1e-9
        base_m = strategy_metrics(
            side_f, pd.Series(np.ones(len(slc), dtype=float), index=slc), fwd_f,
            cost_bps=cost_bps,
            threshold=0.5, years_in_window=years_in_window,
        )
        base_m.pop("per_trade_pnl", None)
        baseline_sharpe.append(base_m["sharpe_net"])
        for m_name in models:
            p = oof_probs.iloc[test_idx][m_name]
            cm = classification_metrics(y_f.values, p.values)
            sel_thr = selected_threshold_per_model_per_fold[m_name][fold_k] \
                if fold_k < len(selected_threshold_per_model_per_fold[m_name]) else 0.5
            sm_selected = strategy_metrics(
                side_f, p, fwd_f, cost_bps=cost_bps,
                threshold=sel_thr, years_in_window=years_in_window,
            )
            pnl_per_model_per_fold[m_name].append(sm_selected.pop("per_trade_pnl"))
            fold_metrics.append({
                "fold": fold_k, "primary": primary_name, "model": m_name,
                "threshold": sel_thr, **cm, **sm_selected,
            })
            sharpe_per_fold_per_model[m_name].append(sm_selected["sharpe_net"])
            n_trades_per_fold_per_model[m_name].append(int(sm_selected["n_trades"]))
            max_dd_per_fold_per_model[m_name].append(float(sm_selected["max_drawdown"]))
            for thr in threshold_grid_cfg:
                sm_grid = strategy_metrics(
                    side_f, p, fwd_f, cost_bps=cost_bps,
                    threshold=thr, years_in_window=years_in_window,
                )
                sm_grid.pop("per_trade_pnl", None)
                grid_metrics.append({
                    "fold": fold_k, "primary": primary_name, "model": m_name,
                    "threshold": thr, **sm_grid,
                })

    pd.DataFrame(fold_metrics).to_json(out_dir / "metrics_per_fold.json", orient="records", indent=2)
    pd.DataFrame(grid_metrics).to_json(out_dir / "threshold_grid_metrics.json", orient="records", indent=2)
    write_oof_parquet(out_dir, oof_probs)

    # 11. Stack decision (same gate as Fase 1 v4).
    oof_clean = oof_probs.dropna()
    if len(oof_clean) >= 2:
        decision = should_stack(
            sharpe_per_fold_per_model, baseline_sharpe,
            oof_clean.corr().values,
            n_trades_per_fold_per_model=n_trades_per_fold_per_model,
            min_models=cfg["stacking"]["min_models_beating_baseline"],
            min_folds=cfg["stacking"]["min_folds_beating_baseline"],
            max_corr=cfg["stacking"]["max_oof_corr"],
            min_trades_per_fold=cfg["stacking"].get("min_trades_per_fold", 30),
        )
        stack_decision = {
            "stack": decision.stack, "reason": decision.reason,
            "n_models_passing": decision.n_models_passing,
            "max_pair_corr": float(decision.max_pair_corr),
        }
    else:
        stack_decision = {"stack": False, "reason": "insufficient OOF data",
                          "n_models_passing": 0, "max_pair_corr": 0.0}

    # 12. PSR/DSR on real per-trade pnl moments.
    pnl_agg = aggregate_per_trade_pnl_metrics(pnl_per_model_per_fold)
    sr_trials = np.array(
        [sr for d in pnl_agg.values() for sr in d["per_fold_sr_per_trade"]],
        dtype=float,
    )
    # B0132 (AFML §11/§13): deflate by the family-wise trial COUNT (models ×
    # folds × threshold-grid points), a per-primary lower bound; sr_trials only
    # supplies the variance estimate. Passed in as `n_trials_familywise`.
    # [R2] When effectively-independent OOS return-series clusters are available
    # (pooled multi-cell run), `dsr_cluster_n` < familywise is the less-conservative
    # count; we still DEFLATE BY THE LARGER (familywise) count for the headline DSR
    # so the per-asset gate never gets easier from a refinement, and report both.
    psr_per_model: dict[str, float] = {}
    dsr_per_model: dict[str, float] = {}
    for m_name in models:
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

    # 13. MDA + best_model (median Sharpe heuristic; T13 selector applied in T9.B.3).
    mda_aggregated = {m: aggregate_mda_across_folds(mda_per_fold[m]) for m in models}
    clustered_mda_aggregated = {m: aggregate_mda_across_folds(clustered_mda_per_fold[m]) for m in models}
    median_sharpes = {
        m: float(np.nanmedian(sharpe_per_fold_per_model[m]))
        if any(np.isfinite(s) for s in sharpe_per_fold_per_model[m])
        else float("nan")
        for m in models
    }
    finite_medians = {m: v for m, v in median_sharpes.items() if np.isfinite(v)}
    best_model = max(finite_medians, key=finite_medians.get) if finite_medians else models[0]

    # 14. Persist outputs.
    (out_dir / "mda_per_fold.json").write_text(
        json.dumps({m: mda_per_fold[m] for m in models}, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "clustered_mda.json").write_text(
        json.dumps({
            "per_fold": {m: clustered_mda_per_fold[m] for m in models},
            "aggregated": clustered_mda_aggregated,
        }, indent=2, default=str),
        encoding="utf-8",
    )
    psr_dsr_payload = {
        "psr": psr_per_model,
        "dsr": dsr_per_model,
        # B0132: empirical trial-Sharpe sample size vs family-wise trial count.
        "n_trial_sharpes_sampled": int(len(sr_trials)),
        "n_trials_familywise": int(n_trials_familywise),
        "n_trials_familywise_note": "per-primary lower bound: models × folds × threshold-grid points",
        "per_model_aggregate": {
            m: {k: v for k, v in d.items() if k != "per_fold_sr_per_trade"}
            for m, d in pnl_agg.items()
        },
        "trial_pool_sr_per_trade": [float(s) for s in sr_trials],
    }
    # [R2] Optional effectively-independent cluster count (pooled multi-cell only).
    if dsr_cluster_n is not None:
        psr_dsr_payload["n_trials_effective_clusters"] = int(dsr_cluster_n)
        psr_dsr_payload["n_trials_effective_clusters_note"] = (
            dsr_cluster_note
            or "AFML §14.7.3 effectively-independent OOS-return-series clusters; "
               "headline DSR deflates by the larger (familywise) count for safety"
        )
    (out_dir / "psr_dsr.json").write_text(
        json.dumps(psr_dsr_payload, indent=2, default=str),
        encoding="utf-8",
    )
    (out_dir / "threshold_selection.json").write_text(
        json.dumps({
            "selected_per_fold": selected_threshold_per_model_per_fold,
            "diagnostics": threshold_selection_diag,
        }, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)),
        encoding="utf-8",
    )

    # Calibration plots per model.
    y_oof = y.loc[oof_probs.index]
    for m_name in models:
        mask = oof_probs[m_name].notna()
        if mask.sum() > 0:
            plot_calibration(
                out_dir,
                y_true=y_oof[mask].values,
                y_prob=oof_probs.loc[mask, m_name].values,
                model_name=m_name,
            )

    median_selected_threshold_per_model = {
        m: float(np.median(selected_threshold_per_model_per_fold[m]))
        if selected_threshold_per_model_per_fold[m] else 0.5
        for m in models
    }
    summary = {
        "primary": primary_name,
        "asset": asset,
        "asset_class": asset_class,
        "bars_per_year": bars_per_year,
        "n_events": int(len(X)),
        "n_folds": len(folds_test_idx),
        "baseline_sharpe_per_fold": baseline_sharpe,
        "sharpe_per_fold_per_model": sharpe_per_fold_per_model,
        "n_trades_per_fold_per_model": n_trades_per_fold_per_model,
        "max_dd_per_fold_per_model": max_dd_per_fold_per_model,
        "selected_threshold_per_fold_per_model": selected_threshold_per_model_per_fold,
        "median_selected_threshold_per_model": median_selected_threshold_per_model,
        "median_sharpe": median_sharpes,
        "best_model": best_model,
        "stack_decision": stack_decision,
        # B0161: threshold provenance + whole-OOF discrimination, so "does the
        # meta discriminate at all?" is answerable from summary.json alone.
        "threshold_rule": cfg["metrics"].get("threshold_rule"),
        "p_star": cfg["metrics"].get("audit_effective_threshold"),
        "threshold_grid": [float(t) for t in threshold_grid_cfg],
        "oof_discrimination_per_model": _oof_discrimination(y, oof_probs, models),
    }
    write_summary_json(out_dir, summary)
    print(
        f"  [done] {asset}/{primary_name}: PSR={psr_per_model}, DSR={dsr_per_model}, "
        f"best={best_model}, stack={stack_decision['stack']}"
    )
    return summary


def _oof_discrimination(y: pd.Series, oof_probs: pd.DataFrame, models: list[str]) -> dict:
    """Whole-OOF (all folds pooled) discrimination per model: ROC AUC, Brier,
    base rate, n. Positional alignment (NaN-prob rows dropped) so it is safe
    for both the per-asset and pooled-slice paths regardless of index quirks.

    AUC ~0.5 here means no fix to the threshold can create an edge — firing
    more would only trade noise at cost. That is the honest-null check B0161
    makes visible without parquet spelunking.
    """
    y_arr = np.asarray(y, dtype=float)
    out: dict[str, dict] = {}
    for m in models:
        p = oof_probs[m].to_numpy(dtype=float)
        mask = ~np.isnan(p)
        if not mask.any():
            out[m] = {"n": 0, "base_rate": float("nan"),
                      "roc_auc": float("nan"), "brier": float("nan")}
            continue
        cm = classification_metrics(y_arr[mask].astype(int), p[mask])
        out[m] = {"n": int(mask.sum()), "base_rate": float(y_arr[mask].mean()),
                  "roc_auc": cm["roc_auc"], "brier": cm["brier"]}
    return out


# ===========================================================================
# POOLED cross-asset meta-learner (B0148 Slice 2) — Phases B / C / D.
#
# Behind `meta_pooling.enabled` (default OFF → never reached). Trains ONE meta
# classifier per (primary, pool-key) on the UNION of member events, then slices
# the pooled OOF back per asset and reuses `_write_per_asset_oos` so the
# per-asset output schema is identical to the baseline → drop-in A/B.
#
# Invariants preserved: primaries rule-based per asset (Phase A unchanged);
# sample weights flow to base fit + RandomizedSearchCV + calibration; calibration
# = sigmoid via FrozenEstimator on a POOLED-WALL-CLOCK-TIME tail (B2); inner CV
# purged in TIME via PurgedTimeGroupSplit; pooled uniqueness via
# pooled_avg_uniqueness (B1); should_stack 30-trade gate kept.
# ===========================================================================

# H4 bar duration per class, used to size the pooled embargo (wall-clock) off the
# COARSEST member's vertical-barrier horizon (spec [R]). FX/metal H4 ≈ 1560
# bars/yr; crypto trades 7d/wk → 2190 bars/yr. Duration = 365.25d / bars_per_year.
_BAR_DURATION_BY_CLASS = {
    "fx": pd.Timedelta(days=365.25 / 1560),
    "metal": pd.Timedelta(days=365.25 / 1560),
    "crypto": pd.Timedelta(days=365.25 / 2190),
}

_POOL_CLASSES = ("fx", "metal", "crypto")


def _pooled_embargo_td(members: list[dict], cfg: dict) -> pd.Timedelta:
    """Embargo as a wall-clock multiple of the triple-barrier vertical-barrier
    horizon, sized off the COARSEST member's bar duration (spec [R]).

    horizon (bars) × coarsest bar-duration covers the longest in-flight label
    across the pool, conservatively. The `embargo_pct`-style fraction is folded
    into the horizon directly: we embargo the full vertical-barrier horizon (the
    max time a label can stay open), which is stricter than LdP's 0.01T heuristic.
    """
    horizon_bars = int(cfg["triple_barrier"]["horizon"])
    coarsest = max(
        (_BAR_DURATION_BY_CLASS.get(m["asset_class"], _BAR_DURATION_BY_CLASS["fx"]) for m in members),
        default=_BAR_DURATION_BY_CLASS["fx"],
    )
    return coarsest * horizon_bars


def _pooled_core_schema(members: list[dict]) -> list[str]:
    """Pooled-core feature schema = shared base columns present in ALL members
    (intersection) + the 3 primary cols, in a deterministic order. The asset-class
    one-hot columns are appended later by `_compose_pooled_X` (they don't exist on
    any member yet). Excludes the 3 primary cols from the intersection step so they
    are always present even if a member somehow lacks one (they never do).
    """
    primary_cols = ["primary_side", "primary_strength", "bars_since_signal"]
    # Intersection of base columns (everything except the primary cols).
    shared: set[str] | None = None
    for m in members:
        base_cols = set(m["X"].columns) - set(primary_cols)
        shared = base_cols if shared is None else (shared & base_cols)
    shared = shared or set()
    # Deterministic order: sorted shared base, then the fixed primary cols.
    ordered_base = sorted(shared)
    return ordered_base + primary_cols


def _compose_pooled_X(
    member: dict, shared_cols: list[str], schema: str
) -> pd.DataFrame:
    """Restrict a member's X to the pooled schema and append the asset-class one-hot.

    core (default/confirmatory): shared_cols only (intersection + primary cols) +
    `is_fx,is_metal,is_crypto` one-hot. No NaN by construction.

    extended (exploratory): NaN-union — every member keeps its own columns; absent
    columns become NaN (xgb/catboost/rf handle NaN natively). The shared core is a
    subset; the one-hot is still appended.
    """
    cls = member["asset_class"]
    if schema == "core":
        Xc = member["X"].reindex(columns=shared_cols).copy()
    elif schema == "extended":
        # NaN-union handled at concat time (pandas aligns columns, fills NaN);
        # keep the member's full column set here.
        Xc = member["X"].copy()
    else:
        raise NotImplementedError(
            f"meta_pooling.schema={schema!r} not implemented (use 'core' or 'extended')"
        )
    for c in _POOL_CLASSES:
        Xc[f"is_{c}"] = 1.0 if cls == c else 0.0
    return Xc


def _run_pooled(
    plan: list[dict],
    cfg: dict,
    asset_dfs: dict[str, pd.DataFrame],
    daily_macro: pd.DataFrame,
    dry_run: bool,
) -> None:
    """Pooled-path orchestrator (B0148 Slice 2). Replaces the per-asset training
    loop when `meta_pooling.enabled` is true. Writes per-asset summary.json under
    `meta_pooling.output_subdir` so the comparison harness can A/B against the
    baseline tree.
    """
    mp = cfg["meta_pooling"]
    scope = mp.get("scope", "within_class")
    schema = mp.get("schema", "core")
    weight_balance = mp.get("weight_balance", "per_class")
    pooled_uniqueness = bool(mp.get("pooled_uniqueness", True))
    train_min_frac = float(mp.get("train_min_frac", 0.5))
    out_root = Path(mp.get("output_subdir", "results/clf_multi_h4_pooled"))
    out_root.mkdir(parents=True, exist_ok=True)

    # --- Phase A: build every plan member's inputs (rule-based, per-asset). ---
    # cfg passed here must already have asset_overrides applied per member.
    built_members: list[dict] = []
    for entry in plan:
        asset = entry["asset"]
        eff_cfg = apply_asset_overrides(cfg, asset)
        try:
            b = _build_asset_primary_inputs(entry, eff_cfg, asset_dfs, daily_macro)
        except Exception as exc:  # noqa: BLE001
            print(f"  Phase-A ERROR {asset}/{entry['primary']}: {type(exc).__name__}: {exc}")
            continue
        if b is None:
            continue
        b["pool_key"] = b["asset_class"] if scope == "within_class" else "all"
        built_members.append(b)

    if not built_members:
        print("  POOLED: no members survived Phase A — nothing to pool.")
        return

    # --- Phase B: group by (primary, pool_key). ---
    groups: dict[tuple[str, str], list[dict]] = {}
    for b in built_members:
        groups.setdefault((b["primary_name"], b["pool_key"]), []).append(b)

    print(f"\nPOOLED path: scope={scope} schema={schema} weight_balance={weight_balance} "
          f"pooled_uniqueness={pooled_uniqueness}; {len(groups)} pool(s).")

    for (primary_name, pool_key), members in groups.items():
        print(f"\n=== POOL ({primary_name}, {pool_key}): "
              f"{[m['asset'] for m in members]} ===")
        _run_one_pool(
            primary_name=primary_name,
            pool_key=pool_key,
            members=members,
            cfg=cfg,
            schema=schema,
            weight_balance=weight_balance,
            pooled_uniqueness=pooled_uniqueness,
            train_min_frac=train_min_frac,
            out_root=out_root,
            dry_run=dry_run,
        )

    print("\nPOOLED training complete.")


def _run_one_pool(
    primary_name: str,
    pool_key: str,
    members: list[dict],
    cfg: dict,
    schema: str,
    weight_balance: str,
    pooled_uniqueness: bool,
    train_min_frac: float,
    out_root: Path,
    dry_run: bool,
) -> None:
    """Phase B concat + Phase C pooled train + Phase D per-asset OOS, for ONE pool."""
    models = cfg["models"]

    # --- Phase B: compose pooled X/y/w + time/asset arrays. ---
    shared_cols = _pooled_core_schema(members)
    X_parts, y_parts, asset_parts = [], [], []
    et_parts, le_parts = [], []
    w_peas_parts = []   # per-asset avg_uniqueness (for none/per_asset balance fallback)
    for m in members:
        Xc = _compose_pooled_X(m, shared_cols, schema)
        X_parts.append(Xc)
        y_parts.append(m["y"])
        asset_parts.append(np.array([m["asset"]] * len(Xc)))
        et_parts.append(pd.DatetimeIndex(m["event_time"]))
        le_parts.append(pd.DatetimeIndex(m["label_end_time"]))
        w_peas_parts.append(np.asarray(m["w"], dtype=float))

    # concat preserves row order = member order; reset to a clean RangeIndex so
    # positional slicing (folds, calib tail) is unambiguous across duplicate
    # timestamps from different assets.
    X = pd.concat(X_parts, axis=0, ignore_index=False)
    if schema == "extended":
        # NaN-union: align all members' columns then re-append one-hot order.
        X = X  # pd.concat already unioned columns, filling absent with NaN
    X = X.reset_index(drop=True)
    y = pd.concat(y_parts, axis=0).reset_index(drop=True)
    asset_row = np.concatenate(asset_parts)
    event_time = pd.DatetimeIndex(np.concatenate([e.values for e in et_parts]))
    label_end_time = pd.DatetimeIndex(np.concatenate([l.values for l in le_parts]))
    w_peas = np.concatenate(w_peas_parts)

    n_pooled = len(X)
    assert len(y) == n_pooled == len(asset_row) == len(event_time) == len(label_end_time)

    # --- B1: pooled cross-asset wall-clock uniqueness weights. ---
    pooled_u = pooled_avg_uniqueness(event_time, label_end_time)
    w = pooled_u.copy() if pooled_uniqueness else w_peas.copy()

    # --- §4: weight balancing on EFFECTIVE event mass. ---
    w = _balance_pool_weights(w, asset_row, members, weight_balance)

    # --- Pre-flight: effective-N (AFML effective sample size = Σ pooled
    # uniqueness weights, balancing-independent) vs the make_folds refusal floor.
    eff_n = float(np.sum(pooled_u))
    raw_n = int(n_pooled)
    n_folds = int(cfg["walk_forward"]["n_folds"])
    floor = wf_event_floor(n_folds, cfg["walk_forward"]["train_min_bars"])
    print(f"  PRE-FLIGHT [{primary_name}/{pool_key}]: raw_N={raw_n}  "
          f"pooled_effective_N={eff_n:.1f}  refusal_floor≈{floor}  "
          f"{'OK (relieves starvation)' if eff_n >= floor else 'BELOW FLOOR — pooling may not relieve constraint'}")

    embargo_td = _pooled_embargo_td(members, cfg)

    # --- Phase C: pooled time-purged folds + per-model training. ---
    bars_per_year_by_member = {m["asset"]: m["bars_per_year"] for m in members}
    # within_class pools share one bars_per_year; global → annualize per-asset in D.
    pool_bars_per_year = members[0]["bars_per_year"] if pool_key != "all" else None

    try:
        folds = make_pooled_time_folds(
            event_time=event_time,
            label_end_time=label_end_time,
            n_folds=(1 if dry_run else n_folds),
            train_min_frac=train_min_frac,
            embargo_td=embargo_td,
            asset=asset_row,
        )
    except ValueError as exc:
        print(f"  POOL ({primary_name},{pool_key}) make_pooled_time_folds refused: {exc}")
        return

    n_iter = cfg["dry_run"]["n_iter"] if dry_run else cfg["hyperparam_search"]["n_iter"]
    threshold_grid_cfg = cfg["metrics"]["threshold_grid"]
    ts_cfg = cfg.get("threshold_selection", {"method": "inner_cv", "inner_splits": 3,
                                             "min_trades_per_inner_fold": 20})
    calib_pct = cfg["calibration"]["calib_holdout_pct"]

    side = X["primary_side"]
    fwd_ret = pd.concat([m["fwd_ret"].reset_index(drop=True) for m in members],
                        axis=0).reset_index(drop=True)
    fwd_ret.index = X.index
    side.index = X.index

    oof_probs = pd.DataFrame(index=X.index, columns=models, dtype=float)
    selected_threshold_per_model_per_fold: dict[str, list[float]] = {m: [] for m in models}
    threshold_selection_diag: dict[str, list[dict]] = {m: [] for m in models}
    mda_per_fold: dict[str, list[dict[str, float]]] = {m: [] for m in models}
    clustered_mda_per_fold: dict[str, list[dict[str, float]]] = {m: [] for m in models}

    et_ns = event_time.asi8

    for model_name in models:
        t0 = time.time()
        for fold in folds:
            tr = np.asarray(fold.train_idx)
            te = np.asarray(fold.test_idx)
            if len(tr) == 0 or len(te) == 0:
                continue
            X_tr_full = X.iloc[tr]
            y_tr_full = y.iloc[tr]
            w_tr_full = w[tr]
            et_tr_full = event_time[tr]
            le_tr_full = label_end_time[tr]

            if len(np.unique(y_tr_full)) < 2:
                continue

            # B2: calibration holdout = chronological tail in POOLED WALL-CLOCK
            # TIME (sort train rows by event_time, take latest calib_pct), and
            # PURGE the calib boundary by label_end_time so no train label leaks
            # into the calib tail.
            order_tr = np.argsort(et_tr_full.asi8, kind="stable")
            n_hold = max(int(len(tr) * calib_pct), 1)
            calib_local = order_tr[-n_hold:]
            train_local = order_tr[:-n_hold]
            calib_first_ns = int(et_tr_full.asi8[calib_local].min())
            # purge: drop train rows whose label_end >= calib window start.
            le_tr_ns = le_tr_full.asi8
            train_local = train_local[le_tr_ns[train_local] < calib_first_ns]
            if len(train_local) == 0 or len(np.unique(y_tr_full.iloc[train_local])) < 2:
                continue

            X_tr = X_tr_full.iloc[train_local]
            y_tr = y_tr_full.iloc[train_local]
            w_tr = w_tr_full[train_local]
            X_ca = X_tr_full.iloc[calib_local]
            y_ca = y_tr_full.iloc[calib_local]
            w_ca = w_tr_full[calib_local]

            # Inner CV purged in TIME, bound to the TRAIN rows used for the search.
            inner_cv = PurgedTimeGroupSplit(
                n_splits=cfg["hyperparam_search"]["cv_splits"],
                event_time=et_tr_full[train_local],
                label_end_time=le_tr_full[train_local],
                embargo_td=embargo_td,
            )
            base = MODEL_FACTORIES[model_name](random_state=cfg["random_seed"])
            # B0156 (fit side): drop purged inner splits whose TRAIN side is
            # single-class; skip the (model, fold) when none remain.
            _y_tr_arr = np.asarray(y_tr)
            inner_splits = [(tr, va) for tr, va in inner_cv.split(X_tr)
                            if len(np.unique(_y_tr_arr[tr])) >= 2]
            if not inner_splits:
                continue
            search = RandomizedSearchCV(
                base, HP_SPACES[model_name], n_iter=n_iter, cv=inner_splits,
                # B0156: callable scorer, robust to single-class inner-CV slices.
                scoring=neg_log_loss_estimator_scorer,
                n_jobs=1, random_state=cfg["random_seed"],
            )
            search.fit(X_tr, y_tr, sample_weight=w_tr)
            best_kwargs = search.best_params_

            # Threshold selection over a time-purged inner CV on the FULL train slice.
            bpy = pool_bars_per_year or members[0]["bars_per_year"]
            if dry_run or ts_cfg.get("method") != "inner_cv":
                # B0161: fixed fallback is p* under ev_breakeven (see per-asset path).
                sel_threshold = float(cfg["metrics"].get("audit_effective_threshold", 0.55))
                sel_diag = {"selected_threshold": sel_threshold,
                            "fallback_reason": "dry_run_or_grid_report_mode"}
            else:
                inner_cv_ts = PurgedTimeGroupSplit(
                    n_splits=ts_cfg.get("inner_splits", 3),
                    event_time=et_tr_full,
                    label_end_time=le_tr_full,
                    embargo_td=embargo_td,
                )
                rcp = RefittingCalibratedPipeline(
                    model_name=model_name,
                    base_kwargs=best_kwargs,
                    calib_holdout_pct=calib_pct,
                    method=cfg["calibration"].get("method", "sigmoid"),
                    random_state=cfg["random_seed"],
                )
                inner_oof_full, inner_val_indices = inner_oof_predict_proba(
                    rcp, X_tr_full, y_tr_full, inner_cv_ts,
                    sample_weight=w_tr_full,
                    return_val_indices=True,
                )
                inner_oof = pd.Series(inner_oof_full[:, 1], index=X_tr_full.index)
                sel_threshold, sel_diag = select_threshold_inner_cv(
                    side.loc[X_tr_full.index],
                    inner_oof,
                    fwd_ret.loc[X_tr_full.index],
                    bars_per_year=bpy,
                    threshold_grid=np.asarray(threshold_grid_cfg, dtype=float),
                    cost_bps=members[0]["cost_bps"],
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
            oof_probs.iloc[te, oof_probs.columns.get_loc(model_name)] = \
                clf.predict_proba(X.iloc[te])[:, 1]

            if not dry_run:
                fold_mda = mda_importance(
                    clf, X.iloc[te], y.iloc[te], neg_log_loss_scorer,
                    sample_weight=w[te], n_repeats=5,
                    random_state=cfg["random_seed"],
                )
                mda_per_fold[model_name].append(fold_mda)
                clusters = cluster_features(X_tr, random_state=cfg["random_seed"])
                cl_imp = clustered_mda_importance(
                    clf, X.iloc[te], y.iloc[te], neg_log_loss_scorer, clusters,
                    sample_weight=w[te], n_repeats=5,
                    random_state=cfg["random_seed"],
                )
                clustered_mda_per_fold[model_name].append({
                    "|".join(sorted(clusters[cid])): float(imp)
                    for cid, imp in cl_imp.items()
                })
        print(f"  [pool {primary_name}/{pool_key}] {model_name}: "
              f"{(time.time() - t0) / 60.0:.1f} min over {len(folds)} fold(s)")

    if dry_run:
        print(f"  [dry-run] pool ({primary_name},{pool_key}) trained 1 fold — "
              f"skipping per-asset OOS write.")
        return

    # --- [R2] DSR effectively-independent cluster count over the pooled cells. ---
    dsr_cluster_n, dsr_cluster_note = _pooled_dsr_cluster_n(oof_probs, fwd_ret, side)

    # --- Phase D: slice pooled OOF per asset → reuse _write_per_asset_oos. ---
    n_folds_eff = len(folds)
    n_trials_familywise = len(models) * n_folds_eff * len(threshold_grid_cfg)
    for m in members:
        asset = m["asset"]
        rows = np.flatnonzero(asset_row == asset)
        if len(rows) == 0:
            continue
        # Map the pooled rows of this asset back to the asset's own X order, and
        # restore the asset's DatetimeIndex (event_time) so _write_per_asset_oos's
        # per-fold calendar-span `years_in_window` matches the baseline path exactly.
        a_index = event_time[rows]
        X_a = X.iloc[rows].copy(); X_a.index = a_index
        y_a = y.iloc[rows].copy(); y_a.index = a_index
        side_a = side.iloc[rows].copy(); side_a.index = a_index
        fwd_a = fwd_ret.iloc[rows].copy(); fwd_a.index = a_index
        w_a = w[rows]
        oof_a = oof_probs.iloc[rows].copy(); oof_a.index = a_index
        # Re-derive per-asset fold test blocks: which of this asset's local rows
        # belong to each pooled test block.
        pooled_to_local = {g: i for i, g in enumerate(rows)}
        folds_test_idx_a = []
        for fold in folds:
            te_global = [pooled_to_local[g] for g in fold.test_idx if g in pooled_to_local]
            folds_test_idx_a.append(np.asarray(sorted(te_global), dtype=int))
        bpy = m["bars_per_year"]   # per-asset annualization (Phase D)
        out_dir = out_root / asset / primary_name
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            _write_per_asset_oos(
                out_dir=out_dir,
                cfg=cfg,
                primary_name=primary_name,
                asset=asset,
                asset_class=m["asset_class"],
                bars_per_year=bpy,
                cost_bps=m["cost_bps"],
                X=X_a, y=y_a, side=side_a, fwd_ret=fwd_a, w=w_a,
                oof_probs=oof_a,
                folds_test_idx=folds_test_idx_a,
                selected_threshold_per_model_per_fold=selected_threshold_per_model_per_fold,
                threshold_selection_diag=threshold_selection_diag,
                mda_per_fold=mda_per_fold,
                clustered_mda_per_fold=clustered_mda_per_fold,
                n_trials_familywise=n_trials_familywise,
                dsr_cluster_n=dsr_cluster_n,
                dsr_cluster_note=dsr_cluster_note,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  Phase-D ERROR {asset}/{primary_name}: {type(exc).__name__}: {exc}")
            continue


def _balance_pool_weights(
    w: np.ndarray, asset_row: np.ndarray, members: list[dict], weight_balance: str
) -> np.ndarray:
    """§4 sample-weight balancing on EFFECTIVE event mass.

    per_class (default): renormalize so each asset CLASS contributes equal total
    effective weight. per_asset: equal total per asset. none: raw (pooled
    uniqueness only). Balancing preserves the overall total weight so absolute
    magnitudes stay comparable across configs.
    """
    if weight_balance == "none":
        return w
    cls_by_asset = {m["asset"]: m["asset_class"] for m in members}
    out = w.astype(float).copy()
    if weight_balance == "per_asset":
        keys = asset_row
    elif weight_balance == "per_class":
        keys = np.array([cls_by_asset[a] for a in asset_row])
    else:
        raise NotImplementedError(f"weight_balance={weight_balance!r}")
    total = out.sum()
    uniq = np.unique(keys)
    if total <= 0 or len(uniq) == 0:
        return out
    target_per_group = total / len(uniq)
    for g in uniq:
        mask = keys == g
        gsum = out[mask].sum()
        if gsum > 0:
            out[mask] *= target_per_group / gsum
    return out


def _pooled_dsr_cluster_n(
    oof_probs: pd.DataFrame, fwd_ret: pd.Series, side: pd.Series
) -> tuple[int | None, str | None]:
    """[R2] Effectively-independent trial count = number of clusters of the
    per-(model) OOS return series. When only one usable cell exists, returns N=1
    with a note; falls back to None (caller keeps familywise) if too few series.

    The OOS "return series" per model = side * fwd_ret on rows where that model's
    OOF prob >= 0.5 (a coarse act mask), aligned across models by row position.
    Correlated cells cluster together → cluster count is the AFML §14.7.3
    independent-trial count.
    """
    series = {}
    for m in oof_probs.columns:
        p = oof_probs[m]
        mask = p.notna() & (p >= 0.5)
        r = (side * fwd_ret).where(mask, 0.0)
        if r.abs().sum() > 0:
            series[m] = r.values
    if len(series) < 2:
        return (1 if len(series) == 1 else None,
                "single usable cell — N=familywise conservative fallback")
    mat = pd.DataFrame(series)
    try:
        clusters = cluster_features(mat, random_state=42)
        n_clusters = len(clusters)
    except Exception:
        return None, "clustering failed — keep familywise count"
    return int(n_clusters), (
        f"AFML §14.7.3 cluster count over {len(series)} per-model OOS return series"
    )


# ===========================================================================
# Deployment-config aggregator (T9.B.3)
#
# Reads the per-(asset, primary) outputs written by _run_one_asset_primary
# and applies the T13 best-model selector + T12 tier function to produce a
# single deployment_config.yaml — the input that Phase 3 (live trading)
# consumes for sizing and asset selection.
# ===========================================================================


def _read_one_pair(output_dir: Path, asset: str, primary: str) -> dict | None:
    """Read summary.json + psr_dsr.json for one (asset, primary). Returns
    None when the pair has no usable summary (skip_reason set or files missing)."""
    pair_dir = Path(output_dir) / asset / primary
    summary_path = pair_dir / "summary.json"
    psr_path = pair_dir / "psr_dsr.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "skip_reason" in summary:
        return None
    psr_dsr = json.loads(psr_path.read_text(encoding="utf-8")) if psr_path.exists() else {}
    return {"summary": summary, "psr_dsr": psr_dsr}


def _max_dd_for_model(summary: dict, model: str) -> float:
    """Return the WORST per-fold max_drawdown for `model` (negative number).
    Convert to positive % for the deployment tier (which expects max_drawdown_pct)."""
    per_fold = summary.get("max_dd_per_fold_per_model", {}).get(model, [])
    if not per_fold:
        return 0.0
    worst = float(min(per_fold))  # max_drawdown is a NEGATIVE number (equity drop)
    return abs(worst)


def _build_deployment_entry(asset: str, asset_class: str, summary: dict, psr_dsr: dict, cfg: dict) -> dict:
    """Compose the per-asset YAML block: apply T13 best_model + T12 tier."""
    from pipeline.best_model import select_best_model
    from pipeline.deployment import AssetResults, asset_deployment_tier

    psr_per_model = dict(psr_dsr.get("psr", {}))
    dsr_per_model = dict(psr_dsr.get("dsr", {}))
    psr_dsr_per_model = {
        m: {"psr": psr_per_model.get(m, float("nan")),
            "dsr": dsr_per_model.get(m, float("nan"))}
        for m in summary.get("sharpe_per_fold_per_model", {})
    }
    median_sharpe = {m: float(np.nanmedian(v)) if any(np.isfinite(s) for s in v) else 0.0
                     for m, v in summary["sharpe_per_fold_per_model"].items()}
    n_trades_per_fold = summary["n_trades_per_fold_per_model"]

    # T13: DSR-aware best-model selector
    best_model, selection_reason = select_best_model(
        psr_dsr_per_model,
        median_sharpe,
        n_trades_per_fold,
        min_trades_per_fold=cfg["best_model"]["min_trades_per_fold"],
        min_folds_with_trades=cfg["best_model"]["min_folds_with_trades"],
    )

    # T12: tier from best_model's PSR/DSR + hard gates
    best_psr = psr_dsr_per_model[best_model]["psr"]
    best_dsr = psr_dsr_per_model[best_model]["dsr"]
    per_model_agg = psr_dsr.get("per_model_aggregate", {}).get(best_model, {})
    n_trades_total = int(per_model_agg.get("n_trades", sum(n_trades_per_fold[best_model])))
    max_dd_pct = _max_dd_for_model(summary, best_model)

    tier = asset_deployment_tier(AssetResults(
        psr=best_psr if best_psr == best_psr else 0.0,  # NaN → 0 for tier check (will hard-gate)
        dsr=best_dsr if best_dsr == best_dsr else 0.0,
        max_drawdown_pct=max_dd_pct,
        n_trades_total=n_trades_total,
    ))

    selected_threshold_per_fold = summary.get("selected_threshold_per_fold_per_model", {}).get(best_model, [])
    median_thr = float(np.median(selected_threshold_per_fold)) if selected_threshold_per_fold else 0.5

    primary_params = cfg["primary"].get(summary["primary"], {})

    return {
        "enabled": tier.tier not in ("disabled",),
        "deployment_tier": tier.tier,
        "kelly_fraction": tier.kelly_fraction,
        "primary": summary["primary"],
        "primary_params": primary_params,
        "best_model": best_model,
        "selection_reason": selection_reason["criterion"],
        "selection_diag": {
            k: v for k, v in selection_reason.items() if k != "criterion"
        },
        "use_stack": summary["stack_decision"]["stack"],
        "stack_reason": summary["stack_decision"]["reason"],
        "threshold_median": median_thr,
        "threshold_per_fold": selected_threshold_per_fold,
        "metrics_oos": {
            "per_trade_sr": per_model_agg.get("sr_per_trade"),
            "psr": best_psr,
            "dsr": best_dsr,
            "n_trades": n_trades_total,
            "max_dd_pct": max_dd_pct,
            "median_annualized_sharpe": median_sharpe.get(best_model, 0.0),
        },
        "asset_class": asset_class,
        "deployment_note": tier.reason,
    }


def aggregate_to_deployment_config(
    output_dir: Path | str,
    plan: list[dict],
    cfg: dict,
) -> dict:
    """Aggregate per-(asset, primary) outputs into the deployment_config dict.

    Returns the dict and ALSO writes it to `{output_dir}/deployment_config.yaml`.
    When an asset has multiple primaries (rare — only when force_both_primaries
    is set since Hurst pre-screening usually picks one), the primary whose
    deployment_tier is highest wins.

    Tier ranking (highest to lowest): full > half > quarter > paper_only > disabled.
    """
    output_dir = Path(output_dir)
    tier_rank = {"full": 4, "half": 3, "quarter": 2, "paper_only": 1, "disabled": 0}

    # Group plan entries by asset.
    by_asset: dict[str, list[dict]] = {}
    for entry in plan:
        by_asset.setdefault(entry["asset"], []).append(entry)

    assets_block: dict[str, dict] = {}
    trial_pool_sr: list[float] = []
    skipped: list[str] = []

    for asset, entries in by_asset.items():
        candidate_blocks: list[dict] = []
        for entry in entries:
            primary = entry["primary"]
            pair = _read_one_pair(output_dir, asset, primary)
            if pair is None:
                skipped.append(f"{asset}/{primary}")
                continue
            block = _build_deployment_entry(
                asset=asset,
                asset_class=entry["asset_class"],
                summary=pair["summary"],
                psr_dsr=pair["psr_dsr"],
                cfg=cfg,
            )
            candidate_blocks.append(block)
            trial_pool_sr.extend(pair["psr_dsr"].get("trial_pool_sr_per_trade", []))

        if not candidate_blocks:
            skipped.append(asset)
            continue
        # Pick the best primary per asset (highest tier rank).
        best_block = max(candidate_blocks, key=lambda b: tier_rank.get(b["deployment_tier"], -1))
        assets_block[asset] = best_block

    out_dict = {
        "last_modified": datetime.now(tz=timezone.utc).isoformat(),
        "trial_pool_size": len(trial_pool_sr),
        "skipped": skipped,
        "assets": assets_block,
    }
    out_path = output_dir / "deployment_config.yaml"
    out_path.write_text(yaml.safe_dump(out_dict, sort_keys=False), encoding="utf-8")
    return out_dict


if __name__ == "__main__":
    main()
