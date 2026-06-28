"""DSR-aware best-model selector (Phase 2 T13).

Motivation: Phase 1 v4 exposed a latent issue with the orchestrator's
default best-model selector (median annualised Sharpe). On XAU D1
ema_cross:
  - catboost: median Sharpe +0.54 → ranked best by old criterion
  - rf: median Sharpe 0.00 but DSR 0.257 (the only model with materially
    positive deflated significance)

For single-asset XAU D1 the discrepancy is borderline either way. For
multi-asset H4 with 8 assets × 3 models × N folds, picking the wrong
criterion can invert deployment decisions wholesale. Phase 2 switches to
DSR-aware ranking with a n_trades-based qualification gate; falls back
to median Sharpe only when no model has enough folds with trades to
support a DSR comparison.
"""
from __future__ import annotations


def select_best_model(
    psr_dsr_per_model: dict,
    sharpe_median_per_model: dict,
    n_trades_per_fold_per_model: dict,
    min_trades_per_fold: int = 30,
    min_folds_with_trades: int = 2,
) -> tuple[str, dict]:
    """Select the best model by DSR among those that clear the trade-count gate.

    Logic (in order):
      1. A model "qualifies" if ≥`min_folds_with_trades` of its folds have
         ≥`min_trades_per_fold` trades. This is the gate that excludes
         models with high DSR but thin distributional support.
      2. ≥2 qualified → rank by DSR. The winner's DSR may or may not
         agree with median Sharpe — the test
         `test_h4_multi_asset_scenario_dsr_aware_overrides_median_sharpe`
         encodes the case where they disagree.
      3. Exactly 1 qualified → that model is the best (no contest).
      4. 0 qualified → fallback to argmax(median Sharpe). The diagnostic
         carries a `warning` so callers can flag this case (typically
         results in `disabled` or `paper_only` tiers in deployment).

    Parameters
    ----------
    psr_dsr_per_model : dict[str, dict[str, float]]
        Per-model {"psr": float, "dsr": float}.
    sharpe_median_per_model : dict[str, float]
        Per-model median annualised Sharpe across folds. Used only in
        the fallback path.
    n_trades_per_fold_per_model : dict[str, list[int]]
        Per-model list of n_trades per outer fold.
    min_trades_per_fold : int
        Per-fold floor: a fold "has trades" if it produces ≥ this many.
        Default 30, matching the stack-decision gate.
    min_folds_with_trades : int
        How many folds-with-trades a model needs to qualify. Default 2.

    Returns
    -------
    (best_model, reason) : tuple[str, dict]
        - best_model: name of the chosen model.
        - reason: diagnostic dict with at least `criterion`
          ("dsr_aware" | "single_qualified" | "median_sharpe_fallback")
          and `qualified_models`. DSR-aware path also includes
          `dsr_ranking`, `best_dsr`, `best_psr`. Fallback includes a
          `warning` string.
    """
    qualified: list[str] = []
    for model, n_trades_list in n_trades_per_fold_per_model.items():
        n_folds_ok = sum(1 for n in n_trades_list if n >= min_trades_per_fold)
        if n_folds_ok >= min_folds_with_trades:
            qualified.append(model)

    if len(qualified) >= 2:
        # DSR-aware ranking among the qualified set.
        best = max(qualified, key=lambda m: psr_dsr_per_model[m]["dsr"])
        return best, {
            "criterion": "dsr_aware",
            "qualified_models": qualified,
            "dsr_ranking": {m: psr_dsr_per_model[m]["dsr"] for m in qualified},
            "best_dsr": psr_dsr_per_model[best]["dsr"],
            "best_psr": psr_dsr_per_model[best]["psr"],
        }

    if len(qualified) == 1:
        only = qualified[0]
        return only, {
            "criterion": "single_qualified",
            "qualified_models": qualified,
            "dsr": psr_dsr_per_model[only]["dsr"],
            "psr": psr_dsr_per_model[only]["psr"],
        }

    # 0 qualified → fallback to median Sharpe (best effort under thin data).
    best = max(sharpe_median_per_model, key=sharpe_median_per_model.get)
    return best, {
        "criterion": "median_sharpe_fallback",
        "qualified_models": [],
        "warning": "no_model_qualified_by_n_trades",
        "median_sharpe": sharpe_median_per_model[best],
    }
