"""Inner-CV threshold selection — Phase 2 refactor (Option B).

The function now receives OOF probabilities pre-computed upstream (typically
via `pipeline.walk_forward.inner_oof_predict_proba` driving a
`RefittingCalibratedPipeline` over `PurgedTimeSeriesSplit`). It does not
train any model or open any CV — it only aggregates `strategy_metrics`
per threshold over chronological sub-blocks with NaN-masking, returning the
best threshold and a diagnostic trail.

Why this refactor (vs Phase 1 v3+, where this function did its own inner
training + CV): separating "produce OOF probs" from "score thresholds on
those probs" makes both pieces independently testable, and reuses the
same inner-CV path for the wrapper that Phase 2 multi-asset needs. See
plan v2.3.1 and changelog for the design rationale.

NaN semantics:
  - `prediction` may contain NaN (rows outside any val fold of the upstream
    helper). `strategy_metrics` treats `NaN >= threshold` as False, so NaN
    rows don't count as trades — they don't pollute per-block Sharpe.
  - When a sub-block has fewer than `min_trades_per_inner_fold` trades at
    a given threshold, that block's Sharpe is masked to NaN before
    averaging across blocks (lesson Phase 1 v3a → v3b).
  - If no threshold ever produced ≥min_trades in any sub-block, falls
    back to the lowest threshold in the grid with `fallback_used=True`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from pipeline.metrics import strategy_metrics


def select_threshold_inner_cv(
    side: pd.Series,
    prediction: pd.Series,
    fwd_return: pd.Series,
    bars_per_year: int,
    threshold_grid: np.ndarray = np.arange(0.50, 0.66, 0.02),
    cost_bps: float = 10.0,
    min_trades_per_inner_fold: int = 20,
    n_sub_blocks: int = 3,
    sub_block_indices: list[np.ndarray] | None = None,
) -> tuple[float, dict]:
    """Select a threshold by scoring each candidate over chronological sub-blocks.

    Parameters
    ----------
    side : pd.Series
        ±1 trade direction per event. Same length and order as `prediction`
        and `fwd_return`.
    prediction : pd.Series
        OOF probability of "take the trade" (typically `predict_proba(...)[:, 1]`).
        May contain NaN for rows outside any val fold of the upstream CV.
    fwd_return : pd.Series
        Realized log-return per event (signed by `side` inside
        `strategy_metrics`).
    bars_per_year : int
        Annualisation factor for Sharpe (252 for D1, ~1560 for FX/Metal H4,
        ~2190 for crypto H4). Replaces the DatetimeIndex-derived
        `years_in_window` of the Phase 1 path.
    threshold_grid : np.ndarray
        Candidate thresholds. Must be sorted ascending if you want the
        fallback (min) semantics to mean "most permissive".
    cost_bps : float
        Per-trade cost in basis points (subtracted inside `strategy_metrics`).
    min_trades_per_inner_fold : int
        A sub-block's per-threshold Sharpe is NaN-masked if its n_trades at
        that threshold is below this floor.
    n_sub_blocks : int
        Number of chronological sub-blocks. Each block evaluates every
        threshold in the grid; per-threshold averages across blocks are
        NaN-safe. 3 is the Phase 1 default; the function is well-defined
        for any `n_sub_blocks ≥ 1`. Ignored when `sub_block_indices` is
        provided.
    sub_block_indices : list[np.ndarray] | None
        Explicit positional index arrays defining the sub-blocks (one
        array per block). When provided, these are used in place of the
        chronological `n_sub_blocks` split. Pass the `val_indices` list
        returned by `inner_oof_predict_proba(..., return_val_indices=True)`
        to evaluate Sharpe over exactly the CV val regions (matching the
        Phase 1 v3+ per-CV-fold averaging). When `prediction.index` is a
        `DatetimeIndex`, the per-block annualisation is derived from
        calendar span (`span_days / 365.25`) to match Phase 1 v3+ exactly;
        otherwise it falls back to `len(block) / bars_per_year`.

    Returns
    -------
    (best_threshold, diagnostic) : tuple[float, dict]
        - best_threshold: float, the threshold from `threshold_grid` with
          the highest NaN-safe mean Sharpe across sub-blocks. If no
          threshold ever cleared the trade floor in any sub-block, falls
          back to `min(threshold_grid)`.
        - diagnostic: dict with keys
            - selected_threshold (float)
            - selected_score (float, only when not fallback)
            - grid_scores_avg (dict[float, float], only when not fallback)
            - grid_n_trades (dict[float, list[int]] across blocks)
            - fallback_used (bool)
            - fallback_reason (str, only when fallback_used)
    """
    n = len(side)
    if sub_block_indices is not None:
        sub_blocks = list(sub_block_indices)
    else:
        if n_sub_blocks < 1:
            raise ValueError(f"n_sub_blocks must be ≥ 1, got {n_sub_blocks}")
        block_size = max(n // n_sub_blocks, 1)
        sub_blocks = []
        for i in range(n_sub_blocks):
            start = i * block_size
            end = (i + 1) * block_size if i < n_sub_blocks - 1 else n
            if start >= n:
                break
            sub_blocks.append(np.arange(start, end))

    # If the prediction Series has a DatetimeIndex, derive years per block
    # from calendar span (Phase 1 v3+ convention). Otherwise use the
    # bars_per_year fallback. This matters when chronological sub-blocks
    # don't align with calendar years (e.g. inner CV val ranges spanning
    # variable numbers of bars).
    is_datetime = isinstance(prediction.index, pd.DatetimeIndex)

    grid_scores: dict[float, list[float]] = {float(thr): [] for thr in threshold_grid}
    grid_n_trades: dict[float, list[int]] = {float(thr): [] for thr in threshold_grid}

    for val_idx in sub_blocks:
        side_val = side.iloc[val_idx]
        pred_val = prediction.iloc[val_idx]
        fwd_val = fwd_return.iloc[val_idx]
        if is_datetime and len(val_idx) > 1:
            val_index = prediction.index[val_idx]
            span_days = (val_index[-1] - val_index[0]).days
            years_in_window = max(span_days / 365.25, 1e-9)
        else:
            years_in_window = max(len(val_idx) / bars_per_year, 1e-9)

        for thr in threshold_grid:
            m = strategy_metrics(
                side_val, pred_val, fwd_val,
                cost_bps=cost_bps,
                threshold=float(thr),
                years_in_window=years_in_window,
                min_trades_for_sharpe=min_trades_per_inner_fold,
            )
            grid_n_trades[float(thr)].append(int(m["n_trades"]))
            if m["n_trades"] < min_trades_per_inner_fold:
                grid_scores[float(thr)].append(float("nan"))
            else:
                grid_scores[float(thr)].append(float(m["sharpe_net"]))

    # NaN-safe per-threshold mean across sub-blocks.
    avg_scores: dict[float, float] = {}
    for thr, scores in grid_scores.items():
        finite = [s for s in scores if np.isfinite(s)]
        if finite:
            avg_scores[thr] = float(np.mean(finite))

    if not avg_scores:
        fallback = float(threshold_grid.min())
        return fallback, {
            "selected_threshold": fallback,
            "fallback_used": True,
            "fallback_reason": (
                "no sub-block reached min_trades_per_inner_fold at any threshold"
            ),
            "grid_n_trades": grid_n_trades,
        }

    best_threshold = max(avg_scores, key=avg_scores.get)
    return float(best_threshold), {
        "selected_threshold": float(best_threshold),
        "selected_score": avg_scores[best_threshold],
        "grid_scores_avg": avg_scores,
        "grid_n_trades": grid_n_trades,
        "fallback_used": False,
    }
