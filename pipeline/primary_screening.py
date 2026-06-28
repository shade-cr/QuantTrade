"""Primary signal pre-screening via Hurst exponent (Phase 2 T7).

Motivation: Phase 1 v4 showed that `momentum_zscore` is dead on XAU D1 —
xgb and catboost get 0 trades across all folds; only rf finds a marginal
signal (per-trade SR 0.029, PSR 0.601). Replicating both candidate
primaries on all 8 H4 assets without pre-screening would waste 30-50% of
the Phase 2 training budget on experiments theory predicts will fail.

The Hurst exponent (Mandelbrot 1971) is a classic measure of persistence:
  - H > 0.5 → trending series (momentum-following works)
  - H = 0.5 → random walk
  - H < 0.5 → mean-reverting

This module computes H on the FIRST `train_min_bars` of the price series
(chronological head, never tail) and maps the result to a primary
shortlist. The look-ahead constraint is the critical invariant — H must
NEVER see the test folds, since the decision "which primaries to train"
becomes a model decision otherwise and leaks future information.

Empirical note on `hurst.compute_Hc(kind='price')`: R/S analysis on
~3000-bar cumulative random walks shows a small-sample upward bias of
~0.05-0.15 (H ≈ 0.55-0.68 instead of the theoretical 0.5). This means
pure-random-walk assets may be classified as `trending` and only get
ema_cross trained. We accept this for Phase 2 because (a) the deflation
test (DSR) downstream catches assets where the chosen primary has no
real edge, and (b) operating with the slightly aggressive default keeps
the budget savings (~30-50%) that motivate pre-screening in the first
place. If we observe many assets falling on the H>0.52 side with weak
post-train metrics, the right response is to widen `h_trending` to
~0.58 in the config — NOT to widen the ambiguous zone, which would
return both candidates and erase the budget benefit.
"""
from __future__ import annotations
import pandas as pd


_MIN_BARS_FOR_HURST = 500


def screen_primaries_for_asset(
    df: pd.DataFrame,
    train_min_bars: int = 3000,
    candidates: list[str] | None = None,
    h_trending: float = 0.52,
    h_mr: float = 0.48,
    force_both_primaries: list[str] | None = None,
    asset: str | None = None,
) -> tuple[list[str], dict]:
    """Pre-screen which primary candidates to train for this asset.

    Computes the Hurst exponent on `df['close'].head(train_min_bars)` —
    CRITICAL: head, not tail. The chronologically earliest bars are the
    only window available before the walk-forward outer's first test
    fold, so this preserves OOS for the screening decision itself.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'close' column. Other columns are ignored.
    train_min_bars : int
        Number of leading bars to use for the Hurst computation. Matches
        the outer WF `train_min_bars` config so the screening window is
        exactly the data available before the first test fold.
    candidates : list[str] | None
        Pool of primary names. Default ["ema_cross", "momentum_zscore"].
    h_trending : float
        H > h_trending → return only `ema_cross`.
    h_mr : float
        H < h_mr → return only `momentum_zscore`.
        For h_mr <= H <= h_trending the regime is ambiguous and both
        candidates are returned.
    force_both_primaries : list[str] | None
        If `asset` is in this list, the function returns all candidates
        regardless of H. Use this for assets where theory predicts a
        specific behaviour but you want empirical validation.
    asset : str | None
        Asset name, used for the `force_both_primaries` lookup only.

    Returns
    -------
    (screened, diagnostic) : tuple[list[str], dict]
        - screened: subset of `candidates` to train.
        - diagnostic: dict with keys
            - hurst (float | None)
            - regime ("trending" | "mean_reverting" | "mixed" |
                     "insufficient_data" | "compute_error")
            - data_window ("pre_fold" — for the audit trail)
            - n_bars_used (int)
            - skipped_primaries (list[str])
            - error (str, only on compute_error)
            - override (str, only when force_both_primaries kicks in)
    """
    if candidates is None:
        candidates = ["ema_cross", "momentum_zscore"]

    # Manual override: assets listed in force_both_primaries skip the
    # Hurst check entirely. We still compute H (cheaply) for the diagnostic
    # if there's enough data, so the audit trail shows what was overridden.
    if asset is not None and force_both_primaries and asset in force_both_primaries:
        # CRITICAL: head(train_min_bars), not tail.
        pre_fold = df["close"].head(train_min_bars).values
        hurst_value = _compute_hurst_safe(pre_fold)
        return list(candidates), {
            "hurst": hurst_value,
            "regime": "overridden",
            "data_window": "pre_fold",
            "n_bars_used": len(pre_fold),
            "skipped_primaries": [],
            "override": "force_both_primaries",
        }

    # CRITICAL: head(train_min_bars), not tail. Using tail leaks the test
    # period into the screening decision.
    pre_fold = df["close"].head(train_min_bars).values

    if len(pre_fold) < _MIN_BARS_FOR_HURST:
        return list(candidates), {
            "hurst": None,
            "regime": "insufficient_data",
            "data_window": "pre_fold",
            "n_bars_used": len(pre_fold),
            "skipped_primaries": [],
            "error": (
                f"only {len(pre_fold)} pre-fold bars (need ≥{_MIN_BARS_FOR_HURST})"
            ),
        }

    try:
        from hurst import compute_Hc
        H, _c, _data = compute_Hc(pre_fold, kind="price", simplified=True)
    except Exception as e:  # noqa: BLE001 — any failure → conservative fallback
        return list(candidates), {
            "hurst": None,
            "regime": "compute_error",
            "data_window": "pre_fold",
            "n_bars_used": len(pre_fold),
            "skipped_primaries": [],
            "error": str(e),
        }

    H = float(H)

    if H > h_trending:
        screened = ["ema_cross"] if "ema_cross" in candidates else list(candidates)
        regime = "trending"
    elif H < h_mr:
        screened = ["momentum_zscore"] if "momentum_zscore" in candidates else list(candidates)
        regime = "mean_reverting"
    else:
        screened = list(candidates)
        regime = "mixed"

    return screened, {
        "hurst": H,
        "regime": regime,
        "data_window": "pre_fold",
        "n_bars_used": len(pre_fold),
        "skipped_primaries": [p for p in candidates if p not in screened],
    }


def _compute_hurst_safe(prices) -> float | None:
    """Best-effort Hurst computation that swallows errors (returns None)."""
    if len(prices) < _MIN_BARS_FOR_HURST:
        return None
    try:
        from hurst import compute_Hc
        H, _c, _data = compute_Hc(prices, kind="price", simplified=True)
        return float(H)
    except Exception:  # noqa: BLE001
        return None
