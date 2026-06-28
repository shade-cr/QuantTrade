"""Tests for primary pre-screening via Hurst exponent (Phase 2 T7).

The most important invariant: the function must compute H on the FIRST
`train_min_bars` of the price series (chronological head), NEVER on the
tail. Using the tail leaks future information into the "which primaries
to train" decision.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from pipeline.primary_screening import screen_primaries_for_asset


def _df_from_close(close: np.ndarray) -> pd.DataFrame:
    """Wrap a price array into a DataFrame with the columns the function needs."""
    n = len(close)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame({"close": close}, index=idx)


def test_trending_series_screens_to_ema_cross_only():
    """A strongly trending series (random walk with positive drift) should
    have H clearly > 0.52 → screen returns ["ema_cross"]."""
    rng = np.random.default_rng(0)
    n = 3000
    # Cumulative drift dominates the noise → strong trending behaviour.
    increments = rng.standard_normal(n) * 0.01 + 0.005
    close = 100.0 + np.cumsum(increments)
    df = _df_from_close(close)

    screened, diag = screen_primaries_for_asset(
        df, train_min_bars=3000, candidates=["ema_cross", "momentum_zscore"],
    )

    assert screened == ["ema_cross"], f"expected only ema_cross, got {screened}"
    assert diag["regime"] == "trending"
    assert diag["hurst"] > 0.52, f"H should be > 0.52 for trending, got {diag['hurst']}"


def test_mean_reverting_series_screens_to_momentum_only():
    """A mean-reverting series (sine + noise around a constant) should have
    H clearly < 0.48 → screen returns ["momentum_zscore"]."""
    rng = np.random.default_rng(1)
    n = 3000
    t = np.arange(n)
    close = 100.0 + 5.0 * np.sin(t * 2 * np.pi / 50) + rng.standard_normal(n) * 0.5
    df = _df_from_close(close)

    screened, diag = screen_primaries_for_asset(
        df, train_min_bars=3000, candidates=["ema_cross", "momentum_zscore"],
    )

    assert screened == ["momentum_zscore"], f"expected only momentum_zscore, got {screened}"
    assert diag["regime"] == "mean_reverting"
    assert diag["hurst"] < 0.48, f"H should be < 0.48 for MR, got {diag['hurst']}"


def test_ambiguous_zone_keeps_both_candidates():
    """When H falls in the ambiguous zone [h_mr, h_trending] the function
    returns BOTH candidates ("mixed" regime).

    Implementation note: `hurst.compute_Hc(kind='price', simplified=True)`
    on cumulative random walks of ~3000 bars is empirically biased upward
    (H ≈ 0.55-0.68 across seeds, not 0.5). This is a well-known
    small-sample artifact of R/S analysis on non-stationary series. So
    "pure random walk gives H ≈ 0.5" is NOT a reliable assumption to
    bake into the test.

    What we actually test here: the function's threshold *logic* — given
    an H value and bracketing h_mr/h_trending, it should classify as
    "mixed" and return both. We measure the empirical H of a synthetic
    series first, then pass thresholds that bracket it.
    """
    from hurst import compute_Hc
    rng = np.random.default_rng(2)
    n = 3000
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    df = _df_from_close(close)
    H_emp, _, _ = compute_Hc(close, kind="price", simplified=True)

    # Bracket the empirical H so it falls strictly inside (h_mr, h_trending).
    screened, diag = screen_primaries_for_asset(
        df, train_min_bars=3000,
        candidates=["ema_cross", "momentum_zscore"],
        h_trending=float(H_emp) + 0.05,
        h_mr=float(H_emp) - 0.05,
    )

    assert set(screened) == {"ema_cross", "momentum_zscore"}, (
        f"expected both candidates, got {screened}"
    )
    assert diag["regime"] == "mixed"
    assert diag["hurst"] == pytest.approx(float(H_emp))


def test_force_both_primaries_overrides_hurst_decision():
    """When the asset is listed in `force_both_primaries`, the function
    returns both candidates regardless of what H says."""
    # Build a trending series — without override would screen to ["ema_cross"].
    rng = np.random.default_rng(0)
    n = 3000
    increments = rng.standard_normal(n) * 0.01 + 0.005
    close = 100.0 + np.cumsum(increments)
    df = _df_from_close(close)

    screened, diag = screen_primaries_for_asset(
        df,
        asset="BTCUSD",
        train_min_bars=3000,
        candidates=["ema_cross", "momentum_zscore"],
        force_both_primaries=["BTCUSD"],
    )

    assert set(screened) == {"ema_cross", "momentum_zscore"}, (
        f"force_both should override and return both, got {screened}"
    )
    assert diag.get("override") == "force_both_primaries"


def test_insufficient_data_falls_back_to_all_candidates():
    """With < 500 pre-fold bars, Hurst is unreliable → return both candidates
    as the conservative fallback. The diagnostic must mark the regime as
    `insufficient_data` so this is auditable."""
    n = 300  # below the 500-bar floor
    close = 100.0 + np.cumsum(np.random.default_rng(0).standard_normal(n) * 0.1)
    df = _df_from_close(close)

    screened, diag = screen_primaries_for_asset(
        df,
        train_min_bars=3000,  # we have only 300 bars — train_min_bars is the request
        candidates=["ema_cross", "momentum_zscore"],
    )

    assert set(screened) == {"ema_cross", "momentum_zscore"}, (
        f"insufficient data should keep both candidates, got {screened}"
    )
    assert diag["regime"] == "insufficient_data"
    assert diag.get("hurst") is None


def test_uses_head_not_tail_no_look_ahead():
    """CRITICAL invariant: H must be computed over `df['close'].head(train_min_bars)`,
    NEVER `df['close'].tail(...)`. Using the tail leaks future information
    into the screening decision.

    Verification: construct a series that is mean-reverting in its FIRST
    half and trending in its SECOND half. If the function uses head(),
    it sees the MR portion → screen=["momentum_zscore"]. If it (wrongly)
    uses tail(), it sees the trending portion → screen=["ema_cross"].
    """
    rng = np.random.default_rng(3)
    n = 6000  # 3000 head + 3000 tail
    half = n // 2
    t = np.arange(half)
    head_close = 100.0 + 5.0 * np.sin(t * 2 * np.pi / 50) + rng.standard_normal(half) * 0.5
    increments = rng.standard_normal(half) * 0.01 + 0.005
    tail_close = head_close[-1] + np.cumsum(increments)
    close = np.concatenate([head_close, tail_close])
    df = _df_from_close(close)

    screened, diag = screen_primaries_for_asset(
        df, train_min_bars=half, candidates=["ema_cross", "momentum_zscore"],
    )

    # head() of the first 3000 rows is MR → screen MR.
    assert screened == ["momentum_zscore"], (
        f"function should see MR head (rows 0..3000), got screen={screened}. "
        f"If you see ['ema_cross'], the function is wrongly using tail() — LOOK-AHEAD LEAK."
    )
    assert diag["data_window"] == "pre_fold"


def test_xau_d1_historical_cross_check():
    """Cross-check against Phase 1 v4: XAU D1 head(1500) should screen to
    ema_cross (regime trending). Phase 1 confirmed empirically that
    momentum_zscore is dead on XAU D1 — the pre-screening should reach
    the same conclusion ex-ante.

    Skipped when data/XAUUSD_D1.csv is missing (CI without bundled data).
    """
    from pathlib import Path
    data_path = Path(__file__).resolve().parents[1] / "data" / "XAUUSD_D1.csv"
    if not data_path.exists():
        pytest.skip(f"missing {data_path} — bundle data for CI to run this check")

    df = pd.read_csv(data_path, parse_dates=[0])
    df = df.set_index(df.columns[0])
    # train_min_bars matches the value configs/xau_d1.yaml uses.
    screened, diag = screen_primaries_for_asset(df, train_min_bars=1500)

    assert screened == ["ema_cross"], (
        f"XAU D1 head(1500) should screen to ema_cross (Phase 1 v4 confirmed "
        f"momentum_zscore is dead). Got: {screened}"
    )
    assert diag["regime"] == "trending"
    assert diag["hurst"] > 0.52, f"H should be > 0.52, got {diag['hurst']}"
