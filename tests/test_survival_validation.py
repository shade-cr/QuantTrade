"""Tests for pipeline.survival_validation — robust validation of the survival book
(B0112). These pin the methodology the risk-officer review demanded:

  * the bootstrap is ALIGNED-MULTIVARIATE and GAP-PRESERVING — it resamples whole
    calendar-aligned rows of the return matrix, never mixing assets across time,
    so cross-asset correlation and weekend-crypto gap (NaN) structure survive;
  * it is a STATIONARY block bootstrap (Politis-Romano 1994) — geometric block
    lengths, wrap-around — so serial dependence (vol clustering) survives;
  * it is deterministic under a seed (backtest/reproducibility parity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.survival_validation import (
    stationary_bootstrap_paths,
    bootstrap_metric_distribution,
    summarize_distribution,
    optimal_block_length,
    recommended_block_length,
    inject_shock,
    scale_vol_window,
    probability_of_backtest_overfitting,
)


def _toy_rets() -> pd.DataFrame:
    """Small multi-asset return matrix with a deliberate GAP: asset C has NaN for
    the first 3 rows (models a late-listing crypto leg) so the gap-preservation
    property is testable."""
    idx = pd.date_range("2020-01-01", periods=12, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "A": rng.normal(0, 0.01, 12),
            "B": rng.normal(0, 0.02, 12),
            "C": rng.normal(0, 0.05, 12),
        },
        index=idx,
    )
    df.loc[df.index[:3], "C"] = np.nan  # late-listing gap
    return df


def _row_keys(df: pd.DataFrame) -> set:
    """Hashable key per row, NaN-safe, so we can test set membership of rows."""
    keys = set()
    for _, row in df.iterrows():
        keys.add(tuple("NaN" if pd.isna(v) else round(float(v), 12) for v in row))
    return keys


def test_bootstrap_preserves_shape_and_columns():
    rets = _toy_rets()
    paths = stationary_bootstrap_paths(rets, expected_block_len=3.0, n_paths=5, seed=1)
    assert len(paths) == 5
    for p in paths:
        assert p.shape == rets.shape
        assert list(p.columns) == list(rets.columns)
        assert p.index.equals(rets.index)


def test_bootstrap_rows_are_verbatim_original_rows():
    """ALIGNED-MULTIVARIATE + GAP-PRESERVING: every resampled row must equal SOME
    original row exactly (including its NaNs). If columns were resampled
    independently, a row would mix assets from different dates and this fails."""
    rets = _toy_rets()
    orig_keys = _row_keys(rets)
    paths = stationary_bootstrap_paths(rets, expected_block_len=3.0, n_paths=10, seed=2)
    for p in paths:
        assert _row_keys(p).issubset(orig_keys)


def test_bootstrap_is_deterministic_under_seed():
    rets = _toy_rets()
    a = stationary_bootstrap_paths(rets, expected_block_len=3.0, n_paths=4, seed=7)
    b = stationary_bootstrap_paths(rets, expected_block_len=3.0, n_paths=4, seed=7)
    for pa, pb in zip(a, b):
        pd.testing.assert_frame_equal(pa, pb)


def test_bootstrap_preserves_serial_blocks_with_long_block_length():
    """With a very long expected block length the bootstrap rarely restarts, so a
    path is dominated by one long contiguous (wrap-around) run of the original
    rows — i.e. consecutive output rows are consecutive original rows. We assert
    the fraction of consecutive-original transitions is high."""
    rets = _toy_rets()
    n = len(rets)
    # Map each original row-key to its position(s) to detect contiguity.
    [path] = stationary_bootstrap_paths(rets, expected_block_len=1000.0, n_paths=1, seed=3)
    # Reconstruct the source position of each output row via verbatim match.
    pos_by_key = {}
    for i, (_, row) in enumerate(rets.iterrows()):
        key = tuple("NaN" if pd.isna(v) else round(float(v), 12) for v in row)
        pos_by_key.setdefault(key, []).append(i)
    src = []
    for _, row in path.iterrows():
        key = tuple("NaN" if pd.isna(v) else round(float(v), 12) for v in row)
        src.append(pos_by_key[key][0])
    consecutive = sum(
        1 for j in range(1, len(src)) if src[j] == (src[j - 1] + 1) % n
    )
    assert consecutive >= len(src) - 2  # at most ~1 block restart over the path


# --------------------------------------------------------------------------- #
# summarize_distribution — percentile summary of a metric's bootstrap samples
# --------------------------------------------------------------------------- #
def test_summarize_distribution_percentiles():
    samples = list(range(101))  # 0..100 -> p5=5, p50=50, p95=95
    s = summarize_distribution(samples, percentiles=(5, 50, 95))
    assert s["p5"] == pytest.approx(5.0)
    assert s["p50"] == pytest.approx(50.0)
    assert s["p95"] == pytest.approx(95.0)
    assert s["mean"] == pytest.approx(50.0)
    assert s["n"] == 101


# --------------------------------------------------------------------------- #
# bootstrap_metric_distribution — run a metric over many bootstrap paths
# --------------------------------------------------------------------------- #
def test_bootstrap_metric_distribution_aggregates_per_metric_key():
    rets = _toy_rets()

    def metric_fn(path: pd.DataFrame) -> dict:
        return {"vol_A": float(path["A"].std(ddof=1))}

    dist = bootstrap_metric_distribution(
        rets, metric_fn, expected_block_len=3.0, n_paths=8, seed=11
    )
    assert set(dist.keys()) == {"vol_A"}
    assert len(dist["vol_A"]) == 8
    assert all(np.isfinite(v) for v in dist["vol_A"])


def test_bootstrap_metric_distribution_is_deterministic():
    rets = _toy_rets()
    mf = lambda p: {"vol_A": float(p["A"].std(ddof=1))}
    a = bootstrap_metric_distribution(rets, mf, expected_block_len=3.0, n_paths=6, seed=5)
    b = bootstrap_metric_distribution(rets, mf, expected_block_len=3.0, n_paths=6, seed=5)
    assert a == b


# --------------------------------------------------------------------------- #
# optimal_block_length — Politis-White (2004) automatic data-driven selection
# --------------------------------------------------------------------------- #
def _ar1(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, 1, n)
    x = np.empty(n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def test_optimal_block_length_iid_is_short():
    """IID (no serial dependence) -> the bootstrap needs no long blocks; PW2004
    returns a short block length."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 3000)
    b = optimal_block_length(x)
    assert 1.0 <= b <= 6.0


def test_optimal_block_length_grows_with_persistence():
    """A strongly autocorrelated AR(1) needs LONGER blocks than IID, and more
    persistence -> longer blocks (monotone in phi)."""
    b_iid = optimal_block_length(_ar1(3000, 0.0, 1))
    b_mid = optimal_block_length(_ar1(3000, 0.5, 1))
    b_high = optimal_block_length(_ar1(3000, 0.9, 1))
    assert b_high > b_mid > b_iid
    assert b_high > 10.0  # strong persistence -> clearly multi-day blocks


def test_recommended_block_length_uses_abs_returns_and_takes_max():
    """For a return matrix, the recommended block length is driven by VOLATILITY
    CLUSTERING (which lives in |returns|, not the near-white returns), taken as the
    max over assets so the most persistent leg's dependence is preserved. A column
    with volatility clustering must drive a LONGER block than an IID-vol column."""
    rng = np.random.default_rng(7)
    n = 3000
    # Column 1: IID returns (no vol clustering). Column 2: vol-clustered (GARCH-like)
    # via an AR(1) in the VARIANCE.
    iid = rng.normal(0, 0.01, n)
    vol = np.empty(n)
    vol[0] = 0.01
    for t in range(1, n):
        vol[t] = np.sqrt(0.9 * vol[t - 1] ** 2 + 0.1 * (0.01 ** 2))
    clustered = rng.normal(0, 1, n) * vol
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    df = pd.DataFrame({"iid": iid, "clustered": clustered}, index=idx)
    b = recommended_block_length(df)
    assert b >= optimal_block_length(np.abs(iid))
    assert b == pytest.approx(
        max(optimal_block_length(np.abs(iid)), optimal_block_length(np.abs(clustered)))
    )
    assert b >= 1.0


# --------------------------------------------------------------------------- #
# Synthetic stress primitives — inject adversarial events into a return path
# --------------------------------------------------------------------------- #
def test_inject_shock_overwrites_only_target_cells():
    """inject_shock sets the given assets' return on the target date and leaves
    every other cell untouched — models a crypto weekend mass-gap or an FX shock."""
    rets = _toy_rets().fillna(0.0)
    date = rets.index[6]
    out = inject_shock(rets, date, {"B": -0.25, "C": -0.30})
    assert out.loc[date, "B"] == pytest.approx(-0.25)
    assert out.loc[date, "C"] == pytest.approx(-0.30)
    assert out.loc[date, "A"] == pytest.approx(rets.loc[date, "A"])  # untouched
    # all other rows identical
    other = rets.index[5]
    assert out.loc[other].equals(rets.loc[other])
    # input not mutated
    assert not rets.loc[date, "B"] == pytest.approx(-0.25)


def test_scale_vol_window_amplifies_only_the_window():
    """scale_vol_window multiplies returns inside [start, end] by `factor` (a vol
    spike / crisis) and leaves returns outside the window unchanged."""
    rets = _toy_rets().fillna(0.0)
    start, end = rets.index[4], rets.index[7]
    out = scale_vol_window(rets, start, end, factor=3.0)
    inside = (rets.index >= start) & (rets.index <= end)
    pd.testing.assert_frame_equal(out.loc[inside], rets.loc[inside] * 3.0)
    pd.testing.assert_frame_equal(out.loc[~inside], rets.loc[~inside])


# --------------------------------------------------------------------------- #
# probability_of_backtest_overfitting — CSCV (Bailey-Borwein-LdP-Zhu 2017)
# --------------------------------------------------------------------------- #
def test_pbo_requires_even_n_splits():
    M = np.random.default_rng(0).normal(0, 1, (600, 10))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(M, n_splits=11)


def test_pbo_random_matrix_is_near_half():
    """THE load-bearing guard (advisor #1): an i.i.d. performance matrix has NO
    config with an edge, so the in-sample winner is no better than chance out of
    sample -> EXPECTED PBO ~= 0.5. PBO for a SINGLE matrix has high variance (a
    realization may by chance contain a full-sample-dominant column), so the ~0.5
    property holds IN EXPECTATION — average over many i.i.d. matrices (per the
    advisor's note). If this is not ~0.5, the CSCV is wrong and nothing downstream
    is trustworthy."""
    rng = np.random.default_rng(0)
    pbos = [
        probability_of_backtest_overfitting(rng.normal(0.0, 1.0, (600, 20)), n_splits=12)["pbo"]
        for _ in range(50)
    ]
    assert 0.45 <= float(np.mean(pbos)) <= 0.55


def test_pbo_planted_dominant_config_is_near_zero():
    """A config with a genuine persistent edge (constant positive drift) is the
    in-sample AND out-of-sample winner on every split -> PBO ~= 0. Guards against a
    bug that always returns ~0.5 regardless of signal."""
    rng = np.random.default_rng(7)
    M = rng.normal(0.0, 1.0, (600, 20))
    M[:, 0] += 1.5  # column 0 dominates persistently
    res = probability_of_backtest_overfitting(M, n_splits=12)
    assert res["pbo"] <= 0.05


def test_pbo_returns_diagnostics():
    rng = np.random.default_rng(1)
    M = rng.normal(0.0, 1.0, (480, 8))
    res = probability_of_backtest_overfitting(M, n_splits=8)
    assert res["n_configs"] == 8
    assert res["n_splits"] == 8
    assert res["n_combos"] == 70  # C(8,4)
    assert len(res["lambdas"]) == 70
