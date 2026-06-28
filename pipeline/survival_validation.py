"""Robust validation of the survival book (B0112).

The historical backtest is ONE trajectory — and ~80% of the 26y XAU/USD-class
window is FX-only (metals/BTC/ETH start 2021, SOL 2025), so a single maxDD / vol
number has no error bar and under-samples the deployed full-universe regime. This
module turns "the backtest runs" into "the risk machinery survives a DISTRIBUTION
of plausible futures."

Methodology (per the risk-officer review of B0112)
--------------------------------------------------
* Stationary block bootstrap (Politis & Romano 1994): resample the return path in
  blocks of GEOMETRIC random length (mean = expected_block_len), wrapping around,
  so serial dependence (volatility clustering) survives — an IID resample would
  destroy the clustering the vol-target and kill-switches exist to handle.
* ALIGNED-MULTIVARIATE + GAP-PRESERVING: we resample whole calendar-aligned ROWS
  of the asset-by-date return matrix, never resampling columns independently. So
  (a) cross-asset correlation within a bar is preserved exactly, and (b) the
  weekend-crypto gap structure (NaN rows for not-yet-listed / non-trading legs) is
  carried verbatim — the maxDD distribution must not understate the gap tail that
  apply_risk_controls' gap-aware split was built for.
* Deterministic under a seed (reproducibility / backtest parity).

Block length is a parameter here; the Politis-White (2004) automatic data-driven
selection is a separate building block (added as its own increment).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def stationary_bootstrap_indices(
    n: int, expected_block_len: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary-bootstrap row indices for one path of length `n`.

    Start at a uniformly random row. At each subsequent step, with probability
    `p = 1 / expected_block_len` start a NEW block (jump to a fresh uniform random
    row); otherwise advance to the next row, wrapping modulo `n`. This yields
    blocks whose lengths are Geometric(p) with mean `expected_block_len`.
    """
    if n <= 0:
        return np.empty(0, dtype=int)
    p = 1.0 / float(expected_block_len)
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)
    for t in range(1, n):
        if rng.random() < p:
            idx[t] = rng.integers(0, n)
        else:
            idx[t] = (idx[t - 1] + 1) % n
    return idx


def stationary_bootstrap_paths(
    rets: pd.DataFrame,
    *,
    expected_block_len: float,
    n_paths: int,
    seed: int,
) -> list[pd.DataFrame]:
    """Generate `n_paths` stationary-block-bootstrap resamples of the multi-asset
    return matrix `rets`.

    Each returned DataFrame has the SAME index and columns as `rets`; its rows are
    a stationary-bootstrap resample of `rets`' rows (aligned-multivariate: the same
    row index is applied across all columns, so a resampled row is a verbatim copy
    of some original row, NaNs and all). Deterministic given `seed`.
    """
    rng = np.random.default_rng(seed)
    n = rets.shape[0]
    values = rets.values
    out: list[pd.DataFrame] = []
    for _ in range(n_paths):
        idx = stationary_bootstrap_indices(n, expected_block_len, rng)
        resampled = values[idx]
        out.append(pd.DataFrame(resampled, index=rets.index, columns=rets.columns))
    return out


def _flat_top_kernel(s: np.ndarray) -> np.ndarray:
    """Politis (2003) flat-top lag window: 1 on [0, 1/2], a linear taper to 0 on
    (1/2, 1], and 0 beyond. Used to weight the autocovariances in the PW2004
    block-length estimator."""
    a = np.abs(s)
    return np.where(a <= 0.5, 1.0, np.where(a <= 1.0, 2.0 * (1.0 - a), 0.0))


def optimal_block_length(x) -> float:
    """Politis-White (2004) automatic block-length selection for the STATIONARY
    bootstrap (Patton-Politis-White 2009 correction), for one series.

    Estimates the optimal EXPECTED block length from the series' autocorrelation
    structure: longer blocks for more persistent series, ~1 for IID. The estimator
    is scale-invariant, so it is computed on autocorrelations directly:

        b_opt = (2 * G^2 / D_SB)^(1/3) * n^(1/3),  D_SB = 2 * g0^2
              = (|G| / |g0|)^(2/3) * n^(1/3)

    with G = 2 Σ_k λ(k/M) k ρ(k), g0 = 1 + 2 Σ_k λ(k/M) ρ(k), and M = 2·m̂ where m̂
    is the smallest lag after which the autocorrelations stay inside the white-noise
    band 2·sqrt(log10 n / n) for K_N consecutive lags (Politis automatic bandwidth).
    Capped at ceil(min(3·sqrt(n), n/3)), floored at 1.
    """
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = x.size
    if n < 16:
        return 1.0
    x = x - x.mean()
    denom = float(np.dot(x, x)) / n
    if denom <= 0:
        return 1.0

    nlag = int(min(n - 1, np.ceil(2.0 * np.sqrt(n)) + 20))
    rho = np.empty(nlag + 1)
    rho[0] = 1.0
    for k in range(1, nlag + 1):
        rho[k] = float(np.dot(x[:-k], x[k:])) / n / denom

    # Politis automatic bandwidth: smallest m̂ s.t. the next K_N autocorrelations
    # are all inside the white-noise band; M = 2·m̂.
    band = 2.0 * np.sqrt(np.log10(n) / n)
    K_N = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    mhat = nlag
    for m in range(1, nlag - K_N + 1):
        if np.all(np.abs(rho[m + 1 : m + 1 + K_N]) < band):
            mhat = m
            break
    M = int(min(2 * mhat, nlag))
    M = max(M, 1)

    ks = np.arange(1, M + 1)
    lam = _flat_top_kernel(ks / M)
    G = 2.0 * float(np.sum(lam * ks * rho[1 : M + 1]))
    g0 = 1.0 + 2.0 * float(np.sum(lam * rho[1 : M + 1]))
    if abs(g0) < 1e-12:
        return 1.0

    b = (abs(G) / abs(g0)) ** (2.0 / 3.0) * n ** (1.0 / 3.0)
    b_max = float(np.ceil(min(3.0 * np.sqrt(n), n / 3.0)))
    return float(min(max(b, 1.0), b_max))


def recommended_block_length(rets: pd.DataFrame) -> float:
    """Data-driven expected block length for bootstrapping a return matrix.

    Computed on ABSOLUTE returns per asset, then the MAX over assets. Rationale:
    the maxDD tail is driven by VOLATILITY CLUSTERING, which lives in |returns|
    (raw returns are near-white and would yield deceptively short blocks); taking
    the max preserves the most persistent leg's dependence (the conservative,
    longest-memory choice the risk-officer's gap/clustering caveat demands).
    """
    lengths = [
        optimal_block_length(np.abs(rets[c].dropna().values)) for c in rets.columns
    ]
    return float(max(lengths)) if lengths else 1.0


def probability_of_backtest_overfitting(
    perf_matrix, n_splits: int = 16
) -> dict:
    """Probability of Backtest Overfitting via CSCV (Bailey, Borwein, López de Prado
    & Zhu 2017).

    `perf_matrix` is a T×N matrix of per-observation returns (T time obs, N swept
    configs). The rows are split into `n_splits` (S, EVEN) contiguous, disjoint,
    equal-size submatrices. Over all C(S, S/2) ways to choose S/2 blocks as the
    in-sample (IS) set (the complement is out-of-sample, OOS):

      * per-config IS / OOS performance = the Sharpe (mean/std) of that config's
        returns POOLED over the chosen blocks (computed from pooled block moments —
        Sharpe of the concatenation, NOT the mean of per-block Sharpes);
      * n* = argmax IS performance (the config you would have picked);
      * ω̄ = rank of n*'s OOS performance / (N+1)  (ascending: 1 = worst);
      * λ = logit(ω̄) = ln(ω̄/(1−ω̄));
      * PBO = fraction of splits with λ ≤ 0, i.e. the IS-best config landing at or
        below the OOS median.

    No purge/embargo: a continuously-held book's M[t,n] is a POINT-IN-TIME return
    with no outcome horizon, so adjacent blocks share no outcome window (the same
    reason the main backtest does not purge). Degenerate (≈0 std) blocks are
    NaN-guarded and ranked worst.

    Returns {"pbo", "n_configs", "n_splits", "n_combos", "lambdas"}.
    """
    M = np.asarray(perf_matrix, dtype="float64")
    if M.ndim != 2:
        raise ValueError("perf_matrix must be 2-D (T x N)")
    S = int(n_splits)
    if S % 2 != 0:
        raise ValueError(f"n_splits must be even; got {S}")
    T, N = M.shape
    rows = (T // S) * S
    if rows < S:
        raise ValueError(f"too few rows ({T}) for {S} splits")
    blocks = np.stack(np.split(M[:rows], S))   # S × (rows/S) × N
    bn = blocks.shape[1]                         # rows per block (equal)

    # Pooled-moment building blocks per submatrix (O(S) per combination).
    bsum = blocks.sum(axis=1)                    # S × N
    bsq = (blocks ** 2).sum(axis=1)              # S × N
    all_blocks = set(range(S))

    def pooled_sharpe(block_ids) -> np.ndarray:
        ids = list(block_ids)
        n = bn * len(ids)
        s1 = bsum[ids].sum(axis=0)
        s2 = bsq[ids].sum(axis=0)
        mean = s1 / n
        var = (s2 - n * mean ** 2) / (n - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sr = np.where(var > 1e-24, mean / np.sqrt(var), np.nan)
        return sr

    lambdas = []
    for J in combinations(range(S), S // 2):
        comp = tuple(all_blocks - set(J))
        sr_is = pooled_sharpe(J)
        sr_oos = pooled_sharpe(comp)
        n_star = int(np.nanargmax(sr_is))
        # Ascending OOS ranks (1 = worst); NaNs ranked worst (−inf).
        oos_filled = np.where(np.isnan(sr_oos), -np.inf, sr_oos)
        order = np.argsort(np.argsort(oos_filled, kind="stable"), kind="stable")
        rank_star = float(order[n_star] + 1)     # 1..N
        omega = rank_star / (N + 1)
        lambdas.append(float(np.log(omega / (1.0 - omega))))

    lambdas = np.asarray(lambdas)
    return {
        "pbo": float(np.mean(lambdas <= 0.0)),
        "n_configs": int(N),
        "n_splits": S,
        "n_combos": int(lambdas.size),
        "lambdas": lambdas,
    }


def inject_shock(
    rets: pd.DataFrame, date, shocks: dict[str, float]
) -> pd.DataFrame:
    """Return a copy of `rets` with the given assets' return on `date` overwritten
    by the shock value. Models a discontinuous event — a crypto weekend mass-gap
    (e.g. {"BTCUSD": -0.30, "ETHUSD": -0.30, "SOLUSD": -0.30}) or an FX shock — so
    a stress run can verify the gap-aware kill / DD latch engage and bound the loss.
    Does not mutate the input.
    """
    out = rets.copy()
    for asset, value in shocks.items():
        if asset in out.columns:
            out.loc[date, asset] = float(value)
    return out


def scale_vol_window(
    rets: pd.DataFrame, start, end, factor: float
) -> pd.DataFrame:
    """Return a copy of `rets` with all returns inside the inclusive date window
    [start, end] multiplied by `factor` (a volatility spike / crisis amplification),
    leaving returns outside the window unchanged. Does not mutate the input.
    """
    out = rets.copy()
    mask = (out.index >= start) & (out.index <= end)
    out.loc[mask] = out.loc[mask] * float(factor)
    return out


def bootstrap_metric_distribution(
    rets: pd.DataFrame,
    metric_fn,
    *,
    expected_block_len: float,
    n_paths: int,
    seed: int,
) -> dict[str, list[float]]:
    """Run `metric_fn` over `n_paths` stationary-bootstrap resamples of `rets` and
    aggregate the results per metric key.

    `metric_fn(path_df) -> dict[str, float]` is the per-path measurement (e.g. run
    the survival backtest on the resampled returns and return its maxDD / realized
    vol). Returns `{metric_key: [value_per_path, ...]}`. Deterministic given `seed`.
    """
    paths = stationary_bootstrap_paths(
        rets, expected_block_len=expected_block_len, n_paths=n_paths, seed=seed
    )
    dist: dict[str, list[float]] = {}
    for p in paths:
        m = metric_fn(p)
        for k, v in m.items():
            dist.setdefault(k, []).append(float(v))
    return dist


def summarize_distribution(
    samples, percentiles: tuple[int, ...] = (5, 50, 95)
) -> dict:
    """Percentile + moment summary of a list of bootstrap metric samples.

    Returns `{"p<q>": value, ..., "mean": ..., "std": ..., "n": ...}`. NaNs are
    dropped before summarizing (a path that produced no measurement should not
    poison the distribution); an all-NaN/empty input yields NaN summary stats.
    """
    arr = np.asarray([s for s in samples], dtype="float64")
    arr = arr[np.isfinite(arr)]
    out: dict = {}
    if arr.size == 0:
        for q in percentiles:
            out[f"p{q}"] = float("nan")
        out.update(mean=float("nan"), std=float("nan"), n=0)
        return out
    for q in percentiles:
        out[f"p{q}"] = float(np.percentile(arr, q))
    out["mean"] = float(arr.mean())
    out["std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    out["n"] = int(arr.size)
    return out
