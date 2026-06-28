"""Implied-dispersion extractor from Polymarket strike ladders (B0174 phase 2).

A binary market quoting P(S_T > K) IS the risk-neutral complementary CDF at K.
This module turns a strike ladder into observed-support QUANTILE measures of
dispersion / skew / tail risk — the soundest read under our small-n (~11) and
truncated-tail data (quant-phd-advisor 2026-06-12, design spec
docs/superpowers/specs/2026-06-12-b0174-polymarket-implied-dispersion-design.md).

DELIBERATELY ABSENT (forbidden by design): parametric (lognormal/mixture) fits
and full-support moments (variance/skew as integrals). With truncated tails and
favorite-longshot compression those are "confidently wrong." Every measure here
lives strictly inside the observed strike band and returns NaN rather than
extrapolate an unobserved tail.

Two input formats canonicalize to one (K, ccdf) representation:
  * above-ladder (crypto "above $K on date"): yes_price IS P(S>K).
  * bucket-ladder (gold "GC settle in $A-$B"): per-bin mass -> accumulate.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.isotonic import IsotonicRegression


def parse_strike(label: str) -> float:
    """'$56,000' / '56,000' / '56-000' / '$60' -> float. Raises on no digits."""
    cleaned = re.sub(r"[,$\s]", "", str(label)).replace("-", "")
    if not re.fullmatch(r"\d+(\.\d+)?", cleaned):
        raise ValueError(f"cannot parse strike from label {label!r}")
    return float(cleaned)


def ccdf_from_above_ladder(strikes, yes_prices) -> tuple[np.ndarray, np.ndarray]:
    """(strikes, P(S>K)) -> (K sorted, monotone-decreasing ccdf in [0,1]).

    Enforces the no-arbitrage constraint P(S>K) non-increasing in K via
    isotonic regression (PAVA), then clips to [0,1]. PAVA is the parameter-free
    L2 projection onto the monotone cone — the right tool at n~11."""
    K = np.asarray(strikes, dtype=float)
    p = np.asarray(yes_prices, dtype=float)
    order = np.argsort(K)
    K, p = K[order], p[order]
    iso = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0)
    ccdf = iso.fit_transform(K, p)
    return K, np.clip(ccdf, 0.0, 1.0)


def parse_bucket_label(label: str) -> tuple[float | None, float | None]:
    """Parse a settle-bucket label into (low, high) edges; None = open end.

    '<$4,350' -> (None, 4350) ; '$4,350-$4,475' -> (4350, 4475) ;
    '>$5,100' -> (5100, None). Raises if no numeric edge is found."""
    s = str(label).strip()
    nums = re.findall(r"\d[\d,]*(?:\.\d+)?", s)
    vals = [float(n.replace(",", "")) for n in nums]
    if not vals:
        raise ValueError(f"cannot parse bucket label {label!r}")
    if s.startswith("<") or "less" in s.lower():
        return (None, vals[0])
    if s.startswith(">") or "more" in s.lower() or "+" in s:
        return (vals[0], None)
    if len(vals) >= 2:
        return (vals[0], vals[1])
    raise ValueError(f"ambiguous bucket label {label!r}")


def ccdf_from_buckets(edges, bucket_probs) -> tuple[np.ndarray, np.ndarray]:
    """Bucket ladder -> (interior edges, complementary CDF at those edges).

    `edges` are the n interior boundaries; `bucket_probs` are the n+1 bin
    masses (including the two open end bins). Mass is normalized, then
    ccdf(edge_j) = P(S > edge_j) = sum of mass strictly above edge_j. Open end
    bins are never assigned a location (no tail extrapolation)."""
    edges = np.asarray(edges, dtype=float)
    mass = np.asarray(bucket_probs, dtype=float)
    if len(mass) != len(edges) + 1:
        raise ValueError(
            f"need {len(edges) + 1} bucket_probs for {len(edges)} edges, got {len(mass)}"
        )
    total = mass.sum()
    if total <= 0:
        raise ValueError("bucket mass sums to non-positive")
    mass = mass / total
    # ccdf at interior edge j = mass in bins j+1 .. end (bins above the edge).
    ccdf = np.array([mass[j + 1:].sum() for j in range(len(edges))], dtype=float)
    return edges, np.clip(ccdf, 0.0, 1.0)


def implied_quantile(strikes, ccdf, q: float) -> float:
    """The q-quantile of the implied distribution, or NaN if q is outside the
    observed band (the unobserved tail — we never extrapolate).

    F(x) = P(S <= x) = 1 - ccdf(x), increasing in x. We solve F(x) = q by
    linear interpolation on the observed (F_i, K_i) points (the conservative,
    n~11-defensible choice; PAVA ties are deduplicated keeping monotonicity)."""
    K = np.asarray(strikes, dtype=float)
    F = 1.0 - np.asarray(ccdf, dtype=float)
    # Deduplicate flat ties in F so np.interp sees a strictly increasing xp.
    keep = np.concatenate(([True], np.diff(F) > 1e-12))
    F, K = F[keep], K[keep]
    if q < F[0] - 1e-12 or q > F[-1] + 1e-12:
        return float("nan")
    return float(np.interp(q, F, K))


def bowley_skew(q10: float, q50: float, q90: float) -> float:
    """Bowley-Hinkley quantile skew: ((q90+q10) - 2*q50) / (q90 - q10).
    Tail-free, bounded in [-1, 1]. NaN if the denominator is degenerate."""
    denom = q90 - q10
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float(((q90 + q10) - 2.0 * q50) / denom)


def dispersion_measures(strikes, ccdf, spot: float) -> dict:
    """Observed-support dispersion read. Each field is NaN if its quantiles
    fall outside the observed band.

    ipr_central : (Q75 - Q25) / spot          — the volatility proxy (primary)
    bowley_skew : Bowley-Hinkley on (Q10,Q50,Q90) — asymmetry (secondary)
    p_move_gt_5pct / p_move_gt_10pct           — direct tail-risk reads (panel)
    """
    K = np.asarray(strikes, dtype=float)
    cc = np.asarray(ccdf, dtype=float)
    q25 = implied_quantile(K, cc, 0.25)
    q75 = implied_quantile(K, cc, 0.75)
    q10 = implied_quantile(K, cc, 0.10)
    q50 = implied_quantile(K, cc, 0.50)
    q90 = implied_quantile(K, cc, 0.90)

    ipr = (q75 - q25) / spot if (np.isfinite(q25) and np.isfinite(q75) and spot > 0) else float("nan")
    skew = bowley_skew(q10, q50, q90) if all(np.isfinite(v) for v in (q10, q50, q90)) else float("nan")

    return {
        "ipr_central": ipr,
        "bowley_skew": skew,
        "p_move_gt_5pct": _p_abs_move_gt(K, cc, spot, 0.05),
        "p_move_gt_10pct": _p_abs_move_gt(K, cc, spot, 0.10),
    }


def _p_abs_move_gt(K, ccdf, spot: float, x: float) -> float:
    """P(|S/spot - 1| > x) read off the interpolated ccdf; NaN if either
    barrier strike spot*(1±x) is outside the observed strike band."""
    K = np.asarray(K, dtype=float)
    cc = np.asarray(ccdf, dtype=float)
    if spot <= 0:
        return float("nan")
    up, dn = spot * (1.0 + x), spot * (1.0 - x)
    if up > K.max() or dn < K.min():
        return float("nan")
    p_up = float(np.interp(up, K, cc))                 # P(S > up)
    p_dn = 1.0 - float(np.interp(dn, K, cc))           # P(S < dn) = 1 - P(S>dn)
    return p_up + p_dn


def longshot_distance(yes_prices) -> float:
    """Mean |price - 0.5| across the ladder — the favorite-longshot placebo
    regressor. High when the book is confident (prices near 0/1), low when
    compressed toward 0.5."""
    p = np.asarray(yes_prices, dtype=float)
    return float(np.mean(np.abs(p - 0.5)))
