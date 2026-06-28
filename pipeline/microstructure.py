"""Microstructural price-impact features (López de Prado, AFML Ch. 19).

This module implements **Kyle's lambda** (AFML §19.4.1) as a rolling, CAUSAL
feature, reported primarily as a **t-value** rather than a raw slope, per the
explicit LdP guidance in §19.4 (book line 6866):

    "In practice, I have observed that the t-values associated with these
     microstructural estimates are more informative than the (mean) estimates
     themselves ... t-values are re-scaled by the standard deviation of the
     estimation error, which incorporates another dimension of information
     absent in mean estimates."

Model (AFML §19.4.1, around book line 6895):

    Δp_t = λ · (b_t · V_t) + ε

    * Δp_t      = close_t − close_{t−1}      (price change)
    * b_t       = aggressor sign ∈ {−1, +1}  (buy-/sell-initiated)
    * V_t       = traded volume
    * b_t · V_t = signed volume / net order flow (the regressor)
    * λ         = price-impact coefficient (OLS slope)

λ is the slope of the OLS regression of Δp on signed volume (with an intercept),
estimated over a trailing window. The reported feature is its t-value,
slope / SE(slope).

PROXY CAVEATS
-------------
* **Aggressor sign**: we only have OHLCV bars, not tick-level buy/sell flags, so
  we proxy b_t with the *tick rule* on bar closes: b_t = sign(close_t − close_{t−1}).
  (When Δp_t == 0 the sign is 0, contributing zero signed volume for that bar.)
  This collapses the aggressor flag and the sign of the response into the same
  source, which mildly biases λ positive; the t-value is still informative as a
  relative liquidity/impact feature, which is how LdP uses it.
* **Volume units**: on XAU/USD (MetaTrader 5) `volume` is *tick count*, not
  dollar or contract volume. Kyle's lambda is therefore expressed per unit of
  tick-volume, not per dollar. Cross-asset comparisons of the raw slope are not
  meaningful for this reason; the t-value (scale-free in the regressor units up
  to the noise level) is the preferred, more comparable feature.

CAUSALITY
---------
Every value at index t is computed from the trailing window of bars ending at t
(bars ≤ t only). No future bar enters the estimate at t. This is verified by the
look-ahead mutation tests in tests/test_microstructure.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["kyle_lambda_tvalue", "kyle_lambda", "amihud_lambda",
           "corwin_schultz_spread"]

_CS_DENOM = 3.0 - 2.0 * np.sqrt(2.0)


def corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
    *,
    beta_window: int = 21,
) -> pd.Series:
    """Corwin-Schultz (2012) high-low bid-ask spread estimate — B0135.

    AFML Ch19 §19.3.4, Snippet 19.1. The microstructure estimator that
    degrades most gracefully to daily bars: needs only high/low. The observed
    range mixes true variance (scales with the 2-bar horizon) and the bid-ask
    bounce (does not); estimating beta from two consecutive 1-bar ranges and
    gamma from the joint 2-bar range isolates the spread term:

        beta_t  = mean over `beta_window` of [ln(H_t/L_t)^2 + ln(H_{t-1}/L_{t-1})^2]
        gamma_t = ln( max(H_{t-1},H_t) / min(L_{t-1},L_t) )^2
        alpha_t = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt2)
                  - sqrt(gamma / (3 - 2 sqrt2)),   clamped at 0 (per the authors)
        S_t     = 2 (e^alpha - 1) / (1 + e^alpha)  -> spread as fraction of price

    Negative alphas are common when true volatility dominates the bounce; the
    authors' clamp makes S a non-negative liquidity PROXY, not a tradeable
    quote. `beta_window` (AFML `sl`) defaults to 21 bars (~1 month on D1).

    CAUSAL by construction: shift(1) looks back, rolling windows are trailing
    and inclusive of t; no forward bar enters S_t.
    """
    h = high.astype(float)
    l = low.astype(float)
    hl2 = np.log(h / l) ** 2
    beta = (hl2 + hl2.shift(1)).rolling(beta_window).mean()
    h2 = np.maximum(h, h.shift(1))
    l2 = np.minimum(l, l.shift(1))
    gamma = np.log(h2 / l2) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _CS_DENOM \
        - np.sqrt(gamma / _CS_DENOM)
    alpha = alpha.clip(lower=0.0)
    s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    return pd.Series(s, index=high.index, name="cs_spread").replace(
        [np.inf, -np.inf], np.nan)


def _signed_volume_and_dp(
    close: pd.Series, volume: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Return (signed_volume, dp) aligned arrays for the Kyle regression.

    dp_t = close_t - close_{t-1}; b_t = sign(dp_t) (tick rule); x_t = b_t * V_t.
    The first element (no prior close) is NaN in dp and x so it is naturally
    excluded from any window's valid points.
    """
    close = close.astype(float)
    volume = volume.astype(float)
    dp = close.diff().to_numpy()
    b = np.sign(dp)  # {-1, 0, +1}; NaN at position 0 stays NaN
    x = b * volume.to_numpy()
    return x, dp


def _rolling_ols_slope_and_t(
    x: np.ndarray, y: np.ndarray, window: int, want_tvalue: bool
) -> np.ndarray:
    """Rolling simple-OLS of y ~ x (with intercept) over a trailing `window`.

    For each end-index t, fit on the valid (finite) points among the last
    `window` observations ending at t and return either the slope or its
    t-value (slope / SE(slope)). Emits NaN when there are < 3 valid points or
    the regressor variance is ~0 (degenerate window).

    CAUSAL by construction: the window for index t spans [t-window+1, t].
    """
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if window < 3:
        return out

    finite = np.isfinite(x) & np.isfinite(y)

    for t in range(window - 1, n):
        lo = t - window + 1
        m = finite[lo : t + 1]
        if m.sum() < 3:
            continue
        xs = x[lo : t + 1][m]
        ys = y[lo : t + 1][m]

        k = xs.size
        xbar = xs.mean()
        sxx = np.sum((xs - xbar) ** 2)
        # Zero / near-zero regressor variance => slope undefined.
        if not np.isfinite(sxx) or sxx <= 1e-12:
            continue
        ybar = ys.mean()
        sxy = np.sum((xs - xbar) * (ys - ybar))
        slope = sxy / sxx

        if not want_tvalue:
            out[t] = slope
            continue

        # Residual standard error of the slope for simple OLS.
        dof = k - 2
        if dof <= 0:
            continue
        intercept = ybar - slope * xbar
        resid = ys - (intercept + slope * xs)
        sse = np.sum(resid ** 2)
        # Perfect fit => infinite t-value; report NaN rather than inf to keep
        # the feature numerically tame for downstream models.
        if sse <= 1e-12:
            continue
        sigma2 = sse / dof
        se_slope = np.sqrt(sigma2 / sxx)
        if not np.isfinite(se_slope) or se_slope <= 0:
            continue
        out[t] = slope / se_slope

    return out


def kyle_lambda_tvalue(
    close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Rolling t-value of Kyle's lambda (AFML §19.4.1) — the primary feature.

    Regresses Δp_t on signed volume (b_t·V_t, b_t via the tick rule on closes)
    over a trailing window ending at t, and returns the slope's t-value,
    slope / SE(slope). This is the LdP-preferred form (§19.4): re-scaled by the
    standard deviation of the estimation error.

    Parameters
    ----------
    close, volume : pd.Series
        Bar close prices and traded volumes (tick count on XAU MT5 — see module
        docstring). Must share the same index.
    window : int
        Trailing-window length in bars. Default 20.

    Returns
    -------
    pd.Series
        Same index/length as `close`. The first ~`window` values are NaN
        (insufficient history); degenerate windows are NaN. CAUSAL: value at t
        uses only bars ≤ t.
    """
    x, dp = _signed_volume_and_dp(close, volume)
    vals = _rolling_ols_slope_and_t(x, dp, window, want_tvalue=True)
    return pd.Series(vals, index=close.index, name="kyle_lambda_tvalue")


def kyle_lambda(
    close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Rolling raw slope λ of Kyle's lambda (AFML §19.4.1) — diagnostic.

    Same trailing-window regression as :func:`kyle_lambda_tvalue` but returns
    the raw OLS slope (price impact per unit of signed volume) for diagnostics
    / comparison. See the module docstring for the volume-unit caveat. CAUSAL.
    """
    x, dp = _signed_volume_and_dp(close, volume)
    vals = _rolling_ols_slope_and_t(x, dp, window, want_tvalue=False)
    return pd.Series(vals, index=close.index, name="kyle_lambda")


def amihud_lambda(
    close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Rolling Amihud illiquidity (AFML §19.4.2) — optional diagnostic.

    Amihud (2002) measures the daily price response per unit of dollar volume:

        ILLIQ_t = |log_ret_t| / (close_t · V_t)

    and we report its trailing-window mean. Larger = more illiquid (bigger price
    move per unit traded). `close · volume` is the dollar-volume proxy; with MT5
    tick-count volume this is price·ticks rather than true notional (see module
    docstring). Bars with zero dollar volume are skipped within the window.
    CAUSAL: value at t uses only bars ≤ t.
    """
    close = close.astype(float)
    volume = volume.astype(float)
    log_ret = np.log(close).diff().abs()
    dollar_vol = close * volume
    illiq = log_ret / dollar_vol.where(dollar_vol > 0)
    illiq = illiq.replace([np.inf, -np.inf], np.nan)
    out = illiq.rolling(window=window, min_periods=3).mean()
    return pd.Series(out.to_numpy(), index=close.index, name="amihud_lambda")
