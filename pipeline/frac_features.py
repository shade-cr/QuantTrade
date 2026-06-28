"""Fractionally differentiated features (AFML Chapter 5).

López de Prado, *Advances in Financial Machine Learning*, Ch5 ("Fractionally
Differentiated Features"). The goal is to find the *minimum* amount of
differentiation ``d`` that renders a price series stationary while preserving
as much *memory* (long-range dependence / predictive signal) as possible.

The book's empirical finding (Figure 5.5 / Table 5.1): on log-prices a
fractional order around ``d ~= 0.35-0.40`` keeps the correlation to the raw
level at ~0.99 while already passing the ADF stationarity test, whereas the
integer order ``d = 1`` (ordinary returns) over-differences — it passes ADF but
destroys essentially all memory (corr-to-level ~ 0).

We implement the **fixed-width-window** variant (FFD, Snippets 5.3 / 5.4), not
the expanding-window ``fracDiff`` (Snippet 5.2). Two reasons:

1. **No negative drift.** The expanding window adds a new (small) weight at the
   far tail on every step, so the effective weight vector — and hence the
   series mean — drifts as history accumulates. FFD uses one fixed weight
   vector for every estimate, giving a driftless "level + noise" blend (see the
   prose preceding Snippet 5.3 in AFML).
2. **Strict causality.** With a fixed window of ``width`` lags, the output at
   time ``t`` is exactly ``dot(weights, x[t-width : t+1])`` — a one-sided
   backward filter that touches only bars at indices ``<= t``. No future bar
   can ever enter a past output. This is the property our pipeline's
   no-lookahead contract depends on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ffd_weights(d: float, thres: float = 1e-5) -> np.ndarray:
    """Fixed-width-window fractional-difference weights (AFML Snippet 5.3,
    ``getWeights_FFD``).

    Recurrence: ``w_0 = 1``; ``w_k = -w_{k-1} / k * (d - k + 1)``. New weights
    are appended until ``|w_k| < thres``, at which point the window is truncated.

    The returned 1-D array is ordered **oldest-lag-first** (the book's
    ``w[::-1]``): index 0 is the weight on the oldest bar ``x_{t-width}`` and the
    last index is the weight on the current bar ``x_t`` (always ``1.0``). This
    ordering lets the caller take a plain dot product against a chronologically
    ordered trailing window ``[x_{t-width}, ..., x_t]``.

    Parameters
    ----------
    d : float
        Fractional differentiation order. May be any non-negative real
        (need not be bounded to ``[0, 1]``).
    thres : float
        Weight-magnitude cut-off that fixes the window width.

    Returns
    -------
    np.ndarray
        1-D weight vector, oldest lag first, newest (==1.0) last.
    """
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    # Book stores w[::-1] so the dot product runs oldest..newest.
    return np.array(w[::-1], dtype=float)


def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-5) -> pd.Series:
    """Fixed-width-window fractional differentiation (AFML Snippet 5.4,
    ``fracDiff_FFD``).

    For every ``t`` that has a full trailing window of ``width`` prior bars, the
    output is ``dot(weights, series[t-width : t+1])`` where ``width =
    len(ffd_weights(d, thres)) - 1``. The first ``width`` outputs are ``NaN``
    (insufficient history).

    **Causal.** The output at ``t`` is a one-sided backward filter over bars at
    indices ``<= t`` only; mutating any future bar leaves all earlier outputs
    bit-for-bit unchanged. This is the fixed-width property that motivates FFD
    over the expanding-window ``fracDiff`` (see module docstring).

    Leading ``NaN`` s in the input are forward-filled then dropped before the
    convolution (per the book), but the returned series is re-indexed onto the
    *original* index so length and labels are preserved; positions with no
    computed value remain ``NaN``.

    Parameters
    ----------
    series : pd.Series
        Input level series (typically log-price). Index is preserved
        (DatetimeIndex supported).
    d : float
        Fractional differentiation order.
    thres : float
        Weight cut-off (sets the window width).

    Returns
    -------
    pd.Series
        Same index/length as ``series``; head warm-up region is ``NaN``.
    """
    w = ffd_weights(d, thres=thres)
    width = len(w) - 1

    name = series.name
    # Forward-fill internal/leading gaps, then drop any still-NaN head, exactly
    # as Snippet 5.4 does (series.fillna(method='ffill').dropna()).
    series_f = series.ffill().dropna()

    out = pd.Series(np.nan, index=series.index, dtype=float, name=name)

    if width >= len(series_f):
        # Not enough history for even one full window.
        return out

    vals = series_f.to_numpy()
    idx = series_f.index
    # Vectorised one-sided convolution over the fixed window.
    for iloc1 in range(width, len(series_f)):
        window = vals[iloc1 - width : iloc1 + 1]  # oldest..newest, length width+1
        out.loc[idx[iloc1]] = float(np.dot(w, window))
    return out


def min_ffd_d(
    series: pd.Series,
    d_grid=None,
    thres: float = 1e-5,
    adf_pmax: float = 0.05,
) -> float:
    """Smallest ``d`` on a grid whose FFD series passes the ADF test
    (AFML Snippet 5.5, ``plotMinFFD``).

    Sweeps ``d`` over ``d_grid`` (default ``np.linspace(0, 1, 11)``), computes
    ``frac_diff_ffd``, and runs an Augmented Dickey-Fuller test
    (``statsmodels.tsa.stattools.adfuller``) on the matured (non-NaN) values.
    Returns the **minimum** ``d`` whose ADF p-value ``< adf_pmax`` — i.e. the
    least differentiation that achieves stationarity, thereby retaining the most
    memory. If no grid value passes, returns ``1.0`` (full first difference).

    **Causality note.** This utility is itself only causal *if the caller passes
    a training slice* — i.e. the ``d`` it selects is fit on the data it sees.
    For walk-forward use, choose ``d`` on the training window and apply
    ``frac_diff_ffd`` (which is unconditionally causal) out-of-sample. This
    function is NOT wired into the always-on feature build; it is a calibration
    utility.

    Parameters
    ----------
    series : pd.Series
        Level series (typically a *training-slice* log-price).
    d_grid : array-like, optional
        Candidate ``d`` values, ascending. Default ``np.linspace(0, 1, 11)``.
    thres : float
        Weight cut-off passed through to ``frac_diff_ffd``.
    adf_pmax : float
        ADF p-value threshold for "stationary" (default 0.05).

    Returns
    -------
    float
        Minimum passing ``d``, or ``1.0`` if none pass.
    """
    from statsmodels.tsa.stattools import adfuller

    if d_grid is None:
        d_grid = np.linspace(0, 1, 11)

    for d in sorted(float(x) for x in d_grid):
        ffd = frac_diff_ffd(series, d=d, thres=thres).dropna()
        if len(ffd) < 10 or ffd.std() == 0:
            continue
        pval = adfuller(ffd.to_numpy(), maxlag=1, regression="c", autolag=None)[1]
        if pval < adf_pmax:
            return d
    return 1.0
