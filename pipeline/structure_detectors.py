"""Causal structure detectors for the "math behind chart patterns" spike.

THREE detectors that LABEL structure (regime / break / cycle-trend). They do
NOT create directional edge by themselves — the spike that uses them tests
whether CONDITIONING a simple primary on the detected structure beats the
unconditional primary OUT-OF-SAMPLE. See scripts/spike_structure_regime.py.

CAUSALITY CONTRACT (the load-bearing property, CLAUDE.md no-look-ahead):
  Every detector emits a label at bar t that is a function of data <= t ONLY.
  This is proved as a black box by PREFIX-INVARIANCE in
  tests/test_structure_detectors_causality.py:
      detector(series[:t+1])[t] == detector(series)[t]   (post-warmup).
  The three classic look-ahead traps and how each is avoided:
    * HMM      — fit params ONCE on a train prefix, then infer the state at t
                 by a FORWARD/FILTERED pass over obs[train:t+1]. We never
                 Viterbi-decode the whole series (the Viterbi/smoother is
                 two-sided: it uses future obs to pick the past state).
    * Change-point — online CUSUM-of-mean over a TRAILING window ending at t.
                 No PELT/BinSeg over the full series (that re-segments the past
                 when the future arrives).
    * Wavelet  — DWT/denoise/IDWT on a TRAILING window ending at t; read the
                 reconstructed trend slope at the LAST point of that window.
                 No whole-series DWT (boundary coefficients smear the future
                 into the present).

Sharpe / metric conventions are NOT here — they live in the spike script and
replicate pipeline/metrics.py strategy_metrics (sqrt(trades_per_year), NaN<30).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _log_returns(close: pd.Series) -> np.ndarray:
    c = close.values.astype(float)
    r = np.zeros(len(c), dtype=float)
    r[1:] = np.log(c[1:] / c[:-1])
    return r


def _realized_vol(log_ret: np.ndarray, window: int) -> np.ndarray:
    """Trailing realized vol of log-returns; NaN until `window` obs available."""
    s = pd.Series(log_ret)
    return s.rolling(window).std(ddof=0).values


# --------------------------------------------------------------------------- #
# 1. HMM regime — Gaussian HMM, frozen params, FORWARD-FILTERED state at t
# --------------------------------------------------------------------------- #
def hmm_regime_labels(
    close: pd.Series,
    train_bars: int = 2000,
    n_states: int = 2,
    vol_window: int = 24,
    seed: int = 0,
) -> pd.Series:
    """Causal Gaussian-HMM regime label per bar, in {0, 1, ...} ∪ {NaN}.

    The state id is REMAPPED so that 0 = "trending" and 1 = "choppy" by the
    TRAIN-window signature: the state with the larger mean |return| / vol ratio
    (stronger drift relative to noise) is the trending state. (For n_states>2
    the remap orders states by that ratio descending.)

    Causality
    ---------
    1. Features = (log_return, trailing realized vol). Standardize using
       TRAIN-window mean/std only (frozen scaler — no future leakage).
    2. Fit GaussianHMM ONCE on the train-window features. Params frozen.
    3. For each online bar t >= train_bars, run the FORWARD algorithm on the
       standardized features from train_start..t and take argmax of the
       FILTERED posterior P(state_t | obs_{<=t}). The forward pass at t is, by
       construction, a function of obs <= t only — appending future obs cannot
       change it. (hmmlearn's `predict` is Viterbi = two-sided; we do NOT use it
       for the online label.)

    Bars in [0, train_bars) are NaN (no label inside the fit window).
    """
    from hmmlearn.hmm import GaussianHMM

    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    if n <= train_bars + 1 or train_bars < 50:
        return pd.Series(out, index=close.index)

    log_ret = _log_returns(close)
    rv = _realized_vol(log_ret, vol_window)
    feats = np.column_stack([log_ret, rv])

    # First vol_window-1 rows have NaN vol; start the usable region after that.
    first_valid = vol_window  # rv[vol_window-1] is the first finite value; use >=
    if train_bars <= first_valid + 10:
        return pd.Series(out, index=close.index)

    # Frozen standardizer from the TRAIN window [first_valid, train_bars).
    train_block = feats[first_valid:train_bars]
    mu = train_block.mean(axis=0)
    sd = train_block.std(axis=0)
    sd[sd <= 1e-12] = 1.0
    feats_std = (feats - mu) / sd

    # Fit HMM once on the standardized TRAIN block (params frozen thereafter).
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=50,
        random_state=seed,
        init_params="stmc",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(train_block_std := feats_std[first_valid:train_bars])
        except Exception:
            return pd.Series(out, index=close.index)

    # --- TRAIN-signature remap: rank states by |mean return| / sqrt(var return).
    # feature 0 is log_return (standardized); larger |mean|/sd => more drift => trending.
    means_ret = model.means_[:, 0]
    var_ret = model.covars_[:, 0, 0] if model.covars_.ndim == 3 else model.covars_[:, 0]
    drift_score = np.abs(means_ret) / np.sqrt(np.maximum(var_ret, 1e-12))
    rank = np.argsort(-drift_score)  # descending: rank[0] = strongest drift
    remap = np.empty(n_states, dtype=int)
    for new_id, old_state in enumerate(rank):
        remap[old_state] = new_id  # new_id 0 = trending, 1 = choppy, ...

    # --- Forward-filtered state at each online bar (params frozen).
    # Precompute per-frame log-likelihoods under the frozen emission model, then
    # run a single forward sweep; the filtered posterior at t depends only on
    # frames <= t, so this matches the per-prefix recompute exactly.
    framelogprob = model._compute_log_likelihood(feats_std[first_valid:])  # (T, K)
    log_startprob = np.log(np.maximum(model.startprob_, 1e-300))
    log_transmat = np.log(np.maximum(model.transmat_, 1e-300))

    T = framelogprob.shape[0]
    log_alpha = np.full((T, n_states), -np.inf)
    log_alpha[0] = log_startprob + framelogprob[0]
    for t in range(1, T):
        # log_alpha[t,j] = framelogprob[t,j] + logsumexp_i(log_alpha[t-1,i] + A[i,j])
        prev = log_alpha[t - 1][:, None] + log_transmat  # (K,K)
        log_alpha[t] = framelogprob[t] + _logsumexp_axis0(prev)

    filtered_state = log_alpha.argmax(axis=1)  # argmax of filtered posterior at t
    # Map frame index back to absolute bar index (frames start at first_valid).
    for fi in range(T):
        abs_t = first_valid + fi
        if abs_t < train_bars:
            continue  # keep train window NaN
        out[abs_t] = float(remap[filtered_state[fi]])

    return pd.Series(out, index=close.index)


def _logsumexp_axis0(a: np.ndarray) -> np.ndarray:
    """logsumexp over axis 0 of a (K, K) matrix -> (K,) vector."""
    m = a.max(axis=0)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    return m_safe + np.log(np.exp(a - m_safe).sum(axis=0))


# --------------------------------------------------------------------------- #
# 2. Change-point — online CUSUM-of-mean over a trailing window
# --------------------------------------------------------------------------- #
def changepoint_labels(
    close: pd.Series,
    window: int = 200,
    k_atr: float = 3.0,
) -> pd.Series:
    """Online mean-shift change-point label per bar, in {-1, 0, +1} ∪ {NaN}.

    Detects a structural shift in the MEAN of log-returns using a two-sided
    CUSUM over a TRAILING window ending at t (López de Prado AFML §3.3 / Page
    CUSUM). Label semantics:
       +1  the trailing window contains an UP mean-shift (positive break),
       -1  a DOWN mean-shift,
        0  no break detected in the window.
    The threshold is volatility-scaled: k_atr * sigma_window, where sigma_window
    is the within-window per-bar std of log-returns. This is the SAME idea as
    pipeline.labels.cusum_filter_signal but operating on the trailing window so
    the label at t uses returns r_{t-window+1 .. t} only — never future returns.

    Bars in [0, window) are NaN.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    if n <= window + 1 or window < 10:
        return pd.Series(out, index=close.index)

    r = _log_returns(close)

    for t in range(window, n):
        seg = r[t - window + 1 : t + 1]  # trailing window ending at t (inclusive)
        sigma = seg.std(ddof=0)
        if not np.isfinite(sigma) or sigma <= 1e-12:
            out[t] = 0.0
            continue
        thr = k_atr * sigma
        mu = seg.mean()
        # Two-sided Page CUSUM on mean-centered increments within the window.
        s_pos = 0.0
        s_neg = 0.0
        label = 0.0
        for x in seg:
            s_pos = max(0.0, s_pos + (x - mu))
            s_neg = min(0.0, s_neg + (x - mu))
            if s_pos >= thr:
                # break direction = sign of the window's net drift
                label = 1.0 if mu >= 0 else -1.0
                break
            if s_neg <= -thr:
                label = -1.0 if mu <= 0 else 1.0
                break
        out[t] = label
    return pd.Series(out, index=close.index)


# --------------------------------------------------------------------------- #
# 3. Wavelet / spectral trend — causal trailing-window DWT denoise + slope sign
# --------------------------------------------------------------------------- #
def wavelet_trend_labels(
    close: pd.Series,
    window: int = 256,
    level: int = 3,
    wavelet: str = "db4",
    slope_lookback: int = 8,
) -> pd.Series:
    """Causal denoised-trend slope label per bar, in {-1, 0, +1} ∪ {NaN}.

    At each bar t, take the trailing `window` log-prices ending at t, run a
    multilevel DWT, zero the detail (high-frequency) coefficients to keep only
    the smooth approximation, reconstruct (IDWT), and read the SLOPE of the
    reconstructed trend over the last `slope_lookback` points of the window:
       +1  reconstructed trend rising at t,
       -1  falling,
        0  flat / undefined.
    Because the DWT/IDWT operate on the trailing window ENDING at t, the label
    at t is a function of prices <= t only. A whole-series DWT would smear future
    detail coefficients into past bars at every boundary (look-ahead) — that is
    exactly the trap the prefix-invariance test guards against.

    Falls back to a causal MA-band slope if PyWavelets is unavailable.

    Bars in [0, window) are NaN.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=float)
    if n <= window + 1 or window < 16:
        return pd.Series(out, index=close.index)

    logp = np.log(close.values.astype(float))

    try:
        import pywt

        have_pywt = True
        max_level = pywt.dwt_max_level(window, pywt.Wavelet(wavelet).dec_len)
        use_level = min(level, max_level)
    except Exception:  # pragma: no cover - exercised only without pywt
        have_pywt = False
        use_level = level

    for t in range(window, n):
        seg = logp[t - window + 1 : t + 1]  # trailing window ending at t
        if have_pywt:
            coeffs = pywt.wavedec(seg, wavelet, level=use_level, mode="periodization")
            # Keep approximation, zero all detail levels -> smooth trend.
            denoised = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
            trend = pywt.waverec(denoised, wavelet, mode="periodization")
            trend = trend[: len(seg)]
        else:  # causal MA-band fallback (centered MA would peek; use trailing)
            k = max(2, window // (2 ** use_level))
            trend = pd.Series(seg).rolling(k).mean().values

        tail = trend[-slope_lookback:]
        if np.any(~np.isfinite(tail)) or len(tail) < 2:
            out[t] = 0.0
            continue
        slope = float(tail[-1] - tail[0])
        # Scale-free dead-zone: compare slope to the window's price std.
        scale = np.nanstd(seg) + 1e-12
        if slope > 0.10 * scale:
            out[t] = 1.0
        elif slope < -0.10 * scale:
            out[t] = -1.0
        else:
            out[t] = 0.0
    return pd.Series(out, index=close.index)
