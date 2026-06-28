"""Phase 5 custom primary for Loop A survivor `20260527-XAUUSD-BEAR_QUI-T015D2`.

Manual materialization of `custom_primary_pseudocode` per B0040 Option B.

Pseudocode (verbatim from the reviewed proposal):
  Maintain a symmetric CUSUM accumulator on log-returns of `close`, with the
  per-event threshold set to `cusum_threshold_atr_mult` (1.0) times a 14-bar
  ATR (computed from `high`,`low`,`close`). When the negative cumulative sum
  breaches the threshold, a downside displacement event fires and the
  accumulator resets. On each fired downside event, compute the percentile
  rank of current-bar `volume` within a trailing `volume_window` (42 bars).
  Emit Short (-1) only if that volume percentile rank >= `volume_pct_threshold`
  (0.60); otherwise emit 0 (no signal). When the positive cumulative sum
  breaches the threshold, reset the accumulator and emit 0 (no long signals in
  this regime — directional short-only bias). No signal on any bar where no
  CUSUM breach occurs.

This is a SHORT-ONLY primary: it emits values in {-1, 0} and never +1.

Causal-window discipline (every computation is strictly backward-looking):
  - The volume percentile rank at bar t is computed by `_rolling_quantile_rank`
    over the trailing window bars t-volume_window .. t-1, compared against the
    current bar's volume. No bar > t and not even bar t itself is in the
    comparison set — only strictly prior bars. NaN until enough history exists,
    and those warm-up NaNs are coerced to "no signal" via `.fillna(False)`.
  - ATR is computed by `_compute_atr`, which uses `prev_close = close.shift(1)`
    and a trailing `rolling(window).mean()` — no future bars.
  - CUSUM events come from `pipeline.labels.cusum_filter_signal`, whose
    threshold at bar t is `threshold_atr * ATR[t-1] / close[t-1]` (strictly
    prior) and whose accumulator only ever sees log-returns up to and including
    bar t. It is strict-causal by construction and is NOT modified here.
  - No `.shift(-k)`, no full-sample `.rank()`/quantile/mean/std, no centered
    rolling windows, no resampling/backfill.

INPUT_COLUMNS is empty because the primary reads `volume` (and OHLC) directly
from the `ohlcv` frame, not from the audit pipeline's `features` frame. The
orchestrator's disjointness check
(scripts/run_backtest.py::assert_primary_inputs_disjoint) therefore trivially
passes against any feature blacklist.

Per B0040 backlog entry (this commit's session).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.labels import cusum_filter_signal


INPUT_COLUMNS: tuple[str, ...] = ()


# Proposal-locked configuration constants (T015D2 primary_params).
CUSUM_THRESHOLD_ATR_MULT = 1.0
ATR_WINDOW = 14
VOLUME_WINDOW = 42
VOLUME_PCT_THRESHOLD = 0.60


def _compute_atr(ohlcv: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def _rolling_quantile_rank(s: pd.Series, window: int) -> pd.Series:
    """Strict-prior rolling quantile rank.

    At bar t, returns the fraction of values in s[t-window..t-1] that are
    <= s[t] (i.e., the rank of the current value within the trailing window).
    The window is strictly prior — bar t itself is the comparison value, not a
    member of the comparison set, so this can never peek at the present-or-future.
    Returns NaN when fewer than `window` prior observations exist.
    """
    n = len(s)
    out = np.full(n, np.nan, dtype=float)
    values = s.values.astype(float)
    for t in range(window, n):
        cur = values[t]
        if not np.isfinite(cur):
            continue
        prior = values[t - window:t]
        # Drop NaNs in the prior window; require >= window/2 valid observations
        # for the rank to be meaningful.
        prior_valid = prior[np.isfinite(prior)]
        if prior_valid.size < window // 2:
            continue
        # Fraction of prior values <= current
        rank = float((prior_valid <= cur).sum()) / float(prior_valid.size)
        out[t] = rank
    return pd.Series(out, index=s.index)


def signal(ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: dict) -> pd.Series:
    """Return side in {-1, 0} for the T015D2 CUSUM-downside + volume-gate primary.

    Short-only. Emits -1 when both:
      - cusum_filter_signal at bar t == -1.0 (negative/downside CUSUM event), and
      - the trailing-42-bar volume percentile rank of the current bar >= 0.60.
    Otherwise 0. Never emits +1.
    """
    # 1. CUSUM events (threshold = 1.0 * ATR(14)[t-1] / close[t-1]) — reuses the
    #    project's canonical AFML §3.3 strict-causal implementation. A downside
    #    displacement event is cusum_sig == -1.0.
    atr = _compute_atr(ohlcv, window=ATR_WINDOW)
    cusum_sig = cusum_filter_signal(
        close=ohlcv["close"].astype(float),
        atr=atr,
        threshold_atr=CUSUM_THRESHOLD_ATR_MULT,
    )

    # 2. Strict-prior trailing-window volume percentile rank.
    volume_rank = _rolling_quantile_rank(ohlcv["volume"], window=VOLUME_WINDOW)

    # 3. Combine: short only when a downside CUSUM event coincides with a
    #    high-volume bar. Warm-up NaNs in the volume rank become "no signal".
    cond_cusum_down = (cusum_sig == -1.0)
    cond_volume = (volume_rank >= VOLUME_PCT_THRESHOLD).fillna(False)

    out = pd.Series(0, index=ohlcv.index, dtype="int8")
    out[cond_cusum_down & cond_volume] = -1
    return out
