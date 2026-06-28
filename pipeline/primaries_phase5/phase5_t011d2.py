"""Phase 5 custom primary for Loop A tick 11 survivor `20260526-XAUUSD-BULL_QUI-T011D2`.

Manual materialization of `custom_primary_pseudocode` per B0040 Option A.

Pseudocode (verbatim from the proposal):
  Long signal at bar t when:
    (1) cot_net_noncomm_z52w rolling-quantile rank over trailing 504 bars >= 0.75
    (2) CUSUM filter with threshold h = 2.0 * ATR(14) at bar t emits an upward event
  Short branch: disabled (always 0).
  CUSUM resets after each emission per canonical AFML §2.

Feature-namespace gap (resolved here): the dossier exposes COT positioning under
the name `cot_net_noncomm_z52w` (forward-filled-from-weekly, publication-lag
aware) — that's what the hypothesizer's pseudocode references. But on D1
XAUUSD audits, `build_tier2_features` does NOT add COT to the features frame
(only the H4 builder does, and under the name `cot_net_noncomm_z52` without
the trailing 'w'). The dossier and the meta-labeler's feature frame have
mismatched namespaces. This primary resolves it by loading COT positioning
directly via `phase5.regime_stats.load_cot_net_noncomm_z` — the same function
the dossier uses — so the primary's input is the same series the hypothesizer
reasoned over.

Causal-window discipline:
  - `cot_net_noncomm_z52w` is publication-lag-aware by construction (uses
    `report_date + publication_lag_days` as the index) — no future leak.
  - Rolling-quantile rank uses `shift(1)` so the rank at bar t is computed
    only from values at bars t-1, t-2, ..., t-504.
  - CUSUM is implemented via `pipeline.labels.cusum_filter_signal` which is
    strict-causal by construction.

INPUT_COLUMNS is empty because the primary reads no feature columns from the
audit pipeline's features frame — it loads COT directly. The orchestrator's
disjointness check (scripts/run_xau_d1.py::assert_primary_inputs_disjoint)
therefore trivially passes against any blacklist.

Per B0040 backlog entry (this commit's session).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.labels import cusum_filter_signal
from phase5.regime_stats import load_cot_net_noncomm_z


INPUT_COLUMNS: tuple[str, ...] = ()


# Proposal-locked configuration constants (T011D2 primary_params).
COT_QUANTILE_FLOOR = 0.75
QUANTILE_LOOKBACK_BARS = 504
CUSUM_H_ATR_MULT = 2.0
ATR_WINDOW = 14


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
    Uses `shift(1)` semantics so the trailing window is causal.
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
    """Return side in {0, +1} for the T011D2 COT-conviction + CUSUM primary.

    Long-only. Emits +1 when both:
      - rolling-quantile rank of `cot_net_noncomm_z52w` (trailing 504 bars) >= 0.75
      - cusum_filter_signal at bar t == +1 (positive CUSUM event)
    Otherwise 0.
    """
    # 1. Load COT positioning under the dossier-aware name. Uses the same
    #    loader that the dossier itself uses — see phase5.regime_stats.
    cot_z = load_cot_net_noncomm_z(asset="XAUUSD", target_index=ohlcv.index)

    # 2. Compute strict-causal rolling-quantile rank over the trailing 504 bars.
    cot_rank = _rolling_quantile_rank(cot_z, window=QUANTILE_LOOKBACK_BARS)

    # 3. CUSUM events (h = 2.0 * ATR(14) / close) — reuses the project's
    #    canonical AFML §3.3 implementation.
    atr = _compute_atr(ohlcv, window=ATR_WINDOW)
    cusum_sig = cusum_filter_signal(
        close=ohlcv["close"].astype(float),
        atr=atr,
        threshold_atr=CUSUM_H_ATR_MULT,
    )

    # 4. Combine: long when rank >= 0.75 AND CUSUM emits +1.
    cond_rank = (cot_rank >= COT_QUANTILE_FLOOR).fillna(False)
    cond_cusum = (cusum_sig == 1.0)

    out = pd.Series(0, index=ohlcv.index, dtype="int8")
    out[cond_rank & cond_cusum] = 1
    return out
