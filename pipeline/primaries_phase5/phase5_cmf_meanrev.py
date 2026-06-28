"""Phase 5 custom primary: Chaikin Money Flow (CMF) MEAN-REVERSION fade.

Materialization of the B0088 feasibility survivor(s) per B0040 Option B. The
logic is lifted verbatim from the reviewed feasibility spike
`scripts/spike_cmf_meanrev.py` (functions `money_flow_multiplier`,
`chaikin_money_flow`, `cmf_meanrev_signal`, `cmf_meanrev_position`) so the
materialized primary is byte-for-byte the same mechanism the spike measured.

HYPOTHESIS (B0088): fade CMF extremes. A money-flow extreme is a short-lived
dislocation; oversold money OUTFLOW (CMF below a locked lower threshold) tends to
revert UP -> go LONG (+1); overbought money INFLOW (CMF above a locked upper
threshold) tends to revert DOWN -> go SHORT (-1). CMF's money-flow premise is
strongest where `volume` is REAL traded volume (crypto, e.g. SOLUSD); FX/metals
carry only MT5 TICK volume (a weak proxy) — that caveat lives in the proposal.

CAUSALITY (CLAUDE.md no-look-ahead invariant; fire on bar close, feed only
closed bars):
  CMF[t] is the standard indicator computed through bar t (uses bars <= t). The
  fade SIGNAL derived from CMF[t] is then SHIFTED ONE BAR (`signal.shift(1)`),
  so the side emitted at bar t was decided from CMF[t-1] — the bar t's own CMF
  can never inform the position entered at bar t. This `.shift(1)` is the
  materialized form of the spike's `cmf_meanrev_position`. Proven by the
  prefix-invariance test in tests/test_cmf_meanrev_primary.py
  (signal[t] is unchanged when all bars > t are truncated).

  The walk_forward purge/embargo is a BACKTEST-only device; it is irrelevant to
  this primary, which is causal by construction via the shift. Live execution
  (fire on bar close, act next bar) reproduces this exact shift, so the live and
  backtest decision sequences are identical on the same data.

ZERO-RANGE GUARD: when high == low the Money-Flow Multiplier ((c-l)-(h-c))/(h-l)
is 0/0; it is DEFINED as 0.0 (no money-flow contribution) so a single doji never
NaN-poisons the rolling CMF window.

LOCKED PARAMS (cfg-driven, deterministic): the lower/upper fade thresholds are
NOT recomputed here as in-sample quantiles — they are passed as the ABSOLUTE
constants the spike locked on its train window (period, lo_thresh, hi_thresh,
allow_short), via cfg["primary"]["cmf_meanrev"]. Passing constants (rather than
recomputing quantiles inside the primary) keeps the primary pure/deterministic
AND strictly causal: there is no in-sample statistic computed over the audit
window that could leak across the walk-forward split.

INPUT_COLUMNS is empty: the primary reads only raw OHLCV (high/low/close/volume)
from `ohlcv`, never the meta `features` frame, so the orchestrator's
assert_primary_inputs_disjoint check passes trivially against any blacklist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INPUT_COLUMNS: tuple[str, ...] = ()

# Defaults mirror the standard CMF period; thresholds default to 0 so a
# mis-wired cfg fires symmetrically rather than silently never firing. The
# proposal ALWAYS supplies the locked params, so these defaults are only a
# defensive fallback.
DEFAULT_PERIOD = 20
DEFAULT_LO_THRESH = 0.0
DEFAULT_HI_THRESH = 0.0
DEFAULT_ALLOW_SHORT = True


def money_flow_multiplier(ohlcv: pd.DataFrame) -> pd.Series:
    """Money-Flow Multiplier per bar: ((close-low)-(high-close)) / (high-low).

    Bounded in [-1, +1]: +1 when close==high (all buying), -1 when close==low
    (all selling), 0 at the mid. ZERO-RANGE GUARD: high==low -> 0.0 (not NaN).
    Verbatim from scripts/spike_cmf_meanrev.py::money_flow_multiplier.
    """
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)
    rng = high - low
    num = (close - low) - (high - close)
    mfm = num / rng
    mfm = mfm.where(rng > 0.0, 0.0)
    return mfm


def chaikin_money_flow(ohlcv: pd.DataFrame, period: int) -> pd.Series:
    """CMF[t] = sum_{t-period+1..t}(MFM_i * vol_i) / sum_i(vol_i).

    Causal: uses only bars <= t (standard indicator). NaN during the first
    period-1 warm-up bars and wherever trailing volume sums to 0.
    Verbatim from scripts/spike_cmf_meanrev.py::chaikin_money_flow.
    """
    mfm = money_flow_multiplier(ohlcv)
    vol = ohlcv["volume"].astype(float)
    mfv = mfm * vol
    sum_mfv = mfv.rolling(period).sum()
    sum_vol = vol.rolling(period).sum()
    cmf = sum_mfv / sum_vol.where(sum_vol > 0.0, np.nan)
    return cmf


def _cmf_meanrev_raw_signal(
    cmf: pd.Series, lo_thresh: float, hi_thresh: float, allow_short: bool
) -> pd.Series:
    """Fade CMF extremes (pre-shift). +1 when CMF < lo_thresh (oversold ->
    revert up), -1 when CMF > hi_thresh (overbought -> revert down) if
    allow_short, else 0. 0 elsewhere and wherever CMF is NaN (warm-up).
    Verbatim from scripts/spike_cmf_meanrev.py::cmf_meanrev_signal."""
    sig = pd.Series(0.0, index=cmf.index)
    valid = cmf.notna()
    sig[valid & (cmf < lo_thresh)] = 1.0
    if allow_short:
        sig[valid & (cmf > hi_thresh)] = -1.0
    return sig


def signal(ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: dict) -> pd.Series:
    """Return the CMF mean-reversion fade side in {-1, 0, +1}, indexed like ohlcv.

    Reads the locked params from cfg["primary"]["cmf_meanrev"]:
      period, lo_thresh, hi_thresh, allow_short.

    The raw fade signal (decided from CMF[t]) is shifted one bar so the side at
    bar t reflects CMF[t-1] — the no-look-ahead invariant. Warm-up / shifted-in
    NaNs collapse to 0 (no signal).
    """
    params = (cfg.get("primary", {}) or {}).get("cmf_meanrev", {}) or {}
    period = int(params.get("period", DEFAULT_PERIOD))
    lo_thresh = float(params.get("lo_thresh", DEFAULT_LO_THRESH))
    hi_thresh = float(params.get("hi_thresh", DEFAULT_HI_THRESH))
    allow_short = bool(params.get("allow_short", DEFAULT_ALLOW_SHORT))

    cmf = chaikin_money_flow(ohlcv, period=period)
    raw = _cmf_meanrev_raw_signal(cmf, lo_thresh, hi_thresh, allow_short)
    # CAUSAL SHIFT: position decided at t uses CMF through t-1.
    shifted = raw.shift(1)
    out = shifted.fillna(0.0).astype("int8")
    return out
