"""Phase 5 custom primary for Loop A proposal 20260611-EURUSD-D1-BULL_QUI-G003.

Long-only dip-buy mean-reversion: 5-bar log-return z-score over a trailing
42-bar window; LONG when z < -0.75; never short.

Pseudocode (verbatim from proposal):
  r5[t] = log(close[t] / close[t-5])
  z[t]  = (r5[t] - mean(r5 over bars t-41..t inclusive))
          / std(r5 over the same 42 bars, population std ddof=0)
  Signal[t] = +1 if z[t] < -0.75 else 0
  Signal[t] = 0 if fewer than 47 bars of history (5 + 42), std == 0, or z >= -0.75.
  Never emit -1. State-based: stays +1 on every consecutive bar the condition
  holds. No date, event, or absolute-price references.

Frozen interpretation decisions (DA review 2026-06-11, PROCEED_WITH_CAVEAT —
locked BEFORE any M3 audit run):
  1. STD CONVENTION: population std (ddof=0), explicit in the pseudocode.
     Pinned by test against the sample-std alternative.
  2. WARMUP: strictly "fewer than 47 bars" -> 0; implemented via min_periods=42
     on the rolling window over r5 (r5 itself needs 5 bars, so the first
     full window ends at bar index 46, the 47th bar).
  3. NaN-WITHIN-WINDOW (DA low objection, convention adopted): any NaN among
     the inputs to r5 or the 42-bar window yields signal 0 at that bar —
     rolling with min_periods=42 over a window containing NaN produces NaN,
     and NaN comparisons emit 0. No imputation.
  4. STRICT INEQUALITY z < -0.75; z == -0.75 exactly emits 0.
  5. cfg/primary_params intentionally IGNORED at runtime: 5 / 42 / -0.75 are
     frozen-by-design to the proposal's committed params (phase5_t137
     convention).

DA-mandated reading obligations (recorded in the batch review file): distinct
entry-CLUSTER counts (first-bar-of-condition runs) reported next to raw trade
counts; per-episode survival unmet for episodes whose trades collapse into <2
clusters; MDA check on cs_spread_21 sign ("wider = better" claimed).

PIT safety: rolling windows trailing and inclusive of t; log/shift only look
back; no forward indexing. Verified by prefix-stability and future-mutation
tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INPUT_COLUMNS: tuple[str, ...] = ()  # reads only ohlcv

RET_LOOKBACK = 5      # frozen: proposal commitment
Z_WINDOW = 42         # frozen
Z_ENTRY = -0.75       # frozen


def signal(ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: dict) -> pd.Series:
    """Long-only dip z-score; +1 while z < -0.75, else 0. Never -1."""
    close = ohlcv["close"].astype(float)
    r5 = np.log(close / close.shift(RET_LOOKBACK))
    mu = r5.rolling(Z_WINDOW, min_periods=Z_WINDOW).mean()
    sd = r5.rolling(Z_WINDOW, min_periods=Z_WINDOW).std(ddof=0)
    z = (r5 - mu) / sd.where(sd > 0)
    out = pd.Series(0.0, index=ohlcv.index)
    out[z < Z_ENTRY] = 1.0   # NaN z compares False -> stays 0
    return out
