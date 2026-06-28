"""Phase 5 custom primary for Loop A proposal 20260611-XAUUSD-D1-BULLUNION-G006.

Long-only vol-scaled momentum: z = r21 / (sd126 * sqrt(21)), LONG when z > 0.5.
Deliberately NOT de-meaned (the v2 momentum-z de-meaning damped trends —
documented prior).

Pseudocode (verbatim from proposal):
  r1[t]  = ln(close[t]) - ln(close[t-1])
  r21[t] = ln(close[t]) - ln(close[t-21])
  sd[t]  = std of trailing 126 one-bar returns r1[t-125..t] (inclusive of t),
           multiplied by sqrt(21)
  z[t]   = r21[t] / sd[t]
  Signal[t] = +1 if z[t] > 0.5 else 0. Never -1 (long-only).
  Signal[t] = 0 if t < 147 bars from series start, or sd[t] is 0 or NaN.

Frozen interpretation decisions (DA review 2026-06-11, BLOCK resolved by
criterion amendments; pseudocode itself judged lookahead-clean — locked
BEFORE any M3 audit run):
  1. STD CONVENTION: the pseudocode does not pin ddof; FROZEN HERE as pandas
     default ddof=1 (sample std), consistent with the built-in momentum_zscore
     convention. Pinned by test.
  2. WARMUP: the committed pseudocode's LITERAL "t < 147 bars -> 0" governs
     (DA re-review 2026-06-11 medium objection: rolling(126) semantics alone
     would fire ~21 bars EARLIER than the spec — the looser convention, not
     the stricter as an earlier draft of this docstring wrongly claimed).
     Implemented as an explicit mask zeroing indices 0..146 on top of the
     rolling min_periods guard. Pinned by test.
  3. STRICT INEQUALITY z > 0.5; z == 0.5 exactly emits 0.
  4. sd == 0 or NaN -> 0 (guarded via .where(sd > 0); NaN comparisons False).
  5. cfg/primary_params intentionally IGNORED at runtime: 21 / 126 / 0.5
     frozen-by-design.

DA must-have mods (applied as extra_falsification_criteria on the proposal,
pre-registered): (1) >=1 surviving BULL_STRESSED episode AND >=10 trades from
BULL_STRESSED bars, else falsified even if the union aggregate passes;
(2) firing fraction Signal=+1 on <=50% of union bars measured at pre-flight,
else reclassified as gated long exposure and falsified as stated.

PIT safety: trailing windows inclusive of t only. Verified by prefix-stability
and future-mutation tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INPUT_COLUMNS: tuple[str, ...] = ()  # reads only ohlcv

LOOKBACK = 21        # frozen: proposal commitment
VOL_WINDOW = 126     # frozen
Z_ENTRY = 0.5        # frozen


def signal(ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: dict) -> pd.Series:
    """Long-only vol-scaled momentum; +1 while z > 0.5, else 0. Never -1."""
    close = ohlcv["close"].astype(float)
    logc = np.log(close)
    r1 = logc.diff()
    r21 = logc - logc.shift(LOOKBACK)
    sd = r1.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * np.sqrt(LOOKBACK)
    z = r21 / sd.where(sd > 0)
    out = pd.Series(0.0, index=ohlcv.index)
    out[z > Z_ENTRY] = 1.0   # NaN z compares False -> stays 0
    # Frozen decision 2: literal pseudocode warmup "t < 147 bars -> 0".
    out.iloc[: LOOKBACK + VOL_WINDOW] = 0.0
    return out
