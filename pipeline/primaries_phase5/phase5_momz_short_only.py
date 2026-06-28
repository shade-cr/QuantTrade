"""Phase 5 custom primary for Loop A proposal 20260611-GBPUSD-D1-BEAR_QUI-F008.

Refinement of F007 (DA BLOCK: short-specific claim vs pooled two-sided
criterion). Resolution is the DA's remediation (a): a SHORT-ONLY primary, so
the pooled falsification criterion tests exactly the claimed side.

Pseudocode (verbatim from proposal):
  r_t  = ln(close_t / close_{t-21})            # 21-bar log return, lookback=21
  mu_t = rolling_mean(r, window=252)           # trailing 252 bars, inclusive of t
  sd_t = rolling_std(r, window=252, ddof=1)    # pandas default ddof=1
  z_t  = (r_t - mu_t) / sd_t
  side_t = -1 if z_t < -0.3 (strict) else 0    # short-only: never emits +1
  if r_t, mu_t, or sd_t is NaN (warmup) -> side_t = 0
  if sd_t == 0 -> side_t = 0

Frozen interpretation decisions (locked BEFORE any M3 audit run):
  1. IDENTITY WITH THE BUILT-IN: the proposal commits that "every -1 it emits
     is a -1 the built-in emits, and it emits nothing else" — the dossier
     baseline inheritance (momentum_zscore n_events=1570, hit_rate_q=1.0 on
     this cell) depends on it. Implemented by CALLING
     pipeline.labels.momentum_zscore_signal(lookback=21, threshold=0.3) and
     clipping to the short branch, so the identity holds by construction and
     cannot drift from a re-implementation. Pinned by test.
  2. SHORT-ONLY CLIP: sig.where(sig == -1, 0) — +1 and 0 both map to 0; -1
     passes through. No other transformation.
  3. WARMUP / ZERO-STD: inherited from the built-in (z NaN during the
     lookback+252 warmup or when sd==0 compares False on both thresholds →
     emits 0). No extra handling needed; matches the pseudocode's NaN→0 rule.
  4. cfg/primary_params are intentionally IGNORED at runtime: lookback=21 and
     threshold=0.3 are frozen-by-design to the proposal's committed params
     (same convention as phase5_t137).

PIT safety: the built-in uses only close_t and trailing rolling windows
inclusive of t; no forward shift inside signal() — execution-at-next-bar
follows the pipeline's standard event convention. No date, calendar, volume,
or absolute-price-level logic.
"""
from __future__ import annotations

import pandas as pd

from pipeline.labels import momentum_zscore_signal

INPUT_COLUMNS: tuple[str, ...] = ()  # reads only ohlcv

LOOKBACK = 21      # frozen: proposal primary commitment
THRESHOLD = 0.3    # frozen: proposal primary commitment


def signal(ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: dict) -> pd.Series:
    """Short branch of the built-in momentum z-score primary; never emits +1."""
    sig = momentum_zscore_signal(
        ohlcv["close"].astype(float), lookback=LOOKBACK, threshold=THRESHOLD
    )
    return sig.where(sig == -1.0, 0.0)
