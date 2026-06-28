"""Phase 5 custom primary for Loop A proposal 20260530-EURUSD-D1-BEAR_STR-T137.

Pseudocode (verbatim from proposal):
  Compute: vol_rank[t] = percentile rank of volume[t] within the trailing
  21-bar window (bars t-20 through t inclusive).
  Compute: ema21[t] = exponential moving average of close with span=21,
  using all bars up to and including t.
  Signal logic:
    if vol_rank[t] > 0.60 AND close[t] < ema21[t]: signal = -1 (short).
    if vol_rank[t] > 0.60 AND close[t] > ema21[t]: signal = +1 (long).
    Otherwise: signal = 0.
  No look-ahead: vol_rank and ema21 use only information available at bar t.

Causal-window discipline:
  - vol_rank uses rolling(21).rank(pct=True): the rank of the CURRENT bar's
    volume within its own trailing window (t-20..t inclusive) - volume[t] is
    known at the close of bar t, when the signal is stamped.
  - ema21 uses ewm(span=21, adjust=False) - recursive, strictly trailing.
  - close[t] == ema21[t] (exact tie) emits 0, matching the pseudocode's
    strict inequalities.
  - No reference to dates, named events, or absolute price levels.

Frozen interpretation decisions (DA review 2026-06-10, PROCEED_WITH_CAVEAT —
locked BEFORE any M3 audit run):
  1. PERCENTILE RANK = pandas pct rank, INCLUSIVE of the current bar,
     denominator n=21: rank k of 21 -> k/21. Boundary consequence at the
     committed 0.60 threshold: rank 13 (13/21 ~= 0.619) FIRES, rank 12
     (12/21 ~= 0.571) does not. The exclusive (k-1)/(n-1) reading would
     move this boundary; it is rejected. Pinned by test.
  2. VOLUME TIES use rank method='average' (explicit in the code). FX tick
     volume ties are common; the averaged fractional rank is the committed
     convention.
  3. EMA CONVENTION: ewm(span=21, adjust=False) — forward-recursive,
     seeded on bar 0. The first ~2*span bars carry seed bias; the audit's
     warmup/fold handling absorbs it. adjust=True is rejected.
  4. cfg/primary_params are intentionally IGNORED at runtime: the module
     constants are frozen-by-design to the proposal's committed params.

DA-mandated audit-reading requirement: the hypothesis narrative argues only
for the SHORT side; the committed pseudocode is two-sided and is implemented
faithfully. The M3 audit reading MUST slice PnL and trade counts per side —
if the long side carries the result, survival is narrative-unfalsified and
routes to a refinement lineage, not promotion.
"""
from __future__ import annotations

import pandas as pd

INPUT_COLUMNS: tuple[str, ...] = ()  # reads only ohlcv

VOLUME_PCT_RANK_LOOKBACK = 21    # primary_params.volume_pct_rank_lookback
VOLUME_PCT_RANK_THRESHOLD = 0.60  # primary_params.volume_pct_rank_threshold
PRICE_EMA_LOOKBACK = 21          # primary_params.price_ema_lookback


def signal(ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: dict) -> pd.Series:
    """Volume-gated EMA-side signal: high participation confirms direction."""
    close = ohlcv["close"].astype(float)
    vol = ohlcv["volume"].astype(float)

    vol_rank = vol.rolling(
        VOLUME_PCT_RANK_LOOKBACK, min_periods=VOLUME_PCT_RANK_LOOKBACK
    ).rank(method="average", pct=True)
    ema = close.ewm(span=PRICE_EMA_LOOKBACK, adjust=False).mean()

    gated = vol_rank > VOLUME_PCT_RANK_THRESHOLD  # NaN warmup compares False
    out = pd.Series(0, index=ohlcv.index, dtype="int8")
    out[gated & (close < ema)] = -1
    out[gated & (close > ema)] = 1
    return out
