"""B0018 — opportunistic-insider event primary (pre-registered 2026-07-04).

Frozen rule (docs/superpowers/specs/2026-07-04-b0018-insider-event-preregistration.md,
committed BEFORE any data look; parameters are NOT tunable without a new DSR trial):

  +1 (long) on the FIRST bar at/after the effective knowledge day of each
  qualifying OPPORTUNISTIC purchase filing (CMP filter — see
  pipeline.insider_events). Multiple filings mapping to the same bar = one
  signal. No re-fire for the same ticker within REFIRE_GAP_BARS bars of a
  prior fire. Never -1. 0 otherwise.

Knowledge discipline: the event time is the Form 4 FILING acceptance
timestamp (never the transaction date), rolled to the next session when
accepted after the close (effective_knowledge_day — the single knowledge-day
convention shared with B0017).

Asset identity: the pooled/backtest runner injects cfg["_current_asset"];
this module fails loud without it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.insider_events import (
    load_insider_purchases,
    opportunistic_knowledge_days,
)

INPUT_COLUMNS: tuple[str, ...] = ()  # reads only ohlcv index + insider cache

REFIRE_GAP_BARS = 10  # matches the h10 holding horizon; FROZEN


def signal(ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: dict) -> pd.Series:
    asset = cfg.get("_current_asset")
    if not asset:
        raise ValueError(
            "phase5_insider_event requires cfg['_current_asset'] (injected by "
            "the runner) to load the ticker's insider cache — refusing to guess"
        )
    admit_unclassifiable = bool(cfg.get("insider_admit_unclassifiable", False))
    kd = opportunistic_knowledge_days(
        load_insider_purchases(asset), admit_unclassifiable=admit_unclassifiable
    )
    out = pd.Series(0, index=ohlcv.index, dtype=int)
    if len(kd) == 0:
        return out

    bar_dates = pd.DatetimeIndex(ohlcv.index).tz_localize(None).normalize()
    # first bar at/after each knowledge day; events past the last bar drop out
    pos = np.searchsorted(bar_dates.values, pd.DatetimeIndex(kd).values, side="left")
    pos = np.unique(pos[pos < len(bar_dates)])

    last_fired = -REFIRE_GAP_BARS - 1
    for p in pos:
        if p - last_fired >= REFIRE_GAP_BARS:
            out.iloc[p] = 1
            last_fired = p
    return out
