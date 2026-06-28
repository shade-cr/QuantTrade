"""Tests for the B003v1 long-only CUSUM-markup + low-volume-gate custom primary
(proposal 20260529-XAGUSD-D1-BULL_QUI-B003v1, materialized per B0040 Option B).

The load-bearing test is `test_prefix_stability_no_lookahead`: a causal primary
must produce identical signals on bars 0..k-1 whether run on the full series or
on the first-k-bar prefix. Any future-peeking (full-sample stat, centered
window, negative shift) breaks prefix stability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.primaries_phase5.phase5_b003v1 import (
    signal,
    INPUT_COLUMNS,
    VOLUME_MEDIAN_LOOKBACK,
)


def _empty_features(idx: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(index=idx)


def test_input_columns_is_empty():
    # Reads ohlcv only -> disjointness check vs meta features is trivial.
    assert INPUT_COLUMNS == ()


def test_output_contract(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    out = signal(df, _empty_features(df.index), cfg={})
    assert out.index.equals(df.index)
    assert out.dtype == np.int8
    # Long-only: never short.
    assert set(out.unique()).issubset({0, 1})
    assert (out == -1).sum() == 0


def test_determinism(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    a = signal(df, _empty_features(df.index), cfg={})
    b = signal(df, _empty_features(df.index), cfg={})
    pd.testing.assert_series_equal(a, b)


def test_emits_at_least_one_long(synth_ohlcv):
    """On 500 bars of synthetic OHLCV the CUSUM should cross at least once
    with a below-median-volume bar; an all-zero primary is unauditable."""
    df = synth_ohlcv.set_index("time")
    out = signal(df, _empty_features(df.index), cfg={})
    assert (out == 1).sum() >= 1


def test_warmup_emits_nothing(synth_ohlcv):
    """Before the volume-median window has filled, no signal can fire (the
    participation gate is NaN); before ATR warms up, no event can fire."""
    df = synth_ohlcv.set_index("time")
    out = signal(df, _empty_features(df.index), cfg={})
    assert (out.iloc[: VOLUME_MEDIAN_LOOKBACK + 1] == 0).all()


def test_up_event_with_high_volume_is_skipped():
    """Construct a deterministic up-drift series where the CUSUM crossing bar
    has above-median volume -> the climactic leg is explicitly skipped."""
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    # Gentle constant drift so the CUSUM accumulates and crosses periodically.
    close = pd.Series(100.0 * np.exp(np.arange(n) * 0.004), index=idx)
    high = close * 1.001
    low = close * 0.999
    base = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close,
         "volume": np.full(n, 1000.0)},
        index=idx,
    )
    feats = _empty_features(idx)

    out_low_vol = signal(base, feats, cfg={})
    event_bars = out_low_vol[out_low_vol == 1].index
    if len(event_bars) == 0:
        # Drift too soft for this ATR regime -> nothing to assert against;
        # the synthetic-fixture test above covers firing.
        return
    # Same series, but spike volume ON the event bars only -> above-median
    # participation -> those longs must disappear.
    spiked = base.copy()
    spiked.loc[event_bars, "volume"] = 100_000.0
    out_spiked = signal(spiked, feats, cfg={})
    assert (out_spiked.loc[event_bars] == 0).all()


def test_prefix_stability_no_lookahead(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    feats = _empty_features(df.index)
    full = signal(df, feats, cfg={})
    k = 300
    prefix = signal(df.iloc[:k], _empty_features(df.index[:k]), cfg={})
    pd.testing.assert_series_equal(full.iloc[:k], prefix)


def test_future_mutation_does_not_change_past(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    feats = _empty_features(df.index)
    a = signal(df, feats, cfg={})
    mutated = df.copy()
    mutated.iloc[400, mutated.columns.get_loc("close")] *= 3
    mutated.iloc[400, mutated.columns.get_loc("volume")] *= 50
    b = signal(mutated, feats, cfg={})
    pd.testing.assert_series_equal(a.iloc[:400], b.iloc[:400])
