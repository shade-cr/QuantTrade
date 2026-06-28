"""Tests for the T015D2 short-only CUSUM-downside + volume-gate custom primary.

The most important test here is `test_prefix_stability_no_lookahead`: a truly
causal primary must produce identical signals on bars 0..k-1 whether it is run
on the full series or on the first-k-bar prefix. Any accidental future-peeking
(full-sample quantile, centered window, negative shift) would make the
full-run value on an early bar differ from the prefix-run value, and this test
would catch it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.primaries_phase5.phase5_t015d2 import (
    signal,
    INPUT_COLUMNS,
    VOLUME_WINDOW,
)


def _empty_features(idx: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(index=idx)


def test_input_columns_is_empty():
    # Reads volume from the ohlcv frame, not features → disjointness check trivial.
    assert INPUT_COLUMNS == ()


def test_output_index_matches_ohlcv(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    out = signal(df, _empty_features(df.index), cfg={})
    assert out.index.equals(df.index)
    assert len(out) == len(df)


def test_values_in_short_only_set(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    out = signal(df, _empty_features(df.index), cfg={})
    assert set(out.unique()).issubset({-1, 0})
    # Never long.
    assert (out == 1).sum() == 0
    assert (out > 0).sum() == 0


def test_dtype_is_int8(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    out = signal(df, _empty_features(df.index), cfg={})
    assert out.dtype == np.int8


def test_determinism(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    a = signal(df, _empty_features(df.index), cfg={})
    b = signal(df, _empty_features(df.index), cfg={})
    pd.testing.assert_series_equal(a, b)


def test_emits_at_least_one_short(synth_ohlcv):
    # Sanity: on the 500-bar synthetic frame the primary should fire at least
    # once, otherwise the test below is vacuous.
    df = synth_ohlcv.set_index("time")
    out = signal(df, _empty_features(df.index), cfg={})
    assert (out == -1).sum() >= 1


def test_warmup_bars_emit_no_signal(synth_ohlcv):
    # Before VOLUME_WINDOW bars of history exist, the volume rank is NaN →
    # coerced to "no signal", so no short can fire in the warm-up region.
    df = synth_ohlcv.set_index("time")
    out = signal(df, _empty_features(df.index), cfg={})
    assert (out.iloc[:VOLUME_WINDOW] == 0).all()


def test_prefix_stability_no_lookahead(synth_ohlcv):
    """Truncating the input to the first k bars must not change the signals on
    bars 0..k-1. This is the core lookahead guard: a causal primary's output at
    bar t depends only on bars <= t, so a prefix run and the full run must agree
    on the overlapping prefix for every prefix length tested.
    """
    df = synth_ohlcv.set_index("time")
    full = signal(df, _empty_features(df.index), cfg={})

    # Test several prefix cutoffs spanning warm-up and steady-state regions.
    for k in (50, 75, 120, 200, 333, 499):
        assert k <= len(df)
        prefix_df = df.iloc[:k]
        prefix_out = signal(prefix_df, _empty_features(prefix_df.index), cfg={})
        # Same length and index as the truncated input.
        assert len(prefix_out) == k
        assert prefix_out.index.equals(df.index[:k])
        # Identical to the full run on the overlapping prefix.
        pd.testing.assert_series_equal(
            prefix_out,
            full.iloc[:k],
            obj=f"prefix k={k}",
        )
