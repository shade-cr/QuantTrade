"""Tests for the T137 volume-gated EMA-side custom primary
(proposal 20260530-EURUSD-D1-BEAR_STR-T137, materialized per B0040 Option B).

The load-bearing test is `test_prefix_stability_no_lookahead` (see
test_phase5_b003v1.py for rationale).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.primaries_phase5.phase5_t137 import (
    signal,
    INPUT_COLUMNS,
    VOLUME_PCT_RANK_LOOKBACK,
    VOLUME_PCT_RANK_THRESHOLD,
    PRICE_EMA_LOOKBACK,
)


def _empty_features(idx: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(index=idx)


def test_input_columns_is_empty():
    assert INPUT_COLUMNS == ()


def test_output_contract(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    out = signal(df, _empty_features(df.index), cfg={})
    assert out.index.equals(df.index)
    assert out.dtype == np.int8
    assert set(out.unique()).issubset({-1, 0, 1})


def test_determinism(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    a = signal(df, _empty_features(df.index), cfg={})
    b = signal(df, _empty_features(df.index), cfg={})
    pd.testing.assert_series_equal(a, b)


def test_emits_both_sides(synth_ohlcv):
    """The pseudocode is two-sided (direction from the EMA side); on 500
    synthetic bars both sides should fire."""
    df = synth_ohlcv.set_index("time")
    out = signal(df, _empty_features(df.index), cfg={})
    assert (out == -1).sum() >= 1
    assert (out == 1).sum() >= 1


def test_warmup_emits_nothing(synth_ohlcv):
    """vol_rank is NaN until the 21-bar window fills; NaN > 0.60 is False."""
    df = synth_ohlcv.set_index("time")
    out = signal(df, _empty_features(df.index), cfg={})
    assert (out.iloc[: VOLUME_PCT_RANK_LOOKBACK - 1] == 0).all()


def test_signal_matches_formula(synth_ohlcv):
    """Recompute the rule independently and compare bar-by-bar."""
    df = synth_ohlcv.set_index("time")
    out = signal(df, _empty_features(df.index), cfg={})
    vol = df["volume"].astype(float)
    close = df["close"].astype(float)
    vol_rank = vol.rolling(
        VOLUME_PCT_RANK_LOOKBACK, min_periods=VOLUME_PCT_RANK_LOOKBACK
    ).rank(pct=True)
    ema = close.ewm(span=PRICE_EMA_LOOKBACK, adjust=False).mean()
    gated = vol_rank > VOLUME_PCT_RANK_THRESHOLD
    expected = pd.Series(0, index=df.index, dtype="int8")
    expected[gated & (close < ema)] = -1
    expected[gated & (close > ema)] = 1
    pd.testing.assert_series_equal(out, expected)


def test_low_volume_bars_emit_nothing(synth_ohlcv):
    """Bars whose volume rank is at/below the threshold never signal."""
    df = synth_ohlcv.set_index("time")
    out = signal(df, _empty_features(df.index), cfg={})
    vol_rank = df["volume"].astype(float).rolling(
        VOLUME_PCT_RANK_LOOKBACK, min_periods=VOLUME_PCT_RANK_LOOKBACK
    ).rank(pct=True)
    below = vol_rank <= VOLUME_PCT_RANK_THRESHOLD
    assert (out[below.fillna(True)] == 0).all()


def test_rank_boundary_semantics_pinned():
    """DA caveat (2026-06-10): percentile rank is INCLUSIVE-of-self, k/n.
    At the 0.60 threshold with n=21: rank 13 (13/21 ~= 0.619) fires,
    rank 12 (12/21 ~= 0.571) does not. Pins the committed reading against
    the exclusive (k-1)/(n-1) alternative, which would silence rank 13."""
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    # Strictly increasing volume 1..21 in the last window would make the
    # final bar rank 21/21. Build windows where the LAST bar has a chosen
    # rank k by giving it value k - 0.5 among 1..21.
    close = pd.Series(np.linspace(100.0, 120.0, n), index=idx)  # above EMA -> +1 side

    def _df_with_final_rank(k: int) -> pd.DataFrame:
        vol = np.ones(n)
        # Final 21-bar window: values 1..20 then the current bar at k-0.5
        vol[-21:-1] = np.arange(1.0, 21.0)
        vol[-1] = k - 0.5
        return pd.DataFrame(
            {"open": close, "high": close * 1.001, "low": close * 0.999,
             "close": close, "volume": vol},
            index=idx,
        )

    out_13 = signal(_df_with_final_rank(13), _empty_features(idx), cfg={})
    out_12 = signal(_df_with_final_rank(12), _empty_features(idx), cfg={})
    assert out_13.iloc[-1] == 1, "rank 13/21 (0.619 > 0.60) must fire"
    assert out_12.iloc[-1] == 0, "rank 12/21 (0.571 <= 0.60) must not fire"


def test_prefix_stability_no_lookahead(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    full = signal(df, _empty_features(df.index), cfg={})
    k = 300
    prefix = signal(df.iloc[:k], _empty_features(df.index[:k]), cfg={})
    pd.testing.assert_series_equal(full.iloc[:k], prefix)


def test_future_mutation_does_not_change_past(synth_ohlcv):
    df = synth_ohlcv.set_index("time")
    a = signal(df, _empty_features(df.index), cfg={})
    mutated = df.copy()
    mutated.iloc[400, mutated.columns.get_loc("close")] *= 3
    mutated.iloc[400, mutated.columns.get_loc("volume")] *= 50
    b = signal(mutated, _empty_features(df.index), cfg={})
    pd.testing.assert_series_equal(a.iloc[:400], b.iloc[:400])
