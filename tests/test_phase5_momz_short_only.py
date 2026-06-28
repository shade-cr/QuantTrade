"""Tests for the F008 short-only momentum z-score custom primary
(proposal 20260611-GBPUSD-D1-BEAR_QUI-F008, refinement of DA-blocked F007,
materialized per B0040 Option B).

The load-bearing tests are `test_identity_with_builtin_short_branch` (the
proposal's dossier-baseline inheritance commits that this primary is EXACTLY
the built-in momentum_zscore restricted to its short branch) and
`test_prefix_stability_no_lookahead`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.labels import momentum_zscore_signal
from pipeline.primaries_phase5.phase5_momz_short_only import (
    signal,
    INPUT_COLUMNS,
    LOOKBACK,
    THRESHOLD,
)


def _empty_features(idx: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(index=idx)


# Golden literals for test_golden_output_pinned — computed once at
# materialization (2026-06-11) on _trending_ohlcv(n=700, seed=7). If these
# break, the built-in momentum_zscore_signal formula has drifted and this
# frozen primary is no longer the audited object.
GOLDEN_N_SHORTS = 118
GOLDEN_FIRST_SHORT_POS = 327
GOLDEN_LAST_SHORT_POS = 695
GOLDEN_POS_CHECKSUM = 51547


def _trending_ohlcv(n: int = 700, seed: int = 7) -> pd.DataFrame:
    """Synthetic series long enough to clear the 21+252-bar warmup, with an
    embedded downtrend segment so the short branch demonstrably fires."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC")
    drift = np.zeros(n)
    drift[350:450] = -0.004   # bear impulse after warmup
    drift[550:650] = 0.004    # bull impulse (must NOT fire short-only)
    ret = drift + rng.normal(0, 0.006, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": np.full(n, 1000.0)},
        index=idx,
    )


def test_input_columns_is_empty():
    assert INPUT_COLUMNS == ()


def test_frozen_params():
    """Proposal commitment: lookback=21, threshold=0.3, ignored cfg."""
    assert LOOKBACK == 21
    assert THRESHOLD == 0.3


def test_identity_with_builtin_short_branch():
    """Frozen decision 1: every -1 the module emits is a -1 the built-in
    emits, and it emits nothing else (no +1, ever)."""
    df = _trending_ohlcv()
    out = signal(df, _empty_features(df.index), cfg={})
    builtin = momentum_zscore_signal(df["close"], lookback=LOOKBACK, threshold=THRESHOLD)
    expected = builtin.where(builtin == -1.0, 0.0)
    pd.testing.assert_series_equal(out, expected)
    assert set(out.unique()).issubset({-1.0, 0.0})
    assert (out == -1).sum() >= 1, "short branch must fire on the bear segment"
    assert (builtin == 1).sum() >= 1, "built-in fires long on the bull segment..."
    assert (out[builtin == 1] == 0).all(), "...but the short-only module must not"


def test_determinism():
    df = _trending_ohlcv()
    a = signal(df, _empty_features(df.index), cfg={})
    b = signal(df, _empty_features(df.index), cfg={})
    pd.testing.assert_series_equal(a, b)


def test_warmup_emits_nothing():
    """z is NaN until lookback+252 bars fill; NaN < -0.3 compares False."""
    df = _trending_ohlcv()
    out = signal(df, _empty_features(df.index), cfg={})
    assert (out.iloc[: LOOKBACK + 252 - 1] == 0).all()


def test_cfg_and_params_ignored():
    """Frozen decision 4: runtime cfg cannot move the committed params."""
    df = _trending_ohlcv()
    a = signal(df, _empty_features(df.index), cfg={})
    b = signal(df, _empty_features(df.index),
               cfg={"primary": {"phase5_momz_short_only": {"lookback": 5, "threshold": 0.01}}})
    pd.testing.assert_series_equal(a, b)


def test_prefix_stability_no_lookahead():
    df = _trending_ohlcv()
    full = signal(df, _empty_features(df.index), cfg={})
    k = 500
    prefix = signal(df.iloc[:k], _empty_features(df.index[:k]), cfg={})
    pd.testing.assert_series_equal(full.iloc[:k], prefix)


def test_future_mutation_does_not_change_past():
    df = _trending_ohlcv()
    a = signal(df, _empty_features(df.index), cfg={})
    mutated = df.copy()
    mutated.iloc[600, mutated.columns.get_loc("close")] *= 3.0
    b = signal(mutated, _empty_features(mutated.index), cfg={})
    pd.testing.assert_series_equal(a.iloc[:600], b.iloc[:600])


def test_constant_price_sd_zero_emits_nothing():
    """DA caveat (F008 review, medium #2): the frozen pseudocode says
    sd_t == 0 -> side 0. On constant close, r is exactly 0 everywhere, so
    mu == r_t and sd == 0 in the SAME inclusive window -> z = 0/0 = NaN ->
    the built-in emits 0. The feared z = -inf branch (r < mu with sd == 0)
    is unreachable because mu/sd share r_t's window: sd == 0 forces
    mu == r_t. This test pins that the freeze holds without an extra mask."""
    n = 400
    idx = pd.date_range("2019-01-01", periods=n, freq="D", tz="UTC")
    close = np.full(n, 100.0)
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close,
         "close": close, "volume": np.full(n, 1000.0)},
        index=idx,
    )
    out = signal(df, _empty_features(idx), cfg={})
    assert (out == 0).all()


def test_golden_output_pinned():
    """DA caveat (F008 review, medium #3): the identity test compares against
    the LIVE built-in, so both sides move together if pipeline.labels.
    momentum_zscore_signal is ever edited. This golden test pins the actual
    output on a fixed deterministic series with committed literals — any
    formula drift in the built-in breaks it."""
    df = _trending_ohlcv(n=700, seed=7)  # deterministic: fixed seed
    out = signal(df, _empty_features(df.index), cfg={})
    fired = np.flatnonzero((out == -1.0).to_numpy())
    # Committed literals (computed once at materialization, 2026-06-11):
    assert int((out == -1.0).sum()) == GOLDEN_N_SHORTS
    assert fired[0] == GOLDEN_FIRST_SHORT_POS
    assert fired[-1] == GOLDEN_LAST_SHORT_POS
    assert int(fired.sum()) == GOLDEN_POS_CHECKSUM
