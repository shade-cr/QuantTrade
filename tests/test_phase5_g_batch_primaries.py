"""Tests for the batch-v3 custom primaries (B0040 Option B materialization):
phase5_dip_meanrev_long (G003, EURUSD BULL_QUIET) and
phase5_volmom_long_only (G006, XAUUSD bull-union).

Load-bearing: formula pins (against the frozen interpretation decisions in
each module docstring) + prefix-stability (no lookahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.primaries_phase5.phase5_dip_meanrev_long import (
    signal as dip_signal,
    INPUT_COLUMNS as DIP_INPUTS,
    RET_LOOKBACK, Z_WINDOW, Z_ENTRY as DIP_ENTRY,
)
from pipeline.primaries_phase5.phase5_volmom_long_only import (
    signal as volmom_signal,
    INPUT_COLUMNS as VOLMOM_INPUTS,
    LOOKBACK, VOL_WINDOW, Z_ENTRY as VOLMOM_ENTRY,
)


def _ohlcv(n=500, seed=3, drift=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="D", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(drift + rng.normal(0, 0.008, n)))
    return pd.DataFrame({"open": close, "high": close * 1.001,
                         "low": close * 0.999, "close": close,
                         "volume": np.full(n, 1000.0)}, index=idx)


_EMPTY = lambda df: pd.DataFrame(index=df.index)  # noqa: E731


# --------------------------- phase5_dip_meanrev_long ----------------------- #
def test_dip_inputs_and_long_only():
    assert DIP_INPUTS == ()
    df = _ohlcv()
    out = dip_signal(df, _EMPTY(df), cfg={})
    assert set(out.unique()).issubset({0.0, 1.0})
    assert (out == 1).sum() >= 1, "dips must fire on 500 random-walk bars"


def test_dip_formula_pinned_ddof0():
    """Frozen decision 1: population std (ddof=0). Scan deterministic seeds
    for a fixture where the ddof=0 vs ddof=1 conventions DISAGREE (z within
    ~1.2% of the -0.75 threshold), then pin equality with ddof=0 there."""
    discriminating = None
    for seed in range(60):
        df = _ohlcv(seed=seed)
        r5 = np.log(df["close"] / df["close"].shift(RET_LOOKBACK))
        mu = r5.rolling(Z_WINDOW, min_periods=Z_WINDOW).mean()
        sd0 = r5.rolling(Z_WINDOW, min_periods=Z_WINDOW).std(ddof=0)
        sd1 = r5.rolling(Z_WINDOW, min_periods=Z_WINDOW).std(ddof=1)
        fire0 = ((r5 - mu) / sd0) < DIP_ENTRY
        fire1 = ((r5 - mu) / sd1) < DIP_ENTRY
        if (fire0 != fire1).any():
            discriminating = (df, fire0)
            break
    assert discriminating is not None, "no seed in 0..59 discriminates ddof"
    df, fire0 = discriminating
    out = dip_signal(df, _EMPTY(df), cfg={})
    expected = pd.Series(0.0, index=df.index)
    expected[fire0] = 1.0
    pd.testing.assert_series_equal(out, expected)


def test_dip_warmup_47_bars():
    df = _ohlcv()
    out = dip_signal(df, _EMPTY(df), cfg={})
    assert (out.iloc[:Z_WINDOW + RET_LOOKBACK - 1] == 0).all()


def test_dip_prefix_stability():
    df = _ohlcv()
    full = dip_signal(df, _EMPTY(df), cfg={})
    prefix = dip_signal(df.iloc[:300], _EMPTY(df.iloc[:300]), cfg={})
    pd.testing.assert_series_equal(full.iloc[:300], prefix)


def test_dip_future_mutation_immune():
    df = _ohlcv()
    a = dip_signal(df, _EMPTY(df), cfg={})
    m = df.copy(); m.iloc[400, m.columns.get_loc("close")] *= 2
    b = dip_signal(m, _EMPTY(m), cfg={})
    pd.testing.assert_series_equal(a.iloc[:400], b.iloc[:400])


# --------------------------- phase5_volmom_long_only ----------------------- #
def test_volmom_inputs_and_long_only():
    assert VOLMOM_INPUTS == ()
    df = _ohlcv(drift=0.002)  # uptrend so momentum fires
    out = volmom_signal(df, _EMPTY(df), cfg={})
    assert set(out.unique()).issubset({0.0, 1.0})
    assert (out == 1).sum() >= 1


def test_volmom_formula_pinned_ddof1():
    """Frozen decision 1: sample std ddof=1 (pandas default), sqrt(21) scale."""
    df = _ohlcv(seed=4, drift=0.001)
    out = volmom_signal(df, _EMPTY(df), cfg={})
    logc = np.log(df["close"].astype(float))
    r1 = logc.diff()
    r21 = logc - logc.shift(LOOKBACK)
    sd = r1.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std(ddof=1) * np.sqrt(LOOKBACK)
    z = r21 / sd.where(sd > 0)
    expected = pd.Series(0.0, index=df.index)
    expected[z > VOLMOM_ENTRY] = 1.0
    expected.iloc[: LOOKBACK + VOL_WINDOW] = 0.0   # literal 147-bar warmup
    pd.testing.assert_series_equal(out, expected)


def test_volmom_never_short_in_downtrend():
    df = _ohlcv(drift=-0.003)
    out = volmom_signal(df, _EMPTY(df), cfg={})
    assert (out >= 0).all()
    assert (out == 1).mean() < 0.05


def test_volmom_warmup_literal_147():
    """Frozen decision 2 (DA re-review): the committed pseudocode's literal
    't < 147 bars -> 0' governs, NOT the looser rolling(126) semantics."""
    df = _ohlcv(drift=0.002)
    out = volmom_signal(df, _EMPTY(df), cfg={})
    assert (out.iloc[:LOOKBACK + VOL_WINDOW] == 0).all()


def test_outputs_are_finite_binary():
    """DA low objection: pin {0.0, 1.0} domain + no NaN for both modules."""
    df = _ohlcv(drift=0.002)
    for sig in (dip_signal, volmom_signal):
        out = sig(df, _EMPTY(df), cfg={})
        assert out.notna().all()
        assert out.isin([0.0, 1.0]).all()


def test_volmom_prefix_stability():
    df = _ohlcv(drift=0.002)
    full = volmom_signal(df, _EMPTY(df), cfg={})
    prefix = volmom_signal(df.iloc[:300], _EMPTY(df.iloc[:300]), cfg={})
    pd.testing.assert_series_equal(full.iloc[:300], prefix)


def test_volmom_future_mutation_immune():
    df = _ohlcv(drift=0.002)
    a = volmom_signal(df, _EMPTY(df), cfg={})
    m = df.copy(); m.iloc[400, m.columns.get_loc("close")] *= 2
    b = volmom_signal(m, _EMPTY(m), cfg={})
    pd.testing.assert_series_equal(a.iloc[:400], b.iloc[:400])
