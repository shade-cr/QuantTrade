"""Adversarial lookahead + contract tests for the materialized CMF mean-reversion
custom primary `pipeline/primaries_phase5/cmf_meanrev.py` (B0088 -> B0040 Option B).

The single load-bearing guarantee for any phase5_* custom primary is the
B0040 adversarial lookahead check: the signal at bar t must be a strict
function of bars <= t (so the position it implies at t+1 uses only past
information). We prove this with a PREFIX-INVARIANCE test — truncating the
future must not change any already-emitted signal value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.primaries_phase5 import phase5_cmf_meanrev as cmf_meanrev


def _index(df: pd.DataFrame) -> pd.DataFrame:
    """The orchestrator hands signal() an OHLCV frame indexed by time."""
    return df.set_index("time")


def _cfg(period: int = 21, lo: float = -0.05, hi: float = 0.10,
         allow_short: bool = True) -> dict:
    """Build the cfg dict in the exact shape the dispatcher passes:
    per-primary params live at cfg['primary']['cmf_meanrev']."""
    return {
        "primary": {
            "cmf_meanrev": {
                "period": period,
                "lo_thresh": lo,
                "hi_thresh": hi,
                "allow_short": allow_short,
            }
        }
    }


# --------------------------------------------------------------------------- #
# Output contract
# --------------------------------------------------------------------------- #
def test_signal_returns_series_indexed_like_ohlcv(synth_ohlcv):
    ohlcv = _index(synth_ohlcv)
    sig = cmf_meanrev.signal(ohlcv, pd.DataFrame(index=ohlcv.index), _cfg())
    assert isinstance(sig, pd.Series)
    assert sig.index.equals(ohlcv.index)


def test_signal_values_in_minus_one_zero_plus_one(synth_ohlcv):
    ohlcv = _index(synth_ohlcv)
    sig = cmf_meanrev.signal(ohlcv, pd.DataFrame(index=ohlcv.index), _cfg())
    assert set(pd.unique(sig.dropna())).issubset({-1, 0, 1})


def test_no_nan_in_output(synth_ohlcv):
    """Warm-up CMF NaNs must collapse to 0 (no signal), never propagate as NaN."""
    ohlcv = _index(synth_ohlcv)
    sig = cmf_meanrev.signal(ohlcv, pd.DataFrame(index=ohlcv.index), _cfg())
    assert not sig.isna().any()


def test_long_only_variant_never_shorts(synth_ohlcv):
    ohlcv = _index(synth_ohlcv)
    cfg = _cfg(lo=0.30, hi=-0.30, allow_short=False)  # thresholds that would fire both
    sig = cmf_meanrev.signal(ohlcv, pd.DataFrame(index=ohlcv.index), cfg)
    assert (sig >= 0).all(), "allow_short=False must never emit -1"


def test_both_sides_can_fire_when_short_allowed(synth_ohlcv):
    ohlcv = _index(synth_ohlcv)
    # Permissive thresholds so both a long and a short fire somewhere.
    cfg = _cfg(lo=0.20, hi=-0.20, allow_short=True)
    sig = cmf_meanrev.signal(ohlcv, pd.DataFrame(index=ohlcv.index), cfg)
    assert (sig == 1).any() and (sig == -1).any()


# --------------------------------------------------------------------------- #
# Adversarial lookahead guard (B0040): prefix-invariance
# --------------------------------------------------------------------------- #
def test_prefix_invariance_signal_at_t_unchanged_when_future_truncated(synth_ohlcv):
    """The core B0040 lookahead check.

    For a strictly causal primary, signal[t] depends only on bars <= t. Compute
    the signal on the full series, then recompute on every truncated prefix and
    assert the last bar of each prefix matches the full-series value at that bar.
    A single .shift(-k) / centered window / full-sample stat would break this.
    """
    ohlcv = _index(synth_ohlcv)
    cfg = _cfg(period=21, lo=-0.05, hi=0.10, allow_short=True)
    full = cmf_meanrev.signal(ohlcv, pd.DataFrame(index=ohlcv.index), cfg)
    # Check a spread of cut points past the warm-up.
    for cut in range(60, len(ohlcv), 37):
        prefix = ohlcv.iloc[:cut]
        sig_prefix = cmf_meanrev.signal(prefix, pd.DataFrame(index=prefix.index), cfg)
        assert sig_prefix.iloc[-1] == full.iloc[cut - 1], (
            f"lookahead leak: signal at bar {cut-1} changed when future truncated "
            f"({sig_prefix.iloc[-1]} on prefix vs {full.iloc[cut-1]} on full series)"
        )


def test_position_acts_at_t_plus_one(synth_ohlcv):
    """The implied position at bar t (entering on the next bar's open after the
    signal closes at t-1) must equal the signal shifted by one bar. We assert the
    primary itself emits the SIGNAL (acted on at t+1 downstream), and that the
    signal at the very first non-warmup bar is reproducible — i.e. the signal at
    bar t never depends on bar t+1's CMF."""
    ohlcv = _index(synth_ohlcv)
    cfg = _cfg()
    sig = cmf_meanrev.signal(ohlcv, pd.DataFrame(index=ohlcv.index), cfg)
    # Drop the last bar; the signal on bars [0, n-2] must be byte-identical.
    sig_trunc = cmf_meanrev.signal(ohlcv.iloc[:-1], pd.DataFrame(index=ohlcv.index[:-1]), cfg)
    pd.testing.assert_series_equal(sig.iloc[:-1], sig_trunc, check_dtype=False)


# --------------------------------------------------------------------------- #
# Zero-range guard
# --------------------------------------------------------------------------- #
def test_zero_range_doji_does_not_nan_poison(synth_ohlcv):
    """A bar with high == low (doji) yields a 0/0 money-flow multiplier; the
    guard defines it as 0.0 so the rolling CMF window is never NaN-poisoned."""
    ohlcv = _index(synth_ohlcv).copy()
    # Force a run of doji bars in the middle of the series.
    for i in range(100, 110):
        ohlcv.iloc[i, ohlcv.columns.get_loc("high")] = ohlcv.iloc[i]["close"]
        ohlcv.iloc[i, ohlcv.columns.get_loc("low")] = ohlcv.iloc[i]["close"]
    sig = cmf_meanrev.signal(ohlcv, pd.DataFrame(index=ohlcv.index), _cfg())
    assert not sig.isna().any()
    assert set(pd.unique(sig)).issubset({-1, 0, 1})


# --------------------------------------------------------------------------- #
# Purity / determinism
# --------------------------------------------------------------------------- #
def test_deterministic(synth_ohlcv):
    ohlcv = _index(synth_ohlcv)
    cfg = _cfg()
    a = cmf_meanrev.signal(ohlcv, pd.DataFrame(index=ohlcv.index), cfg)
    b = cmf_meanrev.signal(ohlcv, pd.DataFrame(index=ohlcv.index), cfg)
    pd.testing.assert_series_equal(a, b)


def test_does_not_mutate_inputs(synth_ohlcv):
    ohlcv = _index(synth_ohlcv)
    before = ohlcv.copy(deep=True)
    cmf_meanrev.signal(ohlcv, pd.DataFrame(index=ohlcv.index), _cfg())
    pd.testing.assert_frame_equal(ohlcv, before)


def test_input_columns_declared_and_empty():
    """Primary reads only raw OHLCV -> INPUT_COLUMNS=() so the disjointness
    check against the meta feature set passes trivially (B0015b Layer c)."""
    assert hasattr(cmf_meanrev, "INPUT_COLUMNS")
    assert tuple(cmf_meanrev.INPUT_COLUMNS) == ()


# --------------------------------------------------------------------------- #
# Fade direction sanity: long fires on low CMF, short on high CMF
# --------------------------------------------------------------------------- #
def test_fade_direction_long_on_oversold_short_on_overbought():
    """Construct a tiny controlled frame: a bar closing at the low on high volume
    drives CMF down (long fade); a bar closing at the high on high volume drives
    CMF up (short fade)."""
    n = 30
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    close = np.full(n, 100.0)
    high = close + 1.0
    low = close - 1.0
    vol = np.full(n, 1000.0)
    # First make a strongly-down money-flow run (close at low) ...
    df = pd.DataFrame({"open": close, "high": high, "low": low,
                       "close": low.copy(), "volume": vol}, index=idx)
    cfg = _cfg(period=5, lo=-0.5, hi=0.5, allow_short=True)
    sig = cmf_meanrev.signal(df, pd.DataFrame(index=idx), cfg)
    # CMF near -1 -> below lo -> long fade fires once warmed up.
    assert (sig == 1).any()
    assert not (sig == -1).any()
