"""Causal-rolling-window tests for the BEARQ-002 phase5_custom primary.

Per the Day-2 skeptic review (Concern D), the BEARQ-002 pseudocode uses
rolling-window percentile bands; these MUST be strictly causal. The bands
at time t must depend only on data through bar t-1 (no self-reference).

These tests verify the no-future-leakage property by comparing rolling-band
values computed on the full series against per-bar recomputation that uses
only the strict prefix [0, t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.primaries_phase5.phase5_xau_bearq_conjunction import (
    _strict_causal_quantile,
    _strict_causal_zscore,
    _compute_features,
    signal,
    QUANTILE_WINDOW,
    ROC_QUANTILE_UPPER,
)


def _make_synth_ohlcv(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Synthetic D1 OHLCV with deterministic noise — no real-asset shape."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=n, freq="D", tz="UTC")
    log_ret = rng.normal(0.0, 0.01, size=n)
    log_ret[300:400] -= 0.005  # inject a slow decline window
    close = 100.0 * np.exp(np.cumsum(log_ret))
    high = close * (1 + rng.uniform(0.0, 0.005, size=n))
    low = close * (1 - rng.uniform(0.0, 0.005, size=n))
    open_ = close + rng.normal(0.0, 0.5, size=n) * 0.005 * close
    volume = rng.lognormal(mean=10.0, sigma=0.3, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_strict_causal_quantile_no_self_reference():
    """Quantile at t MUST equal quantile of values [t-window, t-1]."""
    n, window, q = 400, 100, 0.30
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=n))
    rolled = _strict_causal_quantile(s, window, q)
    # Pick a bar well past warm-up
    for t in (window + 5, window + 50, window + 200, n - 1):
        if t >= len(s):
            continue
        expected = s.iloc[t - window : t].quantile(q)
        actual = rolled.iloc[t]
        if np.isnan(expected):
            assert np.isnan(actual), f"t={t}: actual={actual} should be NaN"
        else:
            assert np.isclose(actual, expected, atol=1e-9), (
                f"t={t}: causal quantile mismatch (actual={actual}, expected={expected})"
            )


def test_strict_causal_quantile_does_not_use_bar_t():
    """Mutating s[t] must NOT change the quantile value at t (leakage check).

    Uses a LOWER outlier on a LOWER-tail percentile so a single mutation
    dramatically changes the 30th-percentile (which is what BEARQ-002 actually
    uses). If the function were t-inclusive, the test at t would fail.
    """
    n, window, q = 300, 50, 0.30
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(size=n))
    s2 = s.copy()
    t = 200
    s2.iloc[t] = -99999.0  # extreme LOWER value distorts the 30th percentile
    rolled1 = _strict_causal_quantile(s, window, q)
    rolled2 = _strict_causal_quantile(s2, window, q)
    # Quantile at t must be identical: the t-th bar is NOT in the window
    assert np.isclose(rolled1.iloc[t], rolled2.iloc[t], atol=1e-9), (
        "Mutating s[t] changed the strict-causal quantile at t - leakage!"
    )
    # Quantile at t+1 MUST differ: s[t] is now in the prior window [t-49..t]
    if t + 1 < n:
        assert not np.isclose(rolled1.iloc[t + 1], rolled2.iloc[t + 1], atol=1e-3), (
            f"Mutating s[t]=-99999 did NOT change quantile at t+1 "
            f"(r1={rolled1.iloc[t+1]}, r2={rolled2.iloc[t+1]}) - window may not include t"
        )


def test_strict_causal_zscore_no_self_reference():
    """z-score at t must use mean/std of [t-window, t-1] only."""
    n, window = 200, 30
    rng = np.random.default_rng(2)
    s = pd.Series(rng.normal(loc=5.0, scale=2.0, size=n))
    z = _strict_causal_zscore(s, window)
    for t in (window + 1, window + 20, n - 1):
        prior = s.iloc[t - window : t]
        expected_mu = prior.mean()
        expected_sd = prior.std(ddof=0)
        expected = (s.iloc[t] - expected_mu) / expected_sd
        actual = z.iloc[t]
        assert np.isclose(actual, expected, atol=1e-9), (
            f"t={t}: z-score mismatch (actual={actual}, expected={expected})"
        )


def test_signal_only_emits_short_or_zero():
    """signal() must return values in {-1, 0} (short-only, no long entries)."""
    ohlcv = _make_synth_ohlcv()
    sig = signal(ohlcv, features=pd.DataFrame(index=ohlcv.index), cfg={})
    unique_vals = set(pd.Series(sig).dropna().unique().tolist())
    assert unique_vals.issubset({-1, 0}), (
        f"signal() returned values outside {{-1,0}}: got {unique_vals}"
    )


def test_signal_warmup_is_nan_or_zero():
    """First QUANTILE_WINDOW + max(ROC, RV, MA_200, 14) bars must be 0 (insufficient lookback)."""
    ohlcv = _make_synth_ohlcv()
    sig = signal(ohlcv, features=pd.DataFrame(index=ohlcv.index), cfg={})
    # The longest lookback is QUANTILE_WINDOW(252) shifted by 1 + ROC(63) = ~315
    warmup = QUANTILE_WINDOW + 63
    head = sig.iloc[:warmup]
    assert (head == 0).all() or head.isna().all(), (
        f"Signal fired during warmup window [0:{warmup}] — leakage suspected"
    )


def test_signal_is_deterministic():
    """Same input must produce identical output (no randomness, no I/O)."""
    ohlcv = _make_synth_ohlcv(seed=7)
    sig1 = signal(ohlcv, features=pd.DataFrame(index=ohlcv.index), cfg={})
    sig2 = signal(ohlcv, features=pd.DataFrame(index=ohlcv.index), cfg={})
    assert sig1.equals(sig2), "signal() is non-deterministic"


def test_signal_does_not_depend_on_future_bars():
    """Removing all bars AFTER t must not change signal[t] for any t."""
    ohlcv = _make_synth_ohlcv()
    full_sig = signal(ohlcv, features=pd.DataFrame(index=ohlcv.index), cfg={})
    # Check at a few interior points well past warm-up
    warmup = QUANTILE_WINDOW + 63
    for t in (warmup + 10, warmup + 100, len(ohlcv) - 50, len(ohlcv) - 5):
        ohlcv_trunc = ohlcv.iloc[: t + 1]
        sig_trunc = signal(ohlcv_trunc, features=pd.DataFrame(index=ohlcv_trunc.index), cfg={})
        assert sig_trunc.iloc[t] == full_sig.iloc[t], (
            f"t={t}: signal differs when future bars truncated (truncated={sig_trunc.iloc[t]}, full={full_sig.iloc[t]}). Future-leak!"
        )
