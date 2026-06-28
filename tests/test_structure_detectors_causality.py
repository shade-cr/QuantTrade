"""Causality (no-look-ahead) tests for the three structure detectors.

THE #1 FAILURE MODE of structure-conditioning spikes is look-ahead via
whole-series fitting:
  - a Gaussian HMM Viterbi-decoded over the FULL series uses future bars to
    infer the state of a past bar (the smoother is two-sided),
  - a whole-series DWT smears future coefficients into the present at every
    boundary,
  - a change-point detector run on the full series re-segments the past when
    the future arrives.
All three manufacture a gorgeous fake edge.

The decisive, detector-agnostic guarantee is PREFIX-INVARIANCE:

    detector(series[:t+1])[t]  ==  detector(series)[t]   for every t in the
    online (post-warmup) region, to floating tolerance.

If appending FUTURE bars (t+1 .. N) ever changes the label at a PAST bar t,
the detector peeked. This is stronger and simpler than auditing the internals:
it treats the detector as a black box and proves the output at t is a function
of data <= t only.

These tests are written FIRST (TDD red) against `pipeline.structure_detectors`,
which does not exist yet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.structure_detectors import (  # noqa: E402
    hmm_regime_labels,
    changepoint_labels,
    wavelet_trend_labels,
)


# --------------------------------------------------------------------------- #
# Fixtures: a deterministic synthetic price series with two regimes
# (a trending block followed by a choppy block) so the detectors have real
# structure to find, and a fixed seed so prefix-invariance is testable.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def synth_close() -> pd.Series:
    rng = np.random.default_rng(7)
    n = 1200
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    # Block 1: strong drift (trend). Block 2: zero-drift mean-reverting (chop).
    r = np.empty(n)
    r[:600] = 0.0008 + 0.004 * rng.standard_normal(600)            # trend up
    chop = np.zeros(600)
    for i in range(1, 600):
        chop[i] = -0.35 * chop[i - 1] + 0.004 * rng.standard_normal()  # AR(1) mean-rev
    r[600:] = chop
    close = 100.0 * np.exp(np.cumsum(r))
    return pd.Series(close, index=idx, name="close")


# Online region: well past the warmup so every detector has emitted labels.
# We probe a handful of bars (full prefix recompute is O(t) per detector — keep
# the probe set small but spread across the series, including the regime break).
PROBE_FRACS = [0.55, 0.62, 0.75, 0.90, 0.99]


def _probe_indices(n: int, warmup: int) -> list[int]:
    idxs = sorted({int(f * n) for f in PROBE_FRACS})
    return [t for t in idxs if t > warmup + 5]


# --------------------------------------------------------------------------- #
# 1. HMM regime — prefix-invariance (filtered/online inference, frozen params)
# --------------------------------------------------------------------------- #
def test_hmm_label_at_t_unchanged_by_future_bars(synth_close):
    """hmm_regime_labels(close[:t+1])[t] must equal hmm_regime_labels(close)[t].

    The HMM must (a) FREEZE parameters from a TRAIN prefix and (b) infer the
    state at t with a FORWARD/FILTERED pass over data <= t only — never Viterbi-
    decode the whole series. Appending future bars must not relabel bar t.
    """
    warmup = 700  # > train window used internally
    full = hmm_regime_labels(synth_close, train_bars=500, n_states=2, seed=0)
    for t in _probe_indices(len(synth_close), warmup):
        prefix = hmm_regime_labels(synth_close.iloc[: t + 1], train_bars=500, n_states=2, seed=0)
        assert prefix.index[-1] == synth_close.index[t]
        a = full.iloc[t]
        b = prefix.iloc[-1]
        # Labels are integer regime ids (or NaN in warmup); compare nan-aware.
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b), f"warmup mismatch at t={t}"
        else:
            assert int(a) == int(b), f"HMM relabelled past bar t={t}: full={a} prefix={b}"


def test_hmm_warmup_region_is_nan(synth_close):
    """Bars before the train window is filled must be NaN (no label emitted)."""
    labels = hmm_regime_labels(synth_close, train_bars=500, n_states=2, seed=0)
    assert labels.iloc[:500].isna().all(), "HMM emitted labels inside the train window"
    assert labels.iloc[500:].notna().any(), "HMM emitted no labels after the train window"


# --------------------------------------------------------------------------- #
# 2. Change-point — prefix-invariance (online/windowed CUSUM-of-mean)
# --------------------------------------------------------------------------- #
def test_changepoint_label_at_t_unchanged_by_future_bars(synth_close):
    """changepoint_labels(close[:t+1])[t] must equal changepoint_labels(close)[t].

    Breaks must be detected ONLINE: a detector that re-segments the whole series
    when new data arrives (e.g. PELT on the full series) would relabel the past.
    """
    warmup = 100
    full = changepoint_labels(synth_close, window=100, k_atr=3.0)
    for t in _probe_indices(len(synth_close), warmup):
        prefix = changepoint_labels(synth_close.iloc[: t + 1], window=100, k_atr=3.0)
        a = full.iloc[t]
        b = prefix.iloc[-1]
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b), f"warmup mismatch at t={t}"
        else:
            assert int(a) == int(b), f"changepoint relabelled past bar t={t}: full={a} prefix={b}"


# --------------------------------------------------------------------------- #
# 3. Wavelet / spectral trend — prefix-invariance (causal MA-band reconstruction)
# --------------------------------------------------------------------------- #
def test_wavelet_trend_at_t_unchanged_by_future_bars(synth_close):
    """wavelet_trend_labels(close[:t+1])[t] must equal wavelet_trend_labels(close)[t].

    A whole-series DWT/IDWT smears future coefficients into past bars at every
    decomposition boundary. The causal version must reconstruct the denoised
    trend at t from a trailing window of data <= t only.
    """
    warmup = 200
    full = wavelet_trend_labels(synth_close, window=128, level=3)
    for t in _probe_indices(len(synth_close), warmup):
        prefix = wavelet_trend_labels(synth_close.iloc[: t + 1], window=128, level=3)
        a = full.iloc[t]
        b = prefix.iloc[-1]
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b), f"warmup mismatch at t={t}"
        else:
            assert int(a) == int(b), f"wavelet relabelled past bar t={t}: full={a} prefix={b}"


# --------------------------------------------------------------------------- #
# Output contract: all three return a Series in {-1, 0, +1} or {0,1,..} ∪ {NaN},
# aligned to the input index, no forward fill across the warmup boundary.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (hmm_regime_labels, dict(train_bars=500, n_states=2, seed=0)),
        (changepoint_labels, dict(window=100, k_atr=3.0)),
        (wavelet_trend_labels, dict(window=128, level=3)),
    ],
)
def test_output_index_aligned_and_valued(synth_close, fn, kwargs):
    out = fn(synth_close, **kwargs)
    assert isinstance(out, pd.Series)
    assert out.index.equals(synth_close.index), f"{fn.__name__} index not aligned to input"
    vals = out.dropna().unique()
    assert set(np.unique(vals.astype(int))).issubset({-1, 0, 1, 2}), (
        f"{fn.__name__} emitted out-of-domain labels: {vals}"
    )
