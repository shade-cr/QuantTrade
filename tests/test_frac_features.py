"""Tests for fractionally differentiated features (AFML Ch5, Snippets 5.3/5.4/5.5).

The load-bearing property under test is CAUSALITY: ``frac_diff_ffd`` at time ``t``
must use only bars at indices ``<= t``. We use the fixed-width-window (FFD) form,
not the expanding-window ``fracDiff``, precisely so that the weight vector is
identical at every ``t`` and no future bar can leak into a past output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.frac_features import ffd_weights, frac_diff_ffd, min_ffd_d


# --------------------------------------------------------------------------- #
# ffd_weights                                                                 #
# --------------------------------------------------------------------------- #
def test_ffd_weights_newest_is_one():
    """The book reverses the weights (w[::-1]); newest lag (last index) == 1.0."""
    w = ffd_weights(0.4)
    assert w.ndim == 1
    assert w[-1] == pytest.approx(1.0)


def test_ffd_weights_d0_is_identity():
    """d=0 => no differencing => single weight [1.0]."""
    w = ffd_weights(0.0)
    assert w.shape == (1,)
    assert w[0] == pytest.approx(1.0)


def test_ffd_weights_d1_is_first_difference():
    """d=1 => ordinary first difference => weights ~= [-1, 1] (oldest..newest)."""
    w = ffd_weights(1.0)
    assert w.shape == (2,)
    np.testing.assert_allclose(w, np.array([-1.0, 1.0]), atol=1e-12)


def test_ffd_weights_monotone_shrink_in_magnitude():
    """Beyond the first couple of terms the weights shrink monotonically in |.|."""
    w = ffd_weights(0.4)
    mags = np.abs(w[::-1])  # newest -> oldest
    # newest weight is 1.0 (largest); each subsequent (older) one is <= prior
    diffs = np.diff(mags)
    assert np.all(diffs <= 1e-12)


def test_ffd_weights_honors_thres():
    """A larger threshold truncates the window to fewer weights."""
    w_tight = ffd_weights(0.4, thres=1e-5)
    w_loose = ffd_weights(0.4, thres=1e-2)
    assert len(w_loose) < len(w_tight)
    # smallest-magnitude retained weight must clear the threshold
    assert np.min(np.abs(w_tight)) >= 1e-5


# --------------------------------------------------------------------------- #
# frac_diff_ffd                                                               #
# --------------------------------------------------------------------------- #
def _logprice_random_walk(n=600, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="D", tz="UTC")
    logp = np.cumsum(rng.normal(0, 0.01, size=n))
    return pd.Series(logp, index=idx, name="logp")


def test_frac_diff_preserves_index_and_length():
    s = _logprice_random_walk()
    out = frac_diff_ffd(s, d=0.4)
    assert len(out) == len(s)
    assert out.index.equals(s.index)


def test_frac_diff_head_is_nan_for_warmup():
    """First ``width`` outputs are NaN (insufficient history).

    Use thres=1e-3 so the FFD window (~54 lags at d=0.4) fits the 600-bar
    series; at the 1e-5 default the power-law weight tail needs ~1457 lags.
    """
    s = _logprice_random_walk()
    d = 0.4
    thres = 1e-3
    w = ffd_weights(d, thres=thres)
    width = len(w) - 1
    out = frac_diff_ffd(s, d=d, thres=thres)
    assert out.iloc[:width].isna().all()
    assert out.iloc[width:].notna().all()


def test_frac_diff_causality_future_does_not_leak():
    """THE no-lookahead test: mutating a FUTURE bar must not change any
    frac_diff_ffd value strictly BEFORE that bar."""
    s = _logprice_random_walk(seed=1)
    d = 0.4
    thres = 1e-3  # ~54-lag window so t=400 is in the mature (non-NaN) region
    out_base = frac_diff_ffd(s, d=d, thres=thres)

    t = 400  # mutate this future bar
    s_mut = s.copy()
    s_mut.iloc[t] += 5.0  # large shock
    out_mut = frac_diff_ffd(s_mut, d=d, thres=thres)

    # the chosen bar must be mature (a real number) for the test to bite
    assert np.isfinite(out_base.iloc[t])

    before = out_base.iloc[:t]
    before_mut = out_mut.iloc[:t]
    # all values strictly before the mutated bar are bit-for-bit unchanged
    pd.testing.assert_series_equal(before, before_mut)
    # and the mutated bar's own output DID change (sanity: filter is live)
    assert out_base.iloc[t] != out_mut.iloc[t]


def test_frac_diff_d1_equals_first_difference():
    """d=1 reproduces the ordinary first difference of the level."""
    s = _logprice_random_walk(seed=2)
    out = frac_diff_ffd(s, d=1.0)
    expected = s.diff()
    # width for d=1 is 1, so only the first value is NaN in both
    np.testing.assert_allclose(
        out.dropna().to_numpy(), expected.iloc[1:].to_numpy(), atol=1e-10
    )


def test_frac_diff_memory_and_stationarity():
    """On a near-random-walk log-price:
      * d~=0.4 keeps HIGH correlation (>0.9) to the level (memory preserved),
      * d=1 (returns) has near-zero corr to the level (memory destroyed),
      * d~=0.4 is MORE stationary than the raw level (lower ADF p-value).
    """
    adfuller = pytest.importorskip("statsmodels.tsa.stattools").adfuller
    s = _logprice_random_walk(n=2500, seed=3)

    # thres=1e-3 => ~54-lag window at d=0.4. Looser thres is exactly what AFML
    # Snippet 5.4 uses; the 1e-5 default would need ~1457 lags. (A pure random
    # walk has weaker level-correlation than the trending ES futures in the
    # book's ~0.99 figure, but d=0.4 still clears 0.9 here.)
    thres = 1e-3
    ffd04 = frac_diff_ffd(s, d=0.4, thres=thres)
    ffd10 = frac_diff_ffd(s, d=1.0, thres=thres)

    mat04 = ffd04.dropna()
    corr04 = np.corrcoef(s.loc[mat04.index].to_numpy(), mat04.to_numpy())[0, 1]
    assert abs(corr04) > 0.9, f"d=0.4 corr-to-level too low: {corr04}"

    mat10 = ffd10.dropna()
    corr10 = np.corrcoef(s.loc[mat10.index].to_numpy(), mat10.to_numpy())[0, 1]
    assert abs(corr10) < 0.3, f"d=1 corr-to-level too high: {corr10}"

    p_level = adfuller(s.to_numpy(), maxlag=1, regression="c", autolag=None)[1]
    p_ffd04 = adfuller(mat04.to_numpy(), maxlag=1, regression="c", autolag=None)[1]
    assert p_ffd04 < p_level, f"d=0.4 not more stationary: {p_ffd04} vs {p_level}"


def test_frac_diff_handles_head_nans():
    """Leading NaNs in the input are tolerated (ffilled/dropped per book)."""
    s = _logprice_random_walk(seed=4)
    s.iloc[:3] = np.nan
    out = frac_diff_ffd(s, d=0.4, thres=1e-3)
    assert len(out) == len(s)
    assert out.index.equals(s.index)
    # at least the mature tail is finite
    assert out.iloc[-1] == out.iloc[-1]  # not NaN


# --------------------------------------------------------------------------- #
# min_ffd_d                                                                   #
# --------------------------------------------------------------------------- #
def test_min_ffd_d_random_walk_returns_fractional():
    """A random-walk log-price should be made stationary by some d in (0,1]."""
    pytest.importorskip("statsmodels.tsa.stattools")
    s = _logprice_random_walk(n=900, seed=5)
    # thres=1e-3 keeps the FFD window short enough across the grid for a
    # meaningful ADF sweep on a 900-bar series.
    d = min_ffd_d(s, thres=1e-3)
    assert 0.0 < d <= 1.0
    assert d < 1.0, "expected some fractional differencing < 1 to suffice"
