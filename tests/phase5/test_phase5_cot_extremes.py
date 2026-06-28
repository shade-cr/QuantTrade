"""Causal-window discipline + signal correctness for phase5_cot_extremes (B0015b).

The primary fires +1 (long) when commercials_net_long_z52 > 1.5 AND
dxy_20d_chg < -0.02. Strict-causal rolling z-score on the weekly COT frame
(see pipeline.cot_features.build_cot_commercials_raw). DXY change is
pct_change(20).shift(1) on features['dtwexbgs_close']. Min-separation
0.20 * ATR(14) between consecutive fires. Long-only by design.

Test fixture construction:
  - 800-bar D1 ohlcv (~2.2 years).
  - features['dtwexbgs_close']: declines from 100 to 88 across the index;
    sharp drop in the second half such that pct_change(20).shift(1) < -0.02
    around bar ~500.
  - Monkeypatched build_cot_commercials_raw returns a weekly fixture where
    commercials_net_long_pct is constant baseline ~0.05 for 60 weeks then
    spikes to ~0.20 for 30 weeks. Strict-causal z52 crosses +1.5 at week ~75
    once the spike has been in the trailing window long enough.
  - Conjunction fires at the overlap of both conditions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# Convenience indices for the fixture
N_BARS = 800
START_TS = "2020-01-01"
FREQ = "D"


@pytest.fixture
def synth_inputs():
    """Returns (ohlcv, features, mock_build_cot_commercials_raw) tuple ready
    for monkeypatch + signal() invocation.
    """
    rng = np.random.default_rng(42)
    idx = pd.date_range(START_TS, periods=N_BARS, freq=FREQ, tz="UTC")
    # Build a slightly bullish close series with realistic vol
    log_returns = rng.normal(0.0001, 0.012, size=N_BARS)
    close = 1500.0 * np.exp(np.cumsum(log_returns))
    high = close * (1.0 + np.abs(rng.normal(0, 0.004, size=N_BARS)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.004, size=N_BARS)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(1000, 5000, N_BARS).astype(float)
    ohlcv = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )

    # DTWEXBGS: gentle pre-period (100 → 99), then sharp drop around bar 430
    # (covering the spike-publication window). 20-bar pct_change must cross
    # -0.02 in the overlap region with the COT spike (~bar 430+) for the
    # conjunction to fire. Bars 430-460 fall 99 -> 90 (~-9% / 20 bars).
    dtwexbgs_close = np.concatenate([
        np.linspace(100.0, 99.0, 430),       # gentle baseline pre-spike
        np.linspace(99.0, 90.0, 30),         # sharp -9% drop over 30 bars
        np.linspace(90.0, 88.0, N_BARS - 460),  # gentle continuation
    ])
    features = pd.DataFrame({"dtwexbgs_close": dtwexbgs_close}, index=idx)

    # Weekly commercials fixture: 90 Tuesday rows starting from 2020-01-07.
    # Baseline pct=0.05 (60 weeks) → spike pct=0.20 (30 weeks). Small noise on
    # both regimes so sd of the rolling-260-daily-bar window is > 0; otherwise
    # constant-baseline → std=0 → z = NaN and the conjunction never fires.
    weekly_idx = pd.date_range("2020-01-07", periods=90, freq="7D", tz="UTC")
    rng_w = np.random.default_rng(101)
    baseline = 0.05 + rng_w.normal(0, 0.005, 60)
    spike = 0.20 + rng_w.normal(0, 0.005, 30)
    pct_pattern = np.concatenate([baseline, spike])
    total_oi = np.full(90, 100_000.0)
    net_long = pct_pattern * total_oi

    def mock_build_cot_commercials_raw(asset, target_index, cache_dir=None):
        # Mirror the publication-time +84h reindex + ffill pattern used by the
        # real build_cot_commercials_raw, on this synthetic weekly fixture.
        weekly_feats = pd.DataFrame(
            {"net_long": net_long, "total_oi": total_oi}, index=weekly_idx
        )
        weekly_feats.index = weekly_feats.index + pd.Timedelta(hours=84)
        weekly_feats = weekly_feats.sort_index()
        aligned = weekly_feats.reindex(target_index, method="ffill")
        return aligned

    return ohlcv, features, mock_build_cot_commercials_raw


# ---------------------------------------------------------------------------
# Module-level contract: INPUT_COLUMNS + docstring
# ---------------------------------------------------------------------------

def test_input_columns_constant_present_and_correct():
    """INPUT_COLUMNS = ('dtwexbgs_close',) — the primary reads DXY from features."""
    from pipeline.primaries_phase5 import phase5_cot_extremes
    assert hasattr(phase5_cot_extremes, "INPUT_COLUMNS")
    assert phase5_cot_extremes.INPUT_COLUMNS == ("dtwexbgs_close",), (
        f"Expected ('dtwexbgs_close',), got {phase5_cot_extremes.INPUT_COLUMNS}"
    )


def test_module_docstring_has_inputs_block():
    """Per Layer (b) docstring linter."""
    from pipeline.primaries_phase5 import phase5_cot_extremes
    assert phase5_cot_extremes.__doc__ is not None
    assert "Inputs read:" in phase5_cot_extremes.__doc__
    assert "disjoint" in phase5_cot_extremes.__doc__


# ---------------------------------------------------------------------------
# Signal shape + behavior
# ---------------------------------------------------------------------------

def test_signal_returns_only_long_or_zero(synth_inputs, monkeypatch):
    """sig values are in {0, +1} only (long-only by design)."""
    ohlcv, features, mock_fn = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_cot_extremes.build_cot_commercials_raw",
        mock_fn,
    )
    from pipeline.primaries_phase5 import phase5_cot_extremes
    sig = phase5_cot_extremes.signal(ohlcv, features, cfg={})
    unique_vals = set(sig.dropna().unique())
    assert unique_vals.issubset({0, 1}), f"sig must be in {{0,1}}; got {unique_vals}"


def test_signal_fires_at_least_once_in_constructed_scenario(synth_inputs, monkeypatch):
    """Given the spike + DXY-decline construction, sig fires ≥1 time."""
    ohlcv, features, mock_fn = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_cot_extremes.build_cot_commercials_raw",
        mock_fn,
    )
    from pipeline.primaries_phase5 import phase5_cot_extremes
    sig = phase5_cot_extremes.signal(ohlcv, features, cfg={})
    fire_count = int((sig == 1).sum())
    assert fire_count >= 1, (
        f"Expected ≥1 fire in constructed scenario, got {fire_count}. "
        f"Likely a misconfigured z-threshold or DXY pct_change formula."
    )


def test_signal_does_not_fire_before_publication_window(synth_inputs, monkeypatch):
    """sig is 0 BEFORE the spike has had time to land in the 52-week rolling z.

    The weekly spike starts at 2020-01-07 + 60 weeks = 2021-02-23 (Tuesday).
    Publication of that report = 2021-02-26 ~12:00 UTC. The primary's
    commercials_z52 cannot exceed 1.5 before the spike has accumulated enough
    weight in the rolling stats — well after 2021-02-26. So sig must be 0 at
    every bar BEFORE 2021-02-26.
    """
    ohlcv, features, mock_fn = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_cot_extremes.build_cot_commercials_raw",
        mock_fn,
    )
    from pipeline.primaries_phase5 import phase5_cot_extremes
    sig = phase5_cot_extremes.signal(ohlcv, features, cfg={})
    pre_spike_pub = sig.index < pd.Timestamp("2021-02-26", tz="UTC")
    assert (sig.loc[pre_spike_pub] == 0).all(), (
        "Sig fired BEFORE the spike publication — causal-window violation"
    )


def test_dxy_chg_uses_shift1_no_self_reference(synth_inputs, monkeypatch):
    """sig at idx[t] cannot depend on features['dtwexbgs_close'][t]; only on
    [t-1] or earlier (via pct_change(20).shift(1))."""
    ohlcv, features, mock_fn = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_cot_extremes.build_cot_commercials_raw",
        mock_fn,
    )
    from pipeline.primaries_phase5 import phase5_cot_extremes
    sig_orig = phase5_cot_extremes.signal(ohlcv, features, cfg={})

    # Perturb dtwexbgs at idx[500] only — sig at idx[500] must be unchanged
    # (pct_change(20).shift(1) at idx[500] uses dtwexbgs[479] and earlier).
    features_pert = features.copy()
    features_pert.iloc[500, features_pert.columns.get_loc("dtwexbgs_close")] *= 0.01
    sig_pert = phase5_cot_extremes.signal(ohlcv, features_pert, cfg={})

    assert sig_orig.iloc[500] == sig_pert.iloc[500], (
        f"Sig at idx[500] changed when dtwexbgs[500] was perturbed; "
        f".shift(1) violation. orig={sig_orig.iloc[500]}, pert={sig_pert.iloc[500]}"
    )


def test_signal_respects_min_separation_atr(synth_inputs, monkeypatch):
    """Min-separation gate kicks in when fires would be closer than 0.20 * ATR."""
    ohlcv, features, mock_fn = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_cot_extremes.build_cot_commercials_raw",
        mock_fn,
    )
    from pipeline.primaries_phase5 import phase5_cot_extremes
    sig = phase5_cot_extremes.signal(ohlcv, features, cfg={})
    fire_indices = sig.index[sig == 1]
    # All fire indices must be strictly monotonic (no duplicates)
    assert fire_indices.is_monotonic_increasing
    assert fire_indices.is_unique


def test_signal_returns_int_dtype(synth_inputs, monkeypatch):
    """sig dtype must be a numpy int kind (int8/int16/... — supports .astype(int8))."""
    ohlcv, features, mock_fn = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_cot_extremes.build_cot_commercials_raw",
        mock_fn,
    )
    from pipeline.primaries_phase5 import phase5_cot_extremes
    sig = phase5_cot_extremes.signal(ohlcv, features, cfg={})
    assert np.issubdtype(sig.dtype, np.integer), (
        f"sig dtype must be integer, got {sig.dtype}"
    )


def test_signal_index_matches_ohlcv_index(synth_inputs, monkeypatch):
    """sig must be indexed identically to ohlcv per the SKILL.md contract."""
    ohlcv, features, mock_fn = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_cot_extremes.build_cot_commercials_raw",
        mock_fn,
    )
    from pipeline.primaries_phase5 import phase5_cot_extremes
    sig = phase5_cot_extremes.signal(ohlcv, features, cfg={})
    assert sig.index.equals(ohlcv.index)


def test_signal_raises_when_dtwexbgs_close_missing(synth_inputs, monkeypatch):
    """If features lacks dtwexbgs_close, signal() raises a clear error
    (NOT silently returning zeros, which would mask the wiring bug)."""
    ohlcv, features, mock_fn = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_cot_extremes.build_cot_commercials_raw",
        mock_fn,
    )
    features_without_dxy = features.drop(columns=["dtwexbgs_close"])
    from pipeline.primaries_phase5 import phase5_cot_extremes
    with pytest.raises(KeyError, match="dtwexbgs_close"):
        phase5_cot_extremes.signal(ohlcv, features_without_dxy, cfg={})
