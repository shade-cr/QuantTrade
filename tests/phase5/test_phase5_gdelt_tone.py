"""Causal-window + signal correctness for phase5_gdelt_tone (B0015c).

Single-condition (tone_z252 < -1.0) primary; mirrors phase5_gld_holdings
structure since both are alt-data single-condition primaries.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


N_BARS = 800
START_TS = "2020-01-01"


@pytest.fixture
def synth_inputs(tmp_path):
    rng = np.random.default_rng(202)
    idx = pd.date_range(START_TS, periods=N_BARS, freq="D", tz="UTC")
    log_ret = rng.normal(0.0001, 0.012, size=N_BARS)
    close = 1500.0 * np.exp(np.cumsum(log_ret))
    high = close * (1.0 + np.abs(rng.normal(0, 0.004, N_BARS)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.004, N_BARS)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(1000, 5000, N_BARS).astype(float)
    ohlcv = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )

    # Build a synthetic tone cache: 1000 daily rows. Baseline mean=-0.5,
    # std=0.5; spike to -3.0 around index ~400 to produce z < -1.0.
    cache_start = pd.Timestamp(START_TS, tz="UTC") - pd.Timedelta(days=300)
    cache_idx = pd.date_range(cache_start, periods=1100, freq="D", tz="UTC")
    tone = rng.normal(-0.5, 0.5, len(cache_idx))
    spike_anchor = pd.Timestamp(START_TS, tz="UTC") + pd.Timedelta(days=400)
    spike_pos = cache_idx.get_indexer([spike_anchor], method="nearest")[0]
    tone[spike_pos:spike_pos + 15] = rng.normal(-3.0, 0.2, 15)
    cache = pd.DataFrame({"tone": tone}, index=cache_idx)
    cache_path = tmp_path / "gdelt_tone_test.parquet"
    cache.to_parquet(cache_path)

    features = pd.DataFrame(index=ohlcv.index)
    return ohlcv, features, cache_path


def test_input_columns_empty_tuple():
    from pipeline.primaries_phase5 import phase5_gdelt_tone
    assert phase5_gdelt_tone.INPUT_COLUMNS == ()


def test_docstring_mentions_gdelt():
    from pipeline.primaries_phase5 import phase5_gdelt_tone
    assert "GDELT" in phase5_gdelt_tone.__doc__


def test_signal_returns_only_long_or_zero(synth_inputs, monkeypatch):
    ohlcv, features, cache = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_gdelt_tone.DEFAULT_CACHE_PATH", cache,
    )
    from pipeline.primaries_phase5 import phase5_gdelt_tone
    sig = phase5_gdelt_tone.signal(ohlcv, features, cfg={})
    assert set(sig.dropna().unique()).issubset({0, 1})


def test_signal_fires_in_spike_region(synth_inputs, monkeypatch):
    ohlcv, features, cache = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_gdelt_tone.DEFAULT_CACHE_PATH", cache,
    )
    from pipeline.primaries_phase5 import phase5_gdelt_tone
    sig = phase5_gdelt_tone.signal(ohlcv, features, cfg={})
    assert int((sig == 1).sum()) >= 1


def test_signal_index_matches_ohlcv(synth_inputs, monkeypatch):
    ohlcv, features, cache = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_gdelt_tone.DEFAULT_CACHE_PATH", cache,
    )
    from pipeline.primaries_phase5 import phase5_gdelt_tone
    sig = phase5_gdelt_tone.signal(ohlcv, features, cfg={})
    assert sig.index.equals(ohlcv.index)


def test_signal_int_dtype(synth_inputs, monkeypatch):
    ohlcv, features, cache = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_gdelt_tone.DEFAULT_CACHE_PATH", cache,
    )
    from pipeline.primaries_phase5 import phase5_gdelt_tone
    sig = phase5_gdelt_tone.signal(ohlcv, features, cfg={})
    assert np.issubdtype(sig.dtype, np.integer)


def test_strict_causal_no_self_reference(synth_inputs, monkeypatch):
    """Perturbing a FUTURE cache row must not change sig in the ohlcv window."""
    ohlcv, features, cache = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_gdelt_tone.DEFAULT_CACHE_PATH", cache,
    )
    from pipeline.primaries_phase5 import phase5_gdelt_tone
    sig_orig = phase5_gdelt_tone.signal(ohlcv, features, cfg={})

    cache_df = pd.read_parquet(cache)
    cache_df.iloc[-1, cache_df.columns.get_loc("tone")] = -10.0  # absurd future spike
    perturbed = cache.parent / "gdelt_tone_perturbed.parquet"
    cache_df.to_parquet(perturbed)
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_gdelt_tone.DEFAULT_CACHE_PATH", perturbed,
    )
    sig_pert = phase5_gdelt_tone.signal(ohlcv, features, cfg={})
    assert sig_orig.equals(sig_pert), "Future-row perturbation changed past sig — lookahead violation"


def test_signal_raises_when_cache_missing(synth_inputs, monkeypatch):
    ohlcv, features, _ = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_gdelt_tone.DEFAULT_CACHE_PATH",
        Path("missing_gdelt_cache.parquet"),
    )
    from pipeline.primaries_phase5 import phase5_gdelt_tone
    from pipeline.alt_data.gdelt_tone import GdeltToneCacheMissing
    with pytest.raises(GdeltToneCacheMissing):
        phase5_gdelt_tone.signal(ohlcv, features, cfg={})
