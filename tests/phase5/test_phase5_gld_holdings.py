"""Causal-window discipline + signal correctness for phase5_gld_holdings (B0015a).

Single-condition primary (gld_holdings_5d_chg_z252 > +1.0) — corrects the
multi-condition AND under-firing failure mode that killed B0015b.

Per docs/superpowers/specs/2026-05-26-gld-holdings-primary.md.
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
    """Returns (ohlcv, features_dummy, cache_path) for monkeypatch + signal().

    Scenario:
      - 800 D1 bars (~2.2 years).
      - GLD holdings: baseline ~30,000,000 oz with small noise; sharp +5%
        spike around bar 400 (cache date 400). After 5d pct_change + z252,
        z > 1.0 around bar ~410 (post pct_change lag + .shift + z window).
    """
    rng = np.random.default_rng(101)
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

    # Build a synthetic GLD cache: 1000 daily rows from a few months before
    # the ohlcv start (so warm-up + z-window are covered).
    cache_start = pd.Timestamp(START_TS, tz="UTC") - pd.Timedelta(days=300)
    cache_idx = pd.date_range(cache_start, periods=1100, freq="D", tz="UTC")
    base_holdings = 30_000_000.0 + rng.normal(0, 50_000, len(cache_idx))
    # Spike ~ +5% around the bar that maps to ohlcv index ~400
    spike_anchor = pd.Timestamp(START_TS, tz="UTC") + pd.Timedelta(days=400)
    spike_pos = cache_idx.get_indexer([spike_anchor], method="nearest")[0]
    base_holdings[spike_pos:spike_pos + 20] *= 1.05  # ramp 5% accumulation over ~20 days
    cache = pd.DataFrame({"gld_oz_held": base_holdings}, index=cache_idx)
    cache_path = tmp_path / "gld_holdings.parquet"
    cache.to_parquet(cache_path)

    # features arg is an empty frame with the ohlcv index (primary doesn't
    # read from features at all; INPUT_COLUMNS=()).
    features = pd.DataFrame(index=ohlcv.index)
    return ohlcv, features, cache_path


# ---------------------------------------------------------------------------
# Module-level contract
# ---------------------------------------------------------------------------

def test_input_columns_is_empty_tuple():
    from pipeline.primaries_phase5 import phase5_gld_holdings
    assert hasattr(phase5_gld_holdings, "INPUT_COLUMNS")
    assert phase5_gld_holdings.INPUT_COLUMNS == (), (
        f"phase5_gld_holdings does not read from features namespace; "
        f"expected INPUT_COLUMNS=(), got {phase5_gld_holdings.INPUT_COLUMNS}"
    )


def test_module_docstring_inputs_read_block_present():
    """Per the spec, primaries with INPUT_COLUMNS=() are EXEMPT from the
    'Inputs read:' docstring requirement (docstring linter checks only
    non-empty INPUT_COLUMNS). But the docstring should still mention the
    alt-data source for clarity."""
    from pipeline.primaries_phase5 import phase5_gld_holdings
    assert phase5_gld_holdings.__doc__ is not None
    assert "GLD" in phase5_gld_holdings.__doc__ or "gld_holdings" in phase5_gld_holdings.__doc__


# ---------------------------------------------------------------------------
# Signal shape + behavior
# ---------------------------------------------------------------------------

def test_signal_returns_only_long_or_zero(synth_inputs, monkeypatch):
    ohlcv, features, cache_path = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_gld_holdings.DEFAULT_CACHE_PATH",
        cache_path,
    )
    from pipeline.primaries_phase5 import phase5_gld_holdings
    sig = phase5_gld_holdings.signal(ohlcv, features, cfg={})
    unique_vals = set(sig.dropna().unique())
    assert unique_vals.issubset({0, 1}), f"sig must be in {{0, 1}}; got {unique_vals}"


def test_signal_fires_at_least_once_in_constructed_scenario(synth_inputs, monkeypatch):
    """The +5% holdings spike over 20 days should produce z > +1.0 → fires."""
    ohlcv, features, cache_path = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_gld_holdings.DEFAULT_CACHE_PATH",
        cache_path,
    )
    from pipeline.primaries_phase5 import phase5_gld_holdings
    sig = phase5_gld_holdings.signal(ohlcv, features, cfg={})
    fire_count = int((sig == 1).sum())
    assert fire_count >= 1, f"Expected >=1 fire in constructed scenario, got {fire_count}"


def test_signal_index_matches_ohlcv(synth_inputs, monkeypatch):
    ohlcv, features, cache_path = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_gld_holdings.DEFAULT_CACHE_PATH",
        cache_path,
    )
    from pipeline.primaries_phase5 import phase5_gld_holdings
    sig = phase5_gld_holdings.signal(ohlcv, features, cfg={})
    assert sig.index.equals(ohlcv.index)


def test_signal_returns_int_dtype(synth_inputs, monkeypatch):
    ohlcv, features, cache_path = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_gld_holdings.DEFAULT_CACHE_PATH",
        cache_path,
    )
    from pipeline.primaries_phase5 import phase5_gld_holdings
    sig = phase5_gld_holdings.signal(ohlcv, features, cfg={})
    assert np.issubdtype(sig.dtype, np.integer)


def test_strict_causal_no_self_reference(synth_inputs, monkeypatch):
    """Perturbing holdings at cache date t must NOT change sig at any bar < t+5
    (5-day pct-change lookback would propagate the change forward by 5 days, but
    NEVER backward — that would be lookahead)."""
    ohlcv, features, cache_path = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_gld_holdings.DEFAULT_CACHE_PATH",
        cache_path,
    )
    from pipeline.primaries_phase5 import phase5_gld_holdings
    sig_orig = phase5_gld_holdings.signal(ohlcv, features, cfg={})

    # Perturb the LAST cache row only
    cache_df = pd.read_parquet(cache_path)
    cache_df.iloc[-1, cache_df.columns.get_loc("gld_oz_held")] *= 1.50
    perturbed_path = cache_path.parent / "gld_holdings_perturbed.parquet"
    cache_df.to_parquet(perturbed_path)
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_gld_holdings.DEFAULT_CACHE_PATH",
        perturbed_path,
    )
    sig_pert = phase5_gld_holdings.signal(ohlcv, features, cfg={})

    # The perturbation is at the last cache row, which corresponds to a future
    # date relative to the ohlcv window. So sig in the ohlcv window must be
    # UNCHANGED. If anything changes, that's a lookahead violation.
    assert sig_orig.equals(sig_pert), (
        "Sig changed when a FUTURE cache row was perturbed -- lookahead violation"
    )


def test_signal_raises_when_cache_missing(synth_inputs, monkeypatch):
    """If the cache parquet is absent, signal raises GldHoldingsCacheMissing."""
    ohlcv, features, _ = synth_inputs
    monkeypatch.setattr(
        "pipeline.primaries_phase5.phase5_gld_holdings.DEFAULT_CACHE_PATH",
        Path("does_not_exist_anywhere.parquet"),
    )
    from pipeline.primaries_phase5 import phase5_gld_holdings
    from pipeline.alt_data.gld_holdings import GldHoldingsCacheMissing
    with pytest.raises(GldHoldingsCacheMissing):
        phase5_gld_holdings.signal(ohlcv, features, cfg={})
