"""B0016 — overnight/intraday decomposition + Kyle/Amihud lambdas in tier2.

Evidence base: Lou-Polk-Skouras (JFE 2019) — anomaly returns split overnight
vs intraday, large-cap momentum is overnight, EWMA components predictive at
the 5-10d horizon; LdP AFML ch.19 — Kyle/Amihud lambdas top his MDA ranking.
All computable from bars we already have. Data audit (2026-07-04) verified
opens are true session opens across the 52-name panel.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.features import build_technical_features

NEW_COLS = [
    "r_overnight", "r_intraday",
    "on_ewma_21", "on_ewma_60", "in_ewma_21", "in_ewma_60",
    "tug_21",
    "amihud_20", "kyle_t_20",
]


@pytest.fixture
def tech(synth_ohlcv):
    return build_technical_features(synth_ohlcv)


def test_new_columns_present(tech):
    for col in NEW_COLS:
        assert col in tech.columns, col


def test_overnight_intraday_identity(synth_ohlcv, tech):
    """r_overnight + r_intraday must equal r_1 exactly (log decomposition)."""
    r1 = np.log(synth_ohlcv["close"] / synth_ohlcv["close"].shift(1))
    recon = tech["r_overnight"] + tech["r_intraday"]
    pd.testing.assert_series_equal(recon.dropna(), r1.dropna(), check_names=False)


def test_overnight_values_exact(synth_ohlcv, tech):
    o, c = synth_ohlcv["open"], synth_ohlcv["close"]
    expected_on = np.log(o / c.shift(1))
    expected_in = np.log(c / o)
    pd.testing.assert_series_equal(tech["r_overnight"].dropna(), expected_on.dropna(), check_names=False)
    pd.testing.assert_series_equal(tech["r_intraday"].dropna(), expected_in.dropna(), check_names=False)
    assert np.isnan(tech["r_overnight"].iloc[0])


def test_tug_is_ewma_spread(tech):
    expected = tech["on_ewma_21"] - tech["in_ewma_21"]
    pd.testing.assert_series_equal(tech["tug_21"].dropna(), expected.dropna(), check_names=False)


def test_all_new_columns_are_causal(synth_ohlcv):
    """PIT: values at t must be identical whether or not future bars exist."""
    full = build_technical_features(synth_ohlcv)
    cut = len(synth_ohlcv) - 60
    trunc = build_technical_features(synth_ohlcv.iloc[:cut])
    for col in NEW_COLS:
        a = full[col].iloc[:cut]
        b = trunc[col]
        pd.testing.assert_series_equal(a, b, check_names=False, atol=1e-12, rtol=1e-9)


def test_lambdas_finite_after_warmup(tech):
    assert tech["amihud_20"].iloc[30:].notna().all()
    assert np.isfinite(tech["kyle_t_20"].iloc[30:]).mean() > 0.95
    # Amihud is an illiquidity magnitude: strictly nonnegative.
    assert (tech["amihud_20"].dropna() >= 0).all()
