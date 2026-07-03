"""B0006 pack: point-in-time construction, rank bounds, panel scalars."""
import numpy as np
import pandas as pd
import pytest

from pipeline.cross_section import build_cross_sectional_features

N, TICKERS = 400, ["AAA", "BBB", "CCC", "DDD", "EEE"]


@pytest.fixture
def panel():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2010-01-04", periods=N, freq="B", tz="UTC")
    out = {}
    for i, t in enumerate(TICKERS):
        r = rng.normal(0.0003 * (i + 1), 0.01 + 0.002 * i, N)
        close = 50.0 * np.exp(np.cumsum(r))
        out[t] = pd.DataFrame({
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": rng.uniform(1e5, 1e6, N),
        }, index=idx)
    return out


EXPECTED = ["cs_mom_12_1_rank", "cs_52wk_high_rank", "cs_st_reversal_rank",
            "cs_idio_vol_rank", "cs_turnover_rank", "cs_basket_beta_rank",
            "cs_dispersion", "cs_breadth_200"]


def test_schema_bounds_and_panel_scalars(panel):
    feats = build_cross_sectional_features(panel, TICKERS)
    assert set(feats) == set(TICKERS)
    for t in TICKERS:
        f = feats[t]
        assert list(f.columns) == EXPECTED
        assert f.index.equals(panel[t].index)
        valid = f.dropna()
        rank_cols = [c for c in EXPECTED if c.endswith("_rank")]
        assert ((valid[rank_cols] >= 0) & (valid[rank_cols] <= 1)).all().all()
    # panel scalars identical across tickers at each t
    for col in ("cs_dispersion", "cs_breadth_200"):
        stacked = pd.concat([feats[t][col] for t in TICKERS], axis=1)
        stacked = stacked.dropna()
        assert (stacked.nunique(axis=1) == 1).all(), f"{col} must be panel-wide"


def test_point_in_time_no_lookahead(panel):
    """Feature values at t must be identical when future rows are truncated."""
    full = build_cross_sectional_features(panel, TICKERS)
    cut = N - 60
    truncated_panel = {t: df.iloc[:cut] for t, df in panel.items()}
    trunc = build_cross_sectional_features(truncated_panel, TICKERS)
    for t in TICKERS:
        a = full[t].iloc[:cut]
        b = trunc[t]
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12)


def test_ranks_are_cross_sectional_not_temporal(panel):
    """At any date, the 5 tickers' momentum ranks must be a permutation of
    the 5 evenly-spaced percentiles — proving the rank is across names."""
    feats = build_cross_sectional_features(panel, TICKERS)
    date = feats[TICKERS[0]]["cs_mom_12_1_rank"].dropna().index[-1]
    vals = sorted(feats[t].loc[date, "cs_mom_12_1_rank"] for t in TICKERS)
    expected = [(i + 1) / len(TICKERS) for i in range(len(TICKERS))]
    assert np.allclose(vals, expected)
