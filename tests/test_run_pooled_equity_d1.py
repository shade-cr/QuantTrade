"""Phase-A member builder mirrors run_backtest._run_one_primary alignment
invariants; runner CLI plumbing (count-events mode)."""
import numpy as np
import pandas as pd
import pytest

from scripts.run_pooled_equity_d1 import build_member_inputs


@pytest.fixture
def cfg():
    return {
        "asset_class": "equity",
        "triple_barrier": {"horizon": 40, "tp_atr_mult": 3.0,
                           "sl_atr_mult": 1.0, "atr_period": 14},
        "primary": {
            "candidates": ["ema_cross", "momentum_zscore"],
            "ema_cross": {"fast": 20, "slow": 50, "dead_zone_atr": 0.25},
            "momentum_zscore": {"lookback": 20, "threshold": 0.3},
        },
        "metrics": {"cost_per_trade_bps": 10, "bars_per_year": 252},
    }


def _features_for(ohlcv: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=ohlcv.index)
    tr = (ohlcv["high"] - ohlcv["low"]).rolling(14).mean()
    feats["_atr_14"] = tr
    r = np.log(ohlcv["close"]).diff()
    feats["z_r20"] = (r - r.rolling(20).mean()) / r.rolling(20).std()
    feats["f_mom"] = r.rolling(5).sum()
    return feats.dropna()


def test_member_alignment_invariants(synth_ohlcv, cfg):
    features = _features_for(synth_ohlcv)
    ohlcv = synth_ohlcv.loc[features.index]
    m = build_member_inputs("TEST", "ema_cross", ohlcv, features, cfg)
    assert m is not None, "synthetic series should produce ema_cross events"
    n = len(m["X"])
    assert n == len(m["y"]) == len(m["w"]) == len(m["fwd_ret"]) \
        == len(m["event_time"]) == len(m["label_end_time"])
    assert not m["X"].isnull().any().any()
    for col in ("primary_side", "primary_strength", "bars_since_signal"):
        assert col in m["X"].columns
    assert "_atr_14" not in m["X"].columns
    assert not m["fwd_ret"].isnull().any()
    assert (m["label_end_time"] >= m["event_time"]).all()
    assert m["asset_class"] == "equity"
    assert m["bars_per_year"] == 252
    assert m["cost_bps"] == 10
    assert set(np.unique(m["y"])) <= {0, 1}
    assert (m["w"] > 0).all() and (m["w"] <= 1).all()


def test_member_returns_none_when_no_signals(synth_ohlcv, cfg):
    features = _features_for(synth_ohlcv)
    ohlcv = synth_ohlcv.loc[features.index]
    cfg["primary"]["momentum_zscore"]["threshold"] = 99.0  # unreachable
    assert build_member_inputs("TEST", "momentum_zscore", ohlcv, features, cfg) is None
