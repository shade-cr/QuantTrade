"""Equity D1 additions to the pooled machinery (B0003): embargo + HP spaces."""
import pandas as pd

from scripts.run_multi_h4 import (
    HP_SPACES,
    _BAR_DURATION_BY_CLASS,
    _pooled_embargo_td,
)
from pipeline.train import MODEL_FACTORIES


def test_equity_bar_duration_registered():
    # D1 equity: 252 bars/year → ~1.45 calendar days per bar.
    assert _BAR_DURATION_BY_CLASS["equity"] == pd.Timedelta(days=365.25 / 252)
    assert _BAR_DURATION_BY_CLASS["equity_index"] == pd.Timedelta(days=365.25 / 252)


def test_pooled_embargo_covers_equity_horizon():
    members = [{"asset_class": "equity"}, {"asset_class": "equity_index"}]
    cfg = {"triple_barrier": {"horizon": 40}}
    td = _pooled_embargo_td(members, cfg)
    # 40 bars × 365.25/252 days ≈ 58 calendar days — NOT the fx fallback (~9.4d).
    assert pd.Timedelta(days=57) < td < pd.Timedelta(days=59)


def test_lgbm_and_lr_hp_spaces_present_and_buildable():
    for name in ("lgbm", "lr"):
        assert name in HP_SPACES
        assert name in MODEL_FACTORIES
