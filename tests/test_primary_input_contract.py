"""Enforce the primary-input contract (CLAUDE.md invariant).

Each primary signal must be a deterministic, rule-based function returning
a pd.Series in {-1, 0, +1}. This test verifies, per primary:

  1. Determinism — calling on identical inputs returns identical outputs.
  2. Output codomain — values are in {-1.0, 0.0, +1.0} (NaNs allowed).
  3. No fitted-classifier dependency — the source code does not contain
     `.fit(` or an sklearn import (rules out ML training on the meta's data).

Also scans `pipeline/primaries_phase5/` for `phase5_*.py` modules and
asserts none of them contain `.fit(` in their source. Phase 5 custom
primaries are auto-discovered so this guardrail keeps working as new
ones are added.

When adding a new built-in primary, register it in REGISTERED_PRIMARIES.
"""
from __future__ import annotations

import inspect
import pathlib

import numpy as np
import pandas as pd
import pytest

from pipeline import labels


# -- Synthetic input fixtures -------------------------------------------------

def _synth_close_atr(n: int = 500) -> tuple[pd.Series, pd.Series]:
    """Synthetic close + ATR with a UTC DatetimeIndex (deterministic seed)."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    log_ret = rng.normal(0, 0.01, size=n)
    close = pd.Series(100.0 * np.exp(np.cumsum(log_ret)), index=idx, name="close")
    high = close * (1 + np.abs(rng.normal(0, 0.005, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, size=n)))
    prev = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return close, atr


def _synth_macro(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Synthetic VIX + DXY daily series spanning the same range as close."""
    rng = np.random.default_rng(7)
    idx_daily = pd.date_range(close.index[0], close.index[-1], freq="D", tz="UTC")
    vix = pd.Series(15 + rng.normal(0, 2, size=len(idx_daily)), index=idx_daily)
    dxy = pd.Series(
        100 + np.cumsum(rng.normal(0, 0.5, size=len(idx_daily))), index=idx_daily
    )
    return vix, dxy


# -- Registry -----------------------------------------------------------------

# Each entry maps a primary function to a builder that produces (args, kwargs)
# from synthetic close + atr. This keeps the test concise across the varied
# signatures the built-in primaries use today.

def _ema_call(close, atr):
    return (close, atr), {}


def _zscore_call(close, atr):
    return (close,), {}


def _bollinger_call(close, atr):
    return (close,), {}


def _cusum_call(close, atr):
    return (close, atr), {}


def _vix_regime_call(close, atr):
    vix, dxy = _synth_macro(close)
    return (close, vix, dxy, "XAUUSD"), {}


REGISTERED_PRIMARIES = [
    pytest.param(labels.ema_crossover_signal, _ema_call, id="ema_crossover_signal"),
    pytest.param(labels.momentum_zscore_signal, _zscore_call, id="momentum_zscore_signal"),
    pytest.param(labels.bollinger_meanrev_signal, _bollinger_call, id="bollinger_meanrev_signal"),
    pytest.param(labels.cusum_filter_signal, _cusum_call, id="cusum_filter_signal"),
    pytest.param(labels.vix_regime_riskflow_signal, _vix_regime_call, id="vix_regime_riskflow_signal"),
]


# -- Tests --------------------------------------------------------------------

@pytest.mark.parametrize("fn,builder", REGISTERED_PRIMARIES)
def test_primary_is_deterministic(fn, builder):
    """Two calls on identical inputs must return identical outputs.

    Catches accidental introduction of stochastic ML, np.random without a
    seed, or any other non-deterministic state.
    """
    close, atr = _synth_close_atr()
    args, kwargs = builder(close, atr)
    out1 = fn(*args, **kwargs)
    out2 = fn(*args, **kwargs)
    pd.testing.assert_series_equal(out1, out2)


@pytest.mark.parametrize("fn,builder", REGISTERED_PRIMARIES)
def test_primary_returns_signed_series(fn, builder):
    """Output is a pd.Series indexed like the input close, values ⊆ {-1, 0, +1}."""
    close, atr = _synth_close_atr()
    args, kwargs = builder(close, atr)
    out = fn(*args, **kwargs)
    assert isinstance(out, pd.Series), f"{fn.__name__} must return pd.Series"
    assert out.index.equals(close.index), (
        f"{fn.__name__} output index must match input close index"
    )
    uniq = set(out.dropna().unique())
    allowed = {-1.0, 0.0, 1.0}
    assert uniq.issubset(allowed), (
        f"{fn.__name__} returned values outside {{-1, 0, +1}}: {uniq - allowed}"
    )


@pytest.mark.parametrize("fn,builder", REGISTERED_PRIMARIES)
def test_primary_source_has_no_fit_call(fn, builder):
    """Static check: primary source must not contain `.fit(` or sklearn imports.

    This is the load-bearing invariant — a fitted classifier as a primary
    re-enters the Francesco "squeeze the orange twice" failure case.
    If a future primary legitimately needs `.fit()` (e.g., a one-shot
    pre-trained scaler), update this test AND revisit the CLAUDE.md
    invariant — don't silently relax it.
    """
    src = inspect.getsource(fn)
    assert ".fit(" not in src, (
        f"{fn.__name__} contains '.fit(' — primaries must be deterministic "
        f"rules, not fitted classifiers. See CLAUDE.md 'Pipeline invariants'."
    )
    assert "import sklearn" not in src and "from sklearn" not in src, (
        f"{fn.__name__} imports sklearn — primaries must remain rule-based. "
        f"See CLAUDE.md 'Pipeline invariants'."
    )


def test_phase5_custom_primaries_have_no_fit_call():
    """Source-level scan of pipeline/primaries_phase5/ for `.fit(` calls.

    Phase 5 customs are dispatched via the `phase5_*` prefix in
    scripts/run_xau_d1.py::_select_primary. They take a richer signature
    (ohlcv, features, cfg) but must remain rule-based — no classifier.fit().
    Auto-discovers new modules so this guardrail stays current.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    pkg_dir = repo_root / "pipeline" / "primaries_phase5"
    if not pkg_dir.exists():
        pytest.skip("pipeline/primaries_phase5/ does not exist yet")
    py_files = [
        p for p in pkg_dir.glob("phase5_*.py") if p.name != "__init__.py"
    ]
    if not py_files:
        pytest.skip("no phase5_* primary modules registered yet")
    offenders = [
        p.name for p in py_files if ".fit(" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"Phase 5 custom primaries contain '.fit(' calls: {offenders}. "
        f"See CLAUDE.md 'Pipeline invariants'."
    )
