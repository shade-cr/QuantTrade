"""Per-side pooled metrics: NaN-not-zero under 30 trades; sides split correctly."""
import numpy as np
import pandas as pd
import pytest

from scripts.report_long_short_split import long_short_split


def _fixtures(n_long=60, n_short=60):
    idx = pd.date_range("2015-01-01", periods=n_long + n_short, freq="B", tz="UTC")
    side = np.array([1] * n_long + [-1] * n_short)
    rng = np.random.default_rng(7)
    # Longs win on average, shorts lose (side * fwd_ret is the trade pnl).
    fwd = np.where(side == 1,
                   rng.normal(0.01, 0.02, n_long + n_short),
                   rng.normal(0.01, 0.02, n_long + n_short))
    events = pd.DataFrame({"side": side, "fwd_ret": fwd}, index=idx)
    oof = pd.DataFrame({"lr": np.full(len(idx), 0.60)}, index=idx)
    return oof, events


def test_sides_are_split_and_signed_correctly():
    oof, events = _fixtures()
    out = long_short_split(oof, events, model="lr", threshold=0.55, cost_bps=0.0)
    assert out["long"]["n_trades"] == 60
    assert out["short"]["n_trades"] == 60
    # Same positive fwd_ret both sides -> longs profit, shorts lose.
    assert out["long"]["mean_pnl_per_trade"] > 0
    assert out["short"]["mean_pnl_per_trade"] < 0


def test_nan_sharpe_below_30_trades():
    oof, events = _fixtures(n_long=60, n_short=10)
    out = long_short_split(oof, events, model="lr", threshold=0.55, cost_bps=0.0)
    assert out["short"]["n_trades"] == 10
    assert np.isnan(out["short"]["sharpe_net"]), "NaN, never 0, under 30 trades"
    assert not np.isnan(out["long"]["sharpe_net"])


def test_threshold_filters_trades():
    oof, events = _fixtures()
    oof.iloc[:30, 0] = 0.40  # below threshold -> dropped
    out = long_short_split(oof, events, model="lr", threshold=0.55, cost_bps=0.0)
    assert out["long"]["n_trades"] == 30


def test_tz_naive_oof_aligns_with_tz_aware_events():
    """Pooled trainer's oof index is tz-naive (np.concatenate strips tz); events
    stays tz-aware UTC. Same instants, same order -> must align, not SKIP."""
    oof, events = _fixtures()
    baseline = long_short_split(oof, events, model="lr", threshold=0.55, cost_bps=0.0)

    oof_naive = oof.copy()
    oof_naive.index = oof_naive.index.tz_convert("UTC").tz_localize(None)

    out = long_short_split(oof_naive, events, model="lr", threshold=0.55, cost_bps=0.0)
    assert out["long"]["n_trades"] == baseline["long"]["n_trades"]
    assert out["short"]["n_trades"] == baseline["short"]["n_trades"]


def test_misaligned_inputs_raise():
    """Verify positional-alignment guard rejects misaligned oof/events."""
    oof, events = _fixtures()

    # Scenario 1: events truncated by one row (different lengths)
    with pytest.raises(ValueError, match="row-aligned"):
        long_short_split(oof, events.iloc[:-1], model="lr", threshold=0.55, cost_bps=0.0)

    # Scenario 2: same length but index order shuffled (same length, different order)
    with pytest.raises(ValueError, match="row-aligned"):
        long_short_split(oof, events.iloc[::-1], model="lr", threshold=0.55, cost_bps=0.0)
