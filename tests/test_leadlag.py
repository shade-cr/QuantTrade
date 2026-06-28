"""Tests for pipeline/leadlag.py — BTC->ETH/SOL lead-lag sleeve building
blocks (B0167), pre-registration §1.2 verbatim.

What is enforced here:

* the Hayashi-Yoshida lead-lag estimator recovers a KNOWN injected lag on
  asynchronous synthetic ticks (right lag AND right direction), and the
  no-lag null (contemporaneous co-movement) does NOT produce a >=30s lead
  verdict;
* the event detector truth table (|r| > k*sigma strict, sign = direction,
  NaN fails closed);
* the follower-not-moved signal truth table (signed same-direction
  displacement strictly < 1 sigma; opposite moves pass the gate; NaN and
  missing-clock-minutes fail closed; codomain {-1, 0, +1});
* no lookahead: truncating the future never changes past signal values;
* the 1-min clock helpers are right-labeled/right-closed (causal) and
  deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.leadlag import (  # noqa: E402
    FOLLOWER_NOT_MOVED_SIGMAS,
    hayashi_yoshida_leadlag,
    ll_events,
    ll_signal,
    minute_bars,
    minute_returns,
)


# ---------------------------------------------------------------------------
# Synthetic asynchronous ticks with a KNOWN injected lag
# ---------------------------------------------------------------------------


def _lagged_tick_pair(
    *,
    lag_s: int,
    total_s: int = 7200,
    leader_step_s: int = 2,
    follower_step_s: int = 3,
    seed: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """Latent 1-second random walk; leader samples it live every
    ``leader_step_s`` seconds, follower samples it ``lag_s`` seconds LATE
    every ``follower_step_s`` seconds (asynchronous clocks). lag_s=0 is the
    contemporaneous null."""
    rng = np.random.default_rng(seed)
    w = np.cumsum(rng.normal(0.0, 1e-3, size=total_s + 1))
    t0 = pd.Timestamp("2026-06-01", tz="UTC")
    lead_t = np.arange(0, total_s + 1, leader_step_s)
    fol_t = np.arange(lag_s, total_s + 1, follower_step_s)
    leader = pd.Series(
        100.0 * np.exp(w[lead_t]),
        index=t0 + pd.to_timedelta(lead_t, unit="s"),
    )
    follower = pd.Series(
        50.0 * np.exp(w[fol_t - lag_s] + rng.normal(0.0, 1e-6, len(fol_t))),
        index=t0 + pd.to_timedelta(fol_t, unit="s"),
    )
    return leader, follower


class TestHayashiYoshida:
    def test_finds_injected_lag_and_direction(self):
        leader, follower = _lagged_tick_pair(lag_s=60, seed=1)
        table, verdict = hayashi_yoshida_leadlag(leader, follower)
        assert verdict.leader_leads is True
        assert verdict.best_lag_s == 60.0
        assert verdict.best_corr > 0.5
        assert verdict.llr > 1.0
        # The table is the per-lag diagnostic the Jun-15 report records.
        peak = table.loc[table["corr"].idxmax()]
        assert peak["lag_s"] == 60.0

    def test_reversed_pair_shows_negative_lag_and_fails(self):
        """Swap the roles: the 'leader' arg now actually FOLLOWS — the
        verdict must say so (best lag negative, leader_leads False)."""
        leader, follower = _lagged_tick_pair(lag_s=60, seed=2)
        table, verdict = hayashi_yoshida_leadlag(follower, leader)
        assert verdict.leader_leads is False
        assert verdict.best_lag_s == -60.0

    def test_no_lag_null_fails_the_30s_lead_bar(self):
        """Contemporaneous co-movement (lag 0): the correlation peak sits at
        lag 0, which is BELOW the registered >=30s lead floor — the verdict
        must fail (this is the §1.2 dead-on-arrival branch)."""
        leader, follower = _lagged_tick_pair(lag_s=0, seed=3)
        table, verdict = hayashi_yoshida_leadlag(leader, follower)
        assert verdict.leader_leads is False
        assert verdict.best_lag_s == 0.0
        assert verdict.best_corr > 0.5  # co-movement IS there, just not led

    def test_table_covers_symmetric_grid_plus_zero(self):
        leader, follower = _lagged_tick_pair(lag_s=60, seed=4)
        lags = [30.0, 60.0, 90.0]
        table, _ = hayashi_yoshida_leadlag(leader, follower, lags=lags)
        assert sorted(table["lag_s"].tolist()) == [
            -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0]

    def test_rejects_tz_naive_ticks(self):
        idx = pd.date_range("2026-06-01", periods=10, freq="1s")  # naive
        s = pd.Series(np.linspace(100, 101, 10), index=idx)
        with pytest.raises(ValueError, match="tz-aware"):
            hayashi_yoshida_leadlag(s, s)

    def test_rejects_too_short_series(self):
        idx = pd.date_range("2026-06-01", periods=1, freq="1s", tz="UTC")
        s = pd.Series([100.0], index=idx)
        with pytest.raises(ValueError, match="at least 2"):
            hayashi_yoshida_leadlag(s, s)

    def test_deterministic(self):
        leader, follower = _lagged_tick_pair(lag_s=60, seed=5)
        t1, v1 = hayashi_yoshida_leadlag(leader, follower)
        t2, v2 = hayashi_yoshida_leadlag(leader, follower)
        pd.testing.assert_frame_equal(t1, t2)
        assert v1 == v2


# ---------------------------------------------------------------------------
# Event detector truth table (§1.2: BTC 1-min |r| > k * sigma_train)
# ---------------------------------------------------------------------------


def _minute_index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-06-01 00:01", periods=n, freq="1min", tz="UTC")


class TestLLEvents:
    def test_truth_table(self):
        idx = _minute_index(5)
        r = pd.Series([0.5e-3, 3.5e-3, -4.0e-3, np.nan, 0.0], index=idx)
        ev = ll_events(r, 1e-3, 3.0)
        # Only |r| STRICTLY above 3 sigma fires; sign = BTC move direction.
        assert list(ev.index) == [idx[1], idx[2]]
        assert ev.loc[idx[1]] == 1.0
        assert ev.loc[idx[2]] == -1.0

    def test_boundary_is_strict(self):
        idx = _minute_index(2)
        r = pd.Series([3.0e-3, -3.0e-3], index=idx)  # exactly 3 sigma
        ev = ll_events(r, 1e-3, 3.0)
        assert len(ev) == 0

    def test_nan_fails_closed(self):
        idx = _minute_index(3)
        r = pd.Series([np.nan, np.nan, np.nan], index=idx)
        assert len(ll_events(r, 1e-3, 3.0)) == 0

    def test_rejects_nonpositive_sigma(self):
        idx = _minute_index(2)
        r = pd.Series([1e-3, 2e-3], index=idx)
        with pytest.raises(ValueError, match="sigma_train"):
            ll_events(r, 0.0, 3.0)
        with pytest.raises(ValueError, match="threshold"):
            ll_events(r, 1e-3, 0.0)


# ---------------------------------------------------------------------------
# Follower signal truth table (§1.2: enter iff concurrent same-direction
# move < 1 sigma)
# ---------------------------------------------------------------------------


class TestLLSignal:
    def test_not_moved_threshold_is_frozen_at_one_sigma(self):
        """The 1-sigma 'not yet moved' bound is REGISTERED — a module
        constant, not a function parameter."""
        assert FOLLOWER_NOT_MOVED_SIGMAS == 1.0

    def test_truth_table(self):
        idx = _minute_index(8)
        sigma_f = 1e-3
        events = pd.Series(
            [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
            index=idx[[0, 1, 2, 3, 4, 5, 6]],
        )
        r_f = pd.Series(
            [
                0.5e-3,   # +0.5 sigma, same dir, < 1 sigma  -> enter +1
                1.5e-3,   # +1.5 sigma, same dir, >= 1 sigma -> 0 (moved)
                -2.0e-3,  # opposite move: same-dir displacement < 0 -> +1
                1.0e-3,   # exactly +1.0 sigma -> 0 (strict <)
                -0.5e-3,  # short event, same dir 0.5 sigma  -> enter -1
                -1.5e-3,  # short event, already moved       -> 0
                np.nan,   # NaN concurrent move -> fail closed 0
                0.0,      # NO event this minute -> 0
            ],
            index=idx,
        )
        sig = ll_signal(r_f, events, sigma_f)
        assert list(sig.index) == list(idx)
        assert sig.tolist() == [1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0]

    def test_codomain(self):
        idx = _minute_index(50)
        rng = np.random.default_rng(0)
        r_f = pd.Series(rng.normal(0, 1e-3, 50), index=idx)
        events = ll_events(pd.Series(rng.normal(0, 1e-3, 50), index=idx),
                           1e-3, 1.0)
        sig = ll_signal(r_f, events, 1e-3)
        assert set(np.unique(sig)) <= {-1.0, 0.0, 1.0}

    def test_event_minute_missing_from_follower_clock_fails_closed(self):
        idx = _minute_index(3)
        off_clock = idx[0] - pd.Timedelta(seconds=30)
        events = pd.Series([1.0], index=pd.DatetimeIndex([off_clock]))
        r_f = pd.Series([0.0, 0.0, 0.0], index=idx)
        sig = ll_signal(r_f, events, 1e-3)
        assert (sig == 0.0).all()

    def test_rejects_nonpositive_sigma(self):
        idx = _minute_index(1)
        with pytest.raises(ValueError, match="sigma_train_follower"):
            ll_signal(pd.Series([0.0], index=idx),
                      pd.Series(dtype=float), 0.0)

    def test_no_lookahead_truncation_invariance(self):
        """The signal at minute t must not change when the future is cut
        off — fire-on-bar-close causality (CLAUDE.md invariant)."""
        idx = _minute_index(200)
        rng = np.random.default_rng(7)
        r_l = pd.Series(rng.normal(0, 1e-3, 200), index=idx)
        r_f = pd.Series(rng.normal(0, 1e-3, 200), index=idx)
        events = ll_events(r_l, 1e-3, 1.5)
        full = ll_signal(r_f, events, 1e-3)
        cut = 120
        ev_trunc = ll_events(r_l.iloc[:cut], 1e-3, 1.5)
        trunc = ll_signal(r_f.iloc[:cut], ev_trunc, 1e-3)
        pd.testing.assert_series_equal(full.iloc[:cut], trunc)


# ---------------------------------------------------------------------------
# 1-min clock helpers (documented choice: TIME bars, right-labeled/closed)
# ---------------------------------------------------------------------------


class TestMinuteClock:
    def test_right_labeled_right_closed_is_causal(self):
        """A tick at 00:00:30 belongs to the bar STAMPED 00:01:00 (the bar
        contains only ticks <= its stamp — fire on bar close)."""
        t0 = pd.Timestamp("2026-06-01 00:00:30", tz="UTC")
        t1 = pd.Timestamp("2026-06-01 00:01:00", tz="UTC")  # boundary tick
        t2 = pd.Timestamp("2026-06-01 00:01:10", tz="UTC")
        prices = pd.Series([100.0, 101.0, 102.0],
                           index=pd.DatetimeIndex([t0, t1, t2]))
        bars = minute_bars(prices)
        b1 = pd.Timestamp("2026-06-01 00:01:00", tz="UTC")
        b2 = pd.Timestamp("2026-06-01 00:02:00", tz="UTC")
        assert list(bars.index) == [b1, b2]
        # Boundary tick (exactly 00:01:00) closes the 00:01:00 bar.
        assert bars.loc[b1, "open"] == 100.0
        assert bars.loc[b1, "close"] == 101.0
        assert bars.loc[b2, "close"] == 102.0

    def test_empty_minutes_carry_last_close_zero_return(self):
        t0 = pd.Timestamp("2026-06-01 00:00:30", tz="UTC")
        t1 = pd.Timestamp("2026-06-01 00:03:30", tz="UTC")  # 3-min gap
        prices = pd.Series([100.0, 110.0], index=pd.DatetimeIndex([t0, t1]))
        bars = minute_bars(prices)
        assert len(bars) == 4  # 00:01 .. 00:04, regular grid
        gap_bar = bars.iloc[1]  # 00:02, no ticks
        assert gap_bar["open"] == gap_bar["close"] == 100.0
        r = minute_returns(prices)
        assert r.iloc[1] == 0.0 and r.iloc[2] == 0.0
        assert r.iloc[3] == pytest.approx(np.log(110.0 / 100.0))

    def test_returns_are_log_close_to_close(self):
        idx = pd.date_range("2026-06-01", periods=180, freq="20s", tz="UTC")
        prices = pd.Series(np.linspace(100.0, 103.0, 180), index=idx)
        r = minute_returns(prices)
        bars = minute_bars(prices)
        expected = np.log(bars["close"]).diff()
        pd.testing.assert_series_equal(r, expected)

    def test_deterministic(self):
        idx = pd.date_range("2026-06-01", periods=500, freq="7s", tz="UTC")
        rng = np.random.default_rng(11)
        prices = pd.Series(100 + np.cumsum(rng.normal(0, 0.01, 500)),
                           index=idx)
        pd.testing.assert_frame_equal(minute_bars(prices),
                                      minute_bars(prices))
