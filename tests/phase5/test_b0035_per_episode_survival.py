"""B0035 — cross-episode survival sign test.

Schema: FalsificationCriterion gains per_episode_survival_fraction (+ min_trades,
margin). Audit: an episode "survives" if its threshold-0.50 net PnL > margin AND
it is "active" (>= per_episode_min_trades trades); the proposal passes the gate
iff survivors >= ceil(fraction * n_active) with n_active >= 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phase5 import run_proposal
from phase5.proposal import FalsificationCriterion, ProposalValidationError


# ---------------- schema validation ----------------

def test_fraction_none_is_valid_and_off():
    fc = FalsificationCriterion()
    fc.validate()  # no raise
    assert fc.per_episode_survival_fraction is None


def test_fraction_in_range_valid():
    FalsificationCriterion(per_episode_survival_fraction=0.667).validate()
    FalsificationCriterion(per_episode_survival_fraction=1.0).validate()


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_fraction_out_of_range_rejected(bad):
    with pytest.raises(ProposalValidationError, match="per_episode_survival_fraction"):
        FalsificationCriterion(per_episode_survival_fraction=bad).validate()


def test_min_trades_must_be_positive_int_when_gate_on():
    with pytest.raises(ProposalValidationError, match="per_episode_min_trades"):
        FalsificationCriterion(
            per_episode_survival_fraction=0.667, per_episode_min_trades=0
        ).validate()


# ---------------- evaluate_per_episode ----------------

def _write_fixtures(tmp_path, monkeypatch, pnl_by_time: pd.Series, regimes: pd.Series,
                    proposal_id="TEST-B0035", primary="ema_cross", asset="XAUUSD", model="xgb"):
    """Persist the per-event pnl artifact + a regimes parquet at monkeypatched dirs."""
    monkeypatch.setattr(run_proposal, "RESULTS_PHASE5_DIR", tmp_path / "results")
    monkeypatch.setattr(run_proposal, "REGIMES_DIR", tmp_path / "regimes")
    out_dir = tmp_path / "results" / proposal_id / primary
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({model: pnl_by_time}).to_parquet(out_dir / "strategy_pnl_threshold50.parquet")
    reg_dir = tmp_path / "regimes"
    reg_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"regime_id": regimes}).to_parquet(reg_dir / f"{asset}_d1_regimes.parquet")


def _criterion(fraction=0.6, min_trades=5, margin=0.0):
    return {
        "per_episode_survival_fraction": fraction,
        "per_episode_min_trades": min_trades,
        "per_episode_net_pnl_margin": margin,
    }


def test_passes_when_majority_episodes_net_positive(tmp_path, monkeypatch):
    # 3 BULL_QUIET episodes of 10 bars each, separated by a different regime.
    idx = pd.date_range("2020-01-01", periods=40, freq="D", tz="UTC")
    regimes = pd.Series(["BEAR_QUIET"] * 40, index=idx, dtype=object)
    regimes.iloc[0:10] = "BULL_QUIET"
    regimes.iloc[15:25] = "BULL_QUIET"
    regimes.iloc[30:40] = "BULL_QUIET"
    # pnl: episodes 1 & 2 net-positive (each 10 trades), episode 3 net-negative.
    pnl = pd.Series(np.nan, index=idx)
    pnl.iloc[0:10] = 0.01      # +
    pnl.iloc[15:25] = 0.01     # +
    pnl.iloc[30:40] = -0.01    # -
    _write_fixtures(tmp_path, monkeypatch, pnl, regimes)

    res = run_proposal.evaluate_per_episode(
        "TEST-B0035", "ema_cross", "XAUUSD", "xgb", _criterion(), ["BULL_QUIET"], "D1"
    )
    assert res["applicable"] is True
    assert res["n_active"] == 3
    assert res["n_survivors"] == 2
    assert res["required_survivors"] == 2  # ceil(0.6 * 3) = 2
    assert res["passed"] is True


def test_fails_when_only_one_episode_positive(tmp_path, monkeypatch):
    idx = pd.date_range("2020-01-01", periods=40, freq="D", tz="UTC")
    regimes = pd.Series(["BEAR_QUIET"] * 40, index=idx, dtype=object)
    regimes.iloc[0:10] = "BULL_QUIET"
    regimes.iloc[15:25] = "BULL_QUIET"
    regimes.iloc[30:40] = "BULL_QUIET"
    pnl = pd.Series(np.nan, index=idx)
    pnl.iloc[0:10] = 0.01       # +  (single-episode-driven edge)
    pnl.iloc[15:25] = -0.01     # -
    pnl.iloc[30:40] = -0.01     # -
    _write_fixtures(tmp_path, monkeypatch, pnl, regimes)

    res = run_proposal.evaluate_per_episode(
        "TEST-B0035", "ema_cross", "XAUUSD", "xgb", _criterion(), ["BULL_QUIET"], "D1"
    )
    assert res["n_survivors"] == 1
    assert res["required_survivors"] == 2
    assert res["passed"] is False


def test_inactive_episodes_excluded_from_denominator(tmp_path, monkeypatch):
    """Episodes with < min_trades trades are not counted as active."""
    idx = pd.date_range("2020-01-01", periods=40, freq="D", tz="UTC")
    regimes = pd.Series(["BEAR_QUIET"] * 40, index=idx, dtype=object)
    regimes.iloc[0:10] = "BULL_QUIET"
    regimes.iloc[15:25] = "BULL_QUIET"
    regimes.iloc[30:40] = "BULL_QUIET"
    pnl = pd.Series(np.nan, index=idx)
    pnl.iloc[0:10] = 0.01            # active (10 trades), +
    pnl.iloc[15:25] = 0.01           # active (10 trades), +
    pnl.iloc[30:32] = 0.01           # only 2 trades -> inactive (min_trades=5)
    _write_fixtures(tmp_path, monkeypatch, pnl, regimes)

    res = run_proposal.evaluate_per_episode(
        "TEST-B0035", "ema_cross", "XAUUSD", "xgb", _criterion(), ["BULL_QUIET"], "D1"
    )
    assert res["n_active"] == 2          # third episode excluded
    assert res["n_survivors"] == 2
    assert res["required_survivors"] == 2  # ceil(0.6 * 2) = 2
    assert res["passed"] is True


def test_fewer_than_two_active_episodes_fails(tmp_path, monkeypatch):
    idx = pd.date_range("2020-01-01", periods=40, freq="D", tz="UTC")
    regimes = pd.Series(["BEAR_QUIET"] * 40, index=idx, dtype=object)
    regimes.iloc[0:10] = "BULL_QUIET"
    pnl = pd.Series(np.nan, index=idx)
    pnl.iloc[0:10] = 0.01  # one active, positive episode — still not assessable
    _write_fixtures(tmp_path, monkeypatch, pnl, regimes)

    res = run_proposal.evaluate_per_episode(
        "TEST-B0035", "ema_cross", "XAUUSD", "xgb", _criterion(), ["BULL_QUIET"], "D1"
    )
    assert res["n_active"] == 1
    assert res["passed"] is False
    assert "fewer than 2 active episodes" in res["reason"]


def test_not_applicable_when_fraction_none(tmp_path, monkeypatch):
    res = run_proposal.evaluate_per_episode(
        "X", "ema_cross", "XAUUSD", "xgb",
        _criterion(fraction=None), ["BULL_QUIET"], "D1",
    )
    assert res == {"applicable": False}


def test_missing_pnl_artifact_fails_gracefully(tmp_path, monkeypatch):
    monkeypatch.setattr(run_proposal, "RESULTS_PHASE5_DIR", tmp_path / "results")
    monkeypatch.setattr(run_proposal, "REGIMES_DIR", tmp_path / "regimes")
    res = run_proposal.evaluate_per_episode(
        "NOPE", "ema_cross", "XAUUSD", "xgb", _criterion(), ["BULL_QUIET"], "D1"
    )
    assert res["applicable"] is True
    assert res["passed"] is False
    assert "missing" in res["reason"]
