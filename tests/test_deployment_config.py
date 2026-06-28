"""Tests for deployment-tier assignment with DSR-aware Kelly sizing (T12).

The deployment writer is the bridge between Phase 2's training metrics
and Phase 3's live execution. It encodes the "what risk per asset"
decision in a small, auditable way: each asset gets a tier
(full/half/quarter/paper_only/disabled) and a Kelly multiplier.

Hard gates override the DSR-based tier:
  - n_trades_total < 100 → disabled (deployability floor)
  - max_drawdown > 0.20 → disabled (retail risk ceiling)
  - PSR < 0.85 → disabled (no statistical edge over random)

Among assets that pass the hard gates, DSR maps to Kelly:
  - DSR ≥ 0.95 → full      kelly=1.0
  - DSR ≥ 0.50 → half      kelly=0.5
  - DSR ≥ 0.20 → quarter   kelly=0.25
  - DSR ≥ 0.05 → paper_only kelly=0.0
  - DSR <  0.05 → disabled  kelly=0.0
"""
from __future__ import annotations
import pytest

from pipeline.deployment import (
    AssetResults,
    asset_deployment_tier,
    session_eligible,
)


def _ar(psr: float, dsr: float, max_dd: float, n_trades: int) -> AssetResults:
    return AssetResults(
        psr=psr, dsr=dsr, max_drawdown_pct=max_dd, n_trades_total=n_trades,
    )


# ---------------------------------------------------------------------------
# DSR → tier mapping (passes all hard gates)
# ---------------------------------------------------------------------------

def test_dsr_above_095_assigns_full_tier():
    result = asset_deployment_tier(_ar(psr=0.99, dsr=0.97, max_dd=0.05, n_trades=400))
    assert result.tier == "full"
    assert result.kelly_fraction == 1.0


def test_dsr_in_050_095_assigns_half_tier():
    result = asset_deployment_tier(_ar(psr=0.97, dsr=0.60, max_dd=0.10, n_trades=300))
    assert result.tier == "half"
    assert result.kelly_fraction == 0.5


def test_dsr_in_020_050_assigns_quarter_tier():
    result = asset_deployment_tier(_ar(psr=0.95, dsr=0.30, max_dd=0.12, n_trades=200))
    assert result.tier == "quarter"
    assert result.kelly_fraction == 0.25


def test_dsr_in_005_020_assigns_paper_only_tier():
    result = asset_deployment_tier(_ar(psr=0.90, dsr=0.10, max_dd=0.15, n_trades=150))
    assert result.tier == "paper_only"
    assert result.kelly_fraction == 0.0


def test_dsr_below_005_assigns_disabled_tier():
    """Phase 1 v4 catboost on XAU D1 ema_cross hits exactly this case:
    PSR=0.981, DSR=0.001, max_dd≈0.18, n_trades=100. DSR<0.05 → disabled."""
    result = asset_deployment_tier(_ar(psr=0.981, dsr=0.001, max_dd=0.18, n_trades=100))
    assert result.tier == "disabled"
    assert result.kelly_fraction == 0.0


# ---------------------------------------------------------------------------
# Hard gates: override the tier to "disabled" regardless of DSR
# ---------------------------------------------------------------------------

def test_n_trades_below_100_overrides_to_disabled_even_with_high_dsr():
    """Even DSR=0.99 doesn't save you if the trade count is too thin to
    deploy. Statistical edge ≠ deployable distribution."""
    result = asset_deployment_tier(_ar(psr=0.99, dsr=0.99, max_dd=0.05, n_trades=50))
    assert result.tier == "disabled"
    assert result.kelly_fraction == 0.0
    assert "n_trades" in result.reason


def test_max_dd_above_020_overrides_to_disabled_even_with_high_dsr():
    """Industry retail rule: 20% drawdown is the ceiling. Edge doesn't
    matter if the path to that edge is unbearable."""
    result = asset_deployment_tier(_ar(psr=0.99, dsr=0.99, max_dd=0.25, n_trades=400))
    assert result.tier == "disabled"
    assert result.kelly_fraction == 0.0
    assert "max_dd" in result.reason


def test_psr_below_085_overrides_to_disabled_even_with_high_dsr():
    """PSR < 0.85 means the observed SR isn't reliably > 0 even before
    deflation. Anything below this is statistical noise."""
    result = asset_deployment_tier(_ar(psr=0.80, dsr=0.99, max_dd=0.10, n_trades=400))
    assert result.tier == "disabled"
    assert result.kelly_fraction == 0.0
    assert "PSR" in result.reason or "psr" in result.reason


# ---------------------------------------------------------------------------
# Session eligibility (separate code path, simpler logic)
# ---------------------------------------------------------------------------

def test_session_eligible_requires_psr_090_and_3_of_4_positive_folds_and_250_events():
    """Per-session criteria (more lax than per-asset because sessions have
    less data per fold):
      - PSR ≥ 0.90
      - ≥3 of 4 folds have positive Sharpe
      - ≥250 events
    """
    assert session_eligible(psr=0.92, sharpe_per_fold=[0.5, 0.3, 0.7, 0.2], n_events=300)


def test_session_eligible_rejects_when_psr_below_090():
    assert not session_eligible(psr=0.85, sharpe_per_fold=[0.5, 0.3, 0.7, 0.2], n_events=300)


def test_session_eligible_rejects_when_fewer_than_3_positive_folds():
    assert not session_eligible(psr=0.95, sharpe_per_fold=[0.5, -0.3, -0.1, 0.2], n_events=300)


def test_session_eligible_rejects_when_events_below_250():
    assert not session_eligible(psr=0.95, sharpe_per_fold=[0.5, 0.3, 0.7, 0.2], n_events=100)
