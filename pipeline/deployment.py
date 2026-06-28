"""Deployment-tier assignment with DSR-aware Kelly sizing (Phase 2 T12).

STUB — implemented in T12.GREEN.

Motivation: Phase 1 v4 showed PSR ≥ 0.95 is achievable for all three
models on XAU D1 ema_cross, but DSR collapses to near-zero for two of
them after trial-pool deflation. Binary `enabled: bool` would either
deploy everything (ignoring deflation) or nothing (DSR ≥ 0.95 is a high
bar with 8 assets × 3 models × 4 folds = 96 trials). The tier system
turns DSR into a Kelly multiplier so borderline assets contribute to
the portfolio with controlled risk.

Tiers (`asset_deployment_tier`):
  - DSR ≥ 0.95  → full          (Kelly 1.0)
  - DSR ≥ 0.50  → half          (Kelly 0.5)
  - DSR ≥ 0.20  → quarter       (Kelly 0.25)
  - DSR ≥ 0.05  → paper_only    (Kelly 0.0, no live)
  - DSR <  0.05 → disabled      (Kelly 0.0, not in hackathon)

Hard gates (override the tier system → disabled):
  - n_trades < 100 (total across folds)
  - max_drawdown > 20%
  - PSR < 0.85

These hard gates are deliberately strict because they reflect retail
industry deployability concerns that DSR alone doesn't capture.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


DeploymentTier = Literal["full", "half", "quarter", "paper_only", "disabled"]


@dataclass
class AssetResults:
    """Inputs to `asset_deployment_tier` — what the orchestrator carries
    forward per asset to the deployment step."""
    psr: float
    dsr: float
    max_drawdown_pct: float
    n_trades_total: int


@dataclass
class TierAssignment:
    """Output of `asset_deployment_tier`."""
    tier: DeploymentTier
    kelly_fraction: float
    psr: float
    dsr: float
    max_dd: float
    n_trades: int
    reason: str


def asset_deployment_tier(asset_results: AssetResults) -> TierAssignment:
    """Map per-asset metrics to a deployment tier with Kelly multiplier.

    Order of checks:
      1. Hard gates (n_trades < 100, max_dd > 0.20, PSR < 0.85) → disabled.
         These reflect retail-deployability concerns DSR alone does not
         capture, so they override the tier.
      2. DSR tier assignment.
    """
    psr = asset_results.psr
    dsr = asset_results.dsr
    max_dd = asset_results.max_drawdown_pct
    n_trades = asset_results.n_trades_total

    def _disabled(reason: str) -> TierAssignment:
        return TierAssignment(
            tier="disabled", kelly_fraction=0.0,
            psr=psr, dsr=dsr, max_dd=max_dd, n_trades=n_trades,
            reason=reason,
        )

    # Hard gates first. Each cites the offending metric in the reason so
    # the deployment config writer can produce an actionable diagnostic.
    if n_trades < 100:
        return _disabled(f"n_trades={n_trades} < 100 (deployability floor)")
    if max_dd > 0.20:
        return _disabled(f"max_dd={max_dd:.1%} > 20% (retail risk ceiling)")
    if psr < 0.85:
        return _disabled(f"PSR={psr:.3f} < 0.85 (no statistical edge)")

    # DSR tier assignment. Bands chosen to combine Kelly-fractional
    # practice with empirical Phase 1 calibration (rf DSR=0.257 → quarter).
    if dsr >= 0.95:
        tier, kelly = "full", 1.0
    elif dsr >= 0.50:
        tier, kelly = "half", 0.5
    elif dsr >= 0.20:
        tier, kelly = "quarter", 0.25
    elif dsr >= 0.05:
        tier, kelly = "paper_only", 0.0
    else:
        tier, kelly = "disabled", 0.0

    return TierAssignment(
        tier=tier, kelly_fraction=kelly,
        psr=psr, dsr=dsr, max_dd=max_dd, n_trades=n_trades,
        reason=f"PSR={psr:.3f}, DSR={dsr:.3f} → tier={tier}",
    )


def session_eligible(psr: float, sharpe_per_fold: list[float], n_events: int) -> bool:
    """Per-session viability gate. Looser than per-asset because sessions
    have less data per fold.

    Criteria (all must hold):
      - PSR ≥ 0.90
      - ≥3 of 4 folds have positive Sharpe (NaN folds excluded)
      - ≥250 events
    """
    if psr < 0.90:
        return False
    n_positive_folds = sum(1 for s in sharpe_per_fold if s is not None and s > 0)
    if n_positive_folds < 3:
        return False
    if n_events < 250:
        return False
    return True
