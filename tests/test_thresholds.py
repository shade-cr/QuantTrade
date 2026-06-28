"""B0161 — EV-breakeven threshold resolution for the multi-asset orchestrator.

The B0155 formula (phase5) becomes the canonical `pipeline.thresholds` module;
`resolve_threshold_grid` derives the orchestrator's threshold grid from barrier
geometry + global cost constants when `metrics.threshold_rule: ev_breakeven_v1`
is set, instead of the legacy fixed grid (whose minimum, 0.50, sat ABOVE the
reachable calibrated-p range for tp=3/sl=1 geometry — the systemic NO_FIRE).
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.thresholds import (
    C_ATR,
    LAMBDA_MARGIN,
    compute_p_star,
    ev_breakeven_grid,
    resolve_threshold_grid,
)


# --------------------------------------------------------------------------- #
# compute_p_star — values pinned to the B0155 pre-registered examples
# --------------------------------------------------------------------------- #

def test_p_star_tp3_sl1_is_0p325():
    assert compute_p_star(3.0, 1.0) == pytest.approx(0.325, abs=1e-9)


def test_p_star_meanrev_tp2_sl1p5():
    assert compute_p_star(2.0, 1.5) == pytest.approx((1.5 + 0.10 + 0.05 * 3.5) / 3.5, abs=1e-9)


def test_p_star_tp1p8_sl1():
    assert compute_p_star(1.8, 1.0) == pytest.approx((1.0 + 0.10 + 0.05 * 2.8) / 2.8, abs=1e-9)


def test_global_constants_unchanged():
    # Amendable only by methodology spec change — pin them so a silent edit fails loudly.
    assert C_ATR == 0.10
    assert LAMBDA_MARGIN == 0.05


def test_phase5_reexports_canonical_formula():
    # phase5.proposal must keep exposing the SAME objects (no drift / duplication).
    from phase5 import proposal

    assert proposal.compute_p_star is compute_p_star
    assert proposal.C_ATR == C_ATR
    assert proposal.LAMBDA_MARGIN == LAMBDA_MARGIN


# --------------------------------------------------------------------------- #
# ev_breakeven_grid
# --------------------------------------------------------------------------- #

def test_grid_is_p_star_plus_offsets_sorted():
    grid = ev_breakeven_grid(3.0, 1.0, offsets=(0.0, 0.03, 0.06, 0.10, 0.15))
    assert grid == pytest.approx([0.325, 0.355, 0.385, 0.425, 0.475])


def test_grid_default_offsets_start_at_p_star():
    grid = ev_breakeven_grid(3.0, 1.0)
    assert grid[0] == pytest.approx(0.325)
    assert grid == sorted(grid)
    assert len(grid) == len(set(np.round(grid, 12)))


def test_grid_clips_above_1():
    # Degenerate geometry pushing p* near 1 must not emit thresholds >= 1.
    grid = ev_breakeven_grid(0.1, 5.0, offsets=(0.0, 0.10, 0.50))
    assert all(t < 1.0 for t in grid)
    assert len(grid) >= 1  # p* itself retained (clipped grid never empty)


# --------------------------------------------------------------------------- #
# resolve_threshold_grid — the orchestrator hook
# --------------------------------------------------------------------------- #

def test_resolve_no_rule_returns_legacy_grid_unchanged():
    metrics_cfg = {"threshold_grid": [0.5, 0.55, 0.6]}
    grid, p_star = resolve_threshold_grid(metrics_cfg, {"tp_atr_mult": 3.0, "sl_atr_mult": 1.0})
    assert grid == [0.5, 0.55, 0.6]
    assert p_star is None


def test_resolve_ev_breakeven_recentres_grid():
    metrics_cfg = {"threshold_grid": [0.5, 0.55, 0.6], "threshold_rule": "ev_breakeven_v1"}
    grid, p_star = resolve_threshold_grid(metrics_cfg, {"tp_atr_mult": 3.0, "sl_atr_mult": 1.0})
    assert p_star == pytest.approx(0.325)
    assert grid[0] == pytest.approx(0.325)
    assert all(t < 0.5 for t in grid[:2])  # the EV-positive zone is now reachable


def test_resolve_respects_explicit_offsets():
    metrics_cfg = {
        "threshold_grid": [0.5],
        "threshold_rule": "ev_breakeven_v1",
        "threshold_grid_offsets": [0.0, 0.2],
    }
    grid, p_star = resolve_threshold_grid(metrics_cfg, {"tp_atr_mult": 3.0, "sl_atr_mult": 1.0})
    assert grid == pytest.approx([0.325, 0.525])


def test_resolve_unknown_rule_raises():
    metrics_cfg = {"threshold_grid": [0.5], "threshold_rule": "best_of_test_set"}
    with pytest.raises(ValueError, match="threshold_rule"):
        resolve_threshold_grid(metrics_cfg, {"tp_atr_mult": 3.0, "sl_atr_mult": 1.0})


def test_resolve_degenerate_geometry_raises():
    # tp=0.1/sl=5 -> p* > 1: no probability can clear breakeven; the orchestrator
    # must refuse loudly instead of running a structurally mute audit.
    metrics_cfg = {"threshold_grid": [0.5], "threshold_rule": "ev_breakeven_v1"}
    with pytest.raises(ValueError, match="degenerate"):
        resolve_threshold_grid(metrics_cfg, {"tp_atr_mult": 0.1, "sl_atr_mult": 5.0})


# --------------------------------------------------------------------------- #
# classification_metrics gains roc_auc (B0161 visibility item)
# --------------------------------------------------------------------------- #

def test_classification_metrics_includes_roc_auc():
    from pipeline.metrics import classification_metrics

    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=200)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, size=200), 0, 1)
    out = classification_metrics(y, p)
    assert "roc_auc" in out
    assert 0.5 < out["roc_auc"] <= 1.0


def test_classification_metrics_roc_auc_nan_single_class():
    from pipeline.metrics import classification_metrics

    y = np.zeros(50, dtype=int)
    p = np.linspace(0.1, 0.4, 50)
    out = classification_metrics(y, p)
    assert np.isnan(out["roc_auc"])
