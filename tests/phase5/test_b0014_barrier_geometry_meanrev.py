"""B0014 — BarrierGeometryAttestation per-archetype R:R bounds.

Trend-following primaries already had an R:R >= 2.5 guard (trend win rate
~30%). Mean-reversion's structural win rate is ~55-60% so the same floor
is the wrong constraint, but the validator must still flag two failure
modes specific to mean-rev:

  - R:R > 2.0 → letting winners run past the mean = trend-in-disguise
  - R:R < 0.5 → break-even requires >67% win rate = overfit territory

cusum_filter remains unconstrained (event-based archetype).
"""
from __future__ import annotations

import pytest

from phase5.proposal import BarrierGeometryAttestation, ProposalValidationError


# ---------------- trend primaries (unchanged behavior, regression coverage) ----------------

def test_trend_primary_rr_below_2p5_rejected():
    bg = BarrierGeometryAttestation(tp_atr_mult=1.5, sl_atr_mult=1.0)
    with pytest.raises(ProposalValidationError, match=r"R:R=1\.50 < 2\.5 for trend primary"):
        bg.validate("ema_cross")


def test_trend_primary_rr_at_2p5_accepted():
    bg = BarrierGeometryAttestation(tp_atr_mult=2.5, sl_atr_mult=1.0)
    bg.validate("ema_cross")  # no raise


def test_phase5_custom_primary_treated_as_trend():
    bg = BarrierGeometryAttestation(tp_atr_mult=2.0, sl_atr_mult=1.0)
    with pytest.raises(ProposalValidationError, match=r"R:R=2\.00 < 2\.5 for trend primary"):
        bg.validate("phase5_custom")


# ---------------- phase5_* mean-reversion custom primaries (B0088 materialization) -------
# A phase5_* custom primary whose name explicitly signals mean-reversion (name
# contains "meanrev") must use the mean-rev R:R band [0.5, 2.0], NOT the trend
# floor (2.5). Otherwise the materialized CMF mean-reversion survivor (B0088)
# could only be audited with a trend geometry that the validator's own docstring
# calls "trend-in-disguise" for a fade. Generic phase5_custom stays trend.

def test_phase5_meanrev_custom_uses_meanrev_band_symmetric_accepted():
    """phase5_cmf_meanrev with symmetric R:R=1.0 must be accepted (mean-rev band)."""
    bg = BarrierGeometryAttestation(tp_atr_mult=1.5, sl_atr_mult=1.5)
    bg.validate("phase5_cmf_meanrev")  # no raise


def test_phase5_meanrev_custom_rr_2p0_accepted():
    bg = BarrierGeometryAttestation(tp_atr_mult=2.0, sl_atr_mult=1.0)
    bg.validate("phase5_cmf_meanrev")  # no raise (would FAIL the trend floor of 2.5)


def test_phase5_meanrev_custom_rr_above_2p0_rejected():
    bg = BarrierGeometryAttestation(tp_atr_mult=3.0, sl_atr_mult=1.0)
    with pytest.raises(ProposalValidationError, match=r"R:R=3\.00 > 2\.0 for mean-rev primary"):
        bg.validate("phase5_cmf_meanrev")


def test_phase5_meanrev_custom_rr_below_0p5_rejected():
    bg = BarrierGeometryAttestation(tp_atr_mult=0.3, sl_atr_mult=1.0)
    with pytest.raises(ProposalValidationError, match=r"R:R=0\.30 < 0\.5 for mean-rev primary"):
        bg.validate("phase5_cmf_meanrev")


# ---------------- mean-reversion primaries (B0014, new behavior) -----------------------

def test_meanrev_symmetric_rr_accepted():
    """R:R=1.0 (symmetric) is the canonical mean-rev geometry."""
    bg = BarrierGeometryAttestation(tp_atr_mult=2.0, sl_atr_mult=2.0)
    bg.validate("bollinger_meanrev")  # no raise


def test_meanrev_rr_at_upper_bound_accepted():
    """R:R=2.0 is the inclusive upper bound."""
    bg = BarrierGeometryAttestation(tp_atr_mult=2.0, sl_atr_mult=1.0)
    bg.validate("bollinger_meanrev")  # no raise


def test_meanrev_rr_above_upper_bound_rejected():
    """R:R > 2.0 = trend-in-disguise on a mean-rev primary."""
    bg = BarrierGeometryAttestation(tp_atr_mult=3.0, sl_atr_mult=1.0)
    with pytest.raises(ProposalValidationError, match=r"R:R=3\.00 > 2\.0 for mean-rev primary"):
        bg.validate("bollinger_meanrev")


def test_meanrev_rr_at_lower_bound_accepted():
    """R:R=0.5 is the inclusive lower bound."""
    bg = BarrierGeometryAttestation(tp_atr_mult=1.0, sl_atr_mult=2.0)
    bg.validate("bollinger_meanrev")  # no raise


def test_meanrev_rr_below_lower_bound_rejected():
    """R:R < 0.5 demands >67% win rate to break even = overfit-risk territory."""
    bg = BarrierGeometryAttestation(tp_atr_mult=0.3, sl_atr_mult=1.0)
    with pytest.raises(ProposalValidationError, match=r"R:R=0\.30 < 0\.5 for mean-rev primary"):
        bg.validate("bollinger_meanrev")


# ---------------- cusum_filter (unchanged: event-based, no archetype constraint) -------

def test_cusum_filter_accepts_any_positive_geometry():
    """cusum_filter is event-based; no per-archetype R:R rule applies."""
    BarrierGeometryAttestation(tp_atr_mult=3.0, sl_atr_mult=1.0).validate("cusum_filter")  # trend-like OK
    BarrierGeometryAttestation(tp_atr_mult=1.0, sl_atr_mult=1.0).validate("cusum_filter")  # symmetric OK
    BarrierGeometryAttestation(tp_atr_mult=0.2, sl_atr_mult=1.0).validate("cusum_filter")  # low R:R OK


# ---------------- positivity guard (unchanged, regression coverage) ---------------------

def test_non_positive_tp_rejected_regardless_of_archetype():
    bg = BarrierGeometryAttestation(tp_atr_mult=0.0, sl_atr_mult=1.0)
    with pytest.raises(ProposalValidationError, match=r"values must be positive"):
        bg.validate("bollinger_meanrev")


def test_non_positive_sl_rejected_regardless_of_archetype():
    bg = BarrierGeometryAttestation(tp_atr_mult=1.0, sl_atr_mult=-0.5)
    with pytest.raises(ProposalValidationError, match=r"values must be positive"):
        bg.validate("bollinger_meanrev")
