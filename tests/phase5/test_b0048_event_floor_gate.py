"""B0048 — pre-flight event-floor gate.

The gate fails a proposal when the post-primary, in-regime event count is below
max(falsification_criterion.n_trades_total_min, walk-forward geometry floor),
converting an opaque subprocess_failed (walk_forward_refusal_cliff) into an
honest event_floor falsification BEFORE the heavy audit subprocess runs.

Motivating cases: T011D2M (35 events, floor 300) and T015D2M (73 events,
n_trades_min 50, floor 300) — both passed the n_trades-only pre-flight then died
in the audit subprocess on the WF refusal cliff.
"""
from __future__ import annotations

from phase5.run_proposal import evaluate_event_floor


def test_passes_when_events_clear_both_floors():
    res = evaluate_event_floor(n_events=320, n_trades_total_min=50, wf_floor=300)
    assert res["passed"] is True
    assert res["required_events"] == 300


def test_fails_below_wf_floor_even_when_above_declared_min():
    # T015D2M: 73 events, declared n_trades_total_min=50, WF floor 300.
    # The n_trades-only pre-flight let this through; the WF floor must catch it.
    res = evaluate_event_floor(n_events=73, n_trades_total_min=50, wf_floor=300)
    assert res["passed"] is False
    assert res["required_events"] == 300
    assert res["n_events"] == 73
    assert "wf" in res["reason"].lower() or "walk-forward" in res["reason"].lower()


def test_fails_below_declared_min_even_when_above_wf_floor():
    # A dense-ish primary that clears WF geometry but undershoots its own
    # committed edge-quality floor is still falsified.
    res = evaluate_event_floor(n_events=320, n_trades_total_min=400, wf_floor=300)
    assert res["passed"] is False
    assert res["required_events"] == 400


def test_boundary_exactly_at_required_passes():
    res = evaluate_event_floor(n_events=300, n_trades_total_min=50, wf_floor=300)
    assert res["passed"] is True
