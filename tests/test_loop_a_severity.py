"""B0015 — DA objection severity vocabulary normalization.

Tick 3 (2026-07-04): the DA emitted severity 'fatal' twice; loop_a_tick
counted da_objections_high=0 and skipped the retry loop. Tick 4 emitted
'high' and the retry engaged. The loop's behavior must not depend on which
synonym vocabulary the agent happened to use, and retry feedback must never
render blank objection lines (tick 4: 'claim' vs 'claim_attacked' key drift).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.loop_a_tick import _format_da_feedback, _is_high_severity, _norm_severity


def test_fatal_and_critical_count_as_high():
    for sev in ("high", "fatal", "critical", "HIGH", "Fatal "):
        assert _is_high_severity({"severity": sev}), sev
    for sev in ("medium", "major", "minor", "low", ""):
        assert not _is_high_severity({"severity": sev}), sev


def test_norm_severity_lowercases_and_strips():
    assert _norm_severity({"severity": " Major "}) == "major"
    assert _norm_severity({}) == ""


def test_feedback_renders_fatal_objections_as_hard_constraints():
    verdict = {
        "verdict": "BLOCK",
        "steel_man": "coherent mechanism",
        "objections": [
            {"severity": "fatal", "claim_attacked": "geometry",
             "objection": "3:1 TP unreachable", "evidence": "hit_rate_q vs ret_q"},
            {"severity": "major", "claim_attacked": "volume leg",
             "objection": "weakly supported"},
        ],
    }
    text = _format_da_feedback(verdict)
    assert "HIGH-severity objections" in text
    assert "geometry" in text and "3:1 TP unreachable" in text
    assert "Medium-severity objections" in text
    assert "volume leg" in text


def test_feedback_never_renders_blank_numbered_items():
    verdict = {
        "verdict": "BLOCK",
        "objections": [
            {"severity": "high", "claim_attacked": "reachability",
             "objection": "floor arithmetic fails", "evidence": "654 < 799"},
        ],
    }
    text = _format_da_feedback(verdict)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("1."):
            assert len(stripped) > len("1. "), "blank objection line rendered"
            assert "reachability" in stripped
