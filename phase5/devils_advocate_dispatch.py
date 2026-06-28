"""Helper to format devil's-advocate decision payloads and persist verdicts.

The Phase 5 orchestrator calls `make_decision_payload(...)` to build the
JSON the phase5-devils-advocate agent receives, and `persist_verdict(...)`
to write the agent's response to the canonical location with the BLOCK/
PROCEED_WITH_CAVEAT contract validation.

The orchestrator does NOT invoke the agent directly from Python — instead,
it writes `phase5/runtime/next_action.json` and pauses until a sibling
Claude Code session uses the Agent tool to dispatch the review and writes
the response back. This Python helper module formats both halves of that
contract.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


SIGNALS_DIR = Path("signals")
DA_REVIEWS_DIR = SIGNALS_DIR / "devils_advocate_reviews"
RUNTIME_DIR = Path("phase5/runtime")


VALID_DECISION_TYPES = (
    "regime_taxonomy_boundaries",
    "hypothesis_proposal_commit",
    "custom_primary_code_commit",
    "refinement_proposal_commit",
    "promote_or_discard",
    "methodology_spec_edit",  # added Day 1 per skeptic recommendation
)

VALID_VERDICTS = ("BLOCK", "PROCEED_WITH_CAVEAT")
INVALID_VERDICT_TOKENS = ("APPROVE",)  # explicit forbid


class DevilsAdvocateContractError(ValueError):
    """Raised when a devil's-advocate verdict violates the schema."""


@dataclass
class ReviewVerdict:
    decision_type: str
    decision_payload_ref: str
    verdict: str
    steel_man: str
    objections: list[dict]
    must_have_mods_before_proceed: list[str]
    reviewer_notes: str = ""

    def validate(self) -> None:
        if self.decision_type not in VALID_DECISION_TYPES:
            raise DevilsAdvocateContractError(
                f"decision_type={self.decision_type!r} not in {VALID_DECISION_TYPES}"
            )
        if self.verdict in INVALID_VERDICT_TOKENS:
            raise DevilsAdvocateContractError(
                f"verdict={self.verdict!r} is forbidden — agent cannot emit APPROVE"
            )
        if self.verdict not in VALID_VERDICTS:
            raise DevilsAdvocateContractError(
                f"verdict={self.verdict!r} not in {VALID_VERDICTS}"
            )
        if not self.objections:
            raise DevilsAdvocateContractError(
                "objections must contain >=1 entry; vacuous approval is impossible by construction"
            )
        for obj in self.objections:
            if obj.get("severity") not in ("high", "medium", "low"):
                raise DevilsAdvocateContractError(
                    f"objection has invalid severity: {obj}"
                )
            if not obj.get("claim") or not obj.get("evidence") or not obj.get("suggested_remediation"):
                raise DevilsAdvocateContractError(
                    f"objection is missing required field (claim/evidence/suggested_remediation): {obj}"
                )
        if self.verdict == "PROCEED_WITH_CAVEAT":
            high_unaddressed = [
                o for o in self.objections
                if o.get("severity") == "high"
            ]
            if high_unaddressed and not self.must_have_mods_before_proceed:
                raise DevilsAdvocateContractError(
                    "PROCEED_WITH_CAVEAT with >=1 high-severity objection requires "
                    "must_have_mods_before_proceed entries documenting how each is addressed"
                )
        if len(self.steel_man) < 50:
            raise DevilsAdvocateContractError(
                f"steel_man must be substantive (>= 50 chars); got {len(self.steel_man)} chars"
            )

    def to_dict(self) -> dict:
        return {
            "decision_type": self.decision_type,
            "decision_payload_ref": self.decision_payload_ref,
            "verdict": self.verdict,
            "steel_man": self.steel_man,
            "objections": self.objections,
            "must_have_mods_before_proceed": self.must_have_mods_before_proceed,
            "reviewer_notes": self.reviewer_notes,
        }


def make_decision_payload(
    decision_type: str,
    decision_payload: dict,
    regime_stats_dossier: Optional[dict] = None,
    audit_audit_findings: Optional[dict] = None,
    prior_reviews_on_lineage: Optional[list[dict]] = None,
) -> dict:
    """Build the payload the devil's-advocate agent receives."""
    return {
        "decision_type": decision_type,
        "decision_payload": decision_payload,
        "prior_reviews_on_lineage": prior_reviews_on_lineage or [],
        "regime_stats_dossier": regime_stats_dossier or {},
        "audit_audit_findings": audit_audit_findings or {},
    }


def persist_verdict(verdict: ReviewVerdict, review_id: str) -> Path:
    """Validate + persist a verdict to signals/devils_advocate_reviews/<id>.json."""
    verdict.validate()
    DA_REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DA_REVIEWS_DIR / f"{review_id}.json"
    out_path.write_text(json.dumps(verdict.to_dict(), indent=2), encoding="utf-8")
    return out_path


def load_verdict_from_dict(payload: dict) -> ReviewVerdict:
    """Parse a JSON response from the agent into a ReviewVerdict (validated)."""
    v = ReviewVerdict(
        decision_type=payload["decision_type"],
        decision_payload_ref=payload.get("decision_payload_ref", ""),
        verdict=payload["verdict"],
        steel_man=payload["steel_man"],
        objections=payload["objections"],
        must_have_mods_before_proceed=payload.get("must_have_mods_before_proceed", []),
        reviewer_notes=payload.get("reviewer_notes", ""),
    )
    v.validate()
    return v


def submit_for_review(payload: dict, action_id: str) -> Path:
    """Write the next_action payload that pauses the orchestrator for review.

    A sibling Claude Code session (or the user running an interactive loop)
    reads `phase5/runtime/next_action_<action_id>.json`, invokes the
    phase5-devils-advocate agent with the payload, and writes the response
    back to `phase5/runtime/agent_response_<action_id>.json`.

    This Python helper does NOT invoke the agent — that's done by the
    parent Claude Code session via the Agent tool.
    """
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNTIME_DIR / f"next_action_{action_id}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out_path
