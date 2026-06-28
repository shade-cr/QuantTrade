"""Regex linter blocking lookahead-leakage tokens in LLM-emitted text.

The Phase 5 hypothesizer must reason about regime mechanisms without
referencing specific historical episodes. The LLM has training data through
~Jan 2026, so any year/event token in its output is suspicious.

This linter scans string fields of a proposal (or any text payload) and
flags forbidden tokens. It is enforced at proposal-commit time — if the
linter raises, the proposal is rejected and the hypothesizer is re-invoked.

The linter is INTENTIONALLY aggressive (some false positives). Better to
re-prompt the hypothesizer than to silently accept memorized history.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable


# Years 1900-2199 anywhere in the text. The 2099 cap is far enough into the
# future to catch any plausible reference and avoid matching version numbers
# like 'sklearn 1.4'.
_YEAR_PATTERN = re.compile(r"\b(19|20|21)\d{2}\b")

# Month names (any case) when they appear near a digit or "of" — usually
# a date reference. Bare "March effect" is OK; "March 2020" is not.
_MONTH_WITH_CONTEXT = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b"
    r"\s*(of\s+\d{4}|\d{4}|\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4})",
    re.IGNORECASE,
)

# Named historical events / figures whose mention implies memorized episodes.
# Each is a literal-match word boundary on a case-insensitive scan.
_NAMED_EVENTS = [
    "COVID",
    "GFC",  # Global Financial Crisis
    "Lehman",
    "Powell",  # Fed chair
    "Yellen",  # Fed chair / Treasury secretary
    "Bernanke",  # Fed chair
    "Greenspan",
    "Fed pivot",
    "taper tantrum",
    "flash crash",
    "crypto winter",
    "LUNA",
    "Terra Luna",
    "FTX",
    "Silicon Valley Bank",
    "SVB collapse",
    "SVB failure",
    "Brexit",
    "Trump tariff",
    "Trump tariffs",
    "pandemic",
    "QE1",
    "QE2",
    "QE3",
    "QE4",
    "Plaza Accord",
    "Volcker",
    "Black Monday",
    "dot-com bubble",
    "dot com bubble",
    "Asian financial crisis",
    "European sovereign debt crisis",
    "eurozone crisis",
    "Greek default",
    "Russia default",
    "subprime",
    "Bear Stearns",
    "AIG bailout",
    "TARP",
    "ZIRP",
    "NIRP",
]

_NAMED_EVENT_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(e) for e in _NAMED_EVENTS) + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LintHit:
    """A single lint violation in a text field."""

    field: str
    rule: str        # which rule fired: "year" / "month_with_date" / "named_event"
    match: str       # the actual token matched (for reporting)
    position: int    # character offset in the field


@dataclass(frozen=True)
class LintResult:
    """Aggregate lint outcome for a payload."""

    passed: bool
    hits: tuple[LintHit, ...]

    def summary(self) -> str:
        if self.passed:
            return "lookahead lint: PASS"
        lines = [f"lookahead lint: FAIL ({len(self.hits)} hit(s))"]
        for h in self.hits:
            lines.append(f"  field={h.field!r} rule={h.rule} match={h.match!r} pos={h.position}")
        return "\n".join(lines)


def lint_text(text: str, field_label: str = "<text>") -> tuple[LintHit, ...]:
    """Scan a single string; return all violations (empty tuple = clean)."""
    if not isinstance(text, str):
        return ()
    hits: list[LintHit] = []
    for m in _YEAR_PATTERN.finditer(text):
        hits.append(LintHit(field=field_label, rule="year", match=m.group(0), position=m.start()))
    for m in _MONTH_WITH_CONTEXT.finditer(text):
        hits.append(LintHit(field=field_label, rule="month_with_date", match=m.group(0), position=m.start()))
    for m in _NAMED_EVENT_PATTERN.finditer(text):
        hits.append(LintHit(field=field_label, rule="named_event", match=m.group(0), position=m.start()))
    return tuple(hits)


def lint_fields(payload: dict, fields: Iterable[str]) -> LintResult:
    """Lint a set of named string fields in a dict payload.

    Missing fields are ignored. Non-string fields are skipped (the schema
    validator should reject those separately).
    """
    all_hits: list[LintHit] = []
    for f in fields:
        val = payload.get(f)
        if isinstance(val, str):
            all_hits.extend(lint_text(val, field_label=f))
    return LintResult(passed=not all_hits, hits=tuple(all_hits))


# Default fields linted on a phase5 proposal payload.
DEFAULT_PROPOSAL_FIELDS = ("hypothesis", "causal_story", "custom_primary_pseudocode")


def lint_proposal(payload: dict) -> LintResult:
    """Lint a proposal JSON for lookahead leakage in narrative fields."""
    return lint_fields(payload, DEFAULT_PROPOSAL_FIELDS)


# ---------------------------------------------------------------------------
# Anti-circularity lint (B0034) — structural, not text-based
# ---------------------------------------------------------------------------
#
# The regime taxonomy is constructed from these features
# (see `.claude/skills/phase5-regime-methodology/SKILL.md` §Regime taxonomy):
#
#   Trend axis: roc_63 (63-day ROC), ma_50 + ma_200 (MA-stack)
#   Vol axis:   rv_20 (20-day realized vol)
#
# Any primary whose entry condition IS one of these features is tautological
# with the regime label and cannot carry an edge above the regime gate.

REGIME_DEFINING_FEATURES = ("roc_63", "ma_50", "ma_200", "rv_20")

# Lookback values that would mechanically reproduce the regime-defining
# windows. Forbidden for `momentum_zscore` / `cusum_filter` primaries.
FORBIDDEN_MOMENTUM_LOOKBACKS = (63,)
FORBIDDEN_RV_WINDOWS = (20,)

# Forbidden (fast, slow) MA pairs for ema_cross. The (50, 200) pair is
# exactly the regime-defining MA stack; (any, 200) and (50, any) are partial
# matches we also flag (high false-positive tolerance: see module docstring).
FORBIDDEN_EMA_CROSS_PAIRS = ((50, 200),)
FORBIDDEN_EMA_CROSS_PARTIAL_LOOKBACKS = (50, 200)


def lint_anti_circularity(payload: dict, dossier: dict | None = None) -> LintResult:
    """Check that the proposal's primary is not structurally equivalent to a
    regime-defining condition.

    Rules (B0034):
      1. momentum_zscore / cusum_filter with lookback == 63 → FAIL (matches roc_63)
      2. ema_cross with (fast=50, slow=200) → FAIL (matches MA-stack exactly)
      3. ema_cross with either lookback in {50, 200} → WARN (partial match)
      4. Any primary with rv_window == 20 as a sole entry filter → FAIL
      5. feature_overrides.add containing ANY regime-defining feature without
         a corresponding justification in causal_story → WARN
      6. (B0044) target_regime_episode_ordinals NOT a subset of the dossier's
         regime_episode_ordinals → FAIL. Only runs when `dossier` is supplied.

    `dossier` is the same regime-aggregate, point-in-time dossier the
    hypothesizer received (threaded in by the caller, e.g. stage_review). It is
    optional and the episode-ordinal rule is skipped when it is None, so
    existing callers that lint without a dossier are unaffected.

    Note on rule 6 (B0044): naming a SUBSET of the regime's episodes is the
    LEGITIMATE, designed shape-leakage tripwire (per phase5-regime-methodology
    SKILL.md §regime_episode_ordinals: "only flag if the listed ordinals are
    NOT a subset"). The genuine leak is the inverse — naming an ordinal that
    does NOT belong to this regime. The count requirement (>=2) is enforced in
    Proposal.validate(), NOT here, and the single-episode-collapse judgment is
    the DA/skeptic's job, NOT this deterministic rule's.
    """
    hits: list[LintHit] = []
    primary = payload.get("primary", "")
    params = payload.get("primary_params", {}) or {}
    overrides = payload.get("feature_overrides", {}) or {}
    add = overrides.get("add", []) or []
    causal = payload.get("causal_story", "") or ""

    # Rule 1 — momentum/cusum lookback collides with roc_63
    if primary in ("momentum_zscore", "cusum_filter"):
        lb = params.get("lookback")
        if isinstance(lb, (int, float)) and int(lb) in FORBIDDEN_MOMENTUM_LOOKBACKS:
            hits.append(LintHit(
                field="primary_params.lookback",
                rule="anti_circularity_momentum_lookback",
                match=f"{primary}.lookback={lb} matches roc_63 regime feature",
                position=0,
            ))

    # Rule 2/3 — ema_cross matches MA-stack
    if primary == "ema_cross":
        fast = params.get("fast")
        slow = params.get("slow")
        if isinstance(fast, (int, float)) and isinstance(slow, (int, float)):
            pair = (int(fast), int(slow))
            if pair in FORBIDDEN_EMA_CROSS_PAIRS:
                hits.append(LintHit(
                    field="primary_params",
                    rule="anti_circularity_ema_cross_pair",
                    match=f"ema_cross(fast={fast}, slow={slow}) matches MA-stack regime feature",
                    position=0,
                ))
            else:
                if int(fast) in FORBIDDEN_EMA_CROSS_PARTIAL_LOOKBACKS:
                    hits.append(LintHit(
                        field="primary_params.fast",
                        rule="anti_circularity_ema_cross_partial",
                        match=f"ema_cross.fast={fast} matches one leg of MA-stack",
                        position=0,
                    ))
                if int(slow) in FORBIDDEN_EMA_CROSS_PARTIAL_LOOKBACKS:
                    hits.append(LintHit(
                        field="primary_params.slow",
                        rule="anti_circularity_ema_cross_partial",
                        match=f"ema_cross.slow={slow} matches one leg of MA-stack",
                        position=0,
                    ))

    # Rule 4 — rv_window == 20 as sole entry filter
    rv = params.get("rv_window") or params.get("rv_lookback")
    if isinstance(rv, (int, float)) and int(rv) in FORBIDDEN_RV_WINDOWS:
        hits.append(LintHit(
            field="primary_params.rv_window",
            rule="anti_circularity_rv_window",
            match=f"rv_window={rv} matches rv_20 regime feature",
            position=0,
        ))

    # Rule 5 — regime-defining features in feature_overrides.add: lenient check.
    # We require ANY of a broad set of justification phrases — the goal is to catch
    # blatant "kitchen sink" feature dumps, not to police natural-language phrasing.
    # The devil's advocate does the semantic correctness review; this lint only
    # catches the absence of ANY attempt at justification.
    overlapping = [f for f in add if f in REGIME_DEFINING_FEATURES]
    if overlapping:
        causal_lower = causal.lower()
        justification_keywords = (
            # Within-regime variation language
            "within-regime", "within regime",
            # Distribution language (tercile/quartile/decile/percentile/half)
            "tercile", "quartile", "decile", "percentile",
            "lower half", "upper half", "lower third", "upper third",
            # Distribution / regime references
            "regime's distribution", "regime distribution",
            "trailing distribution", "trailing-year distribution",
            "trailing year distribution", "trailing-252",
            # "Stricter than" language
            "stricter than the gate", "tighter than the gate",
            "stricter than", "tighter than",
            "above the gate", "below the gate",
            # Orthogonality / decorrelation arguments
            "orthogonal to", "decorrelated from", "decouple",
            "first difference", "differencing", "delta",
            # Explicit correlation/control language
            "correlation", "control test", "control for",
        )
        has_justification = any(kw in causal_lower for kw in justification_keywords)
        if not has_justification:
            hits.append(LintHit(
                field="feature_overrides.add",
                rule="anti_circularity_features_unjustified",
                match=f"regime-defining features in add list with NO justification phrase in causal_story: {overlapping}",
                position=0,
            ))

    # Rule 6 (B0044) — target_regime_episode_ordinals must be a subset of the
    # regime's episodes. Only evaluable when the dossier is supplied.
    if dossier is not None:
        allowed = dossier.get("regime_episode_ordinals")
        shape = payload.get("lookahead_shape_attestation", {}) or {}
        ordinals = shape.get("target_regime_episode_ordinals", []) or []
        if ordinals and allowed is not None:
            extra = sorted(set(ordinals) - set(allowed))
            if extra:
                hits.append(LintHit(
                    field="lookahead_shape_attestation.target_regime_episode_ordinals",
                    rule="anti_circularity_episode_ordinals_not_subset",
                    match=(
                        f"target_regime_episode_ordinals {sorted(ordinals)} contains ordinals "
                        f"{extra} NOT in dossier.regime_episode_ordinals {sorted(allowed)} — "
                        f"ordinals must be a subset of this regime's episodes (naming an "
                        f"out-of-regime episode is a shape-leakage signal)."
                    ),
                    position=0,
                ))

    return LintResult(passed=not hits, hits=tuple(hits))


def lint_full(payload: dict, dossier: dict | None = None) -> LintResult:
    """Run all proposal lint checks (narrative + anti-circularity).

    `dossier` is forwarded to lint_anti_circularity to enable the B0044
    episode-ordinal subset rule; it is optional and back-compatible.
    """
    narrative = lint_proposal(payload)
    circ = lint_anti_circularity(payload, dossier=dossier)
    all_hits = narrative.hits + circ.hits
    return LintResult(passed=not all_hits, hits=all_hits)


def format_lint_feedback(hits: tuple[LintHit, ...]) -> str:
    """Render lint hits as a natural-language preamble for the hypothesizer
    retry loop (B0034 v2).

    The output is meant to be prepended to the JSON payload sent to the
    phase5-hypothesizer agent on retry. The agent receives explicit
    instructions about which constraints its previous output violated and
    what to do instead.
    """
    if not hits:
        return ""
    lines = [
        "PREVIOUS ATTEMPT FAILED THE ANTI-CIRCULARITY LINT.",
        "",
        "Specific violations:",
    ]
    for h in hits:
        lines.append(f"  - [{h.rule}] {h.match}")
    lines.extend([
        "",
        "Hard constraints to satisfy on this retry:",
        f"  - Forbidden lookbacks for momentum_zscore / cusum_filter: {FORBIDDEN_MOMENTUM_LOOKBACKS}",
        f"  - Forbidden (fast, slow) pairs for ema_cross: {FORBIDDEN_EMA_CROSS_PAIRS}",
        f"  - Forbidden partial lookbacks for ema_cross.fast or .slow: {FORBIDDEN_EMA_CROSS_PARTIAL_LOOKBACKS}",
        f"  - Forbidden rv_window / rv_lookback values: {FORBIDDEN_RV_WINDOWS}",
        "  - If you include any of " + str(REGIME_DEFINING_FEATURES) + " in feature_overrides.add,",
        "    your causal_story MUST contain explicit within-regime-variation justification.",
        "    Use phrases like 'within-regime', 'tercile of the regime distribution',",
        "    'stricter than the gate', 'tighter than the gate', or equivalent.",
        "  - target_regime_episode_ordinals MUST be a SUBSET of the regime's episodes",
        "    listed in the dossier's regime_episode_ordinals. A subset is correct and",
        "    expected; naming an ordinal OUTSIDE that set is the violation. Do NOT pad",
        "    the list to the full set to dodge this — pick the episodes where the",
        "    mechanism should have paid, all drawn from regime_episode_ordinals.",
        "",
        "Recommended orthogonal alternatives for primary entry conditions:",
        "  - Use lookbacks NOT in the forbidden list — e.g., 21, 42, 84, 126, 252",
        "  - Use volume-derived features or alt-data features from `orthogonal_features`",
        "    in the dossier (e.g., cot_net_noncomm_z52w, real_yield_5y_z252d)",
        "",
        "Below is the SAME dossier you received on the first attempt. Generate a NEW proposal",
        "that does not repeat the violations above. Output ONLY the JSON, no preamble.",
        "",
        "=== ORIGINAL DOSSIER PAYLOAD ===",
    ])
    return "\n".join(lines)
