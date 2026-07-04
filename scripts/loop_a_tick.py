"""Loop A v1 — single-tick orchestrator (Python staging half).

This module implements the deterministic Python steps of one Loop A tick:

  1. Read signals/loop_a_state.json
  2. Pick next regime to explore via LRU (skip sample_insufficient regimes)
  3. Generate a proposal_id_hint
  4. Stage next_action_<id>.json for the phase5-hypothesizer agent
  5. Return the staging info to the caller (the /loop-a-tick skill)

The Claude Code session driving the tick then:
  - Reads phase5/runtime/next_action_<id>.json
  - Dispatches phase5-hypothesizer via Agent tool with that payload
  - Writes the response to signals/proposals/<id>.json
  - Calls this module again with `stage_review` to stage DA review
  - Dispatches phase5-devils-advocate
  - Calls this module's `record_tick` to write final state + report entry

State machine per tick (driven by the sibling Claude Code session):

  ┌─────────────────────────────────────────────────────────────────┐
  │ stage_propose  →  [dispatch hypothesizer]  →  stage_review      │
  │      ↓                                              ↓           │
  │  next_action for hypothesizer                  next_action for  │
  │                                                devils_advocate  │
  │                                                     ↓           │
  │                                          [dispatch DA agent]    │
  │                                                     ↓           │
  │                                              run_preflight      │
  │                                                     ↓           │
  │                                              record_tick        │
  └─────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SIGNALS_DIR = REPO_ROOT / "signals"
STATE_PATH = SIGNALS_DIR / "loop_a_state.json"
REGIME_STATS_DIR = SIGNALS_DIR / "regime_stats"
RUNTIME_DIR = REPO_ROOT / "phase5" / "runtime"
RESULTS_DIR = REPO_ROOT / "results" / "loop_a"

# B0104: how many tick summaries the bounded `recent` list keeps (spec §Component 3).
RECENT_K = 20

from pipeline.util.state_store import BoundedIndex, append_archive_jsonl  # noqa: E402
from pipeline.primary_contracts import (  # noqa: E402
    primary_param_schema_for_payload,
    normalize_primary_params,
    PrimaryParamError,
)


def _record_to_index(state: dict, *, outcome: str, summary: dict,
                     asset: str, frequency: str, regime: str, last_ticked: str) -> None:
    """B0104 bounded-index mutations on the in-memory `state` dict.

    Wraps `state` in a BoundedIndex (the projection IS this same dict, written
    by save_state) and applies the three context-safe mutations the spec
    prescribes: bump_counter(outcome), push_recent(summary), and
    set_projection('lru_by_cell', cell_key, last_ticked). The full per-tick row
    is NOT stored here — it goes to the JSONL archive. Mutates `state` in place.
    """
    idx = BoundedIndex(STATE_PATH, K=RECENT_K)
    idx._data = state  # operate on the live state dict; save_state owns the write
    idx.bump_counter(outcome)
    idx.push_recent(summary)
    idx.set_projection("lru_by_cell", _cell_key(asset, frequency, regime), last_ticked)

# B0034 v2: how many lint-failure retries before declaring the tick exhausted.
# 1st attempt is the original stage_propose; counts 2, 3, 4 are retries.
MAX_HYPOTHESIZER_RETRIES = 3

# B0037: how many DA-BLOCK retries before declaring the tick terminally da_blocked.
# Retries fire only when DA returns BLOCK with >=1 high-severity objection;
# medium/low-only BLOCKs are not retried (they're already actionable enough
# without burning more LLM tokens).
MAX_DA_RETRIES = 2


def _norm_severity(objection: dict) -> str:
    """B0015: DA agents have emitted both {high,medium,low} and
    {fatal,major,minor}. Normalize at read time — never persist a count
    computed from a single vocabulary (tick 3 recorded 2 fatal objections
    as da_objections_high=0 and skipped the retry loop)."""
    return str(objection.get("severity", "")).strip().lower()


def _is_high_severity(objection: dict) -> bool:
    return _norm_severity(objection) in ("high", "fatal", "critical")


def _objection_text(objection: dict) -> str:
    """DA output has used both 'claim'/'claim_attacked' and carried the
    substance in 'objection' — render whichever is present so retry
    feedback never shows blank numbered items (observed tick 4)."""
    claim = objection.get("claim") or objection.get("claim_attacked") or ""
    body = objection.get("objection") or ""
    if claim and body:
        return f"{claim} — {body}"
    return claim or body


def _format_da_feedback(verdict: dict) -> str:
    """Render a devil's-advocate BLOCK verdict as natural-language feedback
    for the hypothesizer retry loop (B0037). Mirrors the style of
    phase5.lookahead_lint.format_lint_feedback but adapted to DA output:
    high-severity objections become hard constraints, mediums become
    suggestions, the steel_man is included so the agent sees the strongest
    argument FOR its previous attempt.
    """
    objs = verdict.get("objections", [])
    high = [o for o in objs if _is_high_severity(o)]
    medium = [o for o in objs if _norm_severity(o) in ("medium", "major")]
    must_haves = verdict.get("must_have_mods_before_proceed", [])
    steel_man = verdict.get("steel_man", "").strip()

    lines = [
        "PREVIOUS ATTEMPT PASSED THE LINT BUT WAS BLOCKED BY THE DEVIL'S ADVOCATE.",
        "",
    ]
    if steel_man:
        lines.extend([
            "Steel-man of your previous attempt (the strongest argument FOR it):",
            f"  {steel_man[:500]}{'...' if len(steel_man) > 500 else ''}",
            "",
        ])
    if high:
        lines.append("HIGH-severity objections (MUST be addressed in retry):")
        for i, o in enumerate(high, 1):
            lines.append(f"  {i}. {_objection_text(o)[:400]}")
            evidence = o.get("evidence", "")
            if evidence:
                lines.append(f"     evidence: {evidence[:300]}")
            remediation = o.get("suggested_remediation", "")
            if remediation:
                lines.append(f"     remediation: {remediation[:300]}")
        lines.append("")
    if must_haves:
        lines.append("Mandatory mods (verbatim from DA):")
        for i, m in enumerate(must_haves, 1):
            lines.append(f"  {i}. {m}")
        lines.append("")
    if medium:
        lines.append("Medium-severity objections (address if possible):")
        for o in medium:
            lines.append(f"  - {_objection_text(o)[:200]}")
        lines.append("")
    lines.extend([
        "Constraints carried forward from the anti-circularity lint (B0034):",
        "  - Do NOT use lookback=63, ema_cross (50,200), or rv_window=20",
        "  - If you include regime-defining features in feature_overrides.add,",
        "    justify within-regime variation in causal_story",
        "",
        "Below is the SAME dossier you received earlier. Generate a NEW proposal",
        "that addresses the high-severity objections above. Output ONLY the JSON, no preamble.",
        "",
        "=== ORIGINAL DOSSIER PAYLOAD ===",
    ])
    return "\n".join(lines)


def _rel(path: Path) -> str:
    """Repo-relative path string for reporting, falling back to the absolute
    path when `path` is outside the repo (e.g. a redirected test tmp dir)."""
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _migrate_state(state: dict) -> dict:
    """B0070: make Loop A state frequency-aware, idempotently.

    - asset_scope: wrap bare ticker strings as {"asset": t, "frequency": "D1"}.
    - regime_history[*]: backfill missing "frequency" with "D1".
    - current_tick (if non-null): backfill missing "frequency" with "D1".
    Safe to run on every load; already-migrated state is returned unchanged.
    """
    scope = state.get("asset_scope", [])
    state["asset_scope"] = [
        {"asset": s, "frequency": "D1"} if isinstance(s, str) else s
        for s in scope
    ]
    for entry in state.get("regime_history", []):
        entry.setdefault("frequency", "D1")
    ct = state.get("current_tick")
    if isinstance(ct, dict):
        ct.setdefault("frequency", "D1")
    return state


def load_state() -> dict:
    if not STATE_PATH.exists():
        raise FileNotFoundError(
            f"Loop A state not found at {STATE_PATH}. Create it via /capture + initial state."
        )
    return _migrate_state(json.loads(STATE_PATH.read_text(encoding="utf-8")))


def save_state(state: dict) -> None:
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _load_dossier(asset: str, frequency: str, regime: str) -> Optional[dict]:
    from phase5.asset_registry import dossier_dirname
    path = REGIME_STATS_DIR / dossier_dirname(asset, frequency) / f"{regime}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_key(asset: str, frequency: str, regime: str) -> str:
    """The bounded-projection key for lru_by_cell: 'ASSET|FREQ|REGIME'."""
    return f"{asset}|{frequency}|{regime}"


def _lru_by_cell(state: dict) -> dict[str, str]:
    """Return the bounded LRU projection (cell_key -> max last_ticked).

    B0104: the projection IS the state; we read it directly (O(24)) instead of
    scanning regime_history. For backward-compat with v1 state files that predate
    the projection (no `lru_by_cell` key), derive it once from `regime_history`
    if present — the parity test proves both paths give identical picks.
    """
    proj = state.get("lru_by_cell")
    if isinstance(proj, dict):
        return proj
    derived: dict[str, str] = {}
    for entry in state.get("regime_history", []):
        key = _cell_key(entry["asset"], entry.get("frequency", "D1"), entry["regime"])
        if key not in derived or entry["last_ticked"] > derived[key]:
            derived[key] = entry["last_ticked"]
    return derived


def _regime_lru_pick(state: dict) -> tuple[str, str, str]:
    """Pick (asset, frequency, regime) via least-recently-explored, skipping
    sample_insufficient. Raises if no eligible cell exists.

    B0104: reads the bounded `lru_by_cell` projection (<= asset_scope x
    regime_scope = 24 cells) directly rather than scanning the full per-tick
    `regime_history`. Result is IDENTICAL to the old scan (guarded by
    tests/phase5/test_b0104_lru_parity.py)."""
    lru_by_cell = _lru_by_cell(state)

    eligible: list[tuple[str, str, str, str]] = []
    for cell in state["asset_scope"]:
        asset, frequency = cell["asset"], cell["frequency"]
        for regime in state["regime_scope"]:
            dossier = _load_dossier(asset, frequency, regime)
            if dossier is None:
                continue
            if not dossier.get("sample_sufficient", True):
                continue
            last = lru_by_cell.get(_cell_key(asset, frequency, regime), "")
            eligible.append((last, asset, frequency, regime))

    if not eligible:
        raise RuntimeError(
            f"No eligible (asset, frequency, regime) cells. Check signals/regime_stats/ "
            f"dossiers and asset_scope/regime_scope in {STATE_PATH}."
        )
    eligible.sort()
    _, asset, frequency, regime = eligible[0]
    return asset, frequency, regime


def _proposal_id_hint(asset: str, frequency: str, regime: str, tick_number: int) -> str:
    return (datetime.now(timezone.utc).strftime("%Y%m%d")
            + f"-{asset}-{frequency}-{regime[:8]}-T{tick_number:03d}")


def stage_propose() -> dict:
    """Pick next regime + stage hypothesizer next_action. Returns staging info dict."""
    state = load_state()
    if state.get("current_tick") is not None:
        raise RuntimeError(
            f"Refusing to start new tick: current_tick is in stage "
            f"{state['current_tick'].get('stage')!r}. Either finish it (run the appropriate "
            f"next stage), or reset it by setting current_tick=null in "
            f"signals/loop_a_state.json."
        )
    asset, frequency, regime = _regime_lru_pick(state)
    dossier = _load_dossier(asset, frequency, regime)
    if dossier is None:
        raise RuntimeError(f"Dossier missing for {asset} {frequency} {regime} after LRU pick")

    tick_number = state.get("tick_count", 0) + 1
    action_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_propose_{asset}_{regime}"
    id_hint = _proposal_id_hint(asset, frequency, regime, tick_number)

    payload = {
        "agent": "phase5-hypothesizer",
        "action_id": action_id,
        "input": {
            "asset_class": dossier["asset_class"],
            "regime_id": regime,
            "regime_stats_dossier": dossier,
            "available_features": list(dossier["features_quantile_summary"].keys()),
            # B0085: built-in primaries carry their exact canonical param schema so
            # the hypothesizer emits matching keys (phase5_custom stays free-form).
            "available_primaries": primary_param_schema_for_payload(),
            "id_hint": id_hint,
        },
        "output_path": f"signals/proposals/{id_hint}.json",
    }

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    action_path = RUNTIME_DIR / f"next_action_{action_id}.json"
    action_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    state["current_tick"] = {
        "tick_number": tick_number,
        "stage": "awaiting_hypothesizer",
        "asset": asset,
        "frequency": frequency,
        "regime": regime,
        "action_id": action_id,
        "proposal_id_hint": id_hint,
        "next_action_path": str(action_path.relative_to(REPO_ROOT)),
        "started_at": _now_iso(),
    }
    save_state(state)

    return {
        "stage": "awaiting_hypothesizer",
        "asset": asset,
        "frequency": frequency,
        "regime": regime,
        "action_id": action_id,
        "proposal_id_hint": id_hint,
        "next_action_path": str(action_path.relative_to(REPO_ROOT)),
        "payload": payload,
    }


def stage_review(proposal_path: str) -> dict:
    """Stage DA next_action for a proposal that's been written by the hypothesizer.

    B0034: runs anti-circularity lint before staging DA. If the lint fails,
    the tick is recorded as `circularity_violation` and current_tick is reset
    so a fresh tick can be invoked. DA dispatch is skipped — there's no point
    asking DA to review a proposal we already know violates a hard invariant.
    """
    from phase5 import devils_advocate_dispatch as dad
    from phase5 import lookahead_lint

    state = load_state()
    frequency = state["current_tick"].get("frequency", "D1") if state.get("current_tick") else "D1"
    valid_stages = (
        "awaiting_hypothesizer",
        "awaiting_hypothesizer_retry",
        "awaiting_hypothesizer_da_retry",
    )
    if not state.get("current_tick") or state["current_tick"].get("stage") not in valid_stages:
        raise RuntimeError(
            f"State machine error: expected stage in {valid_stages}, got "
            f"{state.get('current_tick', {}).get('stage')!r}. Re-stage with stage_propose."
        )

    prop_path = Path(proposal_path)
    if not prop_path.is_absolute():
        prop_path = REPO_ROOT / prop_path
    if not prop_path.exists():
        raise FileNotFoundError(f"Proposal not found at {prop_path}")

    proposal = json.loads(prop_path.read_text(encoding="utf-8"))

    # B0039 (extended) — Loop-A hypothesizer proposals omit `asset` by design
    # (the agent is asset-blind to preserve the methodology firewall). The
    # Proposal dataclass requires `asset`, so the B0041 fail-fast validate()
    # below — and run_preflight downstream — both need it injected. Inject from
    # current_tick state here, the first place a Proposal object is constructed.
    # Idempotent: skip if already present. (run_preflight repeats this as a
    # redundant safety net for proposals that bypass stage_review.)
    if not proposal.get("asset"):
        proposal["asset"] = state["current_tick"]["asset"]
        prop_path.write_text(json.dumps(proposal, indent=2), encoding="utf-8")

    # B0085 — built-in-primary param-contract gate. The hypothesizer is now given
    # each built-in primary's canonical param schema, but validate here too so a
    # contract violation surfaces at review time with a clear message rather than
    # as an opaque KeyError inside the audit subprocess. Custom phase5_* primaries
    # pass through untouched. Same normalizer the audit path (build_transient_config)
    # uses, so the two enforcement points can never disagree.
    try:
        normalize_primary_params(proposal.get("primary", ""), proposal.get("primary_params", {}))
    except PrimaryParamError as e:
        raise RuntimeError(
            f"B0085 param-contract violation in {proposal.get('id')!r}: {e} "
            f"Fix primary_params to use the canonical keys from available_primaries "
            f"and re-run stage_review."
        ) from e

    # Load the current tick's dossier once: it feeds both the anti-circularity
    # lint (B0044 episode-ordinal subset rule) and the DA decision payload
    # (B0043 dossier threading). Same regime-aggregate, point-in-time dossier
    # the hypothesizer saw — firewall-safe.
    review_dossier = _load_dossier(
        state["current_tick"]["asset"], frequency, state["current_tick"]["regime"]
    )

    # B0034 — anti-circularity lint: short-circuit BEFORE DA dispatch
    circ = lookahead_lint.lint_anti_circularity(proposal, dossier=review_dossier)
    if not circ.passed:
        retry_count = state["current_tick"].get("retry_count", 0)
        lint_hits_payload = [
            {"rule": h.rule, "field": h.field, "match": h.match}
            for h in circ.hits
        ]
        # Track lint history across retries for the daily report
        history = state["current_tick"].setdefault("lint_hits_history", [])
        history.append({
            "retry_count": retry_count,
            "proposal_id": proposal.get("id"),
            "hits": lint_hits_payload,
        })

        if retry_count < MAX_HYPOTHESIZER_RETRIES:
            # B0034 v2 — re-prompt the hypothesizer with explicit lint feedback
            new_retry_count = retry_count + 1
            new_action_id = (
                datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                + f"_propose_retry{new_retry_count}_{state['current_tick']['asset']}_{state['current_tick']['regime']}"
            )
            asset_inner = state["current_tick"]["asset"]
            regime_inner = state["current_tick"]["regime"]
            tick_n = state["current_tick"]["tick_number"]
            da_n = state["current_tick"].get("da_retry_count", 0)
            base_id = datetime.now(timezone.utc).strftime("%Y%m%d") + f"-{asset_inner}-{frequency}-{regime_inner[:8]}-T{tick_n:03d}"
            # Include DA-retry suffix to avoid id collision when a DA-retry's proposal
            # itself triggers lint-retry. Format: T<tick>D<da>R<lint>.
            suffix = f"D{da_n}R{new_retry_count}" if da_n > 0 else f"R{new_retry_count}"
            new_id_hint = f"{base_id}{suffix}"

            # Rebuild dossier payload from current dossier on disk
            asset = state["current_tick"]["asset"]
            regime = state["current_tick"]["regime"]
            dossier = _load_dossier(asset, frequency, regime)
            if dossier is None:
                raise RuntimeError(f"Dossier disappeared during retry: {asset}/{frequency}/{regime}")
            original_payload_input = {
                "asset_class": dossier["asset_class"],
                "regime_id": regime,
                "regime_stats_dossier": dossier,
                "available_features": list(dossier["features_quantile_summary"].keys()),
                # B0085: built-in primaries carry their exact canonical param schema
                # so the hypothesizer emits matching keys (phase5_custom free-form).
                "available_primaries": primary_param_schema_for_payload(),
                "id_hint": new_id_hint,
            }

            feedback_preamble = lookahead_lint.format_lint_feedback(circ.hits)
            # The hypothesizer agent receives the preamble + the JSON. The
            # agent expects JSON but ignores leading natural-language text.
            retry_payload = {
                "agent": "phase5-hypothesizer",
                "action_id": new_action_id,
                "feedback_preamble": feedback_preamble,
                "input": original_payload_input,
                "output_path": f"signals/proposals/{new_id_hint}.json",
            }

            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
            new_action_path = RUNTIME_DIR / f"next_action_{new_action_id}.json"
            new_action_path.write_text(
                json.dumps(retry_payload, indent=2, default=str), encoding="utf-8"
            )

            state["current_tick"]["stage"] = "awaiting_hypothesizer_retry"
            state["current_tick"]["retry_count"] = new_retry_count
            state["current_tick"]["action_id"] = new_action_id
            state["current_tick"]["proposal_id_hint"] = new_id_hint
            state["current_tick"]["next_action_path"] = str(new_action_path.relative_to(REPO_ROOT))
            state["current_tick"]["last_lint_hits"] = lint_hits_payload
            save_state(state)

            return {
                "stage": "awaiting_hypothesizer_retry",
                "retry_count": new_retry_count,
                "max_retries": MAX_HYPOTHESIZER_RETRIES,
                "previous_proposal_id": proposal.get("id"),
                "new_proposal_id_hint": new_id_hint,
                "next_action_path": str(new_action_path.relative_to(REPO_ROOT)),
                "lint_hits": lint_hits_payload,
                "feedback_preamble_chars": len(feedback_preamble),
                "next_step": (
                    f"Re-dispatch phase5-hypothesizer with the new payload "
                    f"(prepend feedback_preamble to the JSON). Save to output_path. "
                    f"Then call stage_review again on the new proposal."
                ),
            }

        # Retries exhausted — record as terminal circularity_violation
        state["current_tick"]["stage"] = "circularity_violation_exhausted"
        state["current_tick"]["proposal_path"] = str(prop_path.relative_to(REPO_ROOT))
        state["current_tick"]["lint_hits"] = lint_hits_payload
        state["current_tick"]["completed_at"] = _now_iso()
        save_state(state)
        return {
            "stage": "circularity_violation_exhausted",
            "proposal_id": proposal.get("id"),
            "retry_count": retry_count,
            "max_retries": MAX_HYPOTHESIZER_RETRIES,
            "lint_hits": lint_hits_payload,
            "next_step": "run `record_tick`; the hypothesizer failed lint 4 times in a row on this regime.",
        }

    # B0041 — fail-fast on schema violations (e.g. hypothesis/causal_story length
    # outside [30, 800]) BEFORE dispatching DA. Pre-flight catches the same
    # errors, but only after a full DA subprocess round-trip — that's wasted
    # compute and a misleading "PROCEED_WITH_CAVEAT" verdict on the daily
    # report. State is unchanged on raise so the caller can re-stage with a
    # trimmed proposal at the same `awaiting_hypothesizer*` checkpoint.
    from phase5.proposal import load_proposal, ProposalValidationError
    try:
        load_proposal(prop_path).validate()
    except ProposalValidationError as e:
        raise RuntimeError(
            f"Proposal {proposal.get('id')} fails schema validation before DA dispatch: {e}. "
            f"Re-dispatch the hypothesizer with the same payload; instruct it to trim "
            f"the offending field to the 30-800 char range (target ~600-700 for causal_story "
            f"to leave room for DA-retry insertions). Then call stage_review again on the new proposal."
        ) from e

    action_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_review_loop_a"
    # B0043 — thread the current tick's dossier into the DA payload. Without it
    # the DA receives regime_stats_dossier={} and CANNOT verify any statistic the
    # proposal's causal_story cites, so every dossier-grounded objection becomes
    # an unfixable high-severity BLOCK that burns the whole DA-retry budget. The
    # DA is a reviewer (it already sees the asset name in the proposal), so giving
    # it the same regime-aggregate, point-in-time dossier the hypothesizer saw is
    # firewall-safe — it strengthens review rather than leaking lookahead.
    # (review_dossier is loaded once near the top of stage_review and reused.)
    da_payload = dad.make_decision_payload(
        decision_type="hypothesis_proposal_commit",
        decision_payload=proposal,
        regime_stats_dossier=review_dossier,
    )
    full_payload = {
        "agent": "phase5-devils-advocate",
        "action_id": action_id,
        "input": da_payload,
        "output_path": f"signals/devils_advocate_reviews/{proposal['id']}.json",
    }
    action_path = dad.submit_for_review(full_payload, action_id)
    if not action_path.is_absolute():
        action_path = (REPO_ROOT / action_path).resolve()

    state["current_tick"]["stage"] = "awaiting_devils_advocate"
    state["current_tick"]["proposal_path"] = str(prop_path.relative_to(REPO_ROOT))
    state["current_tick"]["da_action_id"] = action_id
    state["current_tick"]["da_next_action_path"] = str(action_path.relative_to(REPO_ROOT))
    save_state(state)

    return {
        "stage": "awaiting_devils_advocate",
        "proposal_id": proposal["id"],
        "da_next_action_path": str(action_path.relative_to(REPO_ROOT)),
        "payload": full_payload,
    }


def run_preflight(proposal_path: str, verdict_path: str) -> dict:
    """Run pre-flight check + record tick. Caller has already dispatched DA and saved verdict."""
    import subprocess
    state = load_state()
    frequency = state["current_tick"].get("frequency", "D1") if state.get("current_tick") else "D1"
    if not state.get("current_tick") or state["current_tick"].get("stage") != "awaiting_devils_advocate":
        raise RuntimeError(
            f"State machine error: expected stage 'awaiting_devils_advocate', got "
            f"{state.get('current_tick', {}).get('stage')!r}."
        )

    prop_path = Path(proposal_path)
    if not prop_path.is_absolute():
        prop_path = REPO_ROOT / prop_path
    verd_path = Path(verdict_path)
    if not verd_path.is_absolute():
        verd_path = REPO_ROOT / verd_path

    if not verd_path.exists():
        raise FileNotFoundError(f"DA verdict not found at {verd_path}")
    verdict = json.loads(verd_path.read_text(encoding="utf-8"))

    result: dict = {
        "proposal_id": json.loads(prop_path.read_text(encoding="utf-8"))["id"],
        "da_verdict": verdict["verdict"],
        "da_objections_total": len(verdict.get("objections", [])),
        "da_objections_high": sum(1 for o in verdict.get("objections", []) if _is_high_severity(o)),
    }

    if verdict["verdict"] == "BLOCK":
        # B0037 — DA-feedback retry loop: if there's a high-severity objection
        # AND we haven't burned our DA-retry budget, re-dispatch the hypothesizer
        # with the DA's objections as feedback.
        has_high = any(_is_high_severity(o) for o in verdict.get("objections", []))
        da_retry_count = state["current_tick"].get("da_retry_count", 0)
        if has_high and da_retry_count < MAX_DA_RETRIES:
            new_da_retry_count = da_retry_count + 1
            asset_inner = state["current_tick"]["asset"]
            regime_inner = state["current_tick"]["regime"]
            tick_n = state["current_tick"]["tick_number"]
            base_id = datetime.now(timezone.utc).strftime("%Y%m%d") + f"-{asset_inner}-{frequency}-{regime_inner[:8]}-T{tick_n:03d}"
            new_id_hint = f"{base_id}D{new_da_retry_count}"
            new_action_id = (
                datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                + f"_propose_da_retry{new_da_retry_count}_{asset_inner}_{regime_inner}"
            )

            dossier = _load_dossier(asset_inner, frequency, regime_inner)
            if dossier is None:
                raise RuntimeError(f"Dossier disappeared during DA-retry: {asset_inner}/{frequency}/{regime_inner}")
            original_payload_input = {
                "asset_class": dossier["asset_class"],
                "regime_id": regime_inner,
                "regime_stats_dossier": dossier,
                "available_features": list(dossier["features_quantile_summary"].keys()),
                # B0085: built-in primaries carry their exact canonical param schema
                # so the hypothesizer emits matching keys (phase5_custom free-form).
                "available_primaries": primary_param_schema_for_payload(),
                "id_hint": new_id_hint,
            }

            feedback_preamble = _format_da_feedback(verdict)
            retry_payload = {
                "agent": "phase5-hypothesizer",
                "action_id": new_action_id,
                "feedback_preamble": feedback_preamble,
                "input": original_payload_input,
                "output_path": f"signals/proposals/{new_id_hint}.json",
            }

            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
            new_action_path = RUNTIME_DIR / f"next_action_{new_action_id}.json"
            new_action_path.write_text(
                json.dumps(retry_payload, indent=2, default=str), encoding="utf-8"
            )

            # Reset lint retry_count for the new proposal (lint retries are
            # independent of DA retries). Keep da_retry_count cumulative.
            state["current_tick"]["stage"] = "awaiting_hypothesizer_da_retry"
            state["current_tick"]["da_retry_count"] = new_da_retry_count
            state["current_tick"]["retry_count"] = 0  # reset lint retry budget per DA-retry
            state["current_tick"]["proposal_id_hint"] = new_id_hint
            state["current_tick"]["action_id"] = new_action_id
            state["current_tick"]["next_action_path"] = str(new_action_path.relative_to(REPO_ROOT))
            da_history = state["current_tick"].setdefault("da_verdict_history", [])
            da_history.append({
                "da_retry_count": da_retry_count,
                "proposal_id": result["proposal_id"],
                "verdict": "BLOCK",
                "objections_high": result["da_objections_high"],
                "objections_total": result["da_objections_total"],
            })
            save_state(state)

            return {
                "stage": "awaiting_hypothesizer_da_retry",
                "da_retry_count": new_da_retry_count,
                "max_da_retries": MAX_DA_RETRIES,
                "previous_proposal_id": result["proposal_id"],
                "new_proposal_id_hint": new_id_hint,
                "next_action_path": str(new_action_path.relative_to(REPO_ROOT)),
                "feedback_preamble_chars": len(feedback_preamble),
                "next_step": (
                    f"Re-dispatch phase5-hypothesizer with the new payload "
                    f"(prepend feedback_preamble to the JSON). Save to output_path. "
                    f"Then call stage_review again on the new proposal."
                ),
            }

        result["preflight_status"] = "skipped_due_to_da_block"
    else:
        # B0039 — Loop-A hypothesizer proposals omit `asset` by design (the
        # agent is asset-blind to preserve the methodology firewall). The
        # downstream phase5.run_proposal requires `p.asset` to locate the
        # regime parquet / OHLCV CSV. Inject it here from current_tick state
        # before the subprocess call. Idempotent: skip if already present.
        prop_payload = json.loads(prop_path.read_text(encoding="utf-8"))
        if not prop_payload.get("asset"):
            prop_payload["asset"] = state["current_tick"]["asset"]
            prop_path.write_text(
                json.dumps(prop_payload, indent=2), encoding="utf-8"
            )
        cmd = [
            sys.executable, "-m", "phase5.run_proposal",
            "--proposal", str(prop_path),
            "--preflight-only",
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        result["preflight_returncode"] = proc.returncode
        result["preflight_stdout_tail"] = (proc.stdout or "")[-500:]
        result["preflight_stderr_tail"] = (proc.stderr or "")[-500:]
        result["preflight_status"] = "passed" if proc.returncode == 0 else "failed"

    state["current_tick"]["stage"] = "completed"
    state["current_tick"]["result"] = result
    state["current_tick"]["completed_at"] = _now_iso()
    save_state(state)
    return result


def record_tick() -> dict:
    """Finalize the current tick: append to regime_history, update counters, write report."""
    state = load_state()
    current = state.get("current_tick")
    if not current or current.get("stage") not in (
        "completed", "circularity_violation", "circularity_violation_exhausted"
    ):
        raise RuntimeError(
            f"State machine error: cannot record tick that is not 'completed', "
            f"'circularity_violation', or 'circularity_violation_exhausted'. "
            f"Current stage: {current.get('stage') if current else None!r}"
        )

    # B0034 + B0034 v2 — circularity-violation paths bypass DA/preflight bookkeeping
    if current["stage"] in ("circularity_violation", "circularity_violation_exhausted"):
        outcome = current["stage"]
        proposal_id = current.get("proposal_path", "").split("/")[-1].replace(".json", "")
        # B0104: full per-tick row -> day-partitioned JSONL archive (queryable,
        # never read whole into agent context). The bounded index keeps only the
        # counters + recent[K] + lru_by_cell projection.
        archive_row = {
            "asset": current["asset"],
            "frequency": current.get("frequency", "D1"),
            "regime": current["regime"],
            "tick_number": current["tick_number"],
            "proposal_id": proposal_id,
            "last_ticked": current["started_at"],
            "outcome": outcome,
            "retry_count": current.get("retry_count", 0),
            "lint_hits": current.get("lint_hits", []),
            "lint_hits_history": current.get("lint_hits_history", []),
        }
        append_archive_jsonl(RESULTS_DIR, _today_str(), archive_row)
        _record_to_index(
            state, outcome=outcome,
            summary={
                "tick_number": current["tick_number"],
                "asset": current["asset"],
                "frequency": current.get("frequency", "D1"),
                "regime": current["regime"],
                "proposal_id": proposal_id,
                "outcome": outcome,
                "retry_count": current.get("retry_count", 0),
                "last_ticked": current["started_at"],
            },
            asset=current["asset"], frequency=current.get("frequency", "D1"),
            regime=current["regime"], last_ticked=current["started_at"],
        )
        state["tick_count"] = current["tick_number"]
        state["last_tick_at"] = current["completed_at"]
        state["current_tick"] = None
        save_state(state)

        report_path = RESULTS_DIR / f"{_today_str()}.md"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        is_new = not report_path.exists()
        with report_path.open("a", encoding="utf-8") as f:
            if is_new:
                f.write(f"# Loop A — {_today_str()}\n\n")
            retry_count = current.get("retry_count", 0)
            f.write(
                f"## Tick {current['tick_number']:03d} — {current['completed_at']}\n"
                f"- **Asset / Regime**: {current['asset']} {current.get('frequency', 'D1')} / {current['regime']}\n"
                f"- **Proposal**: `{proposal_id}`\n"
                f"- **Outcome**: **{outcome}** (B0034 lint short-circuit, DA not dispatched)\n"
                f"- **Retries used**: {retry_count} / {MAX_HYPOTHESIZER_RETRIES}\n"
                f"- **Final lint hits**: {len(current.get('lint_hits', []))}\n"
            )
            for hit in current.get("lint_hits", []):
                f.write(f"  - `{hit['rule']}`: {hit['match']}\n")
            history = current.get("lint_hits_history", [])
            if len(history) > 1:
                f.write(f"- **Retry history**:\n")
                for h in history:
                    f.write(f"  - attempt {h['retry_count']}: {h['proposal_id']} — {len(h['hits'])} hits\n")
            f.write("\n")

        return {"tick_number": current["tick_number"], "outcome": outcome, "report_path": _rel(report_path)}

    result = current["result"]
    da_verdict = result["da_verdict"]
    preflight_status = result["preflight_status"]

    if da_verdict == "BLOCK":
        outcome = "da_blocked"
    elif preflight_status == "failed":
        outcome = "preflight_failed"
    elif preflight_status == "passed":
        outcome = "preflight_passed_pending_audit"
    elif preflight_status == "skipped_due_to_da_block":
        outcome = "da_blocked"
    else:
        outcome = "unknown"

    # B0104: full per-tick row -> day-partitioned JSONL archive; bounded index
    # keeps counters + recent[K] + lru_by_cell only (bump_counter replaces the
    # old scattered *_count top-level fields).
    archive_row = {
        "asset": current["asset"],
        "frequency": current.get("frequency", "D1"),
        "regime": current["regime"],
        "tick_number": current["tick_number"],
        "proposal_id": result["proposal_id"],
        "last_ticked": current["started_at"],
        "outcome": outcome,
        "da_verdict": da_verdict,
        "da_objections_high": result["da_objections_high"],
    }
    append_archive_jsonl(RESULTS_DIR, _today_str(), archive_row)
    _record_to_index(
        state, outcome=outcome,
        summary={
            "tick_number": current["tick_number"],
            "asset": current["asset"],
            "frequency": current.get("frequency", "D1"),
            "regime": current["regime"],
            "proposal_id": result["proposal_id"],
            "outcome": outcome,
            "da_verdict": da_verdict,
            "da_objections_high": result["da_objections_high"],
            "last_ticked": current["started_at"],
        },
        asset=current["asset"], frequency=current.get("frequency", "D1"),
        regime=current["regime"], last_ticked=current["started_at"],
    )
    state["tick_count"] = current["tick_number"]
    state["last_tick_at"] = current["completed_at"]
    state["current_tick"] = None
    save_state(state)

    report_path = RESULTS_DIR / f"{_today_str()}.md"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not report_path.exists()
    with report_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write(f"# Loop A — {_today_str()}\n\n")
        f.write(
            f"## Tick {current['tick_number']:03d} — {current['completed_at']}\n"
            f"- **Asset / Regime**: {current['asset']} {current.get('frequency', 'D1')} / {current['regime']}\n"
            f"- **Proposal**: `{result['proposal_id']}`\n"
            f"- **DA verdict**: {da_verdict} ({result['da_objections_high']} high / {result['da_objections_total']} total objections)\n"
            f"- **Pre-flight**: {preflight_status}\n"
            f"- **Outcome**: **{outcome}**\n\n"
        )

    return {"tick_number": current["tick_number"], "outcome": outcome, "report_path": _rel(report_path)}


def status() -> dict:
    """Return a quick state summary. B0104: counters live under `counters`
    (v2); fall back to the v1 scattered `*_count` fields if reading a pre-
    migration state."""
    state = load_state()
    counters = state.get("counters", {})
    return {
        "tick_count": state.get("tick_count", 0),
        "last_tick_at": state.get("last_tick_at"),
        "current_tick": state.get("current_tick"),
        "survivors": len(state.get("survivors", [])),
        "da_blocked": counters.get("da_blocked", state.get("da_blocked_count", 0)),
        "preflight_failed": counters.get("preflight_failed", state.get("preflight_failed_count", 0)),
        "recent_length": len(state.get("recent", [])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="loop_a_tick")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("stage_propose")
    p = sub.add_parser("stage_review")
    p.add_argument("--proposal", required=True)
    p = sub.add_parser("run_preflight")
    p.add_argument("--proposal", required=True)
    p.add_argument("--verdict", required=True)
    sub.add_parser("record_tick")

    args = ap.parse_args()
    if args.cmd == "status":
        out = status()
    elif args.cmd == "stage_propose":
        out = stage_propose()
    elif args.cmd == "stage_review":
        out = stage_review(args.proposal)
    elif args.cmd == "run_preflight":
        out = run_preflight(args.proposal, args.verdict)
    elif args.cmd == "record_tick":
        out = record_tick()
    else:
        return 2
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
