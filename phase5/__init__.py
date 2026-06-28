"""Phase 5 spike: AI-as-hypothesis-generator with regime conditioning.

The pipeline + M3 audit (from Phase 4) is the immutable validator. This
package wraps it with:
  - lookahead_lint: regex enforcement of no-dates/no-events rules
  - proposal: Pydantic schema for hypothesis proposals
  - regime_stats: per-regime aggregate dossier builder (quantile-encoded)
  - orchestrate: deterministic Python driver for the hypothesis loop
  - run_proposal: single-proposal evaluator + auditor
  - devils_advocate_dispatch: helper to format adversarial review payloads
  - audit_audit: 5-probe harness validating the M3 audit itself

See .claude/skills/phase5-regime-methodology/SKILL.md for the full protocol.
"""
