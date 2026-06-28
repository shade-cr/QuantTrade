---
name: phase5-regime-methodology
description: Shared protocol for Phase 5 AI-as-hypothesis-generator spike. Defines the lookahead-bias firewall, regime taxonomy, proposal JSON schema, falsification rules, and adversarial review contract. Referenced by phase5-hypothesizer, phase5-devils-advocate, and phase5-skeptic agents. Read this before any Phase 5 reasoning task.
---

# Phase 5 Regime-Methodology Protocol

This skill encodes the shared knowledge that all Phase 5 agents must operate under. It is the source of truth — if an agent definition contradicts something here, this file wins.

## Mission

Use an LLM as the **source of trading hypotheses**, conditioned on regime labels. The pipeline + M3 audit (from Phase 4) is the immutable validator. The deliverable is methodology + reproducible artifacts. Finding a tradeable edge is NOT a success criterion. **Zero survivors is a successful spike** — it demonstrates the methodology rejects unsubstantiated hypotheses.

## Hard constraints all agents must honor

### 1. Lookahead-bias firewall

The LLM has a knowledge cutoff (Claude Opus 4.7: Jan 2026). The dataset goes through May 2026. Bars stamped before cutoff are at risk of memorization if any identifying information leaks into the prompt.

**Rules:**
- The Hypothesizer NEVER sees raw OHLCV, calendar dates, month names, year numbers, or named historical events.
- The Hypothesizer receives ONLY a `regime_stats` dossier of quantile-encoded aggregates plus the regime ID.
- Forbidden tokens in any LLM-generated `hypothesis` or `causal_story` field (regex-enforced by `phase5/lookahead_lint.py`):
  - `(19|20|21)\d{2}` — any year
  - Month names (January, February, ..., December and lowercase variants) WHEN paired with a year context
  - Named events: `COVID`, `GFC`, `Lehman`, `Powell`, `Yellen`, `Bernanke`, `Fed pivot`, `taper tantrum`, `flash crash`, `crypto winter`, `LUNA`, `FTX`, `silicon valley bank`, `SVB collapse`, `Brexit`, `Trump tariff`, `pandemic`
- The Skeptic (and ONLY the Skeptic) gets the dates — its job is partly to detect "this looks suspiciously crisis-shaped."
- Asset CLASS is disclosed to all agents (low memorization risk, useful prior): one of `fx`, `metal`, `crypto`, `commodity`, `equity_index`. Asset NAME (e.g., "XAUUSD") is disclosed to the orchestrator and to the Skeptic but NOT to the Hypothesizer (only `asset_class`).

**Why quantile-encoded aggregates:** A raw mean return of `+0.42%` for a regime is an absolute fingerprint that may pattern-match a specific historical episode. The same value expressed as "this regime's mean return sits at the 73rd percentile of all regimes' mean returns" is anonymized.

**Prior art**: `pipeline/llm_prompt.py` implemented exactly this discipline for Phase 4's LLM-as-primary backend (`_quantile_encode` at line 62, see docstring on line 14 about "this looks like the COVID crash"). Phase 5 mirrors that pattern at the regime aggregate level.

### 2. Falsification before evaluation

Every proposal MUST specify a `falsification_criterion` in its JSON payload BEFORE the pipeline runs. The criterion is locked at proposal-commit time; it cannot be loosened by a refinement. A refinement (`parent_proposal` link) MUST emit a strictly tighter criterion.

**Default falsification criterion** (proposals may override with strictly stricter values):
```
audit_class ∈ {"STABLE", "MARGINAL_2FOLDS"}
AND median_active_fold_sharpe ≥ 0.5
AND n_trades_total ≥ 50
```

### 3. Asset-agnostic agent definitions

All agent definitions in `.claude/agents/` are parameterized on `asset_class` (and sometimes `asset` for downstream artifacts), but contain NO hardcoded asset-specific logic. The same hypothesizer agent runs on XAUUSD today and XAGUSD tomorrow without modification.

## Regime taxonomy

Four mutually-exclusive regimes labeled at bar `t` using only data available at `t-1` (point-in-time).

**Trend axis** (uses asset's own close):
- `+1` (BULL) if 63-day rate-of-change > 0 AND 50-day MA > 200-day MA
- `-1` (BEAR) if both inverted
- `0` (TRANSITION) — reassigned to the prior regime label (sticky), so we have no third class in the output

**Vol axis** (uses asset's own realized vol):
- `HIGH` if realized 20-day vol ≥ 75th percentile of the trailing 5-year window
- `LOW` otherwise

Cross-product → 4 regimes:
- `BULL_QUIET`
- `BULL_STRESSED`
- `BEAR_QUIET`
- `BEAR_STRESSED`

**Sanity expectations** (XAU D1 22y, post-hysteresis-and-min-dwell):
- ≤30 regime episodes total
- Min episode ≥ `min_dwell_d1` (default 40 bars; sub-dwell flips absorbed into prior label)
- Three of four regimes ≥5% of total bars; the rarest regime MAY fall below 5% if structurally scarce. When `fraction_of_total_bars < 0.05`, the dossier sets `sample_sufficient = False` and the orchestrator tags any proposal whose `regime_scope` includes that regime as `diagnostic_only = True`. The reasoning is asset-agnostic: any regime with fewer than ~200 bars cannot reliably support per-fold M3 metrics at `n_trades >= 30`, regardless of which specific regime it is or which asset is being labeled.
- M3's `n_trades >= 30` per-fold gate prevents low-bar regimes from sneaking through with under-sampled signals.

**Regime gate mechanism** in evaluation: `sample_weight *= regime_mask` where `regime_mask` is 1 inside the proposal's `regime_scope` and 0 outside. This preserves the `avg_uniqueness` invariant — sample weights flow unchanged through fit/search/calibration; we're just multiplying by another deterministic mask.

### Anti-circularity invariant (B0034)

The regime taxonomy is constructed from `roc_63` (trend axis), `ma_50` + `ma_200` (MA-stack), and `rv_20` (vol axis). Any proposal whose `primary` is structurally equivalent to a regime-defining condition is tautological with the regime label and CANNOT carry an edge above the regime gate.

**Hard constraint (enforced by hypothesizer prompt; checked by devil's advocate)**:

- `primary_params.lookback == 63` (matches trend-axis ROC window) → forbidden for `momentum_zscore`/`cusum_filter` primaries inside any regime
- `(primary_params.fast, primary_params.slow) == (50, 200)` (matches MA-stack) → forbidden for `ema_cross` inside any regime
- `primary_params.rv_window == 20` as a sole entry filter → forbidden

The hypothesizer is instructed (see `.claude/agents/phase5-hypothesizer.md` §Anti-circularity rule) to use orthogonal windows for primary entry conditions: prefer 21, 42, 84, 126, 252 for momentum; pick asymmetric MA pairs not matching (50, 200).

Regime-defining features ARE allowed in `feature_overrides.add` for the meta-learner — within-regime variation of `roc_63` or `rv_20` carries genuine discriminative information beyond the gate's binary cut. But the `causal_story` must explicitly justify why the within-regime variation matters when ANY of `{roc_63, ma_50, ma_200, rv_20}` appears in `feature_overrides.add`.

If a proposal genuinely requires a regime-defining feature as the primary signal (cited literature, validated mechanism), it MUST tighten falsification (`n_trades_total_min: 100+`, `median_active_fold_sharpe_min: 0.8+`) to compensate for elevated circularity risk.

**Burn-in**: the trailing 5-year vol percentile window means the first `252 * 5 = 1260` bars produce NaN regime_id. These bars are not labeled and are excluded from any regime-conditional evaluation by the `regime_gate.mode = filter_events` mechanism. The PIT discipline (`min_periods = window`) prevents expanding-window or backfilled percentile behavior — no contamination from pre-burn-in bars. The dossier exposes `n_unlabeled_bars` so any reviewer can confirm the burn-in is acting as documented.

## Regime stats dossier schema

Path: `signals/regime_stats/<asset>_<freq_lower>/<regime_id>.json`  (B0070: e.g. `XAUUSD_d1/BULL_QUIET.json`; frequency is part of the dossier identity)

```json
{
  "asset_class": "metal",
  "regime_id": "BEAR_QUIET",
  "n_bars": 487,
  "n_episodes": 5,
  "median_episode_len_bars": 62,
  "fraction_of_total_bars": 0.18,
  "n_unlabeled_bars": 1281,
  "sample_sufficient": true,
  "sample_insufficient_reason": null,
  "regime_episode_ordinals": [0, 2, 4, 6, 8],
  "features_quantile_summary": {
    "<feature_name>": {
      "median_quantile": 0.42,
      "iqr_quantile_low": 0.21,
      "iqr_quantile_high": 0.68,
      "vs_other_regimes_rank": "lower|similar|higher"
    }
  },
  "return_distribution_quantile": {
    "median_quantile": 0.31,
    "iqr_low": 0.18,
    "iqr_high": 0.49
  },
  "primary_baseline_summary": {
    "<primary_name>": {
      "trade_count_per_year_q": 0.6,
      "hit_rate_q": 0.4,
      "median_per_trade_return_q": 0.5,
      "hit_rate_vs_other_regimes": "higher|lower|similar",
      "return_vs_other_regimes": "higher|lower|similar",
      "n_events": 142,
      "rankable": true,
      "low_confidence": false
    }
  }
}
// primary_baseline_summary _q fields are an ORDINAL rank over the <=4 regimes that
// fired the primary (often 2-3), NOT a percentile: q=1.0 means "highest of the firing
// regimes". null when rankable=false (fewer than 2 regimes fired it). low_confidence is
// driven ONLY by n_events < 30; rankable=false with low_confidence=false means
// measured-but-not-cross-rankable (use the figure as an absolute in-regime value).
```

All `_q`-suffixed and `_quantile` fields are quantile ranks (0.0–1.0) computed across ALL regimes for that asset. NO absolute values appear in the dossier.

**`regime_episode_ordinals` semantics (B0033)**: positions of THIS regime's episodes in the asset's **GLOBAL** episode timeline. Episodes alternate across regimes (BULL_QUIET ↔ BEAR_QUIET ↔ BULL_STRESSED ↔ ...), so for an asset with 10 total episodes where 5 are BEAR_QUIET, the BEAR_QUIET dossier may report `n_episodes: 5` AND `regime_episode_ordinals: [0, 2, 4, 6, 8]` simultaneously — these are positions in the **global timeline**, not local within-BEAR_QUIET indices.

Implication for hypothesizers and adversarial reviewers: when the proposal's `lookahead_shape_attestation.target_regime_episode_ordinals` lists values greater than or equal to `n_episodes`, this is **NOT** a bug — it just means the listed ordinals reference the global timeline. The DA agent must NOT flag ordinal > n_episodes-1 as an inconsistency; only flag if the listed ordinals are not a subset of the dossier's `regime_episode_ordinals`.

**Sample sufficiency**: `sample_sufficient = (n_bars >= 200) AND (n_episodes >= 3) AND (fraction_of_total_bars >= 0.05)`. When False, `sample_insufficient_reason` names the failing condition. Proposals whose `regime_scope` includes any regime with `sample_sufficient == False` are automatically tagged `diagnostic_only = True` by the orchestrator and skip PROMOTE eligibility regardless of M3 outcome. The hypothesizer is told explicitly that a regime is diagnostic-only and must acknowledge this in the `causal_story`.

## Proposal JSON schema

Path: `signals/proposals/<id>.json` (committed BEFORE pipeline evaluation)

```json
{
  "id": "20260526-XAUUSD-R3-cot",
  "asset": "XAUUSD",
  "asset_class": "metal",
  "regime_scope": ["BEAR_QUIET"],
  "hypothesis": "<2-3 sentences, 30-800 chars (HARD pre-flight limit, phase5/proposal.py:231-235); no dates, no named events; lint-enforced>",
  "causal_story": "<mechanism in 2-3 sentences, 30-800 chars (HARD pre-flight limit); same lookahead constraints. Target ~600-700 chars on first draft so DA-retry insertions stay within the cap.>",
  "primary": "ema_cross|momentum_zscore|cusum_filter|bollinger_meanrev|phase5_<id>",
  "primary_params": {},
  "custom_primary_sha256": "<hash if phase5_*>",
  "feature_overrides": {
    "add": [],
    "drop": []
  },
  "regime_gate": {
    "mode": "filter_events",
    "feature_added": true
  },
  "falsification_criterion": {
    "audit_class_in": ["STABLE", "MARGINAL_2FOLDS"],
    "median_active_fold_sharpe_min": 0.5,
    "n_trades_total_min": 50
  },
  "lookahead_attestation": {
    "checklist_version": "v1",
    "linter_passed": true
  },
  "parent_proposal": null,
  "git_sha_at_propose": "<...>"
}
```

The `id` field uses the format `<YYYYMMDD>-<asset>-<regime_short>-<3-letter-mnemonic>`. The YYYYMMDD here is the COMMIT date, not a data date — used purely for filesystem ordering.

## Custom primary contract

Path: `pipeline/primaries_phase5/<proposal_id>.py`

```python
import pandas as pd

def signal(ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: dict) -> pd.Series:
    """Returns a side series in {-1, 0, +1} indexed by ohlcv.index.

    -1 = short entry signal at this bar
     0 = no signal (no event)
    +1 = long entry signal at this bar
    NaN treated as 0.

    Must be pure (no I/O, no global state). Must be deterministic given
    (ohlcv, features, cfg).
    """
    ...
```

The proposal JSON's `custom_primary_sha256` field MUST match `sha256(open(path).read())` at proposal-commit time. The dispatcher patch in `scripts/run_backtest.py` enforces this.

## Adversarial review contract (devil's advocate)

Output schema (`signals/devils_advocate_reviews/<id>.json`):

```json
{
  "decision_type": "regime_taxonomy_boundaries|hypothesis_proposal_commit|custom_primary_code_commit|refinement_proposal_commit|promote_or_discard",
  "decision_payload_ref": "<path to the artifact under review>",
  "git_sha_under_review": "<...>",
  "verdict": "BLOCK | PROCEED_WITH_CAVEAT",
  "steel_man": "<the strongest argument FOR this decision, articulated as if defending it>",
  "objections": [
    {
      "severity": "high|medium|low",
      "claim": "<what could be wrong>",
      "evidence": "<concrete reference to the artifact under review>",
      "suggested_remediation": "<what would resolve this>"
    }
  ],
  "must_have_mods_before_proceed": [
    "<concrete change required if verdict=BLOCK>"
  ],
  "reviewer_notes": "<free text>"
}
```

**Hard rules:**
- MUST emit ≥1 entry in `objections`. The objections array can NEVER be empty. If the reviewer genuinely cannot find a substantive objection, emit one of severity `"low"` with the claim "no concrete objection found; this decision warrants confirmation by the daily skeptic." Vacuous boilerplate is flagged by the skeptic as rubber-stamping.
- `verdict` ∈ {`BLOCK`, `PROCEED_WITH_CAVEAT`}. The string `"APPROVE"` is NOT a valid verdict. The best-case outcome is `PROCEED_WITH_CAVEAT` with documented caveats.
- `verdict` may be `PROCEED_WITH_CAVEAT` only if no `severity: "high"` objection remains unaddressed.
- MUST include a `steel_man` — the strongest argument FOR the decision, articulated as if defending it. This prevents straw-manning.
- The reviewer does NOT have access to raw OHLCV or to the orchestrator's session context. It has read access to: the decision payload, prior reviews on the same proposal lineage, the regime_stats dossier for the relevant asset/regime, and the audit-of-audit findings.

## Skeptic review contract

The daily skeptic runs in clean context over the day's batch of artifacts. It is NOT a per-decision reviewer (that's the devil's advocate). Its job is to identify **methodology drift** across decisions.

Output: `signals/skeptic_reviews/<YYYYMMDD>.md` — free-form markdown with at minimum:
1. **Audit of the day's adversarial reviews**: are devil's advocate objections substantive or vacuous? Any rubber-stamping?
2. **Methodology drift**: does today's batch represent a coherent extension of the prior day's work, or has the search wandered?
3. **Hypothesis pattern signatures**: do today's proposals all share a hidden assumption that the day's audit verdicts can't rule out?
4. **Lookahead pattern smell**: do any proposals "feel" like memorized historical episodes despite passing the linter?
5. **At least one substantive criticism** with a suggested remediation.

The skeptic, like the devil's advocate, cannot emit unconditional approval. The minimum positive verdict is "no actionable concerns identified today; proceed with daily-skeptic-confirmed status."

## Orchestrator contract (Python `phase5/orchestrate.py`)

The orchestrator drives the loop deterministically. It does NO LLM reasoning itself. At each LLM decision point it:

1. Builds a structured payload (regime dossier, decision under review, etc.) and writes it to `phase5/runtime/next_action.json`.
2. Pauses until the corresponding `phase5/runtime/agent_response.json` is written (by the parent Claude Code session invoking the relevant agent via the Agent tool).
3. Validates the response against the expected schema. If invalid, emits an error and pauses again.
4. Persists the response to its canonical location (`signals/proposals/<id>.json` or `signals/devils_advocate_reviews/<id>.json` or `signals/skeptic_reviews/<YYYYMMDD>.md`).
5. Continues the loop.

After a `BLOCK` verdict from the devil's advocate, the orchestrator MUST either:
- (a) regenerate the decision payload (modify proposal text, rework regime boundaries, etc.) and re-submit for review, OR
- (b) emit a `reviewer_response.md` alongside the artifact that refutes each high-severity objection with concrete evidence. The response is visible to the daily skeptic.

The orchestrator NEVER auto-promotes past a BLOCK without one of the above.

## Reproducibility requirements

Every persisted artifact (proposal, review, audit result) MUST include `git_sha_at_propose` (or equivalent timestamp+sha pair). Replays of the same proposal with the same git_sha and same data files must produce identical outputs — this is enforced by `random_seed: 42` in the pipeline config.

Agent definitions live in `.claude/agents/` and are version-controlled. The orchestrator records the `git_sha_at_propose` so any replay can be re-run against the same agent version.

## File layout (canonical paths)

```
.claude/
  agents/phase5-hypothesizer.md
  agents/phase5-devils-advocate.md
  agents/phase5-skeptic.md
  skills/phase5-regime-methodology/SKILL.md          (this file)

pipeline/
  regimes.py                                          (deterministic labeler)
  primaries_phase5/<proposal_id>.py                   (per-custom-primary code)

phase5/
  proposal.py                                         (Pydantic schema)
  lookahead_lint.py                                   (regex enforcement)
  regime_stats.py                                     (dossier builder)
  orchestrate.py                                      (CLI driver)
  run_proposal.py                                     (single-proposal eval+audit)
  devils_advocate_dispatch.py                         (helper)
  audit_audit.py                                      (5-probe harness)
  runtime/                                            (transient orchestrator state)

data/
  regimes/<asset>_<freq_lower>_regimes.parquet     (D1: XAUUSD_d1_regimes.parquet; freq is lowercase)

signals/
  regime_stats/<asset>_<freq_lower>/<regime_id>.json     (D1: XAUUSD_d1/BULL_QUIET.json)
  proposals/<id>.json
  devils_advocate_reviews/<id>.json
  audit_results/<id>.json
  skeptic_reviews/<YYYYMMDD>.md
  index.csv                                           (single source of truth)

cache/alt/
  gld_holdings.parquet                                (Day 2.5 add)
```

## Threshold rule (B0155, 2026-06-11)

The audit's meta-probability threshold is governed by the proposal field `threshold_rule`:

- **`fixed_0.50`** (default) — the pre-B0155 behavior, bit-for-bit: the verdict aggregates the
  threshold grid at 0.50 and the per-episode gate reads `strategy_pnl_threshold50.parquet`.
- **`ev_breakeven_v1`** — the audit evaluates at the pre-registered EV-breakeven threshold

  ```
  p* = (sl_atr_mult + C_ATR + LAMBDA_MARGIN * (tp_atr_mult + sl_atr_mult)) / (tp_atr_mult + sl_atr_mult)
  ```

  where `tp_atr_mult` / `sl_atr_mult` come from the proposal's `barrier_geometry_attestation`
  (locked at commit time) and `C_ATR = 0.10` (round-trip cost in ATR units) and
  `LAMBDA_MARGIN = 0.05` (safety margin over breakeven) are **GLOBAL methodology constants**
  defined in `phase5/proposal.py`. They are amendable ONLY by a spec change to this section —
  **never per proposal**; a per-proposal cost or margin would be a threshold-shopping channel.
  A proposal MAY carry a precomputed `p_star`; `validate()` recomputes from inputs and rejects
  on mismatch (tolerance 1e-9).

**Rationale (Elkan 2001 cost-ratio theorem; AFML ch.3/10).** A fixed 0.50 threshold is correct
only for symmetric payoffs. Triple-barrier geometry with `tp=3 / sl=1` has an EV breakeven of
`p* = (1 + 0.10 + 0.05*4) / 4 = 0.325` — an honestly calibrated probability of 0.40 is a
positive-EV trade that the payoff-blind 0.50 cut discards. The 2026-06-11 batch produced 4/4
NO_FIRE verdicts precisely because calibrated probabilities in [0.25, 0.48] containing
positive-EV trades were discarded wholesale.

**Diagnostic grid.** For `ev_breakeven_v1` proposals the threshold grid in the transient config
is re-centered at p* — `(p*, p*+0.05, p*+0.10, p*+0.15)` — so grid rows exist at exactly p*.
The grid remains **diagnostic-only**: exactly ONE threshold per audit (the effective threshold)
drives the verdict, so the DSR trials count does not widen.

**Verdict continuity.** `fixed_0.50` verdicts recorded before 2026-06-11 STAND as recorded —
the rule change does not retroactively reinterpret them. A re-audit of a previously falsified
proposal under `ev_breakeven_v1` is a NEW audit entry that increments the program-level trials
count (it is an additional trial, and the DSR bookkeeping must see it as one).
