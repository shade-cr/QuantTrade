"""Phase 5 proposal schema (stdlib dataclasses + manual validation).

See `.claude/skills/phase5-regime-methodology/SKILL.md` for the canonical
schema and field semantics. This module enforces at commit time:
  - Required fields present
  - regime_scope is a non-empty list of valid regime IDs
  - falsification_criterion is at-least-as-strict as the default
  - lookahead_lint passes on narrative fields

A proposal is rejected at commit time if any of these fail. We use stdlib
dataclasses (NOT pydantic) to avoid adding a new dep for the spike.
"""
from __future__ import annotations
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from phase5.lookahead_lint import lint_proposal


REGIME_IDS = ("BULL_QUIET", "BULL_STRESSED", "BEAR_QUIET", "BEAR_STRESSED")
ASSET_CLASSES = ("fx", "metal", "crypto", "commodity", "equity", "equity_index")
PRIMARIES = ("ema_cross", "momentum_zscore", "cusum_filter", "bollinger_meanrev", "phase5_custom",
             "supervised_direct")
M3_CLASSES = ("STABLE", "MARGINAL_2FOLDS", "REGIME_LIMITED", "NOT_PROFITABLE",
              "1FOLD_CONCENTRATED", "MULTI_FOLD_BUT_LOW_N", "NO_FIRE")
PASS_CLASSES = ("STABLE", "MARGINAL_2FOLDS")
FAIL_CLASSES = ("REGIME_LIMITED", "NOT_PROFITABLE", "1FOLD_CONCENTRATED",
                "MULTI_FOLD_BUT_LOW_N", "NO_FIRE")

DEFAULT_FALSIFICATION = {
    "audit_class_in": ["STABLE", "MARGINAL_2FOLDS"],
    "median_active_fold_sharpe_min": 0.5,
    "n_trades_total_min": 50,
}

# add_feature (B0014): keep all events, expose in-scope membership to the meta
# as a 0/1 column — the reachable alternative when a minority-regime filter
# would collapse pooled effective-N (T004D1: 2,852 raw -> 521.7 effective).
REGIME_GATE_MODES = ("filter_events", "weight_events", "add_feature")

# --------------------------------------------------------------------------- #
# B0155 — pre-registered EV-breakeven threshold rule (Elkan 2001 cost-ratio
# theorem; AFML ch.3/10). A fixed 0.50 meta threshold is payoff-blind: it is
# correct ONLY for symmetric payoffs. With asymmetric triple-barrier geometry
# (e.g. tp=3 ATR / sl=1 ATR) the EV-breakeven win probability is ~0.33, so
# honest calibrated probabilities in [0.25, 0.48] containing positive-EV
# trades were discarded wholesale (4/4 NO_FIRE audits on 2026-06-11).
# --------------------------------------------------------------------------- #

THRESHOLD_RULES = ("fixed_0.50", "ev_breakeven_v1")

# B0161: the formula and its global constants moved to pipeline.thresholds —
# the single canonical home shared with the multi-asset orchestrator. The
# re-export keeps every existing `from phase5.proposal import ...` working and
# is pinned by tests/test_thresholds.py::test_phase5_reexports_canonical_formula.
from pipeline.thresholds import (  # noqa: E402  (re-export)
    C_ATR,
    LAMBDA_MARGIN,
    compute_p_star,
)


# --------------------------------------------------------------------------- #
# B0155 — proposal-time feature-existence gate (the B004v3 lesson: a primary /
# meta referencing a NONEXISTENT feature produced a structurally dead gate —
# all-NaN -> 0 events — that masqueraded as an honest falsification).
#
# KNOWN_TIER2_FEATURES is a hardcoded frozen list of every column emitted by
# the tier2 builders (D1 technical + H4 technical + macro with all optional
# series + session one-hots). It is kept honest by
# tests/phase5/test_b0155_ev_threshold.py::test_registry_regenerates_from_synthetic_build,
# which regenerates the set from a synthetic build and FAILS whenever tier2
# changes — forcing this list to be updated in the same commit.
# --------------------------------------------------------------------------- #

KNOWN_TIER2_FEATURES: frozenset[str] = frozenset({
    # D1 technical + vol-regime (pipeline.features.build_technical_features)
    "r_1", "r_5", "r_10", "r_20", "z_r20",
    "rsi_14", "macd_signal", "macd_hist",
    "atr_14_norm", "_atr_14", "bb_width_20",
    "rv_20", "rv_regime", "rv_term_structure", "ffd_logclose",
    "cs_spread_21",  # B0135: Corwin-Schultz high-low spread (liquidity)
    "volume_z42", "volume_pct_rank_21", "volume_rel_median_42",
    # B0147: GLD real-volume block (metals-only, config-gated alt-data)
    "gld_dvol_z42", "gld_amihud_z252",
    # H4 technical (pipeline.features._build_h4_technical)
    "r_1bar", "r_6bars", "r_24bars", "r_120bars", "z_r24bars",
    "bb_width_120bars", "rv_24bars",
    # Macro (pipeline.features.build_macro_features, optional series included)
    "dtwexbgs_close", "dxy_z252", "dxy_chg_5",
    "real_yield_5y", "real_yield_5y_chg_5", "real_yield_5y_z252d",
    "breakeven_5y", "nominal_5y_chg_5",
    "vix_level", "vix_chg_5",
    "us2y_level", "us2y_chg_5", "us_2y10y_spread",
    "umcsent_level", "umcsent_chg_3m",
    # Session one-hots (pipeline.features._build_session_one_hot)
    "session_london", "session_overlap", "session_ny",
})

# Regime-defining features + dossier alt-features the hypothesizer is shown by
# name (signals/regime_stats dossiers). Proposals legitimately reference these
# even though some are not tier2 columns verbatim.
DOSSIER_ALT_FEATURES: frozenset[str] = frozenset({
    "cot_net_noncomm_z52w", "real_yield_5y_z252d", "us_5y2y_z252",
    "vix_level", "vix_chg_5", "breakeven_5y_chg5", "dxy_z252",
})

# Heuristic for "this string param looks like a feature name" — used ONLY for
# the lenient (warning-level) scan of phase5_* custom primary_params.
_FEATURE_NAME_RE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")


def known_feature_registry() -> frozenset[str]:
    """Canonical known-feature registry for the proposal-time existence gate.

    Union of the frozen tier2 column list, the FEATURE_ALIASES keys (conceptual
    requests like "volume" that the audit satisfies via derived columns, B0149)
    and the regime/dossier alt-features.
    """
    from pipeline.features import FEATURE_ALIASES  # lazy: keeps proposal.py import-light
    return KNOWN_TIER2_FEATURES | frozenset(FEATURE_ALIASES) | DOSSIER_ALT_FEATURES


def _feature_name_known(name: str, registry: frozenset[str]) -> bool:
    """True iff `name` is in the registry, an alias key, or a trailing-*
    wildcard whose prefix matches at least one known feature (the same
    wildcard convention apply_primary_feature_blacklist implements)."""
    if name.endswith("*"):
        prefix = name[:-1]
        return any(f.startswith(prefix) for f in registry)
    return name in registry


class ProposalValidationError(ValueError):
    """Raised when a proposal violates the schema or commit-time rules."""


@dataclass
class FalsificationCriterion:
    audit_class_in: list[str] = field(
        default_factory=lambda: list(DEFAULT_FALSIFICATION["audit_class_in"])
    )
    median_active_fold_sharpe_min: float = DEFAULT_FALSIFICATION["median_active_fold_sharpe_min"]
    n_trades_total_min: int = DEFAULT_FALSIFICATION["n_trades_total_min"]
    # B0035 — cross-episode survival gate (sign test on per-episode net PnL).
    # When per_episode_survival_fraction is set (not None), the audit additionally
    # requires the strategy to be net-positive in >= ceil(fraction * n_active)
    # regime episodes, where an episode is "active" iff it has at least
    # per_episode_min_trades trades. This defends against a "survivor" whose
    # edge is driven by a single regime episode — the DA's recurring HIGH on
    # low-n_episodes regimes (e.g. BULL_QUIET n_episodes=3). The hypothesizer
    # auto-includes this when the dossier reports n_episodes < 5.
    per_episode_survival_fraction: Optional[float] = None
    per_episode_min_trades: int = 5
    per_episode_net_pnl_margin: float = 0.0
    # B0089 — Deflated Sharpe Ratio (DSR) HARD GATE on the promotion decision.
    # When dsr_min is set (not None), a candidate whose audited DSR (the
    # threshold-0.50 per-trade DSR of the model that drives the verdict, from
    # pipeline.metrics.deflated_sharpe_ratio, persisted to psr_dsr.json) is
    # < dsr_min is FALSIFIED and downgraded to NOT_PROFITABLE even if its raw
    # per-fold Sharpe and total trade count clear their floors. DSR deflates
    # the observed Sharpe by E[max Sharpe over N trials], so a strategy whose
    # edge is an artifact of selecting the best of many model/threshold/regime
    # configurations cannot promote. This subsumes the external
    # "OOS can't beat IS by >30%" heuristic — that heuristic is a crude proxy
    # for the multiple-testing inflation that DSR measures directly.
    #
    # Default None = backward-compatible (gate off, old behavior unchanged).
    # The recommended promotion floor is dsr_min=0.90: the methodology uses
    # DSR>=0.95 for full-tier Kelly sizing (pipeline.deployment), so a 0.90
    # promotion floor admits MARGINAL survivors for paper-trading while still
    # rejecting trial-deflated DSR~0 candidates, and reserves the stricter
    # 0.95 bar for the live capital-allocation tier.
    dsr_min: Optional[float] = None

    def validate(self) -> None:
        bad = [c for c in self.audit_class_in if c not in M3_CLASSES]
        if bad:
            raise ProposalValidationError(
                f"audit_class_in contains unknown classes: {bad}; allowed: {list(M3_CLASSES)}"
            )
        if any(c in FAIL_CLASSES for c in self.audit_class_in):
            raise ProposalValidationError(
                f"audit_class_in must NOT include failure classes; got {self.audit_class_in}"
            )
        if not isinstance(self.median_active_fold_sharpe_min, (int, float)):
            raise ProposalValidationError("median_active_fold_sharpe_min must be numeric")
        if not isinstance(self.n_trades_total_min, int):
            raise ProposalValidationError("n_trades_total_min must be int")
        # B0035 per-episode gate validation
        if self.per_episode_survival_fraction is not None:
            f = self.per_episode_survival_fraction
            if not isinstance(f, (int, float)) or not (0.0 < f <= 1.0):
                raise ProposalValidationError(
                    f"per_episode_survival_fraction must be in (0, 1]; got {f!r}"
                )
            if not isinstance(self.per_episode_min_trades, int) or self.per_episode_min_trades < 1:
                raise ProposalValidationError(
                    f"per_episode_min_trades must be a positive int; got {self.per_episode_min_trades!r}"
                )
            if not isinstance(self.per_episode_net_pnl_margin, (int, float)):
                raise ProposalValidationError("per_episode_net_pnl_margin must be numeric")
        # B0089 DSR gate validation — dsr_min, when set, is a probability in [0, 1].
        if self.dsr_min is not None:
            if not isinstance(self.dsr_min, (int, float)) or not (0.0 <= self.dsr_min <= 1.0):
                raise ProposalValidationError(
                    f"dsr_min must be a probability in [0, 1] (None to disable); got {self.dsr_min!r}"
                )


@dataclass
class FeatureOverrides:
    add: list[str] = field(default_factory=list)
    drop: list[str] = field(default_factory=list)


@dataclass
class RegimeGate:
    mode: str = "filter_events"
    feature_added: bool = True

    def validate(self) -> None:
        if self.mode not in REGIME_GATE_MODES:
            raise ProposalValidationError(f"regime_gate.mode={self.mode!r} not in {REGIME_GATE_MODES}")


@dataclass
class LookaheadAttestation:
    checklist_version: str = "v1"
    linter_passed: Optional[bool] = None


@dataclass
class BarrierGeometryAttestation:
    """Per Day-2 skeptic concern C, propagate project_barrier_geometry_root_cause
    from MEMORY.md into the proposal schema.

    Per-archetype R:R bounds enforced at proposal validation:

    - **Trend primaries** (ema_cross, momentum_zscore, phase5_*): R:R >= 2.5.
      Trend win rate is structurally ~30%; below 2.5 R:R the expectancy is
      negative. Corrected XAU D1 geometry is tp_atr_mult=3.0 / sl_atr_mult=1.0.
    - **Mean-reversion primaries** (bollinger_meanrev): 0.5 <= R:R <= 2.0.
      Mean-rev win rate is structurally ~55-60%; the break-even R:R sits
      in [0.67, 0.82]. R:R > 2.0 means letting winners run past the mean —
      that is trend-in-disguise, not mean-rev. R:R < 0.5 requires >67%
      win rate to clear break-even, which is overfit territory.
    - **cusum_filter**: event-based, accepts default (no archetype constraint).

    This attestation is recorded BY the proposer at draft time; the audit
    chain in run_proposal.py cross-checks it against the actual config the
    pipeline subprocess used.
    """
    tp_atr_mult: float = 3.0
    sl_atr_mult: float = 1.0
    rationale: str = "Default XAU D1 trend-geometry per project_barrier_geometry_root_cause memory."

    def validate(self, primary_name: str) -> None:
        if self.tp_atr_mult <= 0 or self.sl_atr_mult <= 0:
            raise ProposalValidationError(
                f"barrier_geometry_attestation values must be positive; "
                f"got tp={self.tp_atr_mult}, sl={self.sl_atr_mult}"
            )
        rr = self.tp_atr_mult / self.sl_atr_mult
        # B0088 — a phase5_* custom primary whose name explicitly signals
        # mean-reversion ("meanrev" substring, e.g. phase5_cmf_meanrev) is a
        # FADE archetype and must use the mean-rev R:R band, NOT the trend floor.
        # The CMF mean-reversion survivor (B0088) would otherwise be auditable
        # only with R:R >= 2.5 — a geometry this validator's own docstring calls
        # "trend-in-disguise" for a fade, which would mismeasure the edge. The
        # generic phase5_custom (and any other phase5_* that does not name itself
        # mean-reversion) stays on the trend floor, preserving the existing
        # test_phase5_custom_primary_treated_as_trend contract.
        is_phase5_meanrev = (
            primary_name.startswith("phase5_") and "meanrev" in primary_name.lower()
        )
        # Trend-following primaries require R:R >= 2.5
        if (
            primary_name in ("ema_cross", "momentum_zscore")
            or (primary_name.startswith("phase5_") and not is_phase5_meanrev)
        ):
            if rr < 2.5:
                raise ProposalValidationError(
                    f"barrier_geometry_attestation R:R={rr:.2f} < 2.5 for trend primary "
                    f"{primary_name!r}. Per project_barrier_geometry_root_cause memory, "
                    f"trend signals need R:R >= 2.5 to clear break-even at ~30% win rate."
                )
        # B0014 — mean-reversion primaries require 0.5 <= R:R <= 2.0
        # (bollinger_meanrev built-in + any phase5_* mean-reversion custom).
        elif primary_name == "bollinger_meanrev" or is_phase5_meanrev:
            if rr > 2.0:
                raise ProposalValidationError(
                    f"barrier_geometry_attestation R:R={rr:.2f} > 2.0 for mean-rev primary "
                    f"{primary_name!r}. Mean-rev win rate is structurally ~55-60%; "
                    f"R:R > 2.0 means letting winners run past the mean, which is "
                    f"trend-in-disguise rather than mean-reversion."
                )
            if rr < 0.5:
                raise ProposalValidationError(
                    f"barrier_geometry_attestation R:R={rr:.2f} < 0.5 for mean-rev primary "
                    f"{primary_name!r}. R:R < 0.5 requires >67% win rate to clear "
                    f"break-even — that bar is high enough to flag overfit risk."
                )


@dataclass
class LookaheadShapeAttestation:
    """Shape-leakage attestation per Day 1 skeptic concern.

    The lexical linter blocks year tokens and named events but does NOT
    block abstract pattern-matches to memorized historical episodes (the
    "shape-leakage" channel). To close this, the hypothesizer must name
    at least two regime episodes (by ordinal index in the asset's regime
    parquet, NOT by date) where the hypothesis should have paid if the
    mechanism is real. The skeptic then checks whether those ordinals
    collapse to one memorized event.

    Additionally, the hypothesis MUST be phrased as a cross-asset
    falsifiable prediction — i.e., the hypothesizer must commit to other
    assets where the same mechanism should hold.

    sparsity_note: optional narrative the hypothesizer attaches when the
    target regime has few episodes (typically <6) and the falsification
    criterion's audit_class_in includes MARGINAL_2FOLDS. It documents the
    expected count of "active" folds (n_trades >= 30) under the
    project's NaN-Sharpe floor, so the skeptic can evaluate the audit
    verdict against the sparsity-adjusted falsification criterion rather
    than a STABLE-by-default expectation. Schema-permissive; no
    validation. Added per B0038.
    """

    target_regime_episode_ordinals: list[int] = field(default_factory=list)
    cross_asset_falsifiable_in: list[str] = field(default_factory=list)
    sparsity_note: str | None = None

    def validate(self, diagnostic_only: bool = False) -> None:
        # Diagnostic-only proposals MAY have only 1 ordinal (e.g., a regime
        # with a single observed episode). For tradeable proposals, >=2 is
        # required to prevent shape-leakage onto one memorized episode.
        min_ordinals = 1 if diagnostic_only else 2
        if len(self.target_regime_episode_ordinals) < min_ordinals:
            raise ProposalValidationError(
                f"lookahead_shape_attestation.target_regime_episode_ordinals must contain "
                f">={min_ordinals} ordinal(s) (diagnostic_only={diagnostic_only})"
            )
        if not self.cross_asset_falsifiable_in:
            raise ProposalValidationError(
                "lookahead_shape_attestation.cross_asset_falsifiable_in must be non-empty "
                "(name >=1 other asset where the mechanism should hold)"
            )


@dataclass
class Proposal:
    """Phase 5 proposal payload.

    primary_feature_blacklist: list[str], default empty.
        For custom phase5_* primaries, this lists features the meta-labeler must
        NOT see. Enforcement is at orchestration time (assert_primary_inputs_disjoint
        in scripts/run_backtest.py via the orchestrator's apply_primary_feature_blacklist
        call), NOT in this dataclass.validate(). validate() only ensures the
        field shape (list[str]). The schema is intentionally permissive —
        completeness against build_tier2_features outputs is verified by
        tests/test_primary_feature_blacklist.py::test_phase5_cot_extremes_blacklist_completeness.

        Per docs/superpowers/specs/2026-05-26-edge-search-scope-decision.md
        §Precondición Layer (a)+(c) and
        docs/superpowers/plans/2026-05-26-cot-extremes-primary.md Task 6.
    """
    id: str
    asset: str
    asset_class: str
    regime_scope: list[str]
    hypothesis: str
    causal_story: str
    primary: str
    primary_params: dict = field(default_factory=dict)
    custom_primary_pseudocode: Optional[str] = None
    custom_primary_sha256: Optional[str] = None
    primary_feature_blacklist: list[str] = field(default_factory=list)
    feature_overrides: FeatureOverrides = field(default_factory=FeatureOverrides)
    regime_gate: RegimeGate = field(default_factory=RegimeGate)
    falsification_criterion: FalsificationCriterion = field(default_factory=FalsificationCriterion)
    extra_falsification_criteria: list[dict] = field(default_factory=list)
    lookahead_attestation: LookaheadAttestation = field(default_factory=LookaheadAttestation)
    lookahead_shape_attestation: LookaheadShapeAttestation = field(default_factory=LookaheadShapeAttestation)
    barrier_geometry_attestation: BarrierGeometryAttestation = field(default_factory=BarrierGeometryAttestation)
    parent_proposal: Optional[str] = None
    git_sha_at_propose: Optional[str] = None
    diagnostic_only: bool = False
    # B0155 — pre-registered threshold rule. "fixed_0.50" is the default and
    # preserves the pre-B0155 behavior bit-for-bit (audit aggregates at 0.50).
    # "ev_breakeven_v1" derives p* from the locked barrier geometry +
    # GLOBAL constants C_ATR / LAMBDA_MARGIN (see compute_p_star). p_star may
    # be carried precomputed in the JSON; validate() recomputes and raises on
    # mismatch (tolerance 1e-9) so the threshold can never be hand-edited.
    threshold_rule: str = "fixed_0.50"
    p_star: Optional[float] = None

    def effective_threshold(self) -> float:
        """The single meta-probability threshold the audit evaluates at.

        0.50 for fixed_0.50 (backward compatible); compute_p_star(tp, sl)
        from the barrier_geometry_attestation for ev_breakeven_v1. Always
        recomputed from inputs — a stored p_star is validation-only evidence,
        never the source of truth.
        """
        if self.threshold_rule == "ev_breakeven_v1":
            return compute_p_star(
                self.barrier_geometry_attestation.tp_atr_mult,
                self.barrier_geometry_attestation.sl_atr_mult,
            )
        return 0.50

    def validate(self) -> list[str]:
        if not self.id:
            raise ProposalValidationError("id is required and non-empty")
        if not self.asset:
            raise ProposalValidationError("asset is required and non-empty")
        if self.asset_class not in ASSET_CLASSES:
            raise ProposalValidationError(f"asset_class={self.asset_class!r} not in {ASSET_CLASSES}")
        if not self.regime_scope:
            raise ProposalValidationError("regime_scope must be non-empty")
        bad = [r for r in self.regime_scope if r not in REGIME_IDS]
        if bad:
            raise ProposalValidationError(
                f"regime_scope has unknown regimes: {bad}; allowed: {list(REGIME_IDS)}"
            )
        if not (self.primary in PRIMARIES or self.primary.startswith("phase5_")):
            raise ProposalValidationError(
                f"primary={self.primary!r} not in {PRIMARIES} and not phase5_*"
            )
        for narrative_field, val in (("hypothesis", self.hypothesis), ("causal_story", self.causal_story)):
            if not (30 <= len(val) <= 800):
                raise ProposalValidationError(
                    f"{narrative_field} length {len(val)} not in [30, 800]"
                )
        if self.primary.startswith("phase5_") and not self.custom_primary_pseudocode:
            raise ProposalValidationError(
                f"primary={self.primary!r} (phase5_* custom) requires custom_primary_pseudocode"
            )
        if not self.primary.startswith("phase5_") and self.custom_primary_pseudocode:
            raise ProposalValidationError(
                "custom_primary_pseudocode set but primary is not a phase5_* custom primary"
            )
        self.regime_gate.validate()
        self.falsification_criterion.validate()
        self.lookahead_shape_attestation.validate(diagnostic_only=self.diagnostic_only)
        self.barrier_geometry_attestation.validate(primary_name=self.primary)

        # B0155 — threshold rule. Validated AFTER barrier_geometry_attestation
        # so tp/sl positivity is already guaranteed for compute_p_star.
        if self.threshold_rule not in THRESHOLD_RULES:
            raise ProposalValidationError(
                f"threshold_rule={self.threshold_rule!r} not in {THRESHOLD_RULES}"
            )
        if self.threshold_rule == "ev_breakeven_v1":
            expected_p_star = compute_p_star(
                self.barrier_geometry_attestation.tp_atr_mult,
                self.barrier_geometry_attestation.sl_atr_mult,
            )
            if self.p_star is not None and abs(float(self.p_star) - expected_p_star) > 1e-9:
                raise ProposalValidationError(
                    f"p_star={self.p_star!r} does not match the value recomputed from "
                    f"barrier_geometry_attestation (tp={self.barrier_geometry_attestation.tp_atr_mult}, "
                    f"sl={self.barrier_geometry_attestation.sl_atr_mult}) + global "
                    f"C_ATR={C_ATR}/LAMBDA_MARGIN={LAMBDA_MARGIN}: expected {expected_p_star!r} "
                    f"(tolerance 1e-9). p_star is derived, never hand-set."
                )
        elif self.p_star is not None:
            raise ProposalValidationError(
                f"p_star={self.p_star!r} is set but threshold_rule={self.threshold_rule!r}; "
                f"a free-floating p_star is a threshold-shopping channel — use "
                f"threshold_rule='ev_breakeven_v1' (p* derived from barrier geometry) "
                f"or remove p_star."
            )

        # B0155 — feature-existence HARD gate on feature_overrides (the B004v3
        # lesson: a nonexistent feature name produced a structurally dead gate
        # that masqueraded as falsification).
        warnings: list[str] = []
        registry = known_feature_registry()
        unknown = [
            n for n in (list(self.feature_overrides.add) + list(self.feature_overrides.drop))
            if not _feature_name_known(n, registry)
        ]
        if unknown:
            raise ProposalValidationError(
                f"feature_overrides references unknown feature(s): {unknown}. "
                f"Names must be tier2 columns, FEATURE_ALIASES keys, or dossier "
                f"alt-features (see phase5.proposal.known_feature_registry). A "
                f"nonexistent feature silently degrades to a dead gate / no-op "
                f"(B004v3) — fix the name or drop the override."
            )
        # Lenient (warning-level) scan: phase5_* custom primaries may reference
        # feature names inside primary_params values. Unknown feature-shaped
        # strings are surfaced as warnings, NOT hard errors — custom primaries
        # own their signature and may use non-feature string params.
        if self.primary.startswith("phase5_"):
            for key, val in self.primary_params.items():
                if (
                    isinstance(val, str)
                    and _FEATURE_NAME_RE.fullmatch(val)
                    and not _feature_name_known(val, registry)
                ):
                    warnings.append(
                        f"primary_params[{key!r}] = {val!r} looks like a feature name "
                        f"but is not in the known-feature registry (warning only; "
                        f"verify the custom primary actually computes/receives it)"
                    )
        return warnings

    def run_lookahead_lint(self) -> tuple[bool, str]:
        """Re-run lookahead lint at commit time."""
        result = lint_proposal(self.to_dict())
        return result.passed, result.summary()

    def is_falsification_at_least_as_strict_as(self, baseline: dict) -> tuple[bool, list[str]]:
        """Compare this proposal's criterion to a baseline (e.g., default).

        Returns (ok, list_of_violations).
        """
        f = self.falsification_criterion
        violations: list[str] = []
        base_classes = set(baseline.get("audit_class_in", DEFAULT_FALSIFICATION["audit_class_in"]))
        my_classes = set(f.audit_class_in)
        if not my_classes.issubset(base_classes):
            violations.append(
                f"audit_class_in {sorted(my_classes)} is not a subset of baseline {sorted(base_classes)}"
            )
        baseline_sharpe = baseline.get(
            "median_active_fold_sharpe_min",
            DEFAULT_FALSIFICATION["median_active_fold_sharpe_min"],
        )
        if f.median_active_fold_sharpe_min < baseline_sharpe:
            violations.append(
                f"median_active_fold_sharpe_min={f.median_active_fold_sharpe_min} < baseline {baseline_sharpe}"
            )
        baseline_n = baseline.get("n_trades_total_min", DEFAULT_FALSIFICATION["n_trades_total_min"])
        if f.n_trades_total_min < baseline_n:
            violations.append(
                f"n_trades_total_min={f.n_trades_total_min} < baseline {baseline_n}"
            )
        return (not violations), violations

    def to_dict(self) -> dict:
        return asdict(self)


def _build_dataclass(cls, payload: dict):
    """Build a dataclass from a dict, recursively for nested dataclasses."""
    if payload is None:
        return cls()
    kwargs = {}
    annotations = cls.__annotations__
    for field_name, field_type in annotations.items():
        if field_name not in payload:
            continue
        val = payload[field_name]
        # Recurse for nested dataclasses
        if field_name == "feature_overrides":
            kwargs[field_name] = FeatureOverrides(**(val or {}))
        elif field_name == "regime_gate":
            kwargs[field_name] = RegimeGate(**(val or {}))
        elif field_name == "falsification_criterion":
            kwargs[field_name] = FalsificationCriterion(**(val or {}))
        elif field_name == "lookahead_attestation":
            kwargs[field_name] = LookaheadAttestation(**(val or {}))
        elif field_name == "lookahead_shape_attestation":
            kwargs[field_name] = LookaheadShapeAttestation(**(val or {}))
        elif field_name == "barrier_geometry_attestation":
            kwargs[field_name] = BarrierGeometryAttestation(**(val or {}))
        else:
            kwargs[field_name] = val
    return cls(**kwargs)


def compute_custom_primary_sha256(path: str | Path) -> str:
    """SHA-256 of a custom-primary .py file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_proposal(path: str | Path, *, asset_override: str | None = None) -> Proposal:
    """Load a Proposal from a JSON file.

    `asset_override` injects the given asset symbol when the JSON omits `asset`
    (asset-blind Loop-A proposals).  It is a no-op when `asset` is already
    present in the JSON so existing stamped proposals are unaffected (B0107).
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if asset_override and "asset" not in payload:
        payload["asset"] = asset_override
    return _build_dataclass(Proposal, payload)


def save_proposal(proposal: Proposal, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(proposal.to_dict(), indent=2), encoding="utf-8")
