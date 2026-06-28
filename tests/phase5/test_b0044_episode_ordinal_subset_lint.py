"""B0044 — episode-ordinal subset lint rule (anti-circularity).

Loop A tick 13 (BULL_STRESSED) was BLOCKED three times by the devil's
advocate, partly because it read `target_regime_episode_ordinals` naming a
SUBSET of the regime's episodes as "coded episode-targeting / lookahead."

That objection contradicts the methodology spec
(.claude/skills/phase5-regime-methodology/SKILL.md:137):

  "The DA agent must NOT flag ordinal > n_episodes-1 as an inconsistency;
   only flag if the listed ordinals are NOT a subset of the dossier's
   regime_episode_ordinals."

So a subset is the LEGITIMATE, designed shape-leakage tripwire. The genuine
leak is the inverse: naming an ordinal NOT in this regime's episode set. This
test pins the corrected deterministic rule:

  FAIL iff target_regime_episode_ordinals is NOT a subset of the dossier's
  regime_episode_ordinals (when a dossier is supplied). Subsets PASS; the
  count check (>=2) stays in Proposal.validate(); the rule is skipped when no
  dossier is supplied (back-compat for existing callers).
"""
from __future__ import annotations

from phase5.lookahead_lint import lint_anti_circularity

ORDINAL_RULE = "anti_circularity_episode_ordinals_not_subset"


def _proposal(ordinals: list[int]) -> dict:
    # ema_cross(10, 30), no feature adds → clears the OTHER anti-circularity
    # rules, so only the episode-ordinal rule can fire.
    return {
        "primary": "ema_cross",
        "primary_params": {"fast": 10, "slow": 30},
        "feature_overrides": {"add": [], "drop": []},
        "causal_story": "Causal story text well above the validator floor.",
        "lookahead_shape_attestation": {
            "target_regime_episode_ordinals": ordinals,
            "cross_asset_falsifiable_in": ["fx"],
        },
    }


def _dossier(regime_episode_ordinals: list[int], n_episodes: int) -> dict:
    return {
        "asset_class": "metal",
        "regime_id": "BULL_STRESSED",
        "n_episodes": n_episodes,
        "regime_episode_ordinals": regime_episode_ordinals,
        "features_quantile_summary": {},
    }


def _ordinal_hits(result):
    return [h for h in result.hits if h.rule == ORDINAL_RULE]


def test_tick13_subset_now_passes():
    """Regression guard: the exact tick-13 subsets the DA wrongly BLOCKED."""
    dossier = _dossier([1, 13, 15, 21, 23], n_episodes=5)
    for ordinals in ([1, 13, 21], [13, 15, 21]):
        result = lint_anti_circularity(_proposal(ordinals), dossier=dossier)
        assert not _ordinal_hits(result), f"subset {ordinals} must not be flagged"


def test_genuine_non_subset_fails():
    dossier = _dossier([1, 13, 15, 21, 23], n_episodes=5)
    result = lint_anti_circularity(_proposal([1, 13, 99]), dossier=dossier)
    hits = _ordinal_hits(result)
    assert len(hits) == 1
    assert "99" in hits[0].match
    assert not result.passed


def test_ordinal_ge_n_episodes_but_in_set_passes():
    """Global-timeline ordinals (>= n_episodes-1) are valid when in the set."""
    dossier = _dossier([1, 13, 15, 21, 23], n_episodes=5)
    result = lint_anti_circularity(_proposal([21, 23]), dossier=dossier)
    assert not _ordinal_hits(result)


def test_rule_skipped_when_dossier_omitted():
    """Back-compat: existing callers that pass no dossier are unaffected."""
    result = lint_anti_circularity(_proposal([1, 13, 99]))
    assert not _ordinal_hits(result)


def test_empty_ordinals_no_episode_hit():
    """Count enforcement (>=2) lives in Proposal.validate(), not the lint."""
    dossier = _dossier([1, 13, 15, 21, 23], n_episodes=5)
    result = lint_anti_circularity(_proposal([]), dossier=dossier)
    assert not _ordinal_hits(result)
