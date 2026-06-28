"""Phase 5 custom primary signal modules.

Each module in this package implements a single `signal(ohlcv, features, cfg) -> pd.Series`
function returning side values in {-1, 0, +1} indexed identically to ohlcv.

The dispatcher in scripts/run_backtest.py:_select_primary routes any
`primary_name` starting with `phase5_` to the module in this package whose
filename matches the primary name.

Contract requirements per the shared SKILL.md:
- Pure function: no I/O, no global state, deterministic given (ohlcv, features, cfg).
- Returns pd.Series with same index as ohlcv, dtype int-like.
- NaN is treated as 0 (no signal).
- All rolling windows must be strictly causal: no centered windows, no future bars.
  When computing rolling-window percentile bands, the comparison value at t MUST
  be evaluated against the band derived from data up to and including bar t
  (not beyond). When in doubt, use .shift(1) before .rolling(...) to be strict.

Per docs/superpowers/specs/2026-05-26-edge-search-scope-decision.md §Precondición
(B0000), every phase5_* primary module ALSO declares a module-level constant:

    INPUT_COLUMNS: tuple[str, ...] = (...)

Naming the columns the primary reads from the `features` arg. Raw OHLCV is
universal substrate and is exempt — primaries that read ONLY raw OHLCV
declare INPUT_COLUMNS=(). The orchestrator (scripts/run_backtest.py) invokes
`assert_primary_inputs_disjoint(primary.INPUT_COLUMNS, meta_features_columns)`
before calling signal(), raising PrimaryInputContractError if the primary's
declared inputs overlap with the columns the meta-labeler will see.

This is Layer (c) of the three-layer blacklist enforcement (the other two
layers are syntactic blacklist filter in pipeline.features and docstring
linter in tests/test_primary_feature_blacklist.py).
"""
from __future__ import annotations


class PrimaryInputContractError(ValueError):
    """Raised when a custom phase5_* primary's declared INPUT_COLUMNS intersect
    with the meta-labeler's features columns.

    This is the runtime enforcement of the primary_feature_blacklist
    precondition from docs/superpowers/specs/2026-05-26-edge-search-scope-decision.md
    §Precondición Layer (c).
    """


def assert_primary_inputs_disjoint(
    primary_inputs: tuple[str, ...] | set[str] | list[str],
    meta_features_columns: set[str] | list[str],
) -> None:
    """Layer (c) runtime intersection assertion.

    Raises PrimaryInputContractError if the primary's declared INPUT_COLUMNS
    (columns read from the `features` arg of signal()) intersect with the
    columns the meta-labeler will see in build_tier2_features() outputs
    (post-apply_primary_feature_blacklist).

    Primaries that read only raw ohlcv (close/high/low/volume) should declare
    INPUT_COLUMNS=() and pass trivially. The blacklist filtering at orchestrator
    level (apply_primary_feature_blacklist) ensures raw OHLCV access is
    universal substrate, not a contract violation.
    """
    primary_set = set(primary_inputs)
    meta_set = set(meta_features_columns)
    intersection = primary_set & meta_set
    if intersection:
        raise PrimaryInputContractError(
            f"Primary inputs intersect with meta features columns: {sorted(intersection)}. "
            "This violates the primary_feature_blacklist precondition "
            "(docs/superpowers/specs/2026-05-26-edge-search-scope-decision.md §Precondición). "
            "Either widen the proposal's primary_feature_blacklist to cover these columns, "
            "or remove them from the primary's INPUT_COLUMNS."
        )
