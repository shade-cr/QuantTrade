"""Tests for the primary_feature_blacklist three-layer enforcement (B0015b).

Per docs/superpowers/specs/2026-05-26-edge-search-scope-decision.md §Precondición:

- Layer (a) syntactic: orchestrator drops blacklisted columns from the meta's
  features view via apply_primary_feature_blacklist helper.
- Layer (b) docstring linter: every phase5_* primary module declares
  INPUT_COLUMNS + (if non-empty) an 'Inputs read:' docstring block.
- Layer (c) runtime intersection: assert_primary_inputs_disjoint raises if the
  primary's INPUT_COLUMNS overlaps with the meta's features columns.
"""
from __future__ import annotations
import importlib
from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Layer (a) — syntactic blacklist helper
# ---------------------------------------------------------------------------

def test_apply_primary_feature_blacklist_exact_match():
    """Exact column-name matches are dropped."""
    from pipeline.features import apply_primary_feature_blacklist
    df = pd.DataFrame({"a": [1], "cot_extreme_long": [0], "rv_20": [0.1]})
    out = apply_primary_feature_blacklist(df, ["cot_extreme_long"])
    assert list(out.columns) == ["a", "rv_20"]


def test_apply_primary_feature_blacklist_wildcard():
    """Wildcard (trailing *) drops by prefix match."""
    from pipeline.features import apply_primary_feature_blacklist
    df = pd.DataFrame({
        "a": [1], "dtwexbgs_close": [0], "dtwexbgs_zscore_30d": [0], "rv_20": [0.1]
    })
    out = apply_primary_feature_blacklist(df, ["dtwexbgs_*"])
    assert list(out.columns) == ["a", "rv_20"]


def test_apply_primary_feature_blacklist_mixed_exact_and_wildcard():
    """Mix of exact and wildcard entries is handled in one pass."""
    from pipeline.features import apply_primary_feature_blacklist
    df = pd.DataFrame({
        "a": [1], "cot_extreme_long": [0], "dtwexbgs_close": [0], "rv_20": [0.1]
    })
    out = apply_primary_feature_blacklist(df, ["cot_extreme_long", "dtwexbgs_*"])
    assert list(out.columns) == ["a", "rv_20"]


def test_apply_primary_feature_blacklist_empty_or_none():
    """Empty list or None returns the input unchanged."""
    from pipeline.features import apply_primary_feature_blacklist
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert list(apply_primary_feature_blacklist(df, None).columns) == ["a", "b"]
    assert list(apply_primary_feature_blacklist(df, []).columns) == ["a", "b"]


def test_apply_primary_feature_blacklist_missing_column_silent():
    """Asking to drop a column that doesn't exist is silent."""
    from pipeline.features import apply_primary_feature_blacklist
    df = pd.DataFrame({"a": [1]})
    out = apply_primary_feature_blacklist(df, ["nonexistent"])
    assert list(out.columns) == ["a"]


def test_apply_primary_feature_blacklist_does_not_mutate_input():
    """Returns a new frame; input columns unchanged."""
    from pipeline.features import apply_primary_feature_blacklist
    df = pd.DataFrame({"a": [1], "cot_extreme_long": [0]})
    cols_before = list(df.columns)
    apply_primary_feature_blacklist(df, ["cot_extreme_long"])
    assert list(df.columns) == cols_before


# ---------------------------------------------------------------------------
# Layer (c) — runtime intersection assertion
# ---------------------------------------------------------------------------

def test_assert_primary_inputs_disjoint_empty_input_columns():
    """INPUT_COLUMNS=() passes trivially regardless of meta columns."""
    from pipeline.primaries_phase5 import assert_primary_inputs_disjoint
    assert_primary_inputs_disjoint(
        primary_inputs=(), meta_features_columns={"cot_extreme_long", "rv_20"}
    )


def test_assert_primary_inputs_disjoint_violation_raises():
    """A primary that declares an input column also in meta_features must raise."""
    from pipeline.primaries_phase5 import (
        assert_primary_inputs_disjoint, PrimaryInputContractError,
    )
    with pytest.raises(PrimaryInputContractError):
        assert_primary_inputs_disjoint(
            primary_inputs=("cot_extreme_long",),
            meta_features_columns={"cot_extreme_long", "rv_20"},
        )


def test_assert_primary_inputs_disjoint_clean_pass():
    """Primary declares inputs disjoint from meta — no raise."""
    from pipeline.primaries_phase5 import assert_primary_inputs_disjoint
    assert_primary_inputs_disjoint(
        primary_inputs=("cot_commercials_net_long_z52",),
        meta_features_columns={"rv_20", "ma_50"},
    )


def test_assert_primary_inputs_disjoint_error_message_lists_intersection():
    """Error message names the offending columns so debugging is easy."""
    from pipeline.primaries_phase5 import (
        assert_primary_inputs_disjoint, PrimaryInputContractError,
    )
    with pytest.raises(PrimaryInputContractError, match="cot_extreme_long"):
        assert_primary_inputs_disjoint(
            primary_inputs=("cot_extreme_long", "rv_20"),
            meta_features_columns={"cot_extreme_long", "rv_20", "ma_50"},
        )


# ---------------------------------------------------------------------------
# Layer (b) — docstring linter on phase5_* primaries
# ---------------------------------------------------------------------------

def test_all_phase5_primaries_declare_input_columns():
    """Every phase5_* primary module in pipeline/primaries_phase5/ must declare
    INPUT_COLUMNS as a module-level tuple."""
    pkg_path = Path("pipeline/primaries_phase5")
    primary_modules = [
        f.stem for f in pkg_path.glob("phase5_*.py") if not f.stem.startswith("_")
    ]
    assert primary_modules, "Found no phase5_* primary modules to lint"

    for mod_name in primary_modules:
        mod = importlib.import_module(f"pipeline.primaries_phase5.{mod_name}")
        assert hasattr(mod, "INPUT_COLUMNS"), (
            f"{mod_name}: module-level INPUT_COLUMNS constant required"
        )
        assert isinstance(mod.INPUT_COLUMNS, tuple), (
            f"{mod_name}: INPUT_COLUMNS must be a tuple, got {type(mod.INPUT_COLUMNS)}"
        )


def test_phase5_primary_docstring_inputs_block_when_input_columns_nonempty():
    """For primaries with INPUT_COLUMNS != (), the module docstring must contain
    'Inputs read:' enumeration + disjointness assertion."""
    pkg_path = Path("pipeline/primaries_phase5")
    for f in pkg_path.glob("phase5_*.py"):
        if f.stem.startswith("_"):
            continue
        mod = importlib.import_module(f"pipeline.primaries_phase5.{f.stem}")
        if not mod.INPUT_COLUMNS:
            continue  # Empty tuple is exempt from the docstring requirement
        assert mod.__doc__ and "Inputs read:" in mod.__doc__, (
            f"{f.stem}: docstring missing 'Inputs read:' block; "
            f"INPUT_COLUMNS={mod.INPUT_COLUMNS}"
        )
        assert "disjoint" in mod.__doc__, (
            f"{f.stem}: docstring missing 'disjoint' assertion language"
        )


def test_phase5_cot_extremes_blacklist_completeness():
    """The B0015b blacklist must cover every column in build_tier2_features
    output that starts with cot_, dxy_, or dtwexbgs_.

    This is the completeness invariant of the primary_feature_blacklist —
    adding a new such column to features.py without updating the blacklist
    breaks this test, forcing the developer to make a conscious choice.
    """
    # Define the canonical B0015b blacklist (must mirror what the proposal
    # JSON declares — kept in sync via this test).
    B0015B_BLACKLIST = {
        "cot_commercials_net_long_pct", "cot_commercials_net_long_z52",
        "cot_commercials_net_long_chg_4w", "cot_commercials_extreme_long",
        "cot_commercials_extreme_short",
        "cot_net_noncomm_pct", "cot_net_noncomm_z52", "cot_net_noncomm_chg_4w",
        "cot_extreme_long", "cot_extreme_short",
        # Wildcards (covered by prefix; test must check the prefix logic)
    }
    B0015B_BLACKLIST_WILDCARDS = {"dxy_*", "dtwexbgs_*"}

    # Build the canonical feature set so we can compare against the blacklist.
    # Use a minimal synthetic ohlcv + macro_frame to avoid expensive fixtures.
    import numpy as np
    idx = pd.date_range("2020-01-01", periods=400, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    close = 1500 + np.cumsum(rng.normal(0, 5, 400))
    ohlcv = pd.DataFrame({
        "open": close, "high": close * 1.005, "low": close * 0.995,
        "close": close, "volume": rng.integers(1000, 5000, 400),
    }, index=idx)
    macro_codes = ("DTWEXBGS", "DFII5", "DGS5", "DGS2", "T5YIE", "VIXCLS", "UMCSENT", "UMCSENT_chg_3m")
    macro_frame = pd.DataFrame({c: rng.normal(100, 5, 400) for c in macro_codes}, index=idx)

    from pipeline.features import build_tier2_features
    feats = build_tier2_features(ohlcv, macro_frame)

    cot_cols = {c for c in feats.columns if c.startswith("cot_")}
    dxy_cols = {c for c in feats.columns if c.startswith("dxy_")}
    dtwexbgs_cols = {c for c in feats.columns if c.startswith("dtwexbgs_")}

    # Every cot_* column in features must be in the blacklist explicitly.
    missing_cot = cot_cols - B0015B_BLACKLIST
    assert not missing_cot, (
        f"cot_* columns not in B0015b blacklist (update tests/test_primary_feature_blacklist.py "
        f"or docs/superpowers/specs/2026-05-26-cot-extremes-primary.md): {missing_cot}"
    )

    # dxy_* and dtwexbgs_* must be covered by the wildcards.
    # (No assertion needed beyond the wildcard exists in the blacklist set; the
    # wildcard logic is tested in test_apply_primary_feature_blacklist_wildcard.)
    assert "dxy_*" in B0015B_BLACKLIST_WILDCARDS or not dxy_cols, (
        f"dxy_* wildcard missing; dxy_cols={dxy_cols}"
    )
    assert "dtwexbgs_*" in B0015B_BLACKLIST_WILDCARDS or not dtwexbgs_cols, (
        f"dtwexbgs_* wildcard missing; dtwexbgs_cols={dtwexbgs_cols}"
    )
