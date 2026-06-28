"""Tests for pipeline.primary_contracts (B0085 — built-in-primary param contract).

The Loop-A hypothesizer emits primary_params with guessed key names; the pipeline
reads canonical keys. This module is the single source of truth that maps guessed
true-synonyms to canonical names and fails fast on anything ambiguous, so the M3
audit never dies on an opaque KeyError inside the subprocess.
"""
from __future__ import annotations

import pytest

from pipeline.primary_contracts import (
    PRIMARY_CONTRACTS,
    PrimaryParamError,
    normalize_primary_params,
    primary_param_schema_for_payload,
)


# --- alias resolution (true synonyms only) -------------------------------

def test_cusum_threshold_atr_mult_aliases_to_threshold_atr():
    out = normalize_primary_params("cusum_filter", {"threshold_atr_mult": 1.0})
    assert out == {"threshold_atr": 1.0}


def test_bollinger_window_and_nstd_alias_to_canonical():
    out = normalize_primary_params(
        "bollinger_meanrev", {"window": 21, "n_std": 2.0}
    )
    assert out == {"period": 21, "k_stdev": 2.0}


def test_bollinger_lookback_and_numstd_alias_to_canonical():
    out = normalize_primary_params(
        "bollinger_meanrev", {"lookback": 30, "num_std": 2.5}
    )
    assert out == {"period": 30, "k_stdev": 2.5}


# --- canonical passthrough + extra-key preservation ----------------------

def test_canonical_keys_pass_through_unchanged():
    out = normalize_primary_params("cusum_filter", {"threshold_atr": 2.0})
    assert out == {"threshold_atr": 2.0}


def test_unknown_extra_keys_are_preserved_untouched():
    out = normalize_primary_params(
        "cusum_filter", {"threshold_atr": 1.0, "side": "short_only"}
    )
    assert out == {"threshold_atr": 1.0, "side": "short_only"}


def test_canonical_present_alongside_alias_keeps_canonical_value():
    # The R4/R5 batch carries BOTH guessed + hand-added canonical keys. The
    # explicit canonical value wins; the alias must not clobber it.
    out = normalize_primary_params(
        "cusum_filter", {"threshold_atr_mult": 1.5, "threshold_atr": 2.0}
    )
    assert out["threshold_atr"] == 2.0


# --- fail-fast on missing required (no true-synonym) ---------------------

def test_threshold_sigma_alone_raises_not_silently_coerced():
    # threshold_sigma is vol-sigma units, NOT ATR units -> must NOT alias.
    with pytest.raises(PrimaryParamError) as exc:
        normalize_primary_params("cusum_filter", {"threshold_sigma": 1.0})
    msg = str(exc.value)
    assert "threshold_atr" in msg
    assert "cusum_filter" in msg


def test_ema_cross_missing_required_raises():
    with pytest.raises(PrimaryParamError) as exc:
        normalize_primary_params("ema_cross", {})
    msg = str(exc.value)
    assert "fast" in msg and "slow" in msg


# --- optional defaults are NOT silently injected -------------------------

def test_optional_param_left_absent_not_filled():
    # dead_zone_atr has a signature default; the contract must NOT inject it,
    # leaving the signature/template default to apply (no silent value).
    out = normalize_primary_params("ema_cross", {"fast": 10, "slow": 30})
    assert out == {"fast": 10, "slow": 30}
    assert "dead_zone_atr" not in out


# --- custom primaries: identity passthrough ------------------------------

def test_phase5_custom_is_identity_passthrough():
    params = {"foo": 1, "bar": "baz"}
    assert normalize_primary_params("phase5_custom", params) == params


def test_phase5_named_custom_is_identity_passthrough():
    params = {"cusum_threshold_atr_mult": 1.0, "atr_window": 14}
    assert normalize_primary_params("phase5_t015d2", params) == params


def test_unknown_primary_raises():
    with pytest.raises(PrimaryParamError):
        normalize_primary_params("not_a_primary", {})


# --- payload schema for the hypothesizer (Option A) ----------------------

def test_payload_schema_has_all_builtins():
    schema = primary_param_schema_for_payload()
    for name in ("ema_cross", "momentum_zscore", "cusum_filter", "bollinger_meanrev"):
        assert name in schema
        # each built-in exposes its params keyed by canonical name
        assert "threshold_atr" in schema["cusum_filter"] or name != "cusum_filter"


def test_payload_schema_cusum_lists_canonical_param():
    schema = primary_param_schema_for_payload()
    cusum = schema["cusum_filter"]
    assert "threshold_atr" in cusum
    assert cusum["threshold_atr"]["required"] is True


def test_payload_schema_marks_optional_default():
    schema = primary_param_schema_for_payload()
    dz = schema["ema_cross"]["dead_zone_atr"]
    assert dz["required"] is False
    assert dz["default"] == 0.25


def test_payload_schema_includes_phase5_custom_marker():
    schema = primary_param_schema_for_payload()
    assert "phase5_custom" in schema


def test_payload_schema_carries_no_lookahead_fields():
    # firewall: pure param contract, no regime/asset/date leakage.
    import json

    blob = json.dumps(primary_param_schema_for_payload()).lower()
    for forbidden in ("regime", "asset", "date", "quantile", "dossier"):
        assert forbidden not in blob


def test_contracts_cover_exactly_the_four_builtins():
    assert set(PRIMARY_CONTRACTS) == {
        "ema_cross",
        "momentum_zscore",
        "cusum_filter",
        "bollinger_meanrev",
    }
