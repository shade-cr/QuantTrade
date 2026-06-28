"""Single source of truth for built-in-primary parameter contracts (B0085).

The Loop-A hypothesizer is handed `available_primaries` and emits `primary_params`
with guessed key names (``threshold_atr_mult``, ``threshold_sigma``, ``window``,
``n_std`` ...). The pipeline's ``scripts/run_xau_d1.py::_select_primary`` reads
*canonical* keys (``threshold_atr``, ``period``, ``k_stdev`` ...). For
``cusum_filter`` / ``bollinger_meanrev`` the template config ships no default
section, so a missing canonical key raised ``KeyError`` *inside* the audit
subprocess — it died before producing a Sharpe verdict.

This module defines the contract once and serves two consumers:

* :func:`primary_param_schema_for_payload` — feeds the hypothesizer the exact
  canonical param names (Option A: stops future drift). Pure param contract,
  carries zero lookahead/regime/asset/date data → firewall-clean.
* :func:`normalize_primary_params` — resolves *unambiguous true-synonym* aliases
  to canonical names at config-build time (Option B), and raises
  :class:`PrimaryParamError` if a required key is still missing afterwards
  (conservative + fail-fast). It NEVER silently coerces divergent units
  (``threshold_sigma`` is vol-sigma units, NOT ATR) and NEVER silently fills a
  guessed default — that would reintroduce the DAY2 "silent revert to broken
  defaults" hazard (see phase5/DAY2_REPORT.md, project_barrier_geometry_root_cause).

``phase5_custom`` and any ``phase5_*`` primary are NOT in the registry — they ship
their own ``signal()`` with their own param names, so normalization is an identity
passthrough for them.

The canonical names below are derived from the real ``signal()`` signatures in
pipeline/labels.py. Keep this module in sync if those signatures change.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_MISSING = object()


class PrimaryParamError(ValueError):
    """Raised when a built-in primary's params cannot be normalized to the
    canonical contract (a required key is missing and has no true-synonym
    alias, or the primary is unknown)."""


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str
    required: bool
    description: str
    default: object = _MISSING
    # True-synonym aliases ONLY — keys that mean exactly the same thing in the
    # same units. Divergent-unit guesses (threshold_sigma, threshold_mult, ...)
    # are deliberately absent so they fail fast instead of being coerced.
    aliases: tuple[str, ...] = ()

    @property
    def has_default(self) -> bool:
        return self.default is not _MISSING


@dataclass(frozen=True)
class PrimaryContract:
    name: str
    params: tuple[ParamSpec, ...]


PRIMARY_CONTRACTS: dict[str, PrimaryContract] = {
    "ema_cross": PrimaryContract(
        name="ema_cross",
        params=(
            ParamSpec("fast", "int", required=True,
                      description="Fast EMA span (bars)."),
            ParamSpec("slow", "int", required=True,
                      description="Slow EMA span (bars); must exceed fast."),
            ParamSpec("dead_zone_atr", "float", required=False, default=0.25,
                      description="Min EMA separation in ATR units to fire."),
        ),
    ),
    "momentum_zscore": PrimaryContract(
        name="momentum_zscore",
        params=(
            ParamSpec("lookback", "int", required=True,
                      description="Rolling window (bars) for the momentum z-score."),
            ParamSpec("threshold", "float", required=False, default=0.3,
                      description="Absolute z-score required to fire a side."),
        ),
    ),
    "cusum_filter": PrimaryContract(
        name="cusum_filter",
        params=(
            ParamSpec("threshold_atr", "float", required=True,
                      description="CUSUM trigger in ATR-multiple units "
                                  "(threshold_atr * ATR[t-1] / close[t-1]).",
                      aliases=("threshold_atr_mult",)),
        ),
    ),
    "bollinger_meanrev": PrimaryContract(
        name="bollinger_meanrev",
        params=(
            ParamSpec("period", "int", required=True,
                      description="Rolling window (bars) for the band middle/std.",
                      aliases=("window", "lookback")),
            ParamSpec("k_stdev", "float", required=True,
                      description="Band half-width in standard deviations.",
                      aliases=("n_std", "num_std")),
        ),
    ),
}


def _is_custom(primary: str) -> bool:
    return primary.startswith("phase5_")


def normalize_primary_params(primary: str, params: dict) -> dict:
    """Return ``params`` with true-synonym aliases resolved to canonical names.

    * ``phase5_*`` primaries → identity passthrough (they own their signature).
    * ``supervised_direct`` → identity passthrough (no primary signal used in supervised-direct mode).
    * Unknown built-in name → :class:`PrimaryParamError`.
    * Required canonical key still missing after aliasing → :class:`PrimaryParamError`.
    * Unknown extra keys are preserved untouched (``_select_primary`` ignores them).
    * Explicit canonical values win over aliases; optional defaults are NOT injected.
    """
    if _is_custom(primary) or primary == "supervised_direct":
        return dict(params)

    contract = PRIMARY_CONTRACTS.get(primary)
    if contract is None:
        raise PrimaryParamError(
            f"Unknown primary {primary!r}; known built-ins: "
            f"{sorted(PRIMARY_CONTRACTS)} (or a phase5_* custom primary)."
        )

    out = dict(params)
    missing: list[ParamSpec] = []
    for spec in contract.params:
        if spec.name in out:
            # Canonical already present — drop any aliases so they can't clobber.
            for alias in spec.aliases:
                out.pop(alias, None)
            continue
        # Resolve the first matching true-synonym alias.
        for alias in spec.aliases:
            if alias in out:
                out[spec.name] = out.pop(alias)
                break
        else:
            # No canonical, no alias.
            if spec.required:
                missing.append(spec)
            # Optional + absent → leave it out (signature/template default applies).

    if missing:
        details = ", ".join(
            f"{s.name!r} (accepted: {[s.name, *s.aliases]})" for s in missing
        )
        raise PrimaryParamError(
            f"Primary {primary!r} is missing required param(s): {details}; "
            f"received params {sorted(params)}. Divergent-unit synonyms (e.g. "
            f"threshold_sigma) are intentionally NOT auto-mapped — emit the "
            f"canonical key explicitly."
        )
    return out


def primary_param_schema_for_payload() -> dict:
    """Built-in param contract for the hypothesizer payload (Option A).

    Shape: ``{primary: {param: {type, required, default?, description}}}`` plus a
    bare ``phase5_custom`` marker (no fixed contract). Carries no lookahead data.
    """
    schema: dict = {}
    for name, contract in PRIMARY_CONTRACTS.items():
        params: dict = {}
        for spec in contract.params:
            entry = {
                "type": spec.type,
                "required": spec.required,
                "description": spec.description,
            }
            if spec.has_default:
                entry["default"] = spec.default
            params[spec.name] = entry
        schema[name] = params
    schema["phase5_custom"] = {
        "_note": "Custom primary ships its own signal(); param names are free-form.",
    }
    return schema
