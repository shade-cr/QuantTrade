"""EV-breakeven meta-probability thresholds (B0155 formula; B0161 wiring).

Canonical home of the threshold-rule machinery. `phase5.proposal` re-exports
`compute_p_star` / `C_ATR` / `LAMBDA_MARGIN` from here so the proposal schema
and the multi-asset orchestrator can never drift apart on the formula.

Why this exists (measured 2026-06-11, B0148 post-mortem): a fixed 0.50–0.62
threshold grid is payoff-blind. With asymmetric triple-barrier geometry
(tp=3 ATR / sl=1 ATR) the base win rate is ~23% and a correctly calibrated
meta emits p with median ≈ 0.23 and q99 ≈ 0.43–0.51 — the legacy grid sat
entirely ABOVE the reachable range, silencing every EV-positive trade in
[p*, 0.50] by construction (the systemic NO_FIRE). The breakeven must be
derived a-priori from geometry + global cost constants, never tuned on
results (threshold-shopping).
"""
from __future__ import annotations

from typing import Optional

C_ATR = 0.10
"""Round-trip transaction cost in ATR units — GLOBAL methodology constant.

NEVER per-proposal or per-config: letting a run carry its own cost assumption
would be a threshold-shopping channel (pick the cost that makes your p* admit
the trades you want). Amendable only by spec change to the Phase 5 methodology
(.claude/skills/phase5-regime-methodology/SKILL.md, section
"Threshold rule (B0155, 2026-06-11)")."""

LAMBDA_MARGIN = 0.05
"""Safety margin over the EV breakeven, expressed as a fraction of the full
barrier range (tp + sl) — GLOBAL methodology constant.

NEVER per-proposal, for the same threshold-shopping reason as C_ATR. The
margin demands strictly-positive expected value (not mere breakeven) before a
trade is taken, absorbing calibration error in the meta's probabilities."""

DEFAULT_GRID_OFFSETS = (0.0, 0.03, 0.06, 0.10, 0.15)
"""Diagnostic grid offsets above p*. The first entry MUST be 0.0 so the grid
always contains the breakeven itself; the rest exist only so the threshold
grid report shows how the edge decays as the bar rises. Fixed a-priori."""


def compute_p_star(tp_atr_mult: float, sl_atr_mult: float) -> float:
    """EV-breakeven meta-probability threshold (Elkan 2001 cost-ratio theorem).

    p* = (sl + C_ATR + LAMBDA_MARGIN * (tp + sl)) / (tp + sl)

    where tp/sl are the triple-barrier multiples in ATR units. At p = p* the
    expected value of taking the trade, net of the global round-trip cost
    C_ATR, equals LAMBDA_MARGIN * (tp + sl) > 0.

    Example: tp=3, sl=1 -> (1 + 0.10 + 0.05*4) / 4 = 0.325.
    """
    barrier_range = float(tp_atr_mult) + float(sl_atr_mult)
    return (float(sl_atr_mult) + C_ATR + LAMBDA_MARGIN * barrier_range) / barrier_range


def ev_breakeven_grid(
    tp_atr_mult: float,
    sl_atr_mult: float,
    offsets: tuple[float, ...] = DEFAULT_GRID_OFFSETS,
) -> list[float]:
    """Threshold grid centred on the geometry's breakeven: [p*+o for o in offsets].

    Entries at or above 1.0 are dropped (unreachable for a probability); p*
    itself is always retained even if degenerate geometry pushes it near 1,
    so the grid is never empty.
    """
    p_star = compute_p_star(tp_atr_mult, sl_atr_mult)
    grid = sorted({round(p_star + float(o), 12) for o in offsets})
    clipped = [t for t in grid if t < 1.0]
    if not clipped:
        clipped = [min(p_star, 0.999999)]
    return clipped


def resolve_threshold_grid(
    metrics_cfg: dict, triple_barrier_cfg: dict
) -> tuple[list[float], Optional[float]]:
    """Resolve the orchestrator's effective threshold grid from config.

    Returns ``(grid, p_star)``:
    - no ``threshold_rule`` (legacy): the configured ``threshold_grid`` is
      returned verbatim and ``p_star`` is None — bit-identical behavior for
      every existing config.
    - ``threshold_rule: ev_breakeven_v1``: the grid is DERIVED from the run's
      barrier geometry (``tp_atr_mult``/``sl_atr_mult``) + the global cost
      constants; the configured fixed grid is ignored. ``p_star`` is the
      headline/fallback threshold the orchestrator should persist as
      ``audit_effective_threshold``.

    Any other rule raises — silently falling back to a fixed grid would
    reintroduce the exact failure this module exists to prevent.
    """
    rule = metrics_cfg.get("threshold_rule")
    if rule is None:
        return [float(t) for t in metrics_cfg["threshold_grid"]], None
    if rule != "ev_breakeven_v1":
        raise ValueError(
            f"unsupported metrics.threshold_rule={rule!r} (expected absent or 'ev_breakeven_v1')"
        )
    p_star = compute_p_star(triple_barrier_cfg["tp_atr_mult"], triple_barrier_cfg["sl_atr_mult"])
    if p_star >= 1.0:
        raise ValueError(
            f"degenerate barrier geometry: p*={p_star:.4f} >= 1.0 for "
            f"tp={triple_barrier_cfg['tp_atr_mult']}/sl={triple_barrier_cfg['sl_atr_mult']} — "
            f"no probability can clear breakeven; fix the geometry instead of running a mute audit"
        )
    offsets = tuple(float(o) for o in metrics_cfg.get("threshold_grid_offsets", DEFAULT_GRID_OFFSETS))
    grid = ev_breakeven_grid(
        triple_barrier_cfg["tp_atr_mult"], triple_barrier_cfg["sl_atr_mult"], offsets=offsets
    )
    return grid, p_star
