"""Phase 5 audit-of-the-audit harness (Day 1 blocking).

We use the existing M3 v3 classifier (`scripts/analyze_threshold_transferability.py`)
as the immutable validator for Phase 5 proposals. Before running ANY AI-generated
hypothesis through that validator, we test the validator itself with known-truth
probes:

  Probe NEG_random      — synthetic "random ±1 signal": many trades, ~0 Sharpe
                          per fold. Expect NOT_PROFITABLE.
  Probe POS_leakage     — synthetic perfect-foresight signal: many trades, very
                          high Sharpe per fold, regime pass. Expect STABLE.
                          DOCUMENTS the audit's leakage blindspot.
  Probe SHORT_in_decline — profitable short in a strictly-monotone-down OOS:
                          max_dd is large but max_rally is ~0, so regime gate
                          fails on the rally side. Expect REGIME_LIMITED.
                          DOCUMENTS the audit's short-bias (price-not-equity).
  Probe replay_v3_top   — pick a real candidate from threshold_transferability_
                          overnight_v3.json and verify re-classifying its
                          per_fold metrics reproduces the recorded class.
  Probe replay_v3_fail  — pick a NOT_PROFITABLE candidate and verify the same.

Run all probes:
  uv run python -m phase5.audit_audit --probe all

The report is written to ``results/phase5/audit_audit_report.json`` and a
markdown summary to ``results/phase5/audit_audit_report.md``. Both are
required artifacts for the Day 1 go/no-go.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Import the v3 classifier and regime helpers directly. We do NOT modify them.
import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_threshold_transferability import (
    _classify_transferability,
    _regime_diversity,
    _max_drawdown,
    _max_rally,
)


V3_JSON_PATH = _REPO_ROOT / "results" / "threshold_transferability_overnight_v3.json"


@dataclass
class ProbeResult:
    probe: str
    expected: str
    observed: str
    inputs: dict
    outputs: dict
    passed: bool
    note: str = ""


def _probe_neg_random() -> ProbeResult:
    """Random ±1 signal: many trades, ~0 Sharpe each fold, any regime."""
    per_fold_n = [200, 200, 200, 200]
    per_fold_sharpe = [0.05, -0.03, 0.01, -0.02]
    regime_pass = True  # doesn't matter — NOT_PROFITABLE takes priority
    observed = _classify_transferability(per_fold_n, per_fold_sharpe, regime_pass)
    expected = "NOT_PROFITABLE"
    return ProbeResult(
        probe="NEG_random",
        expected=expected,
        observed=observed,
        inputs={"per_fold_n": per_fold_n, "per_fold_sharpe": per_fold_sharpe, "regime_pass": regime_pass},
        outputs={"transferability": observed},
        passed=(observed == expected),
        note="random ±1 signal: classifier MUST reject as NOT_PROFITABLE (median Sharpe ≤ 0).",
    )


def _probe_pos_leakage() -> ProbeResult:
    """Perfect-foresight signal: high Sharpe every fold, regime passes."""
    per_fold_n = [150, 150, 150, 150]
    per_fold_sharpe = [8.0, 7.5, 9.0, 6.5]
    regime_pass = True
    observed = _classify_transferability(per_fold_n, per_fold_sharpe, regime_pass)
    expected = "STABLE"
    return ProbeResult(
        probe="POS_leakage",
        expected=expected,
        observed=observed,
        inputs={"per_fold_n": per_fold_n, "per_fold_sharpe": per_fold_sharpe, "regime_pass": regime_pass},
        outputs={"transferability": observed},
        passed=(observed == expected),
        note=(
            "DOCUMENTED BLINDSPOT: M3 has no mechanism to detect future-window leakage. "
            "A leaking strategy with high per-fold Sharpe is classified STABLE — this is "
            "by design (the audit checks transferability, not leakage). Leakage must be "
            "blocked upstream (in the data pipeline / signal builder), not in M3."
        ),
    )


def _probe_short_in_decline() -> ProbeResult:
    """Profitable short in a strictly-monotone-down OOS window.

    A real short strategy here would have positive Sharpe (because asset is
    declining and we're short). The regime gate, however, measures the
    underlying asset's price diversity — and a monotone decline has near-zero
    max_rally. So regime_diversity.pass = False -> REGIME_LIMITED.

    This DOCUMENTS the M3 v3 short-bias: a profitable short in a sustained
    decline is unfairly flagged because the audit measures asset price, not
    strategy equity. (The audit's own docstring acknowledges this on
    scripts/analyze_threshold_transferability.py:115-119.)
    """
    n_bars = 500
    close = np.linspace(100.0, 60.0, n_bars)  # strict decline -40%
    regime = _regime_diversity(close, min_move=0.15)
    per_fold_n = [120, 120, 120, 120]
    per_fold_sharpe = [1.8, 1.5, 2.0, 1.6]  # profitable short
    observed = _classify_transferability(per_fold_n, per_fold_sharpe, regime["pass"])
    expected = "REGIME_LIMITED"
    return ProbeResult(
        probe="SHORT_in_decline",
        expected=expected,
        observed=observed,
        inputs={
            "per_fold_n": per_fold_n,
            "per_fold_sharpe": per_fold_sharpe,
            "regime_diversity": regime,
        },
        outputs={"transferability": observed, "regime_pass": regime["pass"]},
        passed=(observed == expected),
        note=(
            "DOCUMENTED BIAS: M3's _regime_diversity measures asset price (max_dd / max_rally), "
            "NOT strategy equity. A profitable short during a sustained decline (rally≈0) "
            "fails the rally side and gets flagged REGIME_LIMITED even though the strategy "
            "earned money. Side-branch patch feat/phase5-m3-v3.1-shortbias to be drafted."
        ),
    )


def _probe_mono_up() -> ProbeResult:
    """A monotone uptrend has max_rally large, max_dd ~ 0 -> regime fail.

    Complementary to SHORT_in_decline. A LONG strategy in a sustained uptrend
    would also be REGIME_LIMITED. Same bias, opposite sign.
    """
    n_bars = 500
    close = np.linspace(80.0, 140.0, n_bars)  # strict +75%
    regime = _regime_diversity(close, min_move=0.15)
    per_fold_n = [120, 120, 120, 120]
    per_fold_sharpe = [1.8, 1.5, 2.0, 1.6]
    observed = _classify_transferability(per_fold_n, per_fold_sharpe, regime["pass"])
    expected = "REGIME_LIMITED"
    return ProbeResult(
        probe="LONG_in_uptrend",
        expected=expected,
        observed=observed,
        inputs={
            "per_fold_n": per_fold_n,
            "per_fold_sharpe": per_fold_sharpe,
            "regime_diversity": regime,
        },
        outputs={"transferability": observed, "regime_pass": regime["pass"]},
        passed=(observed == expected),
        note=(
            "Symmetric to SHORT_in_decline: a LONG strategy in a sustained uptrend (dd≈0) "
            "is flagged REGIME_LIMITED. The bias is direction-agnostic; same patch fixes both."
        ),
    )


def _probe_replay_v3(target_class: str) -> ProbeResult:
    """Replay a real candidate from the v3 output and verify reproduction.

    Picks the FIRST candidate in the v3 JSON whose ``transferability`` matches
    ``target_class``, extracts its per_fold metrics + regime_diversity.pass,
    and re-runs ``_classify_transferability``. Pass iff observed == recorded.
    """
    if not V3_JSON_PATH.exists():
        return ProbeResult(
            probe=f"replay_v3_{target_class}",
            expected=target_class,
            observed="N/A",
            inputs={"v3_json_path": str(V3_JSON_PATH)},
            outputs={},
            passed=False,
            note=f"V3 JSON not found at {V3_JSON_PATH}; cannot replay.",
        )
    payload = json.loads(V3_JSON_PATH.read_text(encoding="utf-8"))
    target = None
    for c in payload.get("candidates", []):
        if c.get("threshold_50", {}).get("transferability") == target_class:
            target = c
            break
    if target is None:
        return ProbeResult(
            probe=f"replay_v3_{target_class}",
            expected=target_class,
            observed="N/A",
            inputs={"v3_json_path": str(V3_JSON_PATH)},
            outputs={},
            passed=False,
            note=f"No candidate with transferability={target_class!r} found in v3 output.",
        )
    t50 = target["threshold_50"]
    per_fold_sharpe = [
        float(s) if s is not None and not (isinstance(s, float) and np.isnan(s)) else float("nan")
        for s in t50["per_fold_sharpe"]
    ]
    per_fold_n = list(t50["per_fold_n_trades"])
    regime_pass = t50.get("regime_diversity", {}).get("pass")
    observed = _classify_transferability(per_fold_n, per_fold_sharpe, regime_pass)
    return ProbeResult(
        probe=f"replay_v3_{target_class}",
        expected=target_class,
        observed=observed,
        inputs={
            "candidate": {
                "asset": target.get("asset"),
                "engine_dir": target.get("engine_dir"),
                "primary": target.get("primary"),
                "model": target.get("model"),
            },
            "per_fold_n_trades": per_fold_n,
            "per_fold_sharpe": per_fold_sharpe,
            "regime_pass": regime_pass,
        },
        outputs={"transferability": observed, "recorded": target_class},
        passed=(observed == target_class),
        note="Reproduces recorded v3 classification by re-running the classifier on the same per-fold inputs.",
    )


PROBES = {
    "NEG_random": _probe_neg_random,
    "POS_leakage": _probe_pos_leakage,
    "SHORT_in_decline": _probe_short_in_decline,
    "LONG_in_uptrend": _probe_mono_up,
    "replay_v3_MARGINAL_2FOLDS": lambda: _probe_replay_v3("MARGINAL_2FOLDS"),
    "replay_v3_NOT_PROFITABLE": lambda: _probe_replay_v3("NOT_PROFITABLE"),
    "replay_v3_1FOLD_CONCENTRATED": lambda: _probe_replay_v3("1FOLD_CONCENTRATED"),
}


def _serialize_inputs(d: dict) -> dict:
    """Make probe inputs JSON-serializable (numpy / NaN handling)."""
    def _conv(v):
        if isinstance(v, float):
            return v if np.isfinite(v) else None
        if isinstance(v, np.ndarray):
            return [float(x) if np.isfinite(x) else None for x in v.tolist()]
        if isinstance(v, list):
            return [_conv(x) for x in v]
        if isinstance(v, dict):
            return {k: _conv(x) for k, x in v.items()}
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v) if np.isfinite(v) else None
        return v
    return _conv(d)


def run_all(out_dir: Path) -> dict:
    """Run all probes; persist JSON + markdown report; return summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[ProbeResult] = []
    for name, fn in PROBES.items():
        r = fn()
        results.append(r)
    report = {
        "n_probes": len(results),
        "n_passed": sum(1 for r in results if r.passed),
        "n_failed": sum(1 for r in results if not r.passed),
        "probes": [
            {
                "probe": r.probe,
                "expected": r.expected,
                "observed": r.observed,
                "passed": r.passed,
                "inputs": _serialize_inputs(r.inputs),
                "outputs": _serialize_inputs(r.outputs),
                "note": r.note,
            }
            for r in results
        ],
    }
    json_path = out_dir / "audit_audit_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = ["# Phase 5 Audit-the-Audit Report", "",
                f"Probes run: **{report['n_probes']}**, passed: **{report['n_passed']}**, failed: **{report['n_failed']}**",
                ""]
    for p in report["probes"]:
        status = "PASS" if p["passed"] else "FAIL"
        md_lines.append(f"## {p['probe']} — {status}")
        md_lines.append("")
        md_lines.append(f"- Expected: `{p['expected']}`")
        md_lines.append(f"- Observed: `{p['observed']}`")
        if p["note"]:
            md_lines.append(f"- Note: {p['note']}")
        md_lines.append("")
    md_path = out_dir / "audit_audit_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Probes: {report['n_passed']}/{report['n_probes']} passed")
    for p in report["probes"]:
        marker = "PASS" if p["passed"] else "FAIL"
        print(f"  [{marker}] {p['probe']:30s}  expected={p['expected']}  observed={p['observed']}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="all", help="probe name or 'all'")
    ap.add_argument("--out", default="results/phase5", help="output directory for the report")
    args = ap.parse_args()
    out_dir = Path(args.out)
    if args.probe == "all":
        report = run_all(out_dir)
        return 0 if report["n_failed"] == 0 else 1
    if args.probe not in PROBES:
        print(f"Unknown probe: {args.probe!r}. Available: {list(PROBES)}", flush=True)
        return 2
    r = PROBES[args.probe]()
    print(f"[{ 'PASS' if r.passed else 'FAIL' }] {r.probe}: expected={r.expected} observed={r.observed}")
    print(f"Note: {r.note}")
    return 0 if r.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
