"""Regression test gate for the Option B refactor (Phase 2 v2.3.1).

Compares the current pipeline's outputs against a frozen Phase 1 v4 snapshot
in `tests/fixtures/phase1_v4_baseline.json`. The migration of
`scripts/run_xau_d1.py` from the coupled `select_threshold_inner_cv` to
`inner_oof_predict_proba + RefittingCalibratedPipeline + select_threshold_inner_cv`
changes the order of fits (clone-per-fold via the helper vs in-place loop
inside the old threshold function), which can introduce O(1e-4) numerical
drift on continuous metrics. Discrete decisions (stack flag, thresholds
picked from a discrete grid, n_trades at the chosen threshold) must match
exactly.

Workflow (the snapshot of Phase 1 v4 is generated ONCE, before the migration
is run on the user's branch; the current results come from re-running the
migrated pipeline):

    # 1. On a Phase 1 v4 commit (e.g. main at 17bdf12), the user already has
    #    results/clf_xau_d1/*/summary.json + psr_dsr.json from a prior run.
    #    Extract the snapshot WITHOUT re-running:
    .venv\\Scripts\\python.exe scripts/generate_phase1_baseline.py

    # 2. On the feat branch (current code, Option B):
    .venv\\Scripts\\python.exe scripts/run_xau_d1.py --config configs/xau_d1.yaml

    # 3. Run the regression test:
    .venv\\Scripts\\python.exe -m pytest tests/test_regression_phase1.py -v

If the baseline or current results are missing, this test is **skipped** with
a clear instruction. CI should fail the suite (not skip) once the baseline
is in place — flip `BASELINE_REQUIRED` to True for that.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "tests" / "fixtures" / "phase1_v4_baseline.json"
RESULTS_DIR = REPO_ROOT / "results" / "clf_xau_d1"

# Tolerances (matches plan v2.3.1 Task 14 table).
RTOL_PSR = 5e-3
RTOL_DSR = 5e-3
RTOL_SHARPE = 5e-3
PRIMARIES = ("ema_cross", "momentum_zscore")

# Set to True in CI once the baseline exists so a missing file is a hard fail.
BASELINE_REQUIRED = False


def _load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        msg = (
            f"baseline missing at {BASELINE_PATH}. "
            f"Generate with:\n  .venv\\Scripts\\python.exe scripts/generate_phase1_baseline.py"
        )
        if BASELINE_REQUIRED:
            raise FileNotFoundError(msg)
        pytest.skip(msg)
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _load_current_outputs() -> dict:
    """Read the current pipeline outputs into the same shape as the baseline.

    Assumes `scripts/run_xau_d1.py` has been run on the current branch.
    If results directories are missing, skip — running the pipeline inside
    a test is too expensive (~10-30 min). The regression check gates manual
    or CI re-runs, not unit-test execution.
    """
    if not RESULTS_DIR.exists():
        pytest.skip(
            f"no pipeline outputs at {RESULTS_DIR}. Run scripts/run_xau_d1.py first."
        )
    current: dict = {"primaries": {}}
    for primary in PRIMARIES:
        summary_path = RESULTS_DIR / primary / "summary.json"
        psr_dsr_path = RESULTS_DIR / primary / "psr_dsr.json"
        if not summary_path.exists():
            pytest.skip(f"missing {summary_path} — did the pipeline complete?")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        psr_dsr = (
            json.loads(psr_dsr_path.read_text(encoding="utf-8"))
            if psr_dsr_path.exists() else {}
        )
        current["primaries"][primary] = {
            "stack": (summary.get("stack_decision") or {}).get("stack"),
            "stack_n_models_passing": (summary.get("stack_decision") or {}).get(
                "n_models_passing"
            ),
            "best_model": summary.get("best_model"),
            "psr_per_model": dict(psr_dsr.get("psr", {})),
            "dsr_per_model": dict(psr_dsr.get("dsr", {})),
            "n_trades_per_fold_per_model": summary.get(
                "n_trades_per_fold_per_model", {}
            ),
            "sharpe_per_fold_per_model": summary.get(
                "sharpe_per_fold_per_model", {}
            ),
            "selected_threshold_per_fold_per_model": summary.get(
                "selected_threshold_per_fold_per_model", {}
            ),
        }
    return current


def _approx_equal(a, b, rtol: float, label: str) -> None:
    """Compare two floats with relative tolerance, handling None/NaN sanely."""
    if a is None and b is None:
        return
    if a is None or b is None:
        raise AssertionError(f"{label}: one side is None (baseline={a}, current={b})")
    if isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b):
        return
    if (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        raise AssertionError(f"{label}: one side is NaN (baseline={a}, current={b})")
    if abs(a) < 1e-9 and abs(b) < 1e-9:
        return
    rel = abs(a - b) / max(abs(a), abs(b))
    assert rel <= rtol, (
        f"{label}: drift {rel:.4e} > rtol={rtol:.0e} (baseline={a}, current={b})"
    )


def test_stack_decision_unchanged():
    """Stack decision (bool) must match exactly per primary."""
    baseline = _load_baseline()
    current = _load_current_outputs()
    for primary in PRIMARIES:
        b = baseline["primaries"][primary]["stack"]
        c = current["primaries"][primary]["stack"]
        assert b == c, f"{primary}: stack changed {b!r} → {c!r}"


def test_best_model_unchanged():
    """`best_model` is a discrete selection — must match exactly."""
    baseline = _load_baseline()
    current = _load_current_outputs()
    for primary in PRIMARIES:
        b = baseline["primaries"][primary]["best_model"]
        c = current["primaries"][primary]["best_model"]
        assert b == c, f"{primary}: best_model changed {b!r} → {c!r}"


def test_thresholds_per_model_per_fold_unchanged():
    """Selected thresholds come from a discrete grid. Must match exactly."""
    baseline = _load_baseline()
    current = _load_current_outputs()
    for primary in PRIMARIES:
        b = baseline["primaries"][primary]["selected_threshold_per_fold_per_model"]
        c = current["primaries"][primary]["selected_threshold_per_fold_per_model"]
        assert b.keys() == c.keys(), (
            f"{primary}: model set changed {set(b.keys())} → {set(c.keys())}"
        )
        for model in b:
            assert b[model] == c[model], (
                f"{primary}/{model}: thresholds changed {b[model]} → {c[model]}"
            )


def test_n_trades_per_model_per_fold_unchanged():
    """Trade counts at the selected thresholds. Discrete — must match exactly."""
    baseline = _load_baseline()
    current = _load_current_outputs()
    for primary in PRIMARIES:
        b = baseline["primaries"][primary]["n_trades_per_fold_per_model"]
        c = current["primaries"][primary]["n_trades_per_fold_per_model"]
        for model in b:
            assert b[model] == c[model], (
                f"{primary}/{model}: n_trades changed {b[model]} → {c[model]}"
            )


def test_psr_per_model_within_tolerance():
    """PSR is continuous. Must match within `rtol=5e-3`."""
    baseline = _load_baseline()
    current = _load_current_outputs()
    for primary in PRIMARIES:
        b = baseline["primaries"][primary]["psr_per_model"]
        c = current["primaries"][primary]["psr_per_model"]
        for model in b:
            _approx_equal(b[model], c.get(model), RTOL_PSR, f"{primary}/{model}/psr")


def test_dsr_per_model_within_tolerance():
    """DSR is continuous. Must match within `rtol=5e-3`."""
    baseline = _load_baseline()
    current = _load_current_outputs()
    for primary in PRIMARIES:
        b = baseline["primaries"][primary]["dsr_per_model"]
        c = current["primaries"][primary]["dsr_per_model"]
        for model in b:
            _approx_equal(b[model], c.get(model), RTOL_DSR, f"{primary}/{model}/dsr")


def test_sharpe_per_model_per_fold_within_tolerance():
    """Per-fold Sharpe. Continuous, must match within `rtol=5e-3` for finite
    values; NaN/None match each other."""
    baseline = _load_baseline()
    current = _load_current_outputs()
    for primary in PRIMARIES:
        b = baseline["primaries"][primary]["sharpe_per_fold_per_model"]
        c = current["primaries"][primary]["sharpe_per_fold_per_model"]
        for model in b:
            b_folds = b[model]
            c_folds = c[model]
            assert len(b_folds) == len(c_folds), (
                f"{primary}/{model}: fold count changed {len(b_folds)} → {len(c_folds)}"
            )
            for fold_idx, (bv, cv) in enumerate(zip(b_folds, c_folds)):
                _approx_equal(
                    bv, cv, RTOL_SHARPE,
                    f"{primary}/{model}/fold[{fold_idx}]/sharpe",
                )
