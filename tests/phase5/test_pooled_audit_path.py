"""B0010 — pooled-universe audit mode in run_proposal (Task 4).

Covers the three additive functions:
  (a) build_transient_pooled_config — overlays a Proposal onto the M3 pooled
      template (configs/equity_m3_d1.yaml) and preserves the sections
      scripts/run_multi_h4.py::_run_one_pool requires.
  (b) grade_pooled_v1 — the pooled_v1 grading arithmetic on synthetic
      per-asset artifacts (summary.json + metrics_per_fold.json), independent
      of any subprocess.
  (c) run_pooled_audit's starved path — a pooled event count below
      wf_event_floor must short-circuit to status "event_floor" WITHOUT
      attempting the full (heavy) pooled subprocess.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest
import yaml

from phase5 import run_proposal
from phase5.proposal import load_proposal


def _proposal_dict() -> dict:
    return {
        "id": "TEST-B0010-POOLED",
        "asset": "POOL",
        "asset_class": "equity",
        "regime_scope": ["BULL_QUIET", "BEAR_QUIET"],
        "hypothesis": (
            "Across the M3 large-cap pool, a 20/50 EMA cross with a 3R "
            "asymmetric barrier isolates trend continuations that a "
            "cross-sectional meta-labeler can price better than the raw "
            "primary, because most whipsaw bars are noise the meta learns "
            "to filter."
        ),
        "causal_story": (
            "EMA separation only exceeds the dead-zone after a sustained "
            "directional run, so entries are endogenous to realized price "
            "action; the meta sees cross-sectional features (rank, "
            "dispersion) that a single-name audit cannot construct, letting "
            "it discriminate true trend continuations from pool-wide noise."
        ),
        "primary": "ema_cross",
        "primary_params": {"fast": 20, "slow": 50},
        "feature_overrides": {"add": [], "drop": []},
        "regime_gate": {"mode": "filter_events", "feature_added": False},
        "falsification_criterion": {
            "audit_class_in": ["STABLE", "MARGINAL_2FOLDS"],
            "median_active_fold_sharpe_min": 0.5,
            "n_trades_total_min": 60,
        },
        "lookahead_attestation": {"checklist_version": "v1", "linter_passed": None},
        "lookahead_shape_attestation": {
            "target_regime_episode_ordinals": [1, 2],
            "cross_asset_falsifiable_in": ["XLK"],
        },
        "barrier_geometry_attestation": {"tp_atr_mult": 3.0, "sl_atr_mult": 1.0},
        "parent_proposal": None,
        "git_sha_at_propose": None,
        "diagnostic_only": False,
    }


def _load_proposal(tmp_path: Path):
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(_proposal_dict()), encoding="utf-8")
    return load_proposal(path)


def _load_proposal_with_criterion(tmp_path: Path, criterion: dict, *, name: str = "proposal_fc.json"):
    payload = _proposal_dict()
    payload["falsification_criterion"] = criterion
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_proposal(path)


# --------------------------------------------------------------------------- #
# (a) build_transient_pooled_config
# --------------------------------------------------------------------------- #

def test_build_transient_pooled_config_overlays_and_preserves_template(tmp_path, monkeypatch):
    monkeypatch.setattr(run_proposal, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(run_proposal, "POOLED_RESULTS_DIR", tmp_path / "results" / "phase5_pooled")
    p = _load_proposal(tmp_path)

    cfg_path = run_proposal.build_transient_pooled_config(p)

    assert cfg_path.exists()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Primary + params overlaid
    assert cfg["primary"]["candidates"] == ["ema_cross"]
    assert cfg["primary"]["ema_cross"]["fast"] == 20
    assert cfg["primary"]["ema_cross"]["slow"] == 50

    # Barrier geometry from the proposal; horizon untouched (stays template's 40)
    assert cfg["triple_barrier"]["tp_atr_mult"] == 3.0
    assert cfg["triple_barrier"]["sl_atr_mult"] == 1.0
    assert cfg["triple_barrier"]["horizon"] == 40

    # regime_scope + feature_overrides at top level (run_pooled_equity_d1.py contract)
    assert set(cfg["regime_scope"]) == {"BULL_QUIET", "BEAR_QUIET"}
    # B0014: the proposal's gate mode reaches the pooled runner verbatim.
    assert cfg["regime_gate_mode"] == "filter_events"
    assert cfg["feature_overrides_add"] == []
    assert cfg["feature_overrides_drop"] == []

    # features overlay
    assert cfg["features"]["cross_sectional"] is True
    assert cfg["features"]["gld_volume"] is False

    # fixed-ev threshold wiring reuses effective_threshold()
    assert cfg["threshold_selection"]["method"] == "fixed_ev"
    assert cfg["metrics"]["audit_effective_threshold"] == pytest.approx(p.effective_threshold())

    # output_dir
    assert cfg["output_dir"] == str(tmp_path / "results" / "phase5_pooled" / p.id)

    # Sections _run_one_pool requires must survive untouched from the template
    for section in ("models", "calibration", "stacking", "best_model", "meta_pooling", "dry_run"):
        assert section in cfg, f"template section {section!r} did not survive the overlay"
    assert cfg["models"] == ["xgb", "lgbm", "rf", "lr"]
    assert cfg["meta_pooling"]["scope"] == "within_class"


def test_build_transient_pooled_config_fails_fast_on_missing_required_param(tmp_path, monkeypatch):
    monkeypatch.setattr(run_proposal, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(run_proposal, "POOLED_RESULTS_DIR", tmp_path / "results")
    payload = _proposal_dict()
    payload["primary_params"] = {}  # missing required "fast"/"slow"
    path = tmp_path / "bad_proposal.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    p = load_proposal(path)

    from pipeline.primary_contracts import PrimaryParamError
    with pytest.raises(PrimaryParamError):
        run_proposal.build_transient_pooled_config(p)


# --------------------------------------------------------------------------- #
# (b) grade_pooled_v1 — synthetic per-asset artifacts
# --------------------------------------------------------------------------- #

def _write_asset_artifacts(root: Path, asset: str, primary: str, best_model: str,
                           fold_rows: list[dict]) -> None:
    d = root / asset / primary
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({"best_model": best_model}), encoding="utf-8")
    rows = [
        {"fold": r["fold"], "primary": primary, "model": r["model"],
         "threshold": 0.5, "sharpe_net": r["sharpe_net"], "n_trades": r["n_trades"]}
        for r in fold_rows
    ]
    (d / "metrics_per_fold.json").write_text(json.dumps(rows), encoding="utf-8")


def test_grade_pooled_v1_arithmetic(tmp_path):
    primary = "ema_cross"
    # Asset A: best_model=xgb. fold0 active (n=40, sharpe=1.0); fold1 inactive
    # (n=10 < 30) with a deliberately huge sharpe=5.0 that must NOT leak into
    # the median (regression guard for "NaN cells skipped, never zero-filled
    # or blindly included").
    _write_asset_artifacts(
        tmp_path, "AAA", primary, "xgb",
        [
            {"fold": 0, "model": "xgb", "sharpe_net": 1.0, "n_trades": 40},
            {"fold": 1, "model": "xgb", "sharpe_net": 5.0, "n_trades": 10},
            # non-best-model row must be ignored entirely
            {"fold": 0, "model": "lgbm", "sharpe_net": 99.0, "n_trades": 500},
        ],
    )
    # Asset B: best_model=lgbm. Both folds active but net-negative aggregate
    # -> must fail the breadth gate despite clearing n_trades_total_min.
    _write_asset_artifacts(
        tmp_path, "BBB", primary, "lgbm",
        [
            {"fold": 0, "model": "lgbm", "sharpe_net": -2.0, "n_trades": 35},
            {"fold": 1, "model": "lgbm", "sharpe_net": 0.1, "n_trades": 50},
        ],
    )

    grading = run_proposal.grade_pooled_v1(["AAA", "BBB"], primary, tmp_path)

    # n_trades_total: 40 + 10 (AAA) + 35 + 50 (BBB) = 135
    assert grading["n_trades_total"] == 135

    # Active cells only: AAA fold0 (1.0), BBB fold0 (-2.0), BBB fold1 (0.1)
    # median of [-2.0, 0.1, 1.0] = 0.1
    assert grading["median_active_fold_sharpe"] == pytest.approx(0.1)

    # Breadth: AAA total=50>=30, aggregate=nanmedian([1.0])=1.0>0 -> pass.
    # BBB total=85>=30, aggregate=nanmedian([-2.0,0.1])=-0.95<0 -> fail.
    assert grading["breadth"] == 1
    assert grading["per_asset"]["AAA"]["breadth_pass"] is True
    assert grading["per_asset"]["BBB"]["breadth_pass"] is False
    assert grading["per_asset"]["AAA"]["n_trades_total"] == 50
    assert grading["per_asset"]["BBB"]["n_trades_total"] == 85
    assert grading["per_asset"]["AAA"]["aggregate_sharpe"] == pytest.approx(1.0)
    assert grading["per_asset"]["BBB"]["aggregate_sharpe"] == pytest.approx(-0.95)


def test_grade_pooled_v1_no_active_folds_anywhere_is_nan_not_zero(tmp_path):
    primary = "ema_cross"
    _write_asset_artifacts(
        tmp_path, "CCC", primary, "rf",
        [
            {"fold": 0, "model": "rf", "sharpe_net": 3.0, "n_trades": 5},
            {"fold": 1, "model": "rf", "sharpe_net": -1.0, "n_trades": 2},
        ],
    )
    grading = run_proposal.grade_pooled_v1(["CCC"], primary, tmp_path)
    assert grading["n_trades_total"] == 7
    assert grading["median_active_fold_sharpe"] is None  # NaN, JSON-serialized as null
    assert grading["breadth"] == 0
    assert grading["per_asset"]["CCC"]["aggregate_sharpe"] is None


def test_grade_pooled_v1_missing_asset_artifacts_does_not_raise(tmp_path):
    grading = run_proposal.grade_pooled_v1(["NOPE"], "ema_cross", tmp_path)
    assert grading["n_trades_total"] == 0
    assert grading["breadth"] == 0
    assert grading["per_asset"]["NOPE"]["breadth_pass"] is False


# --------------------------------------------------------------------------- #
# (c) run_pooled_audit starved path
# --------------------------------------------------------------------------- #

def test_run_pooled_audit_event_floor_short_circuits(tmp_path, monkeypatch):
    monkeypatch.setattr(run_proposal, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(run_proposal, "POOLED_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(run_proposal, "AUDIT_RESULTS_DIR", tmp_path / "audit_results")
    p = _load_proposal(tmp_path)

    # Starved counts: far below any plausible wf_event_floor for the
    # template's default walk_forward geometry (n_folds=4, train_min_bars=1500).
    starved_counts = [
        {"asset": "AAA", "primary": "ema_cross", "n_events": 3},
        {"asset": "BBB", "primary": "ema_cross", "n_events": 2},
    ]
    monkeypatch.setattr(run_proposal, "count_pooled_events_subprocess",
                        lambda cfg_path: starved_counts)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("full pooled subprocess must NOT be invoked below the event floor")

    monkeypatch.setattr(run_proposal, "run_pooled_pipeline_subprocess", _fail_if_called)
    monkeypatch.setattr(run_proposal, "run_long_short_split_subprocess", _fail_if_called)

    record = run_proposal.run_pooled_audit(p)

    assert record["status"] == "event_floor"
    assert record["mode"] == "pooled_universe"
    assert record["member_event_counts"] == starved_counts
    assert record["pooled_event_floor"]["total_events"] == 5
    assert record["pooled_event_floor"]["wf_event_floor"] > 5

    out_path = tmp_path / "audit_results" / f"{p.id}.json"
    assert out_path.exists()
    persisted = json.loads(out_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "event_floor"
    assert persisted["mode"] == "pooled_universe"


def test_run_pooled_audit_effective_n_floor_short_circuits(tmp_path, monkeypatch):
    """B0013: the binding pooled quantity is effective-N, not raw N. T004D1
    (2026-07-04) passed the raw gate with 2,852 events but ran the full audit
    at effective_n_rho1 = 521.7 < floor 799 — a formally floor-invalid verdict
    that burned ~30 min. Raw-count pass + effective-N fail must short-circuit
    to event_floor BEFORE the full pipeline subprocess."""
    monkeypatch.setattr(run_proposal, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(run_proposal, "POOLED_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(run_proposal, "AUDIT_RESULTS_DIR", tmp_path / "audit_results")
    p = _load_proposal(tmp_path)

    # Raw counts comfortably above the raw wf_event_floor...
    fat_counts = [
        {"asset": "AAA", "primary": "ema_cross", "n_events": 1500},
        {"asset": "BBB", "primary": "ema_cross", "n_events": 1500},
    ]
    monkeypatch.setattr(run_proposal, "count_pooled_events_subprocess",
                        lambda cfg_path: fat_counts)

    # ...but the concurrency-adjusted effective-N is below the same floor.
    starved_eff = {"primary": "ema_cross", "raw_n": 3000,
                   "effective_n_rho1": 120.5, "fit_weight_mode": "rho1_pooled"}
    monkeypatch.setattr(run_proposal, "effective_n_pooled_subprocess",
                        lambda cfg_path, primary: starved_eff)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("full pooled subprocess must NOT run below the effective-N floor")

    monkeypatch.setattr(run_proposal, "run_pooled_pipeline_subprocess", _fail_if_called)
    monkeypatch.setattr(run_proposal, "run_long_short_split_subprocess", _fail_if_called)

    record = run_proposal.run_pooled_audit(p)

    assert record["status"] == "event_floor"
    assert record["mode"] == "pooled_universe"
    assert record["pooled_effective_n"]["effective_n_rho1"] == 120.5
    assert record["pooled_effective_n"]["wf_event_floor"] > 120.5
    assert any("effective" in e for e in record["errors"])

    persisted = json.loads((tmp_path / "audit_results" / f"{p.id}.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "event_floor"


def test_run_pooled_audit_effective_n_subprocess_failure_is_fail_loud(tmp_path, monkeypatch):
    """A broken effective-N measurement must not silently degrade to the raw
    gate (that would resurrect the exact B0013 hole)."""
    monkeypatch.setattr(run_proposal, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(run_proposal, "POOLED_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(run_proposal, "AUDIT_RESULTS_DIR", tmp_path / "audit_results")
    p = _load_proposal(tmp_path)

    fat_counts = [{"asset": "AAA", "primary": "ema_cross", "n_events": 3000}]
    monkeypatch.setattr(run_proposal, "count_pooled_events_subprocess",
                        lambda cfg_path: fat_counts)

    def _raise(cfg_path, primary):
        raise RuntimeError("boom: no effective_n json emitted")

    monkeypatch.setattr(run_proposal, "effective_n_pooled_subprocess", _raise)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("full pooled subprocess must NOT run when effective-N is unmeasurable")

    monkeypatch.setattr(run_proposal, "run_pooled_pipeline_subprocess", _fail_if_called)

    record = run_proposal.run_pooled_audit(p)
    assert record["status"] == "subprocess_failed"


def test_run_pooled_audit_count_subprocess_failure_persists_subprocess_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(run_proposal, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(run_proposal, "POOLED_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(run_proposal, "AUDIT_RESULTS_DIR", tmp_path / "audit_results")
    p = _load_proposal(tmp_path)

    def _raise(cfg_path):
        raise RuntimeError("boom: subprocess did not emit member_event_counts.json")

    monkeypatch.setattr(run_proposal, "count_pooled_events_subprocess", _raise)

    record = run_proposal.run_pooled_audit(p)

    assert record["status"] == "subprocess_failed"
    assert record["mode"] == "pooled_universe"
    assert any("boom" in e for e in record["errors"])


def test_run_pooled_audit_refuses_phase5_primary_before_any_subprocess(tmp_path, monkeypatch):
    """B0010 review follow-up: pooled_universe v1 has no B0015b input-disjointness
    check and never calls apply_primary_feature_blacklist, so a phase5_* custom
    primary must be refused fail-loud before any subprocess runs — not silently
    audited with weaker guarantees than the single-name path."""
    monkeypatch.setattr(run_proposal, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(run_proposal, "POOLED_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(run_proposal, "AUDIT_RESULTS_DIR", tmp_path / "audit_results")
    payload = _proposal_dict()
    payload["primary"] = "phase5_whatever"
    path = tmp_path / "phase5_proposal.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    p = load_proposal(path)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("no subprocess wrapper may be invoked for a refused phase5_* primary")

    monkeypatch.setattr(run_proposal, "count_pooled_events_subprocess", _fail_if_called)
    monkeypatch.setattr(run_proposal, "run_pooled_pipeline_subprocess", _fail_if_called)
    monkeypatch.setattr(run_proposal, "run_long_short_split_subprocess", _fail_if_called)

    record = run_proposal.run_pooled_audit(p)

    assert record["status"] == "failed_validation"
    assert record["mode"] == "pooled_universe"
    assert any("phase5_" in e and "B0013" in e for e in record["errors"])

    out_path = tmp_path / "audit_results" / f"{p.id}.json"
    assert out_path.exists()
    persisted = json.loads(out_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed_validation"
    assert persisted["mode"] == "pooled_universe"


# --------------------------------------------------------------------------- #
# (d) _criterion_eval_pooled_v1 — direct unit coverage
# --------------------------------------------------------------------------- #

def test_criterion_eval_pooled_v1_direct(tmp_path):
    criterion = {
        "audit_class_in": ["STABLE", "MARGINAL_2FOLDS"],
        "median_active_fold_sharpe_min": 0.5,
        "n_trades_total_min": 50,
        "per_episode_survival_fraction": 0.6,
        "per_episode_min_trades": 5,
    }
    p = _load_proposal_with_criterion(tmp_path, criterion)

    # (a) computable keys pass and record the observed values; non-computable
    # keys are explicitly flagged rather than silently skipped or forced pass.
    grading_ok = {"median_active_fold_sharpe": 0.8, "n_trades_total": 120}
    out_ok = run_proposal._criterion_eval_pooled_v1(p, grading_ok)

    assert out_ok["median_active_fold_sharpe_min"] == {
        "computed": 0.8, "threshold": 0.5, "passed": True,
    }
    assert out_ok["n_trades_total_min"] == {
        "computed": 120, "threshold": 50, "passed": True,
    }
    assert out_ok["audit_class_in"] == "not_applicable_pooled_v1"
    assert out_ok["per_episode_survival_fraction"] == "not_applicable_pooled_v1"
    assert out_ok["dsr_min"] == "not_applicable_pooled_v1"

    # (b) median None (no active fold anywhere in the pool) must not raise and
    # must not claim a pass — it's a "no measurement", not a "0 measurement".
    grading_none = {"median_active_fold_sharpe": None, "n_trades_total": 10}
    out_none = run_proposal._criterion_eval_pooled_v1(p, grading_none)

    assert out_none["median_active_fold_sharpe_min"]["computed"] is None
    assert out_none["median_active_fold_sharpe_min"]["passed"] is False
    assert out_none["median_active_fold_sharpe_min"]["threshold"] == 0.5
    # n_trades_total_min is independent of the sharpe branch and still evaluates.
    assert out_none["n_trades_total_min"] == {
        "computed": 10, "threshold": 50, "passed": False,
    }


# --------------------------------------------------------------------------- #
# (e) run_pooled_audit — full mocked FULL-SUCCESS path
# --------------------------------------------------------------------------- #

def _fake_count_pooled_events_subprocess(assets: list[str], primary: str, n_events: int):
    """Mimics count_pooled_events_subprocess: writes member_event_counts.json
    at the transient config's output_dir and returns the same list, well above
    any plausible wf_event_floor so run_pooled_audit proceeds past the gate.
    """
    def _inner(cfg_path):
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        out_dir = Path(cfg["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        counts = [{"asset": a, "primary": primary, "n_events": n_events} for a in assets]
        (out_dir / "member_event_counts.json").write_text(json.dumps(counts), encoding="utf-8")
        return counts
    return _inner


def _fake_run_pooled_pipeline_subprocess(asset_artifacts: dict, primary: str):
    """Mimics run_pooled_pipeline_subprocess: fabricates the per-asset artifact
    tree (summary.json + metrics_per_fold.json) _run_one_pool would have
    produced, then reports success.
    """
    def _inner(cfg_path, dry_run=False):
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
        out_dir = Path(cfg["output_dir"])
        for asset, (best_model, fold_rows) in asset_artifacts.items():
            _write_asset_artifacts(out_dir, asset, primary, best_model, fold_rows)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    return _inner


def _fake_run_long_short_split_subprocess(payload: dict):
    """Mimics report_long_short_split.py: writes long_short_split.json under
    the pooled output dir and reports success.
    """
    def _inner(out_dir, cost_bps):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "long_short_split.json").write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    return _inner


def test_run_pooled_audit_success_path_end_to_end_mocked(tmp_path, monkeypatch):
    monkeypatch.setattr(run_proposal, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(run_proposal, "POOLED_RESULTS_DIR", tmp_path / "results" / "phase5_pooled")
    monkeypatch.setattr(run_proposal, "AUDIT_RESULTS_DIR", tmp_path / "audit_results")
    p = _load_proposal(tmp_path)
    primary = p.primary  # "ema_cross" per _proposal_dict

    # (1) count-events-only subprocess: well above the event floor for both
    # pool members so the full pooled subprocess IS invoked.
    monkeypatch.setattr(
        run_proposal, "count_pooled_events_subprocess",
        _fake_count_pooled_events_subprocess(["AAA", "BBB"], primary, 5000),
    )

    # (1b) effective-N gate (B0013): healthy — above any plausible floor.
    monkeypatch.setattr(
        run_proposal, "effective_n_pooled_subprocess",
        lambda cfg_path, prim: {"primary": prim, "raw_n": 10000,
                                "effective_n_rho1": 5000.0,
                                "fit_weight_mode": "rho1_pooled"},
    )

    # (2) full pooled subprocess: fabricate 2 assets x 1 primary artifacts,
    # mirroring the real metrics_per_fold.json / summary.json schema.
    asset_artifacts = {
        "AAA": ("xgb", [
            {"fold": 0, "model": "xgb", "sharpe_net": 1.0, "n_trades": 60},
            {"fold": 1, "model": "xgb", "sharpe_net": 0.6, "n_trades": 70},
        ]),
        "BBB": ("lgbm", [
            {"fold": 0, "model": "lgbm", "sharpe_net": 0.9, "n_trades": 50},
            {"fold": 1, "model": "lgbm", "sharpe_net": 1.1, "n_trades": 55},
        ]),
    }
    monkeypatch.setattr(
        run_proposal, "run_pooled_pipeline_subprocess",
        _fake_run_pooled_pipeline_subprocess(asset_artifacts, primary),
    )

    # (3) long/short split subprocess: minimal fabricated payload.
    long_short_payload = {"note": "fabricated for test", "n_long": 10, "n_short": 8}
    monkeypatch.setattr(
        run_proposal, "run_long_short_split_subprocess",
        _fake_run_long_short_split_subprocess(long_short_payload),
    )

    record = run_proposal.run_pooled_audit(p)

    assert record["status"] == "completed_pending_human_read"
    assert record["mode"] == "pooled_universe"
    assert record["grading_version"] == "pooled_v1"

    # Grading arithmetic, hand-computed from the fabricated fold rows above:
    #   n_trades_total = 60+70+50+55 = 235
    #   active sharpes (all n_trades>=30): [1.0, 0.6, 0.9, 1.1] -> median 0.95
    #   AAA: total=130>=30, agg=nanmedian([1.0,0.6])=0.8>0 -> breadth pass
    #   BBB: total=105>=30, agg=nanmedian([0.9,1.1])=1.0>0 -> breadth pass
    grading = record["grading"]
    assert grading["n_trades_total"] == 235
    assert grading["median_active_fold_sharpe"] == pytest.approx(0.95)
    assert grading["breadth"] == 2
    assert grading["per_asset"]["AAA"]["n_trades_total"] == 130
    assert grading["per_asset"]["AAA"]["breadth_pass"] is True
    assert grading["per_asset"]["BBB"]["n_trades_total"] == 105
    assert grading["per_asset"]["BBB"]["breadth_pass"] is True

    # criterion_eval: proposal's own falsification_criterion has
    # median_active_fold_sharpe_min=0.5, n_trades_total_min=60 (see
    # _proposal_dict) -- both clear against the fabricated grading, and the
    # single-name-only keys are flagged not-applicable rather than skipped.
    ce = record["criterion_eval"]
    assert ce["median_active_fold_sharpe_min"] == {
        "computed": pytest.approx(0.95), "threshold": 0.5, "passed": True,
    }
    assert ce["n_trades_total_min"] == {
        "computed": 235, "threshold": 60, "passed": True,
    }
    assert ce["audit_class_in"] == "not_applicable_pooled_v1"
    assert ce["per_episode_survival_fraction"] == "not_applicable_pooled_v1"
    assert ce["dsr_min"] == "not_applicable_pooled_v1"

    assert record["long_short"] == long_short_payload

    out_path = tmp_path / "audit_results" / f"{p.id}.json"
    assert out_path.exists()
    persisted = json.loads(out_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "completed_pending_human_read"
    assert persisted["mode"] == "pooled_universe"
    assert persisted["grading_version"] == "pooled_v1"
    assert persisted["grading"]["n_trades_total"] == 235
    assert persisted["long_short"] == long_short_payload
    assert persisted["criterion_eval"]["audit_class_in"] == "not_applicable_pooled_v1"
