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
