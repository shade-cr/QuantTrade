"""B0155 — ev_breakeven_v1 pre-registered EV threshold + proposal-time
feature-existence gate.

Methodology (quant-phd-advisor verdict 2026-06-11, Elkan 2001 + AFML ch.3/10):
a fixed 0.50 meta threshold is payoff-blind. With tp=3/sl=1 barriers the EV
breakeven is p* = (sl + C_ATR + LAMBDA_MARGIN*(tp+sl)) / (tp+sl) = 0.325, so
honest calibrated probabilities in [0.25, 0.48] containing positive-EV trades
were discarded wholesale (4/4 NO_FIRE audits 2026-06-11).

C_ATR / LAMBDA_MARGIN are GLOBAL methodology constants — never per-proposal
(that would be a threshold-shopping channel).

Also: the B004v3 lesson — a primary/meta referencing a nonexistent feature
produced a structurally dead gate that masqueraded as falsification. validate()
now hard-errors on unknown names in feature_overrides.add/.drop.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from phase5 import run_proposal
from phase5.proposal import (
    C_ATR,
    LAMBDA_MARGIN,
    THRESHOLD_RULES,
    BarrierGeometryAttestation,
    LookaheadShapeAttestation,
    Proposal,
    ProposalValidationError,
    _build_dataclass,
    compute_p_star,
    known_feature_registry,
    KNOWN_TIER2_FEATURES,
)


# ---------------------------------------------------------------- fixtures

def _minimal_proposal(**overrides) -> Proposal:
    defaults = dict(
        id="TEST-B0155",
        asset="XAUUSD",
        asset_class="metal",
        regime_scope=["BULL_QUIET"],
        hypothesis="x" * 50,
        causal_story="y" * 50,
        primary="ema_cross",
        primary_params={"fast": 10, "slow": 50},
        lookahead_shape_attestation=LookaheadShapeAttestation(
            target_regime_episode_ordinals=[0, 2],
            cross_asset_falsifiable_in=["XAGUSD"],
        ),
    )
    defaults.update(overrides)
    return Proposal(**defaults)


# ---------------------------------------------------------------- constants

def test_global_constants_values():
    """C_ATR / LAMBDA_MARGIN are global methodology constants per spec."""
    assert C_ATR == 0.10
    assert LAMBDA_MARGIN == 0.05
    assert THRESHOLD_RULES == ("fixed_0.50", "ev_breakeven_v1")


# ---------------------------------------------------------------- p_star math

def test_p_star_tp3_sl1_is_0325():
    # (1 + 0.10 + 0.05 * 4) / 4 = 1.30 / 4 = 0.325
    assert compute_p_star(3.0, 1.0) == pytest.approx(0.325, abs=1e-12)


def test_p_star_tp25_sl1():
    # (1 + 0.10 + 0.05 * 3.5) / 3.5 = 1.275 / 3.5 = 0.36428571428571427
    assert compute_p_star(2.5, 1.0) == pytest.approx(1.275 / 3.5, abs=1e-12)


def test_p_star_symmetric_payoff_above_half():
    # tp=1/sl=1: (1 + 0.10 + 0.05 * 2) / 2 = 0.60 — costs + margin push the
    # symmetric breakeven ABOVE the naive 0.50.
    assert compute_p_star(1.0, 1.0) == pytest.approx(0.60, abs=1e-12)


# ---------------------------------------------------------------- threshold_rule schema

def test_default_threshold_rule_is_fixed_050_backward_compat():
    p = _minimal_proposal()
    p.validate()
    assert p.threshold_rule == "fixed_0.50"
    assert p.effective_threshold() == pytest.approx(0.50, abs=1e-12)


def test_unknown_threshold_rule_rejected():
    p = _minimal_proposal(threshold_rule="ev_breakeven_v2")
    with pytest.raises(ProposalValidationError, match="threshold_rule"):
        p.validate()


def test_ev_rule_effective_threshold_from_barrier_attestation():
    p = _minimal_proposal(
        threshold_rule="ev_breakeven_v1",
        barrier_geometry_attestation=BarrierGeometryAttestation(
            tp_atr_mult=3.0, sl_atr_mult=1.0
        ),
    )
    p.validate()
    assert p.effective_threshold() == pytest.approx(0.325, abs=1e-12)


def test_precomputed_p_star_match_accepted():
    p = _minimal_proposal(threshold_rule="ev_breakeven_v1", p_star=0.325)
    p.validate()  # no raise — matches recomputation within 1e-9


def test_precomputed_p_star_mismatch_rejected():
    p = _minimal_proposal(threshold_rule="ev_breakeven_v1", p_star=0.33)
    with pytest.raises(ProposalValidationError, match="p_star"):
        p.validate()


def test_p_star_without_ev_rule_rejected():
    """A stray p_star on a fixed_0.50 proposal is a threshold-shopping smell."""
    p = _minimal_proposal(p_star=0.325)
    with pytest.raises(ProposalValidationError, match="p_star"):
        p.validate()


def test_threshold_rule_round_trips_through_build_dataclass():
    payload = {
        "id": "TEST-B0155-RT",
        "asset": "XAUUSD",
        "asset_class": "metal",
        "regime_scope": ["BULL_QUIET"],
        "hypothesis": "x" * 50,
        "causal_story": "y" * 50,
        "primary": "ema_cross",
        "primary_params": {"fast": 10, "slow": 50},
        "threshold_rule": "ev_breakeven_v1",
        "p_star": 0.325,
        "lookahead_shape_attestation": {
            "target_regime_episode_ordinals": [0, 2],
            "cross_asset_falsifiable_in": ["XAGUSD"],
        },
    }
    p = _build_dataclass(Proposal, payload)
    assert p.threshold_rule == "ev_breakeven_v1"
    assert p.p_star == pytest.approx(0.325)
    p.validate()


# ---------------------------------------------------------------- feature-existence gate

def test_known_features_accepted_in_overrides():
    p = _build_dataclass(Proposal, {
        **_minimal_proposal().to_dict(),
        "feature_overrides": {
            "add": ["rv_20", "volume", "cot_net_noncomm_z52w", "us_5y2y_z252",
                    "breakeven_5y_chg5"],
            "drop": ["dxy_z252", "vix_chg_5"],
        },
    })
    p.validate()  # no raise: tier2 names, alias key, dossier alt-features


def test_unknown_feature_in_add_is_hard_error():
    p = _build_dataclass(Proposal, {
        **_minimal_proposal().to_dict(),
        "feature_overrides": {"add": ["ma_50"], "drop": []},
    })
    with pytest.raises(ProposalValidationError, match="ma_50"):
        p.validate()


def test_unknown_feature_in_drop_is_hard_error():
    p = _build_dataclass(Proposal, {
        **_minimal_proposal().to_dict(),
        "feature_overrides": {"add": [], "drop": ["roc_63"]},
    })
    with pytest.raises(ProposalValidationError, match="roc_63"):
        p.validate()


def test_wildcard_drop_with_known_prefix_accepted():
    p = _build_dataclass(Proposal, {
        **_minimal_proposal().to_dict(),
        "feature_overrides": {"add": [], "drop": ["dxy_*"]},
    })
    p.validate()


def test_wildcard_drop_with_unknown_prefix_rejected():
    p = _build_dataclass(Proposal, {
        **_minimal_proposal().to_dict(),
        "feature_overrides": {"add": [], "drop": ["zzz_*"]},
    })
    with pytest.raises(ProposalValidationError, match="zzz_"):
        p.validate()


def test_h4_feature_names_are_known():
    reg = known_feature_registry()
    for name in ("r_24bars", "z_r24bars", "session_london", "bb_width_120bars"):
        assert name in reg, name


def test_custom_primary_param_feature_references_are_warning_level():
    """For phase5_* customs, suspicious feature-name-shaped strings in
    primary_params are LENIENT (warning list), not a hard gate."""
    p = _minimal_proposal(
        primary="phase5_whatever",
        custom_primary_pseudocode="pseudo",
        primary_params={"gate_feature": "totally_unknown_feat_z99"},
    )
    warnings = p.validate()  # must NOT raise
    assert any("totally_unknown_feat_z99" in w for w in warnings)

    p_ok = _minimal_proposal(
        primary="phase5_whatever",
        custom_primary_pseudocode="pseudo",
        primary_params={"gate_feature": "real_yield_5y_z252d"},
    )
    warnings_ok = p_ok.validate()
    assert not any("real_yield_5y_z252d" in w for w in warnings_ok)


def test_registry_regenerates_from_synthetic_build(synth_ohlcv):
    """The hardcoded KNOWN_TIER2_FEATURES list must equal the union of columns
    produced by the D1 + H4 technical builders, the macro builder (all
    optional series present), and the session one-hot block. This test FAILS
    whenever tier2 changes, forcing the frozen list to be updated."""
    from pipeline.features import (
        build_technical_features,
        build_macro_features,
        _build_h4_technical,
        _build_session_one_hot,
    )
    ohlcv = synth_ohlcv.set_index("time")
    idx = ohlcv.index
    rng = np.random.default_rng(0)
    macro = pd.DataFrame(
        {c: rng.normal(size=len(idx)) for c in
         ("DTWEXBGS", "DFII5", "T5YIE", "DGS5", "VIXCLS", "DGS2",
          "UMCSENT", "UMCSENT_chg_3m")},
        index=idx,
    )
    # B0147: the GLD real-volume block is config-gated alt-data joined by the
    # orchestrator (metals only), not a builder output — union its canonical
    # module constant so the registry equality still forces same-commit sync.
    from pipeline.alt_data.gld_volume import GLD_VOLUME_FEATURES
    cols = (
        set(build_technical_features(ohlcv).columns)
        | set(_build_h4_technical(ohlcv).columns)
        | set(build_macro_features(macro, idx).columns)
        | set(_build_session_one_hot(idx).columns)
        | set(GLD_VOLUME_FEATURES)
    )
    assert cols == set(KNOWN_TIER2_FEATURES)


# ---------------------------------------------------------------- aggregate_at_threshold

_GRID_ROWS = [
    {"model": "xgb", "fold": 0, "threshold": 0.325, "sharpe_net": 1.0, "n_trades": 40},
    {"model": "xgb", "fold": 1, "threshold": 0.325, "sharpe_net": 1.2, "n_trades": 35},
    {"model": "xgb", "fold": 0, "threshold": 0.50, "sharpe_net": 0.1, "n_trades": 10},
    {"model": "xgb", "fold": 1, "threshold": 0.50, "sharpe_net": 0.2, "n_trades": 12},
]


def test_aggregate_at_threshold_selects_requested_rows():
    agg = run_proposal.aggregate_at_threshold(_GRID_ROWS, 0.325)
    assert agg["xgb"]["total_n_trades"] == 75
    assert agg["xgb"]["per_fold_sharpe"] == [1.0, 1.2]


def test_aggregate_threshold_50_wrapper_unchanged():
    agg = run_proposal.aggregate_threshold_50(_GRID_ROWS)
    assert agg["xgb"]["total_n_trades"] == 22
    assert agg["xgb"]["per_fold_sharpe"] == [0.1, 0.2]


# ---------------------------------------------------------------- build_transient_config

def _ev_proposal(**overrides) -> Proposal:
    return _minimal_proposal(
        threshold_rule="ev_breakeven_v1",
        barrier_geometry_attestation=BarrierGeometryAttestation(
            tp_atr_mult=3.0, sl_atr_mult=1.0
        ),
        **overrides,
    )


def test_build_transient_config_ev_rule_recenters_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(run_proposal, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(run_proposal, "RESULTS_PHASE5_DIR", tmp_path / "results")
    p = _ev_proposal()
    cfg_path = run_proposal.build_transient_config(p, tmp_path / "mask.parquet")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    grid = cfg["metrics"]["threshold_grid"]
    assert grid == pytest.approx([0.325, 0.375, 0.425, 0.475])
    assert cfg["metrics"]["audit_effective_threshold"] == pytest.approx(0.325)


def test_build_transient_config_fixed_rule_bit_for_bit(tmp_path, monkeypatch):
    """fixed_0.50 proposals keep the template grid and gain NO new keys —
    backward compat for the 2026-06-11 batch is load-bearing."""
    monkeypatch.setattr(run_proposal, "RUNTIME_DIR", tmp_path / "runtime")
    monkeypatch.setattr(run_proposal, "RESULTS_PHASE5_DIR", tmp_path / "results")
    template = yaml.safe_load(run_proposal.TEMPLATE_CONFIG.read_text(encoding="utf-8"))
    p = _minimal_proposal()
    cfg_path = run_proposal.build_transient_config(p, tmp_path / "mask.parquet")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["metrics"]["threshold_grid"] == template["metrics"]["threshold_grid"]
    assert "audit_effective_threshold" not in cfg["metrics"]


# ---------------------------------------------------------------- per-episode artifact

def _write_episode_fixtures(tmp_path, monkeypatch, pnl, regimes, *,
                            effective: float | None = None,
                            sidecar_threshold: float | None = None,
                            proposal_id="TEST-B0155", primary="ema_cross",
                            asset="XAUUSD", model="xgb"):
    monkeypatch.setattr(run_proposal, "RESULTS_PHASE5_DIR", tmp_path / "results")
    monkeypatch.setattr(run_proposal, "REGIMES_DIR", tmp_path / "regimes")
    out_dir = tmp_path / "results" / proposal_id / primary
    out_dir.mkdir(parents=True, exist_ok=True)
    if effective is None:
        pd.DataFrame({model: pnl}).to_parquet(out_dir / "strategy_pnl_threshold50.parquet")
    else:
        pd.DataFrame({model: pnl}).to_parquet(out_dir / "strategy_pnl_effective.parquet")
        sc = sidecar_threshold if sidecar_threshold is not None else effective
        (out_dir / "strategy_pnl_effective.json").write_text(
            json.dumps({"threshold": sc}), encoding="utf-8")
    reg_dir = tmp_path / "regimes"
    reg_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"regime_id": regimes}).to_parquet(reg_dir / f"{asset}_d1_regimes.parquet")


def _episode_series():
    idx = pd.date_range("2020-01-01", periods=40, freq="D", tz="UTC")
    regimes = pd.Series(["BEAR_QUIET"] * 40, index=idx, dtype=object)
    regimes.iloc[0:10] = "BULL_QUIET"
    regimes.iloc[15:25] = "BULL_QUIET"
    pnl = pd.Series(np.nan, index=idx)
    pnl.iloc[0:10] = 0.01
    pnl.iloc[15:25] = 0.01
    return pnl, regimes


_CRIT = {"per_episode_survival_fraction": 0.6, "per_episode_min_trades": 5,
         "per_episode_net_pnl_margin": 0.0}


def test_per_episode_reads_effective_artifact(tmp_path, monkeypatch):
    pnl, regimes = _episode_series()
    _write_episode_fixtures(tmp_path, monkeypatch, pnl, regimes, effective=0.325)
    res = run_proposal.evaluate_per_episode(
        "TEST-B0155", "ema_cross", "XAUUSD", "xgb", _CRIT, ["BULL_QUIET"], "D1",
        effective_threshold=0.325,
    )
    assert res["applicable"] is True
    assert res["passed"] is True


def test_per_episode_effective_sidecar_mismatch_fails_gracefully(tmp_path, monkeypatch):
    pnl, regimes = _episode_series()
    _write_episode_fixtures(tmp_path, monkeypatch, pnl, regimes,
                            effective=0.325, sidecar_threshold=0.40)
    res = run_proposal.evaluate_per_episode(
        "TEST-B0155", "ema_cross", "XAUUSD", "xgb", _CRIT, ["BULL_QUIET"], "D1",
        effective_threshold=0.325,
    )
    assert res["passed"] is False
    assert "mismatch" in res["reason"]


def test_per_episode_effective_artifact_missing_fails_gracefully(tmp_path, monkeypatch):
    pnl, regimes = _episode_series()
    # Only the legacy 0.50 artifact exists; the effective one is missing.
    _write_episode_fixtures(tmp_path, monkeypatch, pnl, regimes, effective=None)
    res = run_proposal.evaluate_per_episode(
        "TEST-B0155", "ema_cross", "XAUUSD", "xgb", _CRIT, ["BULL_QUIET"], "D1",
        effective_threshold=0.325,
    )
    assert res["passed"] is False
    assert "missing" in res["reason"]


def test_per_episode_default_keeps_threshold50_path(tmp_path, monkeypatch):
    pnl, regimes = _episode_series()
    _write_episode_fixtures(tmp_path, monkeypatch, pnl, regimes, effective=None)
    res = run_proposal.evaluate_per_episode(
        "TEST-B0155", "ema_cross", "XAUUSD", "xgb", _CRIT, ["BULL_QUIET"], "D1",
    )
    assert res["passed"] is True


# ---------------------------------------------------------------- audit record fields

def test_persist_record_carries_threshold_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(run_proposal, "AUDIT_RESULTS_DIR", tmp_path / "audit")
    rec = run_proposal._persist_record(_ev_proposal(), status="completed")
    assert rec["threshold_rule"] == "ev_breakeven_v1"
    assert rec["effective_threshold"] == pytest.approx(0.325)
    assert rec["threshold_inputs"] == {
        "tp_atr_mult": 3.0, "sl_atr_mult": 1.0,
        "C_ATR": 0.10, "LAMBDA_MARGIN": 0.05,
    }


def test_persist_record_fixed_rule_records_050(tmp_path, monkeypatch):
    monkeypatch.setattr(run_proposal, "AUDIT_RESULTS_DIR", tmp_path / "audit")
    rec = run_proposal._persist_record(_minimal_proposal(), status="completed")
    assert rec["threshold_rule"] == "fixed_0.50"
    assert rec["effective_threshold"] == pytest.approx(0.50)
    assert rec["threshold_inputs"] is None


# ---------------------------------------------------------------- run() wiring

def test_run_evaluates_at_p_star(tmp_path, monkeypatch):
    """End-to-end (subprocess mocked): an ev_breakeven_v1 proposal whose grid
    rows exist ONLY at p_star must be aggregated at p_star (a 0.50 lookup would
    find no rows and falsify with an empty per_model_audit)."""
    p = _ev_proposal()
    p.falsification_criterion.per_episode_survival_fraction = 0.6
    prop_path = tmp_path / f"20260611-XAUUSD-{p.id}.json"
    from phase5.proposal import save_proposal
    save_proposal(p, prop_path)

    monkeypatch.setattr(run_proposal, "AUDIT_RESULTS_DIR", tmp_path / "audit")
    monkeypatch.setattr(run_proposal, "build_regime_mask", lambda *a, **k: tmp_path / "mask.parquet")
    monkeypatch.setattr(run_proposal, "build_transient_config", lambda *a, **k: tmp_path / "cfg.yaml")
    monkeypatch.setattr(run_proposal, "count_events_subprocess",
                        lambda *a, **k: {"n_events": 1000, "wf_event_floor": 100})
    monkeypatch.setattr(run_proposal, "run_pipeline_subprocess",
                        lambda *a, **k: SimpleNamespace(returncode=0, stderr="", stdout=""))
    monkeypatch.setattr(run_proposal, "parse_pipeline_results",
                        lambda *a, **k: {
                            "summary": {}, "grid_rows": [
                                {"model": "xgb", "fold": 0, "threshold": 0.325,
                                 "sharpe_net": 1.0, "n_trades": 40},
                                {"model": "xgb", "fold": 1, "threshold": 0.325,
                                 "sharpe_net": 1.2, "n_trades": 35},
                            ],
                            "psr_dsr": {"dsr": {"xgb": 0.99}},
                            "output_dir": str(tmp_path),
                            "feature_overrides_status": {},
                        })
    monkeypatch.setattr(run_proposal, "compute_oos_regime_diversity",
                        lambda *a, **k: {"pass": True})
    captured = {}

    def fake_per_episode(*args, **kwargs):
        captured["effective_threshold"] = kwargs.get("effective_threshold")
        return {"applicable": True, "passed": True}

    monkeypatch.setattr(run_proposal, "evaluate_per_episode", fake_per_episode)

    rec = run_proposal.run(prop_path)
    assert rec["status"] == "completed"
    assert "xgb" in rec["per_model_audit"]
    assert rec["per_model_audit"]["xgb"]["falsification_verdict"] == "survives"
    assert rec["overall_verdict"] == "survives"
    assert captured["effective_threshold"] == pytest.approx(0.325)
    assert rec["effective_threshold"] == pytest.approx(0.325)


# ---------------------------------------------------------------- run_backtest persistence

def test_persist_audit_pnl_writes_effective_artifact(tmp_path):
    from scripts.run_backtest import persist_audit_pnl
    idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    oof = pd.DataFrame({"xgb": [0.30, 0.60, 0.40]}, index=idx)
    per_trade = np.array([0.01, -0.02, 0.03])

    persist_audit_pnl(tmp_path, oof, ["xgb"], per_trade, audit_threshold=0.325)

    # Legacy artifact unchanged: NaN where p < 0.50.
    legacy = pd.read_parquet(tmp_path / "strategy_pnl_threshold50.parquet")["xgb"]
    assert np.isnan(legacy.iloc[0]) and np.isnan(legacy.iloc[2])
    assert legacy.iloc[1] == pytest.approx(-0.02)

    # Effective artifact: NaN only where p < 0.325.
    eff = pd.read_parquet(tmp_path / "strategy_pnl_effective.parquet")["xgb"]
    assert np.isnan(eff.iloc[0])
    assert eff.iloc[1] == pytest.approx(-0.02)
    assert eff.iloc[2] == pytest.approx(0.03)

    sidecar = json.loads((tmp_path / "strategy_pnl_effective.json").read_text(encoding="utf-8"))
    assert sidecar["threshold"] == pytest.approx(0.325)


def test_persist_audit_pnl_default_050_writes_only_legacy(tmp_path):
    from scripts.run_backtest import persist_audit_pnl
    idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    oof = pd.DataFrame({"xgb": [0.30, 0.60, 0.40]}, index=idx)
    per_trade = np.array([0.01, -0.02, 0.03])

    persist_audit_pnl(tmp_path, oof, ["xgb"], per_trade, audit_threshold=0.50)

    assert (tmp_path / "strategy_pnl_threshold50.parquet").exists()
    assert not (tmp_path / "strategy_pnl_effective.parquet").exists()
    assert not (tmp_path / "strategy_pnl_effective.json").exists()
