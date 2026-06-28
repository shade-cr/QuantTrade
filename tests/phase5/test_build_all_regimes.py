"""Tests for the multi-asset regime/dossier batch and its dossier additions."""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase5.regime_stats import build_regime_dossiers


def _regimes_with_dominant_episode() -> pd.DataFrame:
    """BULL_QUIET: one 300-bar episode + two 10-bar episodes => dominant frac > 0.9."""
    idx = pd.date_range("2000-01-01", periods=400, freq="D", tz="UTC")
    rid = ["BULL_QUIET"] * 300 + ["BEAR_QUIET"] * 40 + ["BULL_QUIET"] * 10 \
        + ["BEAR_QUIET"] * 40 + ["BULL_QUIET"] * 10
    return pd.DataFrame({"regime_id": rid}, index=idx)


def test_dominant_episode_fraction_present_and_correct():
    regimes_df = _regimes_with_dominant_episode()
    close = pd.Series(np.linspace(100, 200, len(regimes_df)), index=regimes_df.index, name="close")
    features_df = pd.DataFrame({"close": close})
    dossiers = build_regime_dossiers(regimes_df, features_df, asset_class="metal")
    bq = dossiers["BULL_QUIET"]
    # 320 BULL_QUIET bars, largest episode 300 => 300/320 = 0.9375
    assert "dominant_episode_fraction" in bq
    assert abs(bq["dominant_episode_fraction"] - 300 / 320) < 1e-9


from phase5.regime_stats import build_dossier_features


def test_feature_pack_empty_yields_no_alt_columns(tmp_path):
    """A crypto (empty feature_pack) asset gets no COT/real-yield columns."""
    idx = pd.date_range("2000-01-01", periods=50, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": np.linspace(1, 2, 50), "volume": 1.0},
        index=idx,
    )
    regimes_df = pd.DataFrame(
        {"rv_20": 0.1, "ma_50": 1.0, "ma_200": 1.0, "roc_63": 0.0}, index=idx
    )
    feats = build_dossier_features(df, regimes_df, asset="BTCUSD", feature_pack=())
    assert "cot_net_noncomm_z52w" not in feats.columns
    assert "real_yield_5y_z252d" not in feats.columns
    # regime indicators ARE carried through
    assert "rv_20" in feats.columns


def test_feature_pack_metal_requests_real_yield(tmp_path, monkeypatch):
    """A metal feature_pack injects real-yield WHEN the cache exists."""
    idx = pd.date_range("2000-01-01", periods=50, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": np.linspace(1, 2, 50), "volume": 1.0},
        index=idx,
    )
    regimes_df = pd.DataFrame({"rv_20": 0.1, "ma_50": 1.0, "ma_200": 1.0, "roc_63": 0.0}, index=idx)
    # Stub the loader so the test does not depend on a real DFII5 cache.
    import phase5.regime_stats as rs
    monkeypatch.setattr(rs, "load_real_yield_z",
                        lambda target_index, **k: pd.Series(0.5, index=target_index, name="real_yield_5y_z252d"))
    monkeypatch.setattr(rs.Path, "exists", lambda self: True)
    feats = build_dossier_features(df, regimes_df, asset="XAGUSD",
                                   feature_pack=("real_yield_5y_z252d",))
    assert "real_yield_5y_z252d" in feats.columns


import importlib.util


def _load_batch_module():
    spec = importlib.util.spec_from_file_location(
        "build_all_regimes", REPO_ROOT / "scripts" / "build_all_regimes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_one_cell_skips_short_history(tmp_path, synth_ohlcv):
    """A 500-bar CSV is below the D1 min_bars floor => skipped, no parquet."""
    batch = _load_batch_module()
    csv = tmp_path / "SHORT_D1.csv"
    synth_ohlcv.to_csv(csv, index=False)
    regimes_dir = tmp_path / "regimes"
    dossiers_dir = tmp_path / "dossiers"
    result = batch.build_one_cell(
        ticker="SHORTX", frequency="D1", asset_class="crypto", feature_pack=(),
        data_path=csv, regimes_dir=regimes_dir, dossiers_dir=dossiers_dir,
        min_bars=1460, force=False,
    )
    assert result["status"] == "skipped_insufficient_history"
    assert not (regimes_dir / "SHORTX_d1_regimes.parquet").exists()


def test_build_one_cell_skips_when_zero_labeled(tmp_path, synth_ohlcv):
    """B0066: a CSV that CLEARS min_bars but yields 0 labeled regimes (burn-in +
    min_dwell > history) is an honest skip — no parquet, no dossiers, no manifest."""
    batch = _load_batch_module()
    csv = tmp_path / "EMPTYLAB_D1.csv"
    synth_ohlcv.to_csv(csv, index=False)  # 500 bars; D1 burn-in (1260+) => 0 labeled
    regimes_dir = tmp_path / "regimes"
    dossiers_dir = tmp_path / "dossiers"
    result = batch.build_one_cell(
        ticker="EMPTYLAB", frequency="D1", asset_class="crypto", feature_pack=(),
        data_path=csv, regimes_dir=regimes_dir, dossiers_dir=dossiers_dir,
        min_bars=10, force=False,  # clears the gate (500 > 10) but labels 0
    )
    assert result["status"] == "skipped_no_labeled_bars"
    assert result["n_labeled_bars"] == 0
    assert not (regimes_dir / "EMPTYLAB_d1_regimes.parquet").exists()
    assert not (regimes_dir / "EMPTYLAB_d1_manifest.json").exists()
    assert not (dossiers_dir / "EMPTYLAB_d1").exists()


def test_quasi_circular_threshold_catches_moderate_corr_and_excludes_from_orthogonal():
    """B0047: a feature with |spearman rho| in [0.35, 0.5) against a regime-defining
    feature is quasi_circular AND excluded from orthogonal_features. The old 0.5
    threshold missed this band (and orthogonal_features ignored correlation entirely)."""
    rng = np.random.default_rng(0)
    n = 300
    idx = pd.date_range("2000-01-01", periods=n, freq="D", tz="UTC")
    roc = rng.normal(size=n)
    noise = rng.normal(size=n)
    real_yield = 0.5 * roc + 1.0 * noise          # moderately correlated with roc_63
    cot = rng.normal(size=n)                       # ~uncorrelated
    close = np.linspace(100.0, 200.0, n)
    features_df = pd.DataFrame(
        {
            "close": close,
            "roc_63": roc, "ma_50": 1.0, "ma_200": 1.0, "rv_20": 1.0,
            "real_yield_5y_z252d": real_yield,
            "cot_net_noncomm_z52w": cot,
        },
        index=idx,
    )
    regimes_df = pd.DataFrame({"regime_id": ["BULL_QUIET"] * n}, index=idx)
    # Precondition: the engineered correlation actually lands in the target band
    # (deterministic with the fixed seed) — proves we're testing the 0.35-vs-0.5 gap.
    rho = pd.Series(real_yield, index=idx).corr(pd.Series(roc, index=idx), method="spearman")
    assert 0.35 < abs(rho) < 0.5, f"test construction off-band: rho={rho}"

    dossiers = build_regime_dossiers(regimes_df, features_df, asset_class="metal")
    summ = dossiers["BULL_QUIET"]["features_quantile_summary"]
    orth = dossiers["BULL_QUIET"]["orthogonal_features"]
    # The moderately-collinear feature is flagged AND kept out of orthogonal_features.
    assert summ["real_yield_5y_z252d"]["quasi_circular"] is True
    assert "real_yield_5y_z252d" not in orth
    # The uncorrelated feature stays orthogonal.
    assert summ["cot_net_noncomm_z52w"]["quasi_circular"] is False
    assert "cot_net_noncomm_z52w" in orth


def test_build_one_cell_failure_isolated_on_bad_csv(tmp_path):
    """A malformed CSV is recorded as failed, not raised."""
    batch = _load_batch_module()
    bad = tmp_path / "BAD_D1.csv"
    bad.write_text("time,open,high,low,close\n2020-01-01,1,1,1,1\n", encoding="utf-8")  # no volume
    result = batch.build_one_cell(
        ticker="BADX", frequency="D1", asset_class="fx", feature_pack=(),
        data_path=bad, regimes_dir=tmp_path / "r", dossiers_dir=tmp_path / "d",
        min_bars=1, force=False,
    )
    assert result["status"] == "failed"
    assert "volume" in result["detail"].lower() or "missing" in result["detail"].lower()


def test_build_one_cell_builds_on_long_history(tmp_csv_long, tmp_path):
    """A 9000-bar CSV builds a parquet + 4 dossiers and reports 'built'."""
    batch = _load_batch_module()
    regimes_dir = tmp_path / "regimes"
    dossiers_dir = tmp_path / "dossiers"
    result = batch.build_one_cell(
        ticker="LONGX", frequency="D1", asset_class="metal", feature_pack=(),
        data_path=tmp_csv_long, regimes_dir=regimes_dir, dossiers_dir=dossiers_dir,
        min_bars=1460, force=False,
    )
    assert result["status"] == "built"
    assert (regimes_dir / "LONGX_d1_regimes.parquet").exists()
    assert len(list((dossiers_dir / "LONGX_d1").glob("*.json"))) == 4
    assert result["n_labeled_bars"] > 200


def test_idempotency_second_run_skips_then_force_rebuilds(tmp_csv_long, tmp_path):
    batch = _load_batch_module()
    regimes_dir = tmp_path / "regimes"
    dossiers_dir = tmp_path / "dossiers"
    common = dict(ticker="LONGX", frequency="D1", asset_class="metal", feature_pack=(),
                  data_path=tmp_csv_long, regimes_dir=regimes_dir, dossiers_dir=dossiers_dir,
                  min_bars=1460)
    first = batch.build_one_cell(force=False, **common)
    assert first["status"] == "built"
    second = batch.build_one_cell(force=False, **common)
    assert second["status"] == "skipped_up_to_date"
    forced = batch.build_one_cell(force=True, **common)
    assert forced["status"] == "built"


def test_manifest_written_with_csv_hash(tmp_csv_long, tmp_path):
    batch = _load_batch_module()
    regimes_dir = tmp_path / "regimes"
    batch.build_one_cell(ticker="LONGX", frequency="D1", asset_class="metal", feature_pack=(),
                         data_path=tmp_csv_long, regimes_dir=regimes_dir,
                         dossiers_dir=tmp_path / "d", min_bars=1460, force=False)
    manifest = regimes_dir / "LONGX_d1_manifest.json"
    assert manifest.exists()
    import json
    data = json.loads(manifest.read_text())
    assert "csv_sha256" in data and len(data["csv_sha256"]) == 64


def test_write_rollup_emits_md_and_json(tmp_path):
    batch = _load_batch_module()
    rows = [
        {"ticker": "LONGX", "frequency": "D1", "status": "built", "detail": "",
         "n_total_bars": 9000, "n_labeled_bars": 7000, "regimes_sufficient": ["BULL_QUIET"]},
        {"ticker": "SHORTX", "frequency": "D1", "status": "skipped_insufficient_history",
         "detail": "500 bars < min_bars 1460", "n_total_bars": 500, "n_labeled_bars": 0,
         "regimes_sufficient": []},
    ]
    md_path, json_path = batch.write_rollup(rows, out_dir=tmp_path, date_str="20260527")
    assert md_path.exists() and json_path.exists()
    text = md_path.read_text(encoding="utf-8")
    assert "LONGX" in text and "skipped_insufficient_history" in text
    import json
    assert len(json.loads(json_path.read_text())) == 2


from phase5.regime_stats import _primary_raw_metrics, BASELINE_PANEL


def _rising_ohlc(n=60):
    """Monotonically rising close so a long entry always reaches the TP barrier."""
    idx = pd.date_range("2000-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100.0, 160.0, n), index=idx)
    ohlc = pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1.0},
        index=idx,
    )
    return ohlc


def test_primary_raw_metrics_buckets_and_hit_rate():
    ohlc = _rising_ohlc(60)
    atr = pd.Series(1.0, index=ohlc.index)  # constant ATR
    # Two long entries, both in BULL_QUIET; rising price => both TP => hit_rate 1.0
    sig = pd.Series(0.0, index=ohlc.index)
    sig.iloc[5] = 1.0
    sig.iloc[10] = 1.0
    regimes_df = pd.DataFrame({"regime_id": ["BULL_QUIET"] * 60}, index=ohlc.index)
    raw = _primary_raw_metrics(
        ohlc, atr, regimes_df, sig, frequency="D1",
        tp_mult=2.0, sl_mult=1.0, horizon=20,
    )
    assert raw["BULL_QUIET"]["n_events"] == 2
    assert raw["BULL_QUIET"]["hit_rate"] == 1.0
    assert raw["BULL_QUIET"]["median_per_trade_return"] > 0
    # No events bucketed into a regime with no entries
    assert raw["BEAR_QUIET"]["n_events"] == 0
    assert raw["BEAR_QUIET"]["hit_rate"] is None


def test_baseline_panel_has_four_named_primaries():
    names = [name for name, _ in BASELINE_PANEL]
    assert names == ["ema_crossover", "momentum_zscore", "bollinger_meanrev", "cusum_filter"]


from phase5.regime_stats import build_primary_baselines


def _atr_test_series(ohlc):
    from pipeline.features import _atr
    return _atr(ohlc["high"], ohlc["low"], ohlc["close"]).bfill().fillna(1.0)


def test_build_primary_baselines_encoding_and_firewall():
    n = 800
    idx = pd.date_range("2000-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100.0, 300.0, n), index=idx)  # strong uptrend
    ohlc = pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1.0},
        index=idx,
    )
    atr = _atr_test_series(ohlc)
    rid = (["BULL_QUIET"] * 300 + ["BULL_STRESSED"] * 250 + ["BEAR_QUIET"] * 250)
    regimes_df = pd.DataFrame({"regime_id": rid}, index=idx)
    baselines = build_primary_baselines(ohlc, atr, regimes_df, frequency="D1")
    assert set(baselines.keys()) == {"BULL_QUIET", "BULL_STRESSED", "BEAR_QUIET", "BEAR_STRESSED"}
    bq_ema = baselines["BULL_QUIET"]["ema_crossover"]
    assert set(bq_ema) == {
        "trade_count_per_year_q", "hit_rate_q", "median_per_trade_return_q",
        "hit_rate_vs_other_regimes", "return_vs_other_regimes", "n_events",
        "rankable", "low_confidence",
    }
    for k in ("trade_count_per_year_q", "hit_rate_q", "median_per_trade_return_q"):
        v = bq_ema[k]
        assert v is None or (0.0 <= v <= 1.0)
    # BEAR_STRESSED has no bars => 0 events => low_confidence True (events floor), unrankable.
    bs = baselines["BEAR_STRESSED"]["ema_crossover"]
    assert bs["n_events"] == 0
    assert bs["low_confidence"] is True
    assert bs["rankable"] is False
    assert bs["hit_rate_q"] is None


def test_specialist_primary_is_measured_not_low_confidence():
    """B1: a primary that fires only in ONE regime is MEASURED (low_confidence False)
    even though its cross-regime rank is unrankable (rankable False, _q None)."""
    n = 800
    idx = pd.date_range("2000-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100.0, 300.0, n), index=idx)
    ohlc = pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1.0},
        index=idx,
    )
    atr = _atr_test_series(ohlc)
    regimes_df = pd.DataFrame({"regime_id": ["BULL_QUIET"] * n}, index=idx)
    baselines = build_primary_baselines(ohlc, atr, regimes_df, frequency="D1")
    ema = baselines["BULL_QUIET"]["ema_crossover"]
    assert ema["n_events"] >= 30
    assert ema["low_confidence"] is False
    assert ema["rankable"] is False
    assert ema["hit_rate_q"] is None


def test_build_primary_baselines_firewall_leaves_only_safe_types():
    n = 800
    idx = pd.date_range("2000-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100.0, 300.0, n), index=idx)
    ohlc = pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1.0},
        index=idx,
    )
    atr = _atr_test_series(ohlc)
    rid = (["BULL_QUIET"] * 400 + ["BEAR_QUIET"] * 400)
    regimes_df = pd.DataFrame({"regime_id": rid}, index=idx)
    baselines = build_primary_baselines(ohlc, atr, regimes_df, frequency="D1")
    _ALLOWED_TAGS = {"higher", "lower", "similar", "unknown"}
    _Q_KEYS = {"trade_count_per_year_q", "hit_rate_q", "median_per_trade_return_q"}
    for regime, per_primary in baselines.items():
        for primary, d in per_primary.items():
            for key, val in d.items():
                if key in _Q_KEYS:
                    assert val is None or (isinstance(val, float) and 0.0 <= val <= 1.0)
                elif key in ("hit_rate_vs_other_regimes", "return_vs_other_regimes"):
                    assert val in _ALLOWED_TAGS
                elif key == "n_events":
                    assert isinstance(val, int) and not isinstance(val, bool)
                elif key in ("rankable", "low_confidence"):
                    assert isinstance(val, bool)
                else:
                    raise AssertionError(f"unexpected baseline key {key!r} (possible firewall leak)")


def test_build_regime_dossiers_threads_primary_baselines():
    idx = pd.date_range("2000-01-01", periods=400, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100, 200, 400), index=idx, name="close")
    features_df = pd.DataFrame({"close": close})
    regimes_df = pd.DataFrame({"regime_id": ["BULL_QUIET"] * 400}, index=idx)
    pb = {"BULL_QUIET": {"ema_crossover": {"hit_rate_q": 0.9, "n_events": 50, "low_confidence": False}}}
    dossiers = build_regime_dossiers(
        regimes_df, features_df, asset_class="metal", primary_baselines=pb
    )
    assert dossiers["BULL_QUIET"]["primary_baseline_summary"] == pb["BULL_QUIET"]
    # A regime not present in pb still gets an empty dict (no KeyError).
    assert dossiers["BEAR_QUIET"]["primary_baseline_summary"] == {}


def test_build_regime_dossiers_default_baseline_is_empty():
    idx = pd.date_range("2000-01-01", periods=400, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100, 200, 400), index=idx, name="close")
    features_df = pd.DataFrame({"close": close})
    regimes_df = pd.DataFrame({"regime_id": ["BULL_QUIET"] * 400}, index=idx)
    dossiers = build_regime_dossiers(regimes_df, features_df, asset_class="metal")
    assert dossiers["BULL_QUIET"]["primary_baseline_summary"] == {}


def test_build_one_cell_dossiers_carry_primary_baseline(tmp_csv_long, tmp_path):
    """End-to-end: a built cell's dossiers carry a populated primary_baseline_summary."""
    import json
    batch = _load_batch_module()
    regimes_dir = tmp_path / "regimes"
    dossiers_dir = tmp_path / "dossiers"
    result = batch.build_one_cell(
        ticker="LONGX", frequency="D1", asset_class="metal", feature_pack=(),
        data_path=tmp_csv_long, regimes_dir=regimes_dir, dossiers_dir=dossiers_dir,
        min_bars=1460, force=False,
    )
    assert result["status"] == "built"
    bull_quiet = json.loads((dossiers_dir / "LONGX_d1" / "BULL_QUIET.json").read_text())
    pbs = bull_quiet["primary_baseline_summary"]
    assert set(pbs) == {"ema_crossover", "momentum_zscore", "bollinger_meanrev", "cusum_filter"}
    assert "hit_rate_q" in pbs["ema_crossover"]


def test_build_one_cell_d1_and_h4_dossiers_do_not_collide(tmp_csv_long, tmp_path):
    """B0070: building the SAME ticker at D1 then H4 must write to distinct dossier
    dirs — the H4 run must not clobber the D1 dossiers."""
    batch = _load_batch_module()
    regimes_dir = tmp_path / "regimes"
    dossiers_dir = tmp_path / "dossiers"
    common = dict(ticker="DUALX", asset_class="crypto", feature_pack=(),
                  data_path=tmp_csv_long, regimes_dir=regimes_dir, dossiers_dir=dossiers_dir,
                  force=False)
    r_d1 = batch.build_one_cell(frequency="D1", min_bars=1460, **common)
    r_h4 = batch.build_one_cell(frequency="H4", min_bars=7760, **common)
    assert r_d1["status"] == "built"
    assert r_h4["status"] in ("built", "skipped_no_labeled_bars")
    assert (dossiers_dir / "DUALX_d1").is_dir()
    assert len(list((dossiers_dir / "DUALX_d1").glob("*.json"))) == 4
    if r_h4["status"] == "built":
        assert (dossiers_dir / "DUALX_h4").is_dir()
        assert (dossiers_dir / "DUALX_d1") != (dossiers_dir / "DUALX_h4")
