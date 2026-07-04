"""Phase-A member builder mirrors run_backtest._run_one_primary alignment
invariants; runner CLI plumbing (count-events mode)."""
import json
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts.run_pooled_equity_d1 as runner
from scripts.run_pooled_equity_d1 import build_member_inputs


@pytest.fixture
def cfg():
    return {
        "asset_class": "equity",
        "triple_barrier": {"horizon": 40, "tp_atr_mult": 3.0,
                           "sl_atr_mult": 1.0, "atr_period": 14},
        "primary": {
            "candidates": ["ema_cross", "momentum_zscore"],
            "ema_cross": {"fast": 20, "slow": 50, "dead_zone_atr": 0.25},
            "momentum_zscore": {"lookback": 20, "threshold": 0.3},
        },
        "metrics": {"cost_per_trade_bps": 10, "bars_per_year": 252},
    }


def _features_for(ohlcv: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=ohlcv.index)
    tr = (ohlcv["high"] - ohlcv["low"]).rolling(14).mean()
    feats["_atr_14"] = tr
    r = np.log(ohlcv["close"]).diff()
    feats["z_r20"] = (r - r.rolling(20).mean()) / r.rolling(20).std()
    feats["f_mom"] = r.rolling(5).sum()
    return feats.dropna()


def test_member_alignment_invariants(synth_ohlcv, cfg):
    features = _features_for(synth_ohlcv)
    ohlcv = synth_ohlcv.loc[features.index]
    m = build_member_inputs("TEST", "ema_cross", ohlcv, features, cfg)
    assert m is not None, "synthetic series should produce ema_cross events"
    n = len(m["X"])
    assert n == len(m["y"]) == len(m["w"]) == len(m["fwd_ret"]) \
        == len(m["event_time"]) == len(m["label_end_time"])
    assert not m["X"].isnull().any().any()
    for col in ("primary_side", "primary_strength", "bars_since_signal"):
        assert col in m["X"].columns
    assert "_atr_14" not in m["X"].columns
    assert not m["fwd_ret"].isnull().any()
    assert (m["label_end_time"] >= m["event_time"]).all()
    assert m["asset_class"] == "equity"
    assert m["bars_per_year"] == 252
    assert m["cost_bps"] == 10
    assert set(np.unique(m["y"])) <= {0, 1}
    assert (m["w"] > 0).all() and (m["w"] <= 1).all()


def test_member_returns_none_when_no_signals(synth_ohlcv, cfg):
    features = _features_for(synth_ohlcv)
    ohlcv = synth_ohlcv.loc[features.index]
    cfg["primary"]["momentum_zscore"]["threshold"] = 99.0  # unreachable
    assert build_member_inputs("TEST", "momentum_zscore", ohlcv, features, cfg) is None


def test_keep_mask_realignment_with_mid_series_nan(synth_ohlcv, cfg):
    """A NaN injected into a non-ATR feature column, at timestamps that
    coincide with real triple-barrier events, must drop exactly those events
    via the pre_drop_index.isin(X.index) keep_mask — and the surviving
    sample weights must equal the full (no-NaN) run's weights at those same
    timestamps, since avg_uniqueness (w_all) is computed BEFORE the dropna
    mask is applied. A mutant that replaces keep_mask with np.ones(...)
    (i.e. never actually filters w) would still pass length checks by luck
    in the no-collision case, but corrupts the w<->event alignment here."""
    features = _features_for(synth_ohlcv)
    ohlcv = synth_ohlcv.loc[features.index]

    baseline = build_member_inputs("TEST", "ema_cross", ohlcv, features, cfg)
    assert baseline is not None
    event_index = baseline["X"].index  # event timestamps (this suite's synth
    # fixture keeps the default integer index end-to-end, so these are the
    # positional "timestamps" — real datetime-indexed ohlcv behaves identically)
    assert len(event_index) >= 10, "need enough events to pick a mid-series window"

    mid = len(event_index) // 2
    nan_hit = event_index[mid: mid + 3]
    assert len(nan_hit) == 3
    assert nan_hit.isin(event_index).all()

    features_nan = features.copy()
    features_nan["extra_feat"] = 0.0
    features_nan.loc[nan_hit, "extra_feat"] = np.nan

    m = build_member_inputs("TEST", "ema_cross", ohlcv, features_nan, cfg)
    assert m is not None

    expected_index = event_index.difference(nan_hit)
    surviving_index = m["X"].index
    assert surviving_index.equals(expected_index)
    assert not surviving_index.isin(nan_hit).any()

    n = len(m["X"])
    assert n == len(m["y"]) == len(m["w"]) == len(m["fwd_ret"]) == len(expected_index)

    baseline_w = pd.Series(np.asarray(baseline["w"]), index=event_index)
    np.testing.assert_allclose(np.asarray(m["w"]), baseline_w.loc[surviving_index].values)


def test_regime_gate_filters_member_events(synth_ohlcv, cfg, tmp_path, monkeypatch):
    """With regime_scope set and a synthetic regime parquet marking only the
    first half of bars BULL_QUIET, the member's events all fall in that half
    and the count is strictly below the ungated count."""
    features = _features_for(synth_ohlcv)
    ohlcv = synth_ohlcv.loc[features.index]

    ungated = build_member_inputs("TEST", "ema_cross", ohlcv, features, cfg)
    assert ungated is not None

    n = len(ohlcv.index)
    midpoint = ohlcv.index[n // 2]
    regime_id = np.where(np.arange(n) < n // 2, "BULL_QUIET", "BEAR_QUIET")
    regimes = pd.DataFrame({"regime_id": regime_id}, index=ohlcv.index)
    regime_path = tmp_path / "TEST_d1_regimes.parquet"
    regimes.to_parquet(regime_path)
    monkeypatch.setattr(runner, "_regimes_path", lambda asset: regime_path)

    gated_cfg = dict(cfg)
    gated_cfg["regime_scope"] = ["BULL_QUIET"]
    gated = build_member_inputs("TEST", "ema_cross", ohlcv, features, gated_cfg)

    assert gated is not None
    assert (gated["X"].index < midpoint).all()
    assert len(gated["X"]) < len(ungated["X"])


def test_feature_overrides_drop_removes_column(synth_ohlcv, cfg):
    """cfg feature_overrides_drop=['f_mom'] -> 'f_mom' absent from member X;
    baseline run has it present."""
    features = _features_for(synth_ohlcv)
    ohlcv = synth_ohlcv.loc[features.index]

    baseline = build_member_inputs("TEST", "ema_cross", ohlcv, features, cfg)
    assert baseline is not None
    assert "f_mom" in baseline["X"].columns

    dropped_cfg = dict(cfg)
    dropped_cfg["feature_overrides_drop"] = ["f_mom"]
    dropped = build_member_inputs("TEST", "ema_cross", ohlcv, features, dropped_cfg)
    assert dropped is not None
    assert "f_mom" not in dropped["X"].columns


def test_count_events_only_writes_artifacts_and_skips_training(tmp_path, monkeypatch, synth_ohlcv, cfg):
    """--count-events-only must build members, write member_event_counts.json
    and per-(asset, primary) events_side_fwd.parquet artifacts, and return
    before ever invoking the (expensive, untested-here) pooled trainer."""
    ohlcv_full = (
        synth_ohlcv.set_index("time")[["open", "high", "low", "close", "volume"]]
        .astype("float64")
    )

    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text(
        yaml.safe_dump(
            {
                "selection_rule": "test fixture",
                "selected_at": "2026-07-03",
                "stocks": ["AAA"],
                "etfs": [],
                "alternates": [],
                "excluded_delistees": {},
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    full_cfg = dict(cfg)
    full_cfg.update(
        {
            "universe_path": str(universe_path),
            "universe_segment": "stocks",
            "date_range": {"start": "2010-01-01", "end": "2013-01-01"},
            "meta_pooling": {
                "schema": "core",
                "weight_balance": "per_class",
                "pooled_uniqueness": True,
                "train_min_frac": 0.5,
            },
            "output_dir": str(out_dir),
        }
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(full_cfg), encoding="utf-8")

    monkeypatch.setattr(runner, "load_dataset", lambda path: ohlcv_full)
    monkeypatch.setattr(runner, "build_macro_frame", lambda s, e, cache_dir: pd.DataFrame())
    monkeypatch.setattr(runner, "build_tier2_features", lambda ohlcv, macro: _features_for(ohlcv))

    def _must_not_train(*args, **kwargs):
        raise AssertionError("must not train")

    monkeypatch.setattr(runner, "_run_one_pool", _must_not_train)
    monkeypatch.setattr(
        sys, "argv",
        ["run_pooled_equity_d1.py", "--config", str(config_path), "--count-events-only"],
    )

    rc = runner.main()
    assert rc == 0

    counts_path = out_dir / "member_event_counts.json"
    assert counts_path.exists()
    counts = json.loads(counts_path.read_text(encoding="utf-8"))
    assert len(counts) > 0
    for c in counts:
        assert c["n_events"] > 0
        parquet_path = out_dir / c["asset"] / c["primary"] / "events_side_fwd.parquet"
        assert parquet_path.exists()
        df = pd.read_parquet(parquet_path)
        assert set(df.columns) == {"side", "fwd_ret"}


def test_feature_overrides_status_json_written(tmp_path, monkeypatch, synth_ohlcv, cfg):
    """main() must write feature_overrides_status.json per (asset, primary),
    mirroring run_backtest.py:1158-1168: add_requested/add_status computed
    against the POST-drop meta-feature columns, and drop_applied recorded
    verbatim. 'f_mom' is both dropped and requested-as-add, so its add_status
    must be 'not_in_tier2_skipped' (it's no longer available once dropped)."""
    ohlcv_full = (
        synth_ohlcv.set_index("time")[["open", "high", "low", "close", "volume"]]
        .astype("float64")
    )

    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text(
        yaml.safe_dump(
            {
                "selection_rule": "test fixture",
                "selected_at": "2026-07-03",
                "stocks": ["AAA"],
                "etfs": [],
                "alternates": [],
                "excluded_delistees": {},
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    full_cfg = dict(cfg)
    full_cfg.update(
        {
            "universe_path": str(universe_path),
            "universe_segment": "stocks",
            "date_range": {"start": "2010-01-01", "end": "2013-01-01"},
            "meta_pooling": {
                "schema": "core",
                "weight_balance": "per_class",
                "pooled_uniqueness": True,
                "train_min_frac": 0.5,
            },
            "output_dir": str(out_dir),
            "feature_overrides_add": ["f_mom", "z_r20"],
            "feature_overrides_drop": ["f_mom"],
        }
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(full_cfg), encoding="utf-8")

    monkeypatch.setattr(runner, "load_dataset", lambda path: ohlcv_full)
    monkeypatch.setattr(runner, "build_macro_frame", lambda s, e, cache_dir: pd.DataFrame())
    monkeypatch.setattr(runner, "build_tier2_features", lambda ohlcv, macro: _features_for(ohlcv))

    def _must_not_train(*args, **kwargs):
        raise AssertionError("must not train")

    monkeypatch.setattr(runner, "_run_one_pool", _must_not_train)
    monkeypatch.setattr(
        sys, "argv",
        ["run_pooled_equity_d1.py", "--config", str(config_path), "--count-events-only"],
    )

    rc = runner.main()
    assert rc == 0

    counts = json.loads((out_dir / "member_event_counts.json").read_text(encoding="utf-8"))
    assert len(counts) > 0
    for c in counts:
        status_path = out_dir / c["asset"] / c["primary"] / "feature_overrides_status.json"
        assert status_path.exists()
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["add_requested"] == ["f_mom", "z_r20"]
        assert status["add_status"]["f_mom"] == "not_in_tier2_skipped"
        assert status["add_status"]["z_r20"] == "present"
        assert status["drop_applied"] == ["f_mom"]
        assert "meta_feature_count" in status


@pytest.mark.parametrize("cs_enabled", [True, False])
def test_cross_sectional_join_is_config_gated(tmp_path, monkeypatch, synth_ohlcv, cfg, cs_enabled):
    """With features.cross_sectional true, every member's features frame gains
    the 8 cs_* columns before build_member_inputs is called; with it false (or
    absent), no cs_ columns appear. Uses the count-events mocked-main pattern
    with a 2-ticker universe so the CS panel is non-degenerate (a 1-ticker
    panel would make cs ranks/breadth trivially constant)."""
    ohlcv_full = (
        synth_ohlcv.set_index("time")[["open", "high", "low", "close", "volume"]]
        .astype("float64")
    )

    universe_path = tmp_path / "universe.yaml"
    universe_path.write_text(
        yaml.safe_dump(
            {
                "selection_rule": "test fixture",
                "selected_at": "2026-07-03",
                "stocks": ["AAA", "BBB"],
                "etfs": [],
                "alternates": [],
                "excluded_delistees": {},
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    full_cfg = dict(cfg)
    full_cfg.update(
        {
            "universe_path": str(universe_path),
            "universe_segment": "stocks",
            "date_range": {"start": "2010-01-01", "end": "2013-01-01"},
            "meta_pooling": {
                "schema": "core",
                "weight_balance": "per_class",
                "pooled_uniqueness": True,
                "train_min_frac": 0.5,
            },
            "output_dir": str(out_dir),
            "features": {"cross_sectional": cs_enabled},
        }
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(full_cfg), encoding="utf-8")

    monkeypatch.setattr(runner, "load_dataset", lambda path: ohlcv_full)
    monkeypatch.setattr(runner, "build_macro_frame", lambda s, e, cache_dir: pd.DataFrame())
    monkeypatch.setattr(runner, "build_tier2_features", lambda ohlcv, macro: _features_for(ohlcv))

    def _must_not_train(*args, **kwargs):
        raise AssertionError("must not train")

    monkeypatch.setattr(runner, "_run_one_pool", _must_not_train)
    monkeypatch.setattr(
        sys, "argv",
        ["run_pooled_equity_d1.py", "--config", str(config_path), "--count-events-only"],
    )

    seen_cols: dict[str, list[str]] = {}
    orig = runner.build_member_inputs

    def spy(asset, primary_name, ohlcv, features, cfg_):
        seen_cols[asset] = list(features.columns)
        return orig(asset, primary_name, ohlcv, features, cfg_)

    monkeypatch.setattr(runner, "build_member_inputs", spy)

    rc = runner.main()
    assert rc == 0

    counts_path = out_dir / "member_event_counts.json"
    assert counts_path.exists()
    counts = json.loads(counts_path.read_text(encoding="utf-8"))
    assert len(counts) > 0
    for c in counts:
        parquet_path = out_dir / c["asset"] / c["primary"] / "events_side_fwd.parquet"
        assert parquet_path.exists()

    assert len(seen_cols) > 0, "spy never observed a build_member_inputs call"
    for asset, cols in seen_cols.items():
        if cs_enabled:
            assert "cs_mom_12_1_rank" in cols and "cs_breadth_200" in cols, \
                f"{asset}: expected cs_ columns when cross_sectional=True, got {cols}"
        else:
            assert not any(c.startswith("cs_") for c in cols), \
                f"{asset}: unexpected cs_ columns when cross_sectional=False: {cols}"


# --- B0012 v2 fit-weight mode wiring (Task 3) --------------------------------


def _write_universe(tmp_path, tickers, name="universe.yaml"):
    universe_path = tmp_path / name
    universe_path.write_text(
        yaml.safe_dump(
            {
                "selection_rule": "test fixture",
                "selected_at": "2026-07-03",
                "stocks": list(tickers),
                "etfs": [],
                "alternates": [],
                "excluded_delistees": {},
            }
        ),
        encoding="utf-8",
    )
    return universe_path


def _run_main_capture(tmp_path, monkeypatch, ohlcv_by_ticker, meta_pooling_extra, cfg,
                      date_range, out_dir):
    """Runs the real main() (not --count-events-only) with load_dataset /
    build_macro_frame / build_tier2_features mocked and _run_one_pool replaced
    by a kwargs-capturing stub. Returns the list of captured _run_one_pool
    call-kwargs dicts."""
    universe_path = _write_universe(tmp_path, list(ohlcv_by_ticker), name=f"universe_{out_dir.name}.yaml")
    full_cfg = dict(cfg)
    full_cfg.update(
        {
            "universe_path": str(universe_path),
            "universe_segment": "stocks",
            "date_range": date_range,
            "meta_pooling": {
                "schema": "core",
                "weight_balance": "per_class",
                "pooled_uniqueness": True,
                "train_min_frac": 0.5,
                **meta_pooling_extra,
            },
            "output_dir": str(out_dir),
        }
    )
    config_path = tmp_path / f"config_{out_dir.name}.yaml"
    config_path.write_text(yaml.safe_dump(full_cfg), encoding="utf-8")

    def _load(path):
        path_str = str(path)
        for t, df in ohlcv_by_ticker.items():
            if f"{t}_D1" in path_str:
                return df.copy()
        raise AssertionError(f"unexpected load_dataset path {path_str}")

    monkeypatch.setattr(runner, "load_dataset", _load)
    monkeypatch.setattr(runner, "build_macro_frame", lambda s, e, cache_dir: pd.DataFrame())
    monkeypatch.setattr(runner, "build_tier2_features", lambda ohlcv, macro: _features_for(ohlcv))

    calls = []

    def _capture(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(runner, "_run_one_pool", _capture)
    monkeypatch.setattr(
        sys, "argv", ["run_pooled_equity_d1.py", "--config", str(config_path)],
    )

    rc = runner.main()
    assert rc == 0
    return calls


def _two_ticker_ohlcv_variants():
    """Two tickers with DIFFERENT price paths (BBB scaled + independent noise
    added on top of a shared factor) sharing a common date grid, long enough
    (1500 bars) to clear rolling_panel_rho's 252-bar window at least once."""
    rng = np.random.default_rng(123)
    n = 1500
    dates = pd.date_range("2005-01-03", periods=n, freq="B", tz="UTC")
    factor = rng.normal(0.0001, 0.01, size=n)
    close_a = 100.0 * np.exp(np.cumsum(factor))
    noise_b = rng.normal(0.0, 0.015, size=n)
    close_b = 50.0 * np.exp(np.cumsum(0.3 * factor + noise_b))

    def _mk(close):
        spread = np.abs(rng.normal(0.0, 0.005, size=n)) * close
        open_ = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum.reduce([close + spread, open_, close])
        low = np.minimum.reduce([close - spread, open_, close])
        volume = rng.integers(50_000, 500_000, size=n).astype(float)
        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

    return {"AAA": _mk(close_a), "BBB": _mk(close_b)}


def test_fit_weight_default_is_byte_identical(tmp_path, monkeypatch, synth_ohlcv, cfg):
    ohlcv = (
        synth_ohlcv.set_index("time")[["open", "high", "low", "close", "volume"]]
        .astype("float64")
    )
    date_range = {"start": "2010-01-01", "end": "2013-01-01"}

    out_absent = tmp_path / "out_absent"
    calls_absent = _run_main_capture(
        tmp_path, monkeypatch, {"AAA": ohlcv}, {}, cfg, date_range, out_absent,
    )
    out_rho1 = tmp_path / "out_rho1"
    calls_rho1 = _run_main_capture(
        tmp_path, monkeypatch, {"AAA": ohlcv}, {"fit_weight": "rho1_pooled"}, cfg,
        date_range, out_rho1,
    )

    assert len(calls_absent) == len(calls_rho1) == len(cfg["primary"]["candidates"])
    for c in calls_absent + calls_rho1:
        assert c["pooled_uniqueness"] is True

    by_primary_absent = {c["primary_name"]: c for c in calls_absent}
    by_primary_rho1 = {c["primary_name"]: c for c in calls_rho1}
    for primary in by_primary_absent:
        members_a = by_primary_absent[primary]["members"]
        members_b = by_primary_rho1[primary]["members"]
        assert len(members_a) == len(members_b)
        for ma, mb in zip(members_a, members_b):
            np.testing.assert_allclose(np.asarray(ma["w"]), np.asarray(mb["w"]))

    counts_absent = json.loads((out_absent / "member_event_counts.json").read_text(encoding="utf-8"))
    for out_dir in (out_absent, out_rho1):
        for primary in cfg["primary"]["candidates"]:
            p = out_dir / f"effective_n_{primary}.json"
            assert p.exists()
            diag = json.loads(p.read_text(encoding="utf-8"))
            assert diag["fit_weight_mode"] == "rho1_pooled"
            assert diag["effective_n_rho1"] > 0
            total = sum(c["n_events"] for c in counts_absent if c["primary"] == primary)
            assert diag["raw_n"] == total


def test_fit_weight_v2_overrides_weights_and_flags(tmp_path, monkeypatch, cfg):
    ohlcv_by_ticker = _two_ticker_ohlcv_variants()
    date_range = {"start": "2005-01-01", "end": "2011-06-01"}

    out_base = tmp_path / "out_base"
    calls_base = _run_main_capture(
        tmp_path, monkeypatch, ohlcv_by_ticker, {}, cfg, date_range, out_base,
    )
    out_v2 = tmp_path / "out_v2"
    calls_v2 = _run_main_capture(
        tmp_path, monkeypatch, ohlcv_by_ticker, {"fit_weight": "corr_discounted_v2"}, cfg,
        date_range, out_v2,
    )

    for c in calls_v2:
        assert c["pooled_uniqueness"] is False

    by_primary_base = {c["primary_name"]: c for c in calls_base}
    by_primary_v2 = {c["primary_name"]: c for c in calls_v2}

    for primary in by_primary_base:
        base_w = np.concatenate([np.asarray(m["w"]) for m in by_primary_base[primary]["members"]])
        v2_w = np.concatenate([np.asarray(m["w"]) for m in by_primary_v2[primary]["members"]])
        assert not np.allclose(base_w, v2_w), "v2 must overwrite member weights"

        diag = json.loads((out_v2 / f"effective_n_{primary}.json").read_text(encoding="utf-8"))
        assert diag["fit_weight_mode"] == "corr_discounted_v2"
        assert diag["fit_weight_sum"] > diag["effective_n_rho1"], \
            "v2 must unstarve the fit relative to the rho=1 conservative sum"
        assert diag["enb_ceiling"] is not None
        assert 1.0 - 1e-6 <= diag["enb_ceiling"] <= 2.0 + 1e-6


def test_gate_inputs_unchanged_by_fit_weight(tmp_path, monkeypatch, cfg):
    ohlcv_by_ticker = _two_ticker_ohlcv_variants()
    date_range = {"start": "2005-01-01", "end": "2011-06-01"}

    diag_by_mode: dict[str, dict] = {}
    for mode_key, extra in (
        ("default", {}),
        ("v2", {"fit_weight": "corr_discounted_v2"}),
        ("per_asset", {"fit_weight": "per_asset"}),
    ):
        out_dir = tmp_path / f"out_{mode_key}"
        _run_main_capture(tmp_path, monkeypatch, ohlcv_by_ticker, extra, cfg, date_range, out_dir)
        diag_by_mode[mode_key] = {
            primary: json.loads((out_dir / f"effective_n_{primary}.json").read_text(encoding="utf-8"))
            for primary in cfg["primary"]["candidates"]
        }

    for primary in cfg["primary"]["candidates"]:
        rho1 = diag_by_mode["default"][primary]["effective_n_rho1"]
        v2 = diag_by_mode["v2"][primary]["effective_n_rho1"]
        per_asset = diag_by_mode["per_asset"][primary]["effective_n_rho1"]
        assert rho1 == pytest.approx(v2)
        assert rho1 == pytest.approx(per_asset)
