import numpy as np
import pandas as pd
import pytest
from phase5.regime_stats import load_macro_pack


@pytest.fixture
def fred_cache(tmp_path):
    idx = pd.date_range("2020-01-01", periods=800, freq="D", tz="UTC")
    for code, vals in {
        "VIXCLS": np.linspace(15, 25, 800),
        "DTWEXBGS": np.linspace(90, 110, 800),
        "T5YIE": np.linspace(2.0, 2.5, 800),
        "DGS5": np.linspace(1.0, 3.0, 800),
        "DGS2": np.linspace(0.5, 2.0, 800),
    }.items():
        pd.DataFrame({code: vals}, index=idx).to_parquet(tmp_path / f"{code}.parquet")
    return tmp_path


def test_load_macro_pack_returns_requested_present_members(fred_cache):
    target = pd.date_range("2021-06-01", periods=200, freq="D", tz="UTC")
    members = ["vix_level", "vix_chg_5", "dxy_z252", "breakeven_5y_chg5", "us_5y2y_z252"]
    out = load_macro_pack(target, members, fred_cache_dir=fred_cache)
    assert list(out.columns) == members
    assert out.index.equals(target)


def test_load_macro_pack_skips_absent_cache(fred_cache):
    target = pd.date_range("2021-06-01", periods=200, freq="D", tz="UTC")
    (fred_cache / "VIXCLS.parquet").unlink()
    out = load_macro_pack(target, ["vix_level", "dxy_z252"], fred_cache_dir=fred_cache)
    assert "vix_level" not in out.columns
    assert "dxy_z252" in out.columns


def test_load_macro_pack_pit_shift1_d1(fred_cache):
    target = pd.date_range("2021-06-01", periods=50, freq="D", tz="UTC")
    out = load_macro_pack(target, ["vix_level"], fred_cache_dir=fred_cache)
    raw = pd.read_parquet(fred_cache / "VIXCLS.parquet")["VIXCLS"]
    raw.index = raw.index.tz_localize("UTC") if raw.index.tz is None else raw.index
    t = target[10]
    assert out.loc[t, "vix_level"] == pytest.approx(raw.asof(t - pd.Timedelta(days=1)))


def test_load_macro_pack_pit_on_h4_index(fred_cache):
    target = pd.date_range("2021-06-01", periods=300, freq="4h", tz="UTC")
    out = load_macro_pack(target, ["vix_level"], fred_cache_dir=fred_cache)
    raw = pd.read_parquet(fred_cache / "VIXCLS.parquet")["VIXCLS"]
    raw.index = raw.index.tz_localize("UTC") if raw.index.tz is None else raw.index
    t = target[100]
    assert out.loc[t, "vix_level"] == pytest.approx(raw.asof(t.normalize() - pd.Timedelta(days=1)))


from phase5.regime_stats import build_dossier_features


def _ohlcv(n=300):
    idx = pd.date_range("2021-06-01", periods=n, freq="D", tz="UTC")
    c = pd.Series(np.linspace(100, 120, n), index=idx)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)


def test_build_dossier_features_injects_macro(tmp_path, monkeypatch):
    (tmp_path / "cache" / "fred").mkdir(parents=True)
    idx = pd.date_range("2020-01-01", periods=800, freq="D", tz="UTC")
    pd.DataFrame({"VIXCLS": np.linspace(15, 25, 800)}, index=idx).to_parquet(tmp_path / "cache" / "fred" / "VIXCLS.parquet")
    pd.DataFrame({"DTWEXBGS": np.linspace(90, 110, 800)}, index=idx).to_parquet(tmp_path / "cache" / "fred" / "DTWEXBGS.parquet")
    monkeypatch.chdir(tmp_path)
    df = _ohlcv()
    feats = build_dossier_features(df, pd.DataFrame(index=df.index), asset="BTCUSD", feature_pack=("vix_level", "dxy_z252"))
    assert "vix_level" in feats.columns and "dxy_z252" in feats.columns


def test_build_dossier_features_skips_absent_macro(tmp_path, monkeypatch):
    (tmp_path / "cache" / "fred").mkdir(parents=True)  # empty cache dir
    monkeypatch.chdir(tmp_path)
    df = _ohlcv()
    feats = build_dossier_features(df, pd.DataFrame(index=df.index), asset="BTCUSD", feature_pack=("vix_level",))
    assert "vix_level" not in feats.columns


from phase5.regime_stats import build_regime_dossiers, B0036_MVP_FEATURES


def test_macro_features_registered_for_vetting():
    for m in ("vix_level", "vix_chg_5", "dxy_z252", "breakeven_5y_chg5", "us_5y2y_z252"):
        assert m in B0036_MVP_FEATURES


def test_effective_n_is_frequency_aware_on_h4():
    idx = pd.date_range("2021-01-01", periods=360, freq="4h", tz="UTC")
    regimes = pd.DataFrame({"regime_id": ["BULL_QUIET"] * 360,
                            "roc_63": 0.1, "ma_50": 1.0, "ma_200": 0.9, "rv_20": 0.2}, index=idx)
    feats = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                          "vix_level": np.linspace(15, 25, 360)}, index=idx)
    dossiers = build_regime_dossiers(regimes, feats, asset_class="crypto", frequency="H4")
    eff = dossiers["BULL_QUIET"]["features_quantile_summary"]["vix_level"]["effective_n"]
    assert 50 <= eff <= 70  # ~360/6


def _regimes_frame(n=400):
    idx = pd.date_range("2018-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"regime_id": ["BULL_QUIET"] * n,
                         "roc_63": np.linspace(-1, 1, n), "ma_50": np.linspace(1, 2, n),
                         "ma_200": np.linspace(1, 1.5, n), "rv_20": np.linspace(0.1, 0.3, n)}, index=idx)


def test_quasi_circular_feature_excluded_from_orthogonal():
    reg = _regimes_frame()
    feats = pd.DataFrame(index=reg.index)
    feats[["open", "high", "low", "close"]] = 1.0
    feats["vix_level"] = reg["roc_63"].values  # IS roc_63 -> rho=1 -> circular
    feats[["roc_63", "ma_50", "ma_200", "rv_20"]] = reg[["roc_63", "ma_50", "ma_200", "rv_20"]].values
    d = build_regime_dossiers(reg, feats, asset_class="crypto", frequency="D1")["BULL_QUIET"]
    assert d["features_quantile_summary"]["vix_level"]["quasi_circular"] is True
    assert "vix_level" not in d["orthogonal_features"]


def test_fail_open_guard_excludes_when_correlation_inconclusive():
    reg = _regimes_frame(n=40)
    feats = pd.DataFrame(index=reg.index)
    feats[["open", "high", "low", "close"]] = 1.0
    feats[["roc_63", "ma_50", "ma_200", "rv_20"]] = reg[["roc_63", "ma_50", "ma_200", "rv_20"]].values
    v = np.full(40, np.nan); v[:5] = [1, 2, 3, 4, 5]  # only 5 non-NaN -> <30 overlap
    feats["vix_level"] = v
    d = build_regime_dossiers(reg, feats, asset_class="crypto", frequency="D1")["BULL_QUIET"]
    s = d["features_quantile_summary"]["vix_level"]
    assert s["vetting_inconclusive"] is True
    assert "vix_level" not in d["orthogonal_features"]
