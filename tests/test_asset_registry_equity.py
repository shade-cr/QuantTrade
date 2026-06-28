from phase5.asset_registry import ASSET_REGISTRY


def test_nvda_registered_as_equity():
    spec = ASSET_REGISTRY["NVDA"]
    assert spec.asset_class == "equity"
    assert spec.frequencies == ("D1",)


def test_nvda_dossier_pack_inherits_macro():
    pack = ASSET_REGISTRY["NVDA"].dossier_feature_pack()
    # own base feature
    assert "cs_spread_21" in pack
    # macro members auto-appended for non-fx classes
    for m in ("vix_level", "vix_chg_5", "dxy_z252", "real_yield_5y_z252d"):
        assert m in pack
    # metals-only alt-data must NOT leak into equities
    assert "cot_net_noncomm_z52w" not in pack
    assert "gld_dvol_z42" not in pack
