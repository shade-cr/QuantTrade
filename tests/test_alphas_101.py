"""Tests for pipeline/alphas_101.py — registry plumbing and tier classifier."""
import numpy as np
import pandas as pd
import pytest
from pipeline.alpha_ops import AlphaContext
from pipeline.alphas_101 import REGISTRY, classify_tier


def test_registry_has_all_101_after_transcription_is_complete():
    from pipeline.alphas_101 import REGISTRY
    assert set(REGISTRY) == set(range(1, 102))  # all 101 registered
    # every id has either a callable fn (computable) or a Tier-C stub (fn is None)
    for aid, spec in REGISTRY.items():
        assert (spec.fn is not None) or (classify_tier(spec.fields) == "C")


def test_tier_classification():
    assert classify_tier(("close", "open")) == "A"
    assert classify_tier(("close", "volume")) == "B"
    assert classify_tier(("close", "vwap")) == "B"
    assert classify_tier(("close", "cap")) == "C"
    assert classify_tier(("close", "indclass")) == "C"


def test_tier_adv_variants_are_proxy():
    assert classify_tier(("close", "adv20")) == "B"
    assert classify_tier(("vwap", "adv180")) == "B"


def test_tier_na_trumps_proxy():
    assert classify_tier(("close", "cap", "volume")) == "C"
    assert classify_tier(("close", "indclass.sector", "adv20")) == "C"


def test_register_na_registers_tier_c_stub():
    from pipeline.alphas_101 import register_na, REGISTRY, AlphaSpec
    register_na(999, ("close", "indclass"))
    spec = REGISTRY[999]
    assert spec.fn is None
    assert classify_tier(spec.fields) == "C"
    del REGISTRY[999]  # don't pollute the registry for other tests


def _ctx(n=40, syms=("A", "B", "C", "D"), mode="xs"):
    """Build a test AlphaContext with random OHLCV data."""
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    fields = {f: pd.DataFrame(rng.uniform(1, 100, (n, len(syms))), index=idx,
                              columns=list(syms)) for f in
              ("open", "high", "low", "close", "volume")}
    return AlphaContext(fields, mode=mode)


def test_alpha101_intrabar_ratio_matches_formula():
    ctx = _ctx()
    out = REGISTRY[101].fn(ctx)
    expected = (ctx.close - ctx.open) / ((ctx.high - ctx.low) + 0.001)
    pd.testing.assert_frame_equal(out, expected)


def test_alpha6_is_neg_corr_open_volume():
    ctx = _ctx()
    out = REGISTRY[6].fn(ctx)
    expected = -1 * ctx.correlation(ctx.open, ctx.volume, 10)
    pd.testing.assert_frame_equal(out, expected)


def test_alpha1_matches_formula():
    ctx = _ctx()
    base = ctx.stddev(ctx.returns, 20).where(ctx.returns < 0, ctx.close)
    expected = ctx.rank(ctx.ts_argmax(ctx.signedpower(base, 2.0), 5)) - 0.5
    pd.testing.assert_frame_equal(REGISTRY[1].fn(ctx), expected)


def test_alpha12_matches_formula():
    ctx = _ctx()
    expected = ctx.sign(ctx.delta(ctx.volume, 1)) * (-1 * ctx.delta(ctx.close, 1))
    pd.testing.assert_frame_equal(REGISTRY[12].fn(ctx), expected)


def test_alpha53_matches_formula():
    ctx = _ctx()
    inner = ((ctx.close - ctx.low) - (ctx.high - ctx.close)) / (ctx.close - ctx.low)
    expected = -1 * ctx.delta(inner, 9)
    pd.testing.assert_frame_equal(REGISTRY[53].fn(ctx), expected)


def test_worked_alphas_are_causal():
    for aid in (1, 6, 12, 53, 101):
        base = REGISTRY[aid].fn(_ctx(n=40))
        # rebuild with corrupted future: construct fresh field frames, corrupt them,
        # then rebuild ctx so all cached properties (returns, etc) recompute
        idx = pd.date_range("2020-01-01", periods=40, freq="D", tz="UTC")
        rng = np.random.default_rng(3)
        syms = ("A", "B", "C", "D")
        fields = {f: pd.DataFrame(rng.uniform(1, 100, (40, len(syms))), index=idx,
                                  columns=list(syms)) for f in
                  ("open", "high", "low", "close", "volume")}
        # corrupt the future (rows 30 onward)
        for f in ("open", "high", "low", "close", "volume"):
            fields[f].iloc[30:] *= 7.0
        # rebuild context from corrupted fields — cached properties recompute
        ctx2 = AlphaContext(fields, mode="xs")
        after = REGISTRY[aid].fn(ctx2)
        pd.testing.assert_frame_equal(base.iloc[:25], after.iloc[:25],
                                      check_names=False)


# ===== Task 8: alphas 2–25 tests =====

# Alphas with fn is not None in the 2–25 range (all 22 — none are Tier-C in this range)
_BATCH_IDS = [2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]


@pytest.mark.parametrize("aid", _BATCH_IDS)
def test_batch_alpha_runs_and_returns_correct_shape(aid):
    """Every new alpha in 2–25 with fn runs without raising and returns panel shape."""
    ctx = _ctx(n=210)  # large enough for adv20, sum(returns,250) warm-up is within reason
    spec = REGISTRY[aid]
    assert spec.fn is not None, f"Alpha#{aid} unexpectedly has fn=None"
    out = spec.fn(ctx)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == ctx.close.shape
    assert list(out.columns) == list(ctx.close.columns)


@pytest.mark.parametrize("aid", _BATCH_IDS)
def test_batch_alpha_is_causal(aid):
    """Corrupting rows 30+ must not change rows 0–24 for any batch alpha."""
    n = 210
    base = REGISTRY[aid].fn(_ctx(n=n))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    syms = ("A", "B", "C", "D")
    fields = {f: pd.DataFrame(rng.uniform(1, 100, (n, len(syms))), index=idx,
                              columns=list(syms)) for f in
              ("open", "high", "low", "close", "volume")}
    for f in ("open", "high", "low", "close", "volume"):
        fields[f].iloc[30:] *= 7.0
    ctx2 = AlphaContext(fields, mode="xs")
    after = REGISTRY[aid].fn(ctx2)
    pd.testing.assert_frame_equal(base.iloc[:25], after.iloc[:25], check_names=False)


def test_alpha_011_matches_formula():
    """Alpha#11 hand-check: (rank(ts_max(vwap-close,3))+rank(ts_min(vwap-close,3)))*rank(delta(vol,3))"""
    ctx = _ctx(n=40)
    diff = ctx.vwap - ctx.close
    expected = (
        (ctx.rank(ctx.ts_max(diff, 3)) + ctx.rank(ctx.ts_min(diff, 3)))
        * ctx.rank(ctx.delta(ctx.volume, 3))
    )
    pd.testing.assert_frame_equal(REGISTRY[11].fn(ctx), expected)


def test_alpha_009_ternary_logic():
    """Alpha#9 ternary: returns delta when all-positive min, or when all-negative max, else -delta."""
    ctx = _ctx(n=40)
    d = ctx.delta(ctx.close, 1)
    cond1 = 0 < ctx.ts_min(d, 5)
    cond2 = ctx.ts_max(d, 5) < 0
    inner = d.where(cond2, -1 * d)
    expected = d.where(cond1, inner)
    pd.testing.assert_frame_equal(REGISTRY[9].fn(ctx), expected)


def test_alpha_021_three_way_ternary():
    """Alpha#21 three-way ternary produces only {-1, 1} values (ignoring NaNs)."""
    ctx = _ctx(n=40)
    out = REGISTRY[21].fn(ctx)
    non_nan = out.stack().dropna()
    assert set(non_nan.unique()).issubset({-1.0, 1.0})


def test_batch_tier_c_alphas_none():
    """No alphas in 2–25 should be Tier-C (all are implementable); all have fn is not None."""
    for aid in _BATCH_IDS:
        spec = REGISTRY[aid]
        tier = classify_tier(spec.fields)
        assert tier in ("A", "B"), f"Alpha#{aid} unexpectedly classified as Tier-{tier}"
        assert spec.fn is not None, f"Alpha#{aid} unexpectedly has fn=None"


# ===== Task 9: alphas 26–50 tests =====

# Tier-C ids in 26–50 (register_na only, no fn)
_BATCH9_TIER_C_IDS = [48]

# Computable ids in 26–50 (all except Tier-C)
_BATCH9_COMPUTABLE_IDS = [
    aid for aid in range(26, 51) if aid not in _BATCH9_TIER_C_IDS
]


@pytest.mark.parametrize("aid", _BATCH9_COMPUTABLE_IDS)
def test_batch9_alpha_runs_and_returns_correct_shape(aid):
    """Every computable alpha in 26–50 runs without raising and returns panel shape."""
    ctx = _ctx(n=260)  # large enough for sum(returns,250), correlation(...,230), sum(close,200)
    spec = REGISTRY[aid]
    assert spec.fn is not None, f"Alpha#{aid} unexpectedly has fn=None"
    out = spec.fn(ctx)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == ctx.close.shape
    assert list(out.columns) == list(ctx.close.columns)


@pytest.mark.parametrize("aid", _BATCH9_COMPUTABLE_IDS)
def test_batch9_alpha_is_causal(aid):
    """Corrupting rows 30+ must not change rows 0–24 for any batch-9 alpha."""
    n = 260
    base = REGISTRY[aid].fn(_ctx(n=n))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    syms = ("A", "B", "C", "D")
    fields = {f: pd.DataFrame(rng.uniform(1, 100, (n, len(syms))), index=idx,
                              columns=list(syms)) for f in
              ("open", "high", "low", "close", "volume")}
    for f in ("open", "high", "low", "close", "volume"):
        fields[f].iloc[30:] *= 7.0
    ctx2 = AlphaContext(fields, mode="xs")
    after = REGISTRY[aid].fn(ctx2)
    pd.testing.assert_frame_equal(base.iloc[:25], after.iloc[:25], check_names=False)


@pytest.mark.parametrize("aid", _BATCH9_TIER_C_IDS)
def test_batch9_tier_c_register_na(aid):
    """Each Tier-C alpha in 26–50 has fn=None and classifies as Tier C."""
    spec = REGISTRY[aid]
    assert spec.fn is None, f"Alpha#{aid} should have fn=None (Tier C)"
    assert classify_tier(spec.fields) == "C", f"Alpha#{aid} should be Tier C"


def test_alpha_046_ternary_logic():
    """Alpha#46 ternary: three-branch slope-based logic; spot-check formula equivalence."""
    ctx = _ctx(n=260)
    slope = (
        (ctx.delay(ctx.close, 20) - ctx.delay(ctx.close, 10)) / 10
        - (ctx.delay(ctx.close, 10) - ctx.close) / 10
    )
    neg_ones = pd.DataFrame(-1.0, index=ctx.close.index, columns=ctx.close.columns)
    ones = pd.DataFrame(1.0, index=ctx.close.index, columns=ctx.close.columns)
    inner = -1 * (ctx.close - ctx.delay(ctx.close, 1))
    inner2 = ones.where(slope < 0, inner)
    expected = neg_ones.where(0.25 < slope, inner2)
    pd.testing.assert_frame_equal(REGISTRY[46].fn(ctx), expected)


def test_alpha_027_returns_only_neg1_or_1():
    """Alpha#27 ternary produces only {-1, 1} values (ignoring NaNs)."""
    ctx = _ctx(n=260)
    out = REGISTRY[27].fn(ctx)
    non_nan = out.stack().dropna()
    assert set(non_nan.unique()).issubset({-1.0, 1.0})


def test_alpha_043_matches_formula():
    """Alpha#43 hand-check: ts_rank(vol/adv20, 20) * ts_rank(-1*delta(close,7), 8)."""
    ctx = _ctx(n=260)
    adv20 = ctx.adv(20)
    expected = ctx.ts_rank(ctx.volume / adv20, 20) * ctx.ts_rank(-1 * ctx.delta(ctx.close, 7), 8)
    pd.testing.assert_frame_equal(REGISTRY[43].fn(ctx), expected)


# ===== Task 10: alphas 51–75 tests =====

# Tier-C ids in 51–75 (register_na only, no fn)
_BATCH10_TIER_C_IDS = [56, 58, 59, 63, 67, 69, 70]

# Computable ids in 51–75 (all except Tier-C and #53 which was done in Task 7)
_BATCH10_COMPUTABLE_IDS = [
    aid for aid in range(51, 76) if aid not in _BATCH10_TIER_C_IDS and aid != 53
]


@pytest.mark.parametrize("aid", _BATCH10_COMPUTABLE_IDS)
def test_batch10_alpha_runs_and_returns_correct_shape(aid):
    """Every computable alpha in 51–75 runs without raising and returns panel shape."""
    ctx = _ctx(n=260)
    spec = REGISTRY[aid]
    assert spec.fn is not None, f"Alpha#{aid} unexpectedly has fn=None"
    out = spec.fn(ctx)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == ctx.close.shape
    assert list(out.columns) == list(ctx.close.columns)


@pytest.mark.parametrize("aid", _BATCH10_COMPUTABLE_IDS)
def test_batch10_alpha_is_causal(aid):
    """Corrupting rows 30+ must not change rows 0–24 for any batch-10 alpha."""
    n = 260
    base = REGISTRY[aid].fn(_ctx(n=n))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    syms = ("A", "B", "C", "D")
    fields = {f: pd.DataFrame(rng.uniform(1, 100, (n, len(syms))), index=idx,
                              columns=list(syms)) for f in
              ("open", "high", "low", "close", "volume")}
    for f in ("open", "high", "low", "close", "volume"):
        fields[f].iloc[30:] *= 7.0
    ctx2 = AlphaContext(fields, mode="xs")
    after = REGISTRY[aid].fn(ctx2)
    pd.testing.assert_frame_equal(base.iloc[:25], after.iloc[:25], check_names=False)


@pytest.mark.parametrize("aid", _BATCH10_TIER_C_IDS)
def test_batch10_tier_c_register_na(aid):
    """Each Tier-C alpha in 51–75 has fn=None and classifies as Tier C."""
    spec = REGISTRY[aid]
    assert spec.fn is None, f"Alpha#{aid} should have fn=None (Tier C)"
    assert classify_tier(spec.fields) == "C", f"Alpha#{aid} should be Tier C"


def test_alpha_051_matches_formula():
    """Alpha#51 structural formula check: slope-threshold ternary, verbatim re-derivation."""
    ctx = _ctx(n=260)
    slope = (
        (ctx.delay(ctx.close, 20) - ctx.delay(ctx.close, 10)) / 10
        - (ctx.delay(ctx.close, 10) - ctx.close) / 10
    )
    ones = pd.DataFrame(1.0, index=ctx.close.index, columns=ctx.close.columns)
    false_branch = -1 * (ctx.close - ctx.delay(ctx.close, 1))
    expected = ones.where(slope < (-1 * 0.05), false_branch)
    pd.testing.assert_frame_equal(REGISTRY[51].fn(ctx), expected)


def test_alpha_055_matches_formula():
    """Alpha#55 structural formula check: ternary-free but multi-op composition with normalised range."""
    ctx = _ctx(n=260)
    numer = ctx.close - ctx.ts_min(ctx.low, 12)
    denom = ctx.ts_max(ctx.high, 12) - ctx.ts_min(ctx.low, 12)
    expected = -1 * ctx.correlation(ctx.rank(numer / denom), ctx.rank(ctx.volume), 6)
    pd.testing.assert_frame_equal(REGISTRY[55].fn(ctx), expected)


# ===== Task 11: alphas 76–100 tests =====

# Tier-C ids in 76–100 (register_na, fn is None)
_BATCH11_TIER_C_IDS = [76, 79, 80, 82, 87, 89, 90, 91, 93, 97, 100]

# Computable ids in 76–100 (all except Tier-C)
_BATCH11_COMPUTABLE_IDS = [
    aid for aid in range(76, 101) if aid not in _BATCH11_TIER_C_IDS
]


@pytest.mark.parametrize("aid", _BATCH11_COMPUTABLE_IDS)
def test_batch11_alpha_runs_and_returns_correct_shape(aid):
    """Every computable alpha in 76–100 runs without raising and returns panel shape."""
    ctx = _ctx(n=300)  # large enough for ts_sum(adv5,26)+correlation windows
    spec = REGISTRY[aid]
    assert spec.fn is not None, f"Alpha#{aid} unexpectedly has fn=None"
    out = spec.fn(ctx)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == ctx.close.shape
    assert list(out.columns) == list(ctx.close.columns)


@pytest.mark.parametrize("aid", _BATCH11_COMPUTABLE_IDS)
def test_batch11_alpha_is_causal(aid):
    """Corrupting rows 30+ must not change rows 0–24 for any batch-11 alpha."""
    n = 300
    base = REGISTRY[aid].fn(_ctx(n=n))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    syms = ("A", "B", "C", "D")
    fields = {f: pd.DataFrame(rng.uniform(1, 100, (n, len(syms))), index=idx,
                              columns=list(syms)) for f in
              ("open", "high", "low", "close", "volume")}
    for f in ("open", "high", "low", "close", "volume"):
        fields[f].iloc[30:] *= 7.0
    ctx2 = AlphaContext(fields, mode="xs")
    after = REGISTRY[aid].fn(ctx2)
    pd.testing.assert_frame_equal(base.iloc[:25], after.iloc[:25], check_names=False)


@pytest.mark.parametrize("aid", _BATCH11_TIER_C_IDS)
def test_batch11_tier_c_register_na(aid):
    """Each Tier-C alpha in 76–100 has fn=None and classifies as Tier C."""
    spec = REGISTRY[aid]
    assert spec.fn is None, f"Alpha#{aid} should have fn=None (Tier C)"
    assert classify_tier(spec.fields) == "C", f"Alpha#{aid} should be Tier C"


def test_alpha_099_matches_formula():
    """Alpha#99 structural formula check: rank(corr(sum(HL/2), sum(adv60))) < rank(corr(low, vol)) then *-1."""
    ctx = _ctx(n=300)
    adv60 = ctx.adv(60)
    lhs = ctx.rank(ctx.correlation(
        ctx.ts_sum((ctx.high + ctx.low) / 2, 19),
        ctx.ts_sum(adv60, 19),
        8
    ))
    rhs = ctx.rank(ctx.correlation(ctx.low, ctx.volume, 6))
    expected = ((lhs < rhs).astype(float)) * -1
    pd.testing.assert_frame_equal(REGISTRY[99].fn(ctx), expected)
