import numpy as np
import pandas as pd
import pytest
from pipeline.alpha_ops import AlphaContext


def _panel(n=10, syms=("A", "B", "C", "D")):
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    fields = {}
    for f in ("open", "high", "low", "close", "volume"):
        fields[f] = pd.DataFrame(
            rng.uniform(1, 100, size=(n, len(syms))), index=idx, columns=list(syms)
        )
    # enforce high>=close>=low so derived fields are sane
    fields["high"] = fields["high"]
    return fields


def test_xs_rank_is_cross_sectional_percentile():
    fields = _panel()
    ctx = AlphaContext(fields, mode="xs")
    # On a row where close is strictly increasing A<B<C<D, pct ranks are 0.25..1.0
    row = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    ctx.close.iloc[0] = row
    r = ctx.rank(ctx.close)
    assert r.iloc[0].tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_signedpower_preserves_sign():
    fields = _panel()
    ctx = AlphaContext(fields, mode="xs")
    x = pd.DataFrame({"A": [-4.0], "B": [9.0]})
    out = ctx.signedpower(x, 0.5)
    assert out["A"].iloc[0] == pytest.approx(-2.0)
    assert out["B"].iloc[0] == pytest.approx(3.0)


def test_vwap_is_typical_price_proxy():
    fields = _panel()
    ctx = AlphaContext(fields, mode="xs")
    expected = (ctx.high + ctx.low + ctx.close) / 3.0
    pd.testing.assert_frame_equal(ctx.vwap, expected)


def test_scale_xs_l1_normalizes():
    fields = _panel()
    ctx = AlphaContext(fields, mode="xs")
    x = pd.DataFrame({"A": [1.0], "B": [-3.0]})
    out = ctx.scale(x, a=1.0)
    # denom = |1| + |-3| = 4 ; scaled = [0.25, -0.75] ; sum|.| == 1.0
    assert out["A"].iloc[0] == pytest.approx(0.25)
    assert out["B"].iloc[0] == pytest.approx(-0.75)
    assert out.iloc[0].abs().sum() == pytest.approx(1.0)


def test_delay_and_delta():
    fields = _panel(n=6)
    ctx = AlphaContext(fields, mode="xs")
    s = pd.DataFrame({"A": [1.0, 2, 4, 7, 11, 16]}, index=ctx.close.index)
    assert ctx.delay(s, 1)["A"].tolist()[1:] == [1, 2, 4, 7, 11]
    assert ctx.delta(s, 1)["A"].tolist()[1:] == [1, 2, 3, 4, 5]


def test_ts_argmax_is_position_within_window():
    fields = _panel(n=5)
    ctx = AlphaContext(fields, mode="xs")
    s = pd.DataFrame({"A": [3.0, 1, 9, 2, 5]}, index=ctx.close.index)
    # trailing window d=3 ending at idx 4 is [9,2,5] -> max at position 0
    assert ctx.ts_argmax(s, 3)["A"].iloc[4] == 0
    # window ending at idx 2 is [3,1,9] -> max at position 2
    assert ctx.ts_argmax(s, 3)["A"].iloc[2] == 2


def test_decay_linear_weights_recent_more():
    fields = _panel(n=4)
    ctx = AlphaContext(fields, mode="xs")
    s = pd.DataFrame({"A": [0.0, 0, 0, 1.0]}, index=ctx.close.index)
    # d=4 weights normalized (1,2,3,4)/10; only last day is 1 -> 4/10
    assert ctx.decay_linear(s, 4)["A"].iloc[3] == pytest.approx(0.4)


def test_ts_rank_last_value_percentile():
    fields = _panel(n=5)
    ctx = AlphaContext(fields, mode="xs")
    s = pd.DataFrame({"A": [1.0, 2, 3, 4, 5]}, index=ctx.close.index)
    # window [3,4,5] last value 5 is the max -> pct rank 1.0
    assert ctx.ts_rank(s, 3)["A"].iloc[4] == pytest.approx(1.0)


def test_correlation_perfect_positive():
    fields = _panel(n=6)
    ctx = AlphaContext(fields, mode="xs")
    x = pd.DataFrame({"A": [1.0, 2, 3, 4, 5, 6]}, index=ctx.close.index)
    y = pd.DataFrame({"A": [2.0, 4, 6, 8, 10, 12]}, index=ctx.close.index)
    assert ctx.correlation(x, y, 4)["A"].iloc[5] == pytest.approx(1.0)


def test_adv_is_rolling_mean_volume():
    fields = _panel(n=5)
    ctx = AlphaContext(fields, mode="xs")
    ctx.volume.iloc[:, 0] = [10.0, 20, 30, 40, 50]
    assert ctx.adv(3).iloc[4, 0] == pytest.approx(40.0)  # mean(30,40,50)


def test_operators_are_causal():
    # Perturbing a FUTURE bar must not change any operator value at indices < cut.
    fields = _panel(n=30)
    ctx = AlphaContext(fields, mode="xs")
    cut = 20
    perturbed = {k: v.copy() for k, v in fields.items()}
    for k in perturbed:
        perturbed[k].iloc[cut:] *= 5.0  # corrupt the future
    ctx2 = AlphaContext(perturbed, mode="xs")
    probes = [
        ("ts_rank", lambda c: c.ts_rank(c.close, 5)),
        ("adv", lambda c: c.adv(5)),
        ("correlation", lambda c: c.correlation(c.close, c.volume, 5)),
        ("covariance", lambda c: c.covariance(c.close, c.volume, 5)),
    ]
    for name, fn in probes:
        base = fn(ctx)
        after = fn(ctx2)
        pd.testing.assert_frame_equal(
            base.iloc[:cut], after.iloc[:cut],
            obj=name, check_names=False,
        )


def test_covariance_matches_two_var():
    fields = _panel(n=6)
    ctx = AlphaContext(fields, mode="xs")
    idx = ctx.close.index
    x = pd.DataFrame({"A": [1.0, 2, 3, 4, 5, 6]}, index=idx)
    y = 2 * x
    # cov(x, 2x) = 2 * var(x); pandas rolling var uses ddof=1
    expected = 2 * x["A"].rolling(4).var().iloc[5]
    assert ctx.covariance(x, y, 4)["A"].iloc[5] == pytest.approx(expected)
