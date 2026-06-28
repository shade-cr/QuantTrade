"""WorldQuant-101 formulaic alphas (Kakushadze 2016, arXiv:1601.00991).

Each alpha is a pure function of an AlphaContext returning a wide (time x symbol)
frame of alpha values. The REGISTRY + field manifest drive the computability
partition (Tier A clean / B proxy / C not-computable). Source formulas live in
cache/papers/alpha101_formulas.md.

Field tokens use base names; adv{d} and indclass* variants are matched by prefix
in classify_tier.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from pipeline.alpha_ops import AlphaContext

FIELDS_CLEAN = {"open", "high", "low", "close", "returns"}
FIELDS_PROXY = {"volume", "vwap", "adv"}
FIELDS_NA = {"cap", "indclass"}


@dataclass(frozen=True)
class AlphaSpec:
    id: int
    fields: tuple[str, ...]
    fn: Callable[[AlphaContext], pd.DataFrame] | None


REGISTRY: dict[int, AlphaSpec] = {}


def classify_tier(fields: tuple[str, ...]) -> str:
    """Return tier for an alpha given its field dependencies.

    Returns:
        "C" if any field in FIELDS_NA (cap, indclass) or matches indclass*/adv* prefix
        "B" if any field in FIELDS_PROXY (volume, vwap, adv) or matches adv{d} variant
        "A" if all fields in FIELDS_CLEAN — directly computable
    """
    fs = set(fields)
    # Tier C: market cap or industry-class neutralization (variants like
    # "indclass.sector" / "indclass_industry" all count).
    if (fs & FIELDS_NA) or any(f.startswith("indclass") for f in fs):
        return "C"
    # Tier B (proxy): volume/vwap, or any adv{d} variant (adv20, adv180, ...).
    if (fs & FIELDS_PROXY) or any(f.startswith("adv") for f in fs):
        return "B"
    return "A"


def register(id: int, fields: tuple[str, ...]):
    """Decorator that registers an alpha function into REGISTRY.

    Usage:
        @register(1, ("close", "returns", "volume"))
        def alpha_001(ctx: AlphaContext) -> pd.DataFrame:
            ...
    """
    def deco(fn):
        REGISTRY[id] = AlphaSpec(id=id, fields=fields, fn=fn)
        return fn
    return deco


def register_na(alpha_id: int, fields: tuple[str, ...]) -> None:
    """Register a Tier-C (not-computable) alpha — no callable.

    Used for alphas that require IndNeutralize/IndClass or market cap, which
    our data cannot compute. classify_tier(fields) must return "C".
    """
    REGISTRY[alpha_id] = AlphaSpec(id=alpha_id, fields=fields, fn=None)


# === Worked-example alphas (Task 7) ===


@register(1, ("returns", "close"))
def alpha_001(c: AlphaContext) -> pd.DataFrame:
    """Alpha#1: (rank(Ts_ArgMax(SignedPower((returns<0?stddev(returns,20):close),2),5))-0.5)"""
    base = c.stddev(c.returns, 20).where(c.returns < 0, c.close)
    return c.rank(c.ts_argmax(c.signedpower(base, 2.0), 5)) - 0.5


@register(6, ("open", "volume"))
def alpha_006(c: AlphaContext) -> pd.DataFrame:
    """Alpha#6: (-1 * correlation(open, volume, 10))"""
    return -1 * c.correlation(c.open, c.volume, 10)


@register(12, ("volume", "close"))
def alpha_012(c: AlphaContext) -> pd.DataFrame:
    """Alpha#12: (sign(delta(volume,1)) * (-1 * delta(close,1)))"""
    return c.sign(c.delta(c.volume, 1)) * (-1 * c.delta(c.close, 1))


@register(53, ("close", "low", "high"))
def alpha_053(c: AlphaContext) -> pd.DataFrame:
    """Alpha#53: (-1 * delta((((close-low)-(high-close))/(close-low)),9))"""
    inner = ((c.close - c.low) - (c.high - c.close)) / (c.close - c.low)
    return -1 * c.delta(inner, 9)


@register(101, ("close", "open", "high", "low"))
def alpha_101(c: AlphaContext) -> pd.DataFrame:
    """Alpha#101: ((close-open)/((high-low)+.001))"""
    return (c.close - c.open) / ((c.high - c.low) + 0.001)


# === Alphas 2–25 (Task 8) ===


@register(2, ("close", "open", "volume"))
def alpha_002(c: AlphaContext) -> pd.DataFrame:
    """Alpha#2: (-1 * correlation(rank(delta(log(volume),2)),rank(((close-open)/open)),6))"""
    x = c.rank(c.delta(c.log(c.volume), 2))
    y = c.rank((c.close - c.open) / c.open)
    return -1 * c.correlation(x, y, 6)


@register(3, ("open", "volume"))
def alpha_003(c: AlphaContext) -> pd.DataFrame:
    """Alpha#3: (-1 * correlation(rank(open),rank(volume),10))"""
    return -1 * c.correlation(c.rank(c.open), c.rank(c.volume), 10)


@register(4, ("low",))
def alpha_004(c: AlphaContext) -> pd.DataFrame:
    """Alpha#4: (-1 * Ts_Rank(rank(low),9))"""
    return -1 * c.ts_rank(c.rank(c.low), 9)


@register(5, ("open", "close", "vwap"))
def alpha_005(c: AlphaContext) -> pd.DataFrame:
    """Alpha#5: (rank((open-(sum(vwap,10)/10)))*(-1*abs(rank((close-vwap)))))"""
    return c.rank(c.open - c.ts_sum(c.vwap, 10) / 10) * (-1 * c.abs_(c.rank(c.close - c.vwap)))


@register(7, ("close", "volume", "adv"))
def alpha_007(c: AlphaContext) -> pd.DataFrame:
    """Alpha#7: ((adv20<volume)?((-1*ts_rank(abs(delta(close,7)),60))*sign(delta(close,7))):(-1*1))"""
    adv20 = c.adv(20)
    cond = adv20 < c.volume
    true_branch = (-1 * c.ts_rank(c.abs_(c.delta(c.close, 7)), 60)) * c.sign(c.delta(c.close, 7))
    false_branch = pd.DataFrame(-1.0, index=c.close.index, columns=c.close.columns)
    return true_branch.where(cond, false_branch)


@register(8, ("open", "returns"))
def alpha_008(c: AlphaContext) -> pd.DataFrame:
    """Alpha#8: (-1*rank(((sum(open,5)*sum(returns,5))-delay((sum(open,5)*sum(returns,5)),10))))"""
    inner = c.ts_sum(c.open, 5) * c.ts_sum(c.returns, 5)
    return -1 * c.rank(inner - c.delay(inner, 10))


@register(9, ("close",))
def alpha_009(c: AlphaContext) -> pd.DataFrame:
    """Alpha#9: ((0<ts_min(delta(close,1),5))?delta(close,1):((ts_max(delta(close,1),5)<0)?delta(close,1):(-1*delta(close,1))))"""
    d = c.delta(c.close, 1)
    cond1 = 0 < c.ts_min(d, 5)
    cond2 = c.ts_max(d, 5) < 0
    inner = d.where(cond2, -1 * d)
    return d.where(cond1, inner)


@register(10, ("close",))
def alpha_010(c: AlphaContext) -> pd.DataFrame:
    """Alpha#10: rank(((0<ts_min(delta(close,1),4))?delta(close,1):((ts_max(delta(close,1),4)<0)?delta(close,1):(-1*delta(close,1)))))"""
    d = c.delta(c.close, 1)
    cond1 = 0 < c.ts_min(d, 4)
    cond2 = c.ts_max(d, 4) < 0
    inner = d.where(cond2, -1 * d)
    return c.rank(d.where(cond1, inner))


@register(11, ("close", "vwap", "volume"))
def alpha_011(c: AlphaContext) -> pd.DataFrame:
    """Alpha#11: ((rank(ts_max((vwap-close),3))+rank(ts_min((vwap-close),3)))*rank(delta(volume,3)))"""
    diff = c.vwap - c.close
    return (c.rank(c.ts_max(diff, 3)) + c.rank(c.ts_min(diff, 3))) * c.rank(c.delta(c.volume, 3))


@register(13, ("close", "volume"))
def alpha_013(c: AlphaContext) -> pd.DataFrame:
    """Alpha#13: (-1*rank(covariance(rank(close),rank(volume),5)))"""
    return -1 * c.rank(c.covariance(c.rank(c.close), c.rank(c.volume), 5))


@register(14, ("open", "volume", "returns"))
def alpha_014(c: AlphaContext) -> pd.DataFrame:
    """Alpha#14: ((-1*rank(delta(returns,3)))*correlation(open,volume,10))"""
    return (-1 * c.rank(c.delta(c.returns, 3))) * c.correlation(c.open, c.volume, 10)


@register(15, ("high", "volume"))
def alpha_015(c: AlphaContext) -> pd.DataFrame:
    """Alpha#15: (-1*sum(rank(correlation(rank(high),rank(volume),3)),3))"""
    return -1 * c.ts_sum(c.rank(c.correlation(c.rank(c.high), c.rank(c.volume), 3)), 3)


@register(16, ("high", "volume"))
def alpha_016(c: AlphaContext) -> pd.DataFrame:
    """Alpha#16: (-1*rank(covariance(rank(high),rank(volume),5)))"""
    return -1 * c.rank(c.covariance(c.rank(c.high), c.rank(c.volume), 5))


@register(17, ("close", "volume", "adv"))
def alpha_017(c: AlphaContext) -> pd.DataFrame:
    """Alpha#17: (((-1*rank(ts_rank(close,10)))*rank(delta(delta(close,1),1)))*rank(ts_rank((volume/adv20),5)))"""
    adv20 = c.adv(20)
    return (
        (-1 * c.rank(c.ts_rank(c.close, 10)))
        * c.rank(c.delta(c.delta(c.close, 1), 1))
        * c.rank(c.ts_rank(c.volume / adv20, 5))
    )


@register(18, ("close", "open"))
def alpha_018(c: AlphaContext) -> pd.DataFrame:
    """Alpha#18: (-1*rank(((stddev(abs((close-open)),5)+(close-open))+correlation(close,open,10))))"""
    diff = c.close - c.open
    return -1 * c.rank(c.stddev(c.abs_(diff), 5) + diff + c.correlation(c.close, c.open, 10))


@register(19, ("close", "returns"))
def alpha_019(c: AlphaContext) -> pd.DataFrame:
    """Alpha#19: ((-1*sign(((close-delay(close,7))+delta(close,7))))*(1+rank((1+sum(returns,250)))))"""
    combined = (c.close - c.delay(c.close, 7)) + c.delta(c.close, 7)
    return (-1 * c.sign(combined)) * (1 + c.rank(1 + c.ts_sum(c.returns, 250)))


@register(20, ("open", "high", "low", "close"))
def alpha_020(c: AlphaContext) -> pd.DataFrame:
    """Alpha#20: (((-1*rank((open-delay(high,1))))*rank((open-delay(close,1))))*rank((open-delay(low,1))))"""
    return (
        (-1 * c.rank(c.open - c.delay(c.high, 1)))
        * c.rank(c.open - c.delay(c.close, 1))
        * c.rank(c.open - c.delay(c.low, 1))
    )


@register(21, ("close", "volume", "adv"))
def alpha_021(c: AlphaContext) -> pd.DataFrame:
    """Alpha#21: (((sum(close,8)/8)+stddev(close,8))<(sum(close,2)/2))?(-1*1):(((sum(close,2)/2)<((sum(close,8)/8)-stddev(close,8)))?1:(((1<(volume/adv20))||((volume/adv20)==1))?1:(-1*1)))"""
    adv20 = c.adv(20)
    mean8 = c.ts_sum(c.close, 8) / 8
    std8 = c.stddev(c.close, 8)
    mean2 = c.ts_sum(c.close, 2) / 2
    vol_ratio = c.volume / adv20
    cond1 = (mean8 + std8) < mean2
    cond2 = mean2 < (mean8 - std8)
    cond3 = (1 < vol_ratio) | (vol_ratio == 1)
    ones = pd.DataFrame(1.0, index=c.close.index, columns=c.close.columns)
    neg_ones = pd.DataFrame(-1.0, index=c.close.index, columns=c.close.columns)
    inner2 = ones.where(cond3, neg_ones)
    inner1 = ones.where(cond2, inner2)
    return neg_ones.where(cond1, inner1)


@register(22, ("high", "close", "volume"))
def alpha_022(c: AlphaContext) -> pd.DataFrame:
    """Alpha#22: (-1*(delta(correlation(high,volume,5),5)*rank(stddev(close,20))))"""
    return -1 * (c.delta(c.correlation(c.high, c.volume, 5), 5) * c.rank(c.stddev(c.close, 20)))


@register(23, ("high", "close"))
def alpha_023(c: AlphaContext) -> pd.DataFrame:
    """Alpha#23: (((sum(high,20)/20)<high)?(-1*delta(high,2)):0)"""
    cond = (c.ts_sum(c.high, 20) / 20) < c.high
    true_branch = -1 * c.delta(c.high, 2)
    zeros = pd.DataFrame(0.0, index=c.close.index, columns=c.close.columns)
    return true_branch.where(cond, zeros)


@register(24, ("close",))
def alpha_024(c: AlphaContext) -> pd.DataFrame:
    """Alpha#24: ((((delta((sum(close,100)/100),100)/delay(close,100))<0.05)||((delta((sum(close,100)/100),100)/delay(close,100))==0.05))?(-1*(close-ts_min(close,100))):(-1*delta(close,3)))"""
    ratio = c.delta(c.ts_sum(c.close, 100) / 100, 100) / c.delay(c.close, 100)
    cond = (ratio < 0.05) | (ratio == 0.05)
    true_branch = -1 * (c.close - c.ts_min(c.close, 100))
    false_branch = -1 * c.delta(c.close, 3)
    return true_branch.where(cond, false_branch)


@register(25, ("returns", "close", "high", "vwap", "adv"))
def alpha_025(c: AlphaContext) -> pd.DataFrame:
    """Alpha#25: rank(((((-1*returns)*adv20)*vwap)*(high-close)))"""
    adv20 = c.adv(20)
    return c.rank((-1 * c.returns) * adv20 * c.vwap * (c.high - c.close))


# === Alphas 26–50 (Task 9) ===


@register(26, ("volume", "high"))
def alpha_026(c: AlphaContext) -> pd.DataFrame:
    """Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))"""
    return -1 * c.ts_max(c.correlation(c.ts_rank(c.volume, 5), c.ts_rank(c.high, 5), 5), 3)


@register(27, ("volume", "vwap"))
def alpha_027(c: AlphaContext) -> pd.DataFrame:
    """Alpha#27: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)"""
    inner = c.ts_sum(c.correlation(c.rank(c.volume), c.rank(c.vwap), 6), 2) / 2.0
    cond = 0.5 < c.rank(inner)
    true_branch = pd.DataFrame(-1.0, index=c.vwap.index, columns=c.vwap.columns)
    false_branch = pd.DataFrame(1.0, index=c.vwap.index, columns=c.vwap.columns)
    return true_branch.where(cond, false_branch)


@register(28, ("high", "low", "close", "adv"))
def alpha_028(c: AlphaContext) -> pd.DataFrame:
    """Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))"""
    adv20 = c.adv(20)
    return c.scale(c.correlation(adv20, c.low, 5) + (c.high + c.low) / 2 - c.close)


@register(29, ("close", "returns"))
def alpha_029(c: AlphaContext) -> pd.DataFrame:
    """Alpha#29: (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))"""
    innermost = -1 * c.rank(c.delta(c.close - 1, 5))
    step1 = c.ts_min(c.rank(c.rank(innermost)), 2)
    step2 = c.rank(c.rank(c.scale(c.log(c.ts_sum(step1, 1)))))
    part1 = c.ts_min(c.product(step2, 1), 5)
    part2 = c.ts_rank(c.delay(-1 * c.returns, 6), 5)
    return part1 + part2


@register(30, ("close", "volume"))
def alpha_030(c: AlphaContext) -> pd.DataFrame:
    """Alpha#30: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))"""
    s1 = c.sign(c.close - c.delay(c.close, 1))
    s2 = c.sign(c.delay(c.close, 1) - c.delay(c.close, 2))
    s3 = c.sign(c.delay(c.close, 2) - c.delay(c.close, 3))
    return ((1.0 - c.rank(s1 + s2 + s3)) * c.ts_sum(c.volume, 5)) / c.ts_sum(c.volume, 20)


@register(31, ("close", "low", "adv"))
def alpha_031(c: AlphaContext) -> pd.DataFrame:
    """Alpha#31: ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))"""
    adv20 = c.adv(20)
    part1 = c.rank(c.rank(c.rank(c.decay_linear(-1 * c.rank(c.rank(c.delta(c.close, 10))), 10))))
    part2 = c.rank(-1 * c.delta(c.close, 3))
    part3 = c.sign(c.scale(c.correlation(adv20, c.low, 12)))
    return part1 + part2 + part3


@register(32, ("close", "vwap"))
def alpha_032(c: AlphaContext) -> pd.DataFrame:
    """Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))"""
    part1 = c.scale(c.ts_sum(c.close, 7) / 7 - c.close)
    part2 = 20 * c.scale(c.correlation(c.vwap, c.delay(c.close, 5), 230))
    return part1 + part2


@register(33, ("open", "close"))
def alpha_033(c: AlphaContext) -> pd.DataFrame:
    """Alpha#33: rank((-1 * ((1 - (open / close))^1)))"""
    return c.rank(-1 * c.signedpower(1 - (c.open / c.close), 1))


@register(34, ("returns", "close"))
def alpha_034(c: AlphaContext) -> pd.DataFrame:
    """Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))"""
    return c.rank((1 - c.rank(c.stddev(c.returns, 2) / c.stddev(c.returns, 5))) + (1 - c.rank(c.delta(c.close, 1))))


@register(35, ("volume", "close", "high", "low", "returns"))
def alpha_035(c: AlphaContext) -> pd.DataFrame:
    """Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))"""
    return (
        c.ts_rank(c.volume, 32)
        * (1 - c.ts_rank((c.close + c.high) - c.low, 16))
        * (1 - c.ts_rank(c.returns, 32))
    )


@register(36, ("close", "open", "volume", "returns", "vwap", "adv"))
def alpha_036(c: AlphaContext) -> pd.DataFrame:
    """Alpha#36: (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))"""
    adv20 = c.adv(20)
    p1 = 2.21 * c.rank(c.correlation(c.close - c.open, c.delay(c.volume, 1), 15))
    p2 = 0.7 * c.rank(c.open - c.close)
    p3 = 0.73 * c.rank(c.ts_rank(c.delay(-1 * c.returns, 6), 5))
    p4 = c.rank(c.abs_(c.correlation(c.vwap, adv20, 6)))
    p5 = 0.6 * c.rank((c.ts_sum(c.close, 200) / 200 - c.open) * (c.close - c.open))
    return p1 + p2 + p3 + p4 + p5


@register(37, ("open", "close"))
def alpha_037(c: AlphaContext) -> pd.DataFrame:
    """Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))"""
    return c.rank(c.correlation(c.delay(c.open - c.close, 1), c.close, 200)) + c.rank(c.open - c.close)


@register(38, ("close", "open"))
def alpha_038(c: AlphaContext) -> pd.DataFrame:
    """Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))"""
    return (-1 * c.rank(c.ts_rank(c.close, 10))) * c.rank(c.close / c.open)


@register(39, ("close", "volume", "returns", "adv"))
def alpha_039(c: AlphaContext) -> pd.DataFrame:
    """Alpha#39: ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))"""
    adv20 = c.adv(20)
    return (
        (-1 * c.rank(c.delta(c.close, 7) * (1 - c.rank(c.decay_linear(c.volume / adv20, 9)))))
        * (1 + c.rank(c.ts_sum(c.returns, 250)))
    )


@register(40, ("high", "volume"))
def alpha_040(c: AlphaContext) -> pd.DataFrame:
    """Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))"""
    return (-1 * c.rank(c.stddev(c.high, 10))) * c.correlation(c.high, c.volume, 10)


@register(41, ("high", "low", "vwap"))
def alpha_041(c: AlphaContext) -> pd.DataFrame:
    """Alpha#41: (((high * low)^0.5) - vwap)"""
    return c.signedpower(c.high * c.low, 0.5) - c.vwap


@register(42, ("vwap", "close"))
def alpha_042(c: AlphaContext) -> pd.DataFrame:
    """Alpha#42: (rank((vwap - close)) / rank((vwap + close)))"""
    return c.rank(c.vwap - c.close) / c.rank(c.vwap + c.close)


@register(43, ("volume", "close", "adv"))
def alpha_043(c: AlphaContext) -> pd.DataFrame:
    """Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))"""
    adv20 = c.adv(20)
    return c.ts_rank(c.volume / adv20, 20) * c.ts_rank(-1 * c.delta(c.close, 7), 8)


@register(44, ("high", "volume"))
def alpha_044(c: AlphaContext) -> pd.DataFrame:
    """Alpha#44: (-1 * correlation(high, rank(volume), 5))"""
    return -1 * c.correlation(c.high, c.rank(c.volume), 5)


@register(45, ("close", "volume"))
def alpha_045(c: AlphaContext) -> pd.DataFrame:
    """Alpha#45: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))"""
    p1 = c.rank(c.ts_sum(c.delay(c.close, 5), 20) / 20)
    p2 = c.correlation(c.close, c.volume, 2)
    p3 = c.rank(c.correlation(c.ts_sum(c.close, 5), c.ts_sum(c.close, 20), 2))
    return -1 * (p1 * p2 * p3)


@register(46, ("close",))
def alpha_046(c: AlphaContext) -> pd.DataFrame:
    """Alpha#46: ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))"""
    slope = (c.delay(c.close, 20) - c.delay(c.close, 10)) / 10 - (c.delay(c.close, 10) - c.close) / 10
    neg_ones = pd.DataFrame(-1.0, index=c.close.index, columns=c.close.columns)
    ones = pd.DataFrame(1.0, index=c.close.index, columns=c.close.columns)
    inner = (-1 * (c.close - c.delay(c.close, 1)))
    inner2 = ones.where(slope < 0, inner)
    return neg_ones.where(0.25 < slope, inner2)


@register(47, ("close", "volume", "high", "vwap", "adv"))
def alpha_047(c: AlphaContext) -> pd.DataFrame:
    """Alpha#47: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))"""
    adv20 = c.adv(20)
    part1 = (c.rank(1 / c.close) * c.volume) / adv20
    part2 = (c.high * c.rank(c.high - c.close)) / (c.ts_sum(c.high, 5) / 5)
    return part1 * part2 - c.rank(c.vwap - c.delay(c.vwap, 5))


register_na(48, ("indclass", "close"))


@register(49, ("close",))
def alpha_049(c: AlphaContext) -> pd.DataFrame:
    """Alpha#49: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))"""
    slope = (c.delay(c.close, 20) - c.delay(c.close, 10)) / 10 - (c.delay(c.close, 10) - c.close) / 10
    ones = pd.DataFrame(1.0, index=c.close.index, columns=c.close.columns)
    false_branch = -1 * (c.close - c.delay(c.close, 1))
    return ones.where(slope < (-1 * 0.1), false_branch)


@register(50, ("volume", "vwap"))
def alpha_050(c: AlphaContext) -> pd.DataFrame:
    """Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))"""
    return -1 * c.ts_max(c.rank(c.correlation(c.rank(c.volume), c.rank(c.vwap), 5)), 5)


# === Alphas 51–75 (Task 10) ===


@register(51, ("close",))
def alpha_051(c: AlphaContext) -> pd.DataFrame:
    """Alpha#51: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))"""
    slope = (c.delay(c.close, 20) - c.delay(c.close, 10)) / 10 - (c.delay(c.close, 10) - c.close) / 10
    ones = pd.DataFrame(1.0, index=c.close.index, columns=c.close.columns)
    false_branch = -1 * (c.close - c.delay(c.close, 1))
    return ones.where(slope < (-1 * 0.05), false_branch)


@register(52, ("low", "volume", "returns"))
def alpha_052(c: AlphaContext) -> pd.DataFrame:
    """Alpha#52: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))"""
    low_min5 = c.ts_min(c.low, 5)
    part1 = (-1 * low_min5) + c.delay(low_min5, 5)
    part2 = c.rank((c.ts_sum(c.returns, 240) - c.ts_sum(c.returns, 20)) / 220)
    return part1 * part2 * c.ts_rank(c.volume, 5)


@register(54, ("low", "close", "open", "high"))
def alpha_054(c: AlphaContext) -> pd.DataFrame:
    """Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))"""
    return (-1 * (c.low - c.close) * c.signedpower(c.open, 5)) / ((c.low - c.high) * c.signedpower(c.close, 5))


@register(55, ("close", "low", "high", "volume"))
def alpha_055(c: AlphaContext) -> pd.DataFrame:
    """Alpha#55: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))"""
    numer = c.close - c.ts_min(c.low, 12)
    denom = c.ts_max(c.high, 12) - c.ts_min(c.low, 12)
    return -1 * c.correlation(c.rank(numer / denom), c.rank(c.volume), 6)


register_na(56, ("returns", "cap"))


@register(57, ("close", "vwap"))
def alpha_057(c: AlphaContext) -> pd.DataFrame:
    """Alpha#57: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))"""
    return 0 - (1 * ((c.close - c.vwap) / c.decay_linear(c.rank(c.ts_argmax(c.close, 30)), 2)))


register_na(58, ("vwap", "volume", "indclass"))

register_na(59, ("vwap", "volume", "indclass"))


@register(60, ("close", "high", "low", "volume"))
def alpha_060(c: AlphaContext) -> pd.DataFrame:
    """Alpha#60: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))"""
    inner = ((c.close - c.low) - (c.high - c.close)) / (c.high - c.low) * c.volume
    return 0 - (1 * (2 * c.scale(c.rank(inner)) - c.scale(c.rank(c.ts_argmax(c.close, 10)))))


@register(61, ("vwap", "adv"))
def alpha_061(c: AlphaContext) -> pd.DataFrame:
    """Alpha#61: (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))"""
    adv180 = c.adv(180)
    lhs = c.rank(c.vwap - c.ts_min(c.vwap, int(16.1219)))
    rhs = c.rank(c.correlation(c.vwap, adv180, int(17.9282)))
    return (lhs < rhs).astype(float)


@register(62, ("vwap", "open", "high", "low", "adv"))
def alpha_062(c: AlphaContext) -> pd.DataFrame:
    """Alpha#62: ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)"""
    adv20 = c.adv(20)
    lhs = c.rank(c.correlation(c.vwap, c.ts_sum(adv20, int(22.4101)), int(9.91009)))
    rhs = c.rank(
        (c.rank(c.open) + c.rank(c.open)) < (c.rank((c.high + c.low) / 2) + c.rank(c.high))
    )
    return ((lhs < rhs).astype(float)) * -1


register_na(63, ("close", "open", "vwap", "adv", "indclass"))


@register(64, ("open", "low", "high", "vwap", "adv"))
def alpha_064(c: AlphaContext) -> pd.DataFrame:
    """Alpha#64: ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)"""
    adv120 = c.adv(120)
    blend_ol = c.open * 0.178404 + c.low * (1 - 0.178404)
    blend_hv = (c.high + c.low) / 2 * 0.178404 + c.vwap * (1 - 0.178404)
    lhs = c.rank(c.correlation(c.ts_sum(blend_ol, int(12.7054)), c.ts_sum(adv120, int(12.7054)), int(16.6208)))
    rhs = c.rank(c.delta(blend_hv, int(3.69741)))
    return ((lhs < rhs).astype(float)) * -1


@register(65, ("open", "vwap", "adv"))
def alpha_065(c: AlphaContext) -> pd.DataFrame:
    """Alpha#65: ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)"""
    adv60 = c.adv(60)
    blend = c.open * 0.00817205 + c.vwap * (1 - 0.00817205)
    lhs = c.rank(c.correlation(blend, c.ts_sum(adv60, int(8.6911)), int(6.40374)))
    rhs = c.rank(c.open - c.ts_min(c.open, int(13.635)))
    return ((lhs < rhs).astype(float)) * -1


@register(66, ("low", "vwap", "open", "high"))
def alpha_066(c: AlphaContext) -> pd.DataFrame:
    """Alpha#66: ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)"""
    part1 = c.rank(c.decay_linear(c.delta(c.vwap, int(3.51013)), int(7.23052)))
    # (low * 0.96633) + (low * (1 - 0.96633)) simplifies to low
    inner2 = (c.low - c.vwap) / (c.open - (c.high + c.low) / 2)
    part2 = c.ts_rank(c.decay_linear(inner2, int(11.4157)), int(6.72611))
    return (part1 + part2) * -1


register_na(67, ("high", "vwap", "adv", "indclass"))


@register(68, ("high", "close", "low", "adv"))
def alpha_068(c: AlphaContext) -> pd.DataFrame:
    """Alpha#68: ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)"""
    adv15 = c.adv(15)
    lhs = c.ts_rank(c.correlation(c.rank(c.high), c.rank(adv15), int(8.91644)), int(13.9333))
    blend = c.close * 0.518371 + c.low * (1 - 0.518371)
    rhs = c.rank(c.delta(blend, int(1.06157)))
    return ((lhs < rhs).astype(float)) * -1


register_na(69, ("vwap", "close", "adv", "indclass"))

register_na(70, ("vwap", "close", "adv", "indclass"))


@register(71, ("close", "low", "open", "vwap", "adv"))
def alpha_071(c: AlphaContext) -> pd.DataFrame:
    """Alpha#71: max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))"""
    adv180 = c.adv(180)
    part1 = c.ts_rank(
        c.decay_linear(
            c.correlation(c.ts_rank(c.close, int(3.43976)), c.ts_rank(adv180, int(12.0647)), int(18.0175)),
            int(4.20501)
        ),
        int(15.6948)
    )
    part2 = c.ts_rank(
        c.decay_linear(
            c.signedpower(c.rank((c.low + c.open) - (c.vwap + c.vwap)), 2),
            int(16.4662)
        ),
        int(4.4388)
    )
    return part1.where(part1 >= part2, part2)


@register(72, ("high", "low", "vwap", "volume", "adv"))
def alpha_072(c: AlphaContext) -> pd.DataFrame:
    """Alpha#72: (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))"""
    adv40 = c.adv(40)
    numer = c.rank(c.decay_linear(c.correlation((c.high + c.low) / 2, adv40, int(8.93345)), int(10.1519)))
    denom = c.rank(c.decay_linear(c.correlation(c.ts_rank(c.vwap, int(3.72469)), c.ts_rank(c.volume, int(18.5188)), int(6.86671)), int(2.95011)))
    return numer / denom


@register(73, ("open", "low", "vwap"))
def alpha_073(c: AlphaContext) -> pd.DataFrame:
    """Alpha#73: (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)"""
    blend = c.open * 0.147155 + c.low * (1 - 0.147155)
    part1 = c.rank(c.decay_linear(c.delta(c.vwap, int(4.72775)), int(2.91864)))
    part2 = c.ts_rank(
        c.decay_linear((c.delta(blend, int(2.03608)) / blend) * -1, int(3.33829)),
        int(16.7411)
    )
    return part1.where(part1 >= part2, part2) * -1


@register(74, ("close", "high", "vwap", "volume", "adv"))
def alpha_074(c: AlphaContext) -> pd.DataFrame:
    """Alpha#74: ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)"""
    adv30 = c.adv(30)
    lhs = c.rank(c.correlation(c.close, c.ts_sum(adv30, int(37.4843)), int(15.1365)))
    blend = c.high * 0.0261661 + c.vwap * (1 - 0.0261661)
    rhs = c.rank(c.correlation(c.rank(blend), c.rank(c.volume), int(11.4791)))
    return ((lhs < rhs).astype(float)) * -1


@register(75, ("vwap", "volume", "low", "adv"))
def alpha_075(c: AlphaContext) -> pd.DataFrame:
    """Alpha#75: (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))"""
    adv50 = c.adv(50)
    lhs = c.rank(c.correlation(c.vwap, c.volume, int(4.24304)))
    rhs = c.rank(c.correlation(c.rank(c.low), c.rank(adv50), int(12.4413)))
    return (lhs < rhs).astype(float)


# === Alphas 76–100 (Task 11) ===

# Tier-C stubs (IndNeutralize / IndClass — not computable without industry data)
register_na(76, ("vwap", "low", "adv", "indclass"))     # IndNeutralize(low, IndClass.sector)
register_na(79, ("vwap", "close", "open", "adv", "indclass"))  # IndNeutralize(…, IndClass.sector)
register_na(80, ("open", "high", "adv", "indclass"))     # IndNeutralize(…, IndClass.industry)
register_na(82, ("open", "volume", "indclass"))          # IndNeutralize(volume, IndClass.sector)
register_na(87, ("close", "vwap", "adv", "indclass"))    # IndNeutralize(adv81, IndClass.industry)
register_na(89, ("low", "vwap", "adv", "indclass"))      # IndNeutralize(vwap, IndClass.industry)
register_na(90, ("close", "low", "adv", "indclass"))     # IndNeutralize(adv40, IndClass.subindustry)
register_na(91, ("close", "volume", "vwap", "adv", "indclass"))  # IndNeutralize(close, IndClass.industry)
register_na(93, ("close", "vwap", "adv", "indclass"))    # IndNeutralize(vwap, IndClass.industry)
register_na(97, ("low", "vwap", "adv", "indclass"))      # IndNeutralize(…, IndClass.industry)
register_na(100, ("close", "high", "low", "volume", "adv", "indclass"))  # double IndNeutralize(…, IndClass.subindustry)


@register(77, ("high", "low", "vwap", "adv"))
def alpha_077(c: AlphaContext) -> pd.DataFrame:
    """Alpha#77: min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))"""
    adv40 = c.adv(40)
    inner = ((c.high + c.low) / 2 + c.high) - (c.vwap + c.high)  # simplifies to (high+low)/2 - vwap
    part1 = c.rank(c.decay_linear(inner, int(20.0451)))
    part2 = c.rank(c.decay_linear(c.correlation((c.high + c.low) / 2, adv40, int(3.1614)), int(5.64125)))
    return part1.where(part1 <= part2, part2)


@register(78, ("low", "vwap", "volume", "adv"))
def alpha_078(c: AlphaContext) -> pd.DataFrame:
    """Alpha#78: (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))"""
    adv40 = c.adv(40)
    blend = c.low * 0.352233 + c.vwap * (1 - 0.352233)
    base = c.rank(c.correlation(c.ts_sum(blend, int(19.7428)), c.ts_sum(adv40, int(19.7428)), int(6.83313)))
    exp = c.rank(c.correlation(c.rank(c.vwap), c.rank(c.volume), int(5.77492)))
    # rank values are in [0,1] (non-negative), so signedpower reduces to base**exp
    return base ** exp


@register(81, ("vwap", "volume", "adv"))
def alpha_081(c: AlphaContext) -> pd.DataFrame:
    """Alpha#81: ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)"""
    adv10 = c.adv(10)
    inner_corr = c.rank(c.correlation(c.vwap, c.ts_sum(adv10, int(49.6054)), int(8.47743)))
    inner_pow = c.signedpower(inner_corr, 4)
    lhs = c.rank(c.log(c.product(c.rank(inner_pow), int(14.9655))))
    rhs = c.rank(c.correlation(c.rank(c.vwap), c.rank(c.volume), int(5.07914)))
    return ((lhs < rhs).astype(float)) * -1


@register(83, ("high", "low", "close", "volume", "vwap"))
def alpha_083(c: AlphaContext) -> pd.DataFrame:
    """Alpha#83: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))"""
    hl_ratio = (c.high - c.low) / (c.ts_sum(c.close, 5) / 5)
    numer = c.rank(c.delay(hl_ratio, 2)) * c.rank(c.rank(c.volume))
    denom = hl_ratio / (c.vwap - c.close)
    return numer / denom


@register(84, ("vwap", "close"))
def alpha_084(c: AlphaContext) -> pd.DataFrame:
    """Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))"""
    base = c.ts_rank(c.vwap - c.ts_max(c.vwap, int(15.3217)), int(20.7127))
    exp = c.delta(c.close, int(4.96796))
    # SignedPower(x, a) with DataFrame a: sign(x) * |x|^a
    return np.sign(base) * (base.abs() ** exp)


@register(85, ("high", "close", "low", "volume", "adv"))
def alpha_085(c: AlphaContext) -> pd.DataFrame:
    """Alpha#85: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))"""
    adv30 = c.adv(30)
    blend = c.high * 0.876703 + c.close * (1 - 0.876703)
    base = c.rank(c.correlation(blend, adv30, int(9.61331)))
    exp = c.rank(c.correlation(c.ts_rank((c.high + c.low) / 2, int(3.70596)), c.ts_rank(c.volume, int(10.1595)), int(7.11408)))
    # rank values in [0,1] (non-negative); ^ operator from paper = element-wise power
    return base ** exp


@register(86, ("close", "vwap", "open", "adv"))
def alpha_086(c: AlphaContext) -> pd.DataFrame:
    """Alpha#86: ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)"""
    adv20 = c.adv(20)
    lhs = c.ts_rank(c.correlation(c.close, c.ts_sum(adv20, int(14.7444)), int(6.00049)), int(20.4195))
    rhs = c.rank((c.open + c.close) - (c.vwap + c.open))  # simplifies to rank(close - vwap)
    return ((lhs < rhs).astype(float)) * -1


@register(88, ("open", "low", "high", "close", "volume", "adv"))
def alpha_088(c: AlphaContext) -> pd.DataFrame:
    """Alpha#88: min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))"""
    adv60 = c.adv(60)
    inner = (c.rank(c.open) + c.rank(c.low)) - (c.rank(c.high) + c.rank(c.close))
    part1 = c.rank(c.decay_linear(inner, int(8.06882)))
    part2 = c.ts_rank(
        c.decay_linear(
            c.correlation(c.ts_rank(c.close, int(8.44728)), c.ts_rank(adv60, int(20.6966)), int(8.01266)),
            int(6.65053)
        ),
        int(2.61957)
    )
    return part1.where(part1 <= part2, part2)


@register(92, ("high", "low", "close", "open", "volume", "adv"))
def alpha_092(c: AlphaContext) -> pd.DataFrame:
    """Alpha#92: min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))"""
    adv30 = c.adv(30)
    cond = ((c.high + c.low) / 2 + c.close) < (c.low + c.open)
    inner1 = cond.astype(float)
    part1 = c.ts_rank(c.decay_linear(inner1, int(14.7221)), int(18.8683))
    part2 = c.ts_rank(
        c.decay_linear(c.correlation(c.rank(c.low), c.rank(adv30), int(7.58555)), int(6.94024)),
        int(6.80584)
    )
    return part1.where(part1 <= part2, part2)


@register(94, ("vwap", "adv"))
def alpha_094(c: AlphaContext) -> pd.DataFrame:
    """Alpha#94: ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)"""
    adv60 = c.adv(60)
    base = c.rank(c.vwap - c.ts_min(c.vwap, int(11.5783)))
    exp = c.ts_rank(
        c.correlation(c.ts_rank(c.vwap, int(19.6462)), c.ts_rank(adv60, int(4.02992)), int(18.0926)),
        int(2.70756)
    )
    # rank values in [0,1]; ^ operator = element-wise power
    return (base ** exp) * -1


@register(95, ("open", "high", "low", "adv"))
def alpha_095(c: AlphaContext) -> pd.DataFrame:
    """Alpha#95: (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))"""
    adv40 = c.adv(40)
    lhs = c.rank(c.open - c.ts_min(c.open, int(12.4105)))
    inner_corr = c.rank(c.correlation(
        c.ts_sum((c.high + c.low) / 2, int(19.1351)),
        c.ts_sum(adv40, int(19.1351)),
        int(12.8742)
    ))
    rhs = c.ts_rank(c.signedpower(inner_corr, 5), int(11.7584))
    return (lhs < rhs).astype(float)


@register(96, ("vwap", "volume", "close", "adv"))
def alpha_096(c: AlphaContext) -> pd.DataFrame:
    """Alpha#96: (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)"""
    adv60 = c.adv(60)
    part1 = c.ts_rank(
        c.decay_linear(c.correlation(c.rank(c.vwap), c.rank(c.volume), int(3.83878)), int(4.16783)),
        int(8.38151)
    )
    inner_corr = c.correlation(c.ts_rank(c.close, int(7.45404)), c.ts_rank(adv60, int(4.13242)), int(3.65459))
    part2 = c.ts_rank(
        c.decay_linear(c.ts_argmax(inner_corr, int(12.6556)), int(14.0365)),
        int(13.4143)
    )
    return part1.where(part1 >= part2, part2) * -1


@register(98, ("vwap", "open", "adv"))
def alpha_098(c: AlphaContext) -> pd.DataFrame:
    """Alpha#98: (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))"""
    adv5 = c.adv(5)
    adv15 = c.adv(15)
    part1 = c.rank(c.decay_linear(c.correlation(c.vwap, c.ts_sum(adv5, int(26.4719)), int(4.58418)), int(7.18088)))
    inner = c.ts_rank(
        c.ts_argmin(c.correlation(c.rank(c.open), c.rank(adv15), int(20.8187)), int(8.62571)),
        int(6.95668)
    )
    part2 = c.rank(c.decay_linear(inner, int(8.07206)))
    return part1 - part2


@register(99, ("high", "low", "volume", "adv"))
def alpha_099(c: AlphaContext) -> pd.DataFrame:
    """Alpha#99: ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)"""
    adv60 = c.adv(60)
    lhs = c.rank(c.correlation(
        c.ts_sum((c.high + c.low) / 2, int(19.8975)),
        c.ts_sum(adv60, int(19.8975)),
        int(8.8136)
    ))
    rhs = c.rank(c.correlation(c.low, c.volume, int(6.28259)))
    return ((lhs < rhs).astype(float)) * -1
