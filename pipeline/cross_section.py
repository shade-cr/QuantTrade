"""B0006 cross-sectional feature pack v1.

Binding spec: docs/superpowers/specs/2026-07-03-b0006-cross-sectional-features.md.
6 stock-level percentile-rank features + 2 panel scalars, computed point-in-time
within the frozen universe from OHLCV only. Residualization is vs the equal-weight
basket (~first PC); NEVER sector one-hots (collider guard, LdP-Zoonekynd 2024).
All ranks are cross-sectional percentiles in [1/N, 1] via rank(pct=True) at each
date — stationary by construction on a 35-wide panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _panel_frame(panel: dict[str, pd.DataFrame], col: str,
                 tickers: list[str]) -> pd.DataFrame:
    """Union-index frame of one column across tickers (columns = tickers)."""
    return pd.DataFrame({t: panel[t][col] for t in tickers})


def build_cross_sectional_features(
    panel: dict[str, pd.DataFrame], tickers: list[str],
) -> dict[str, pd.DataFrame]:
    close = _panel_frame(panel, "close", tickers).sort_index()
    volume = _panel_frame(panel, "volume", tickers).sort_index()
    logc = np.log(close)
    r1 = logc.diff()

    # Equal-weight basket return (~first PC of a 35-wide large-cap panel).
    basket_r = r1.mean(axis=1)
    # Market-residual daily return: r_i - beta-free demeaning (v1: simple excess
    # vs basket; a rolling-beta residual is the #6 feature's job, not needed here).
    resid_r = r1.sub(basket_r, axis=0)

    # 1. 12-1 momentum, market-residualized: sum of residual log-returns t-252..t-21.
    mom_12_1 = resid_r.rolling(252).sum().shift(21)
    # 2. 52-week-high proximity: close / rolling 252d max (no residualization).
    prox_52wk = close / close.rolling(252).max()
    # 3. Short-term residual reversal: 21d residual return (sign kept raw; the
    #    meta learns the direction — do not pre-negate).
    st_rev = resid_r.rolling(21).sum()
    # 4. Idiosyncratic vol: 63d std of residual returns.
    idio_vol = resid_r.rolling(63).std()
    # 5. Turnover: 21d dollar volume vs own trailing 252d median dollar volume.
    dollar_vol = close * volume
    turnover = dollar_vol.rolling(21).mean() / dollar_vol.rolling(252).median()
    # 6. Rolling 63d beta to the equal-weight basket.
    cov = r1.rolling(63).cov(basket_r)
    beta = cov.div(basket_r.rolling(63).var(), axis=0)

    def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
        return df.rank(axis=1, pct=True)

    ranks = {
        "cs_mom_12_1_rank": cs_rank(mom_12_1),
        "cs_52wk_high_rank": cs_rank(prox_52wk),
        "cs_st_reversal_rank": cs_rank(st_rev),
        "cs_idio_vol_rank": cs_rank(idio_vol),
        "cs_turnover_rank": cs_rank(turnover),
        "cs_basket_beta_rank": cs_rank(beta),
    }
    # Panel scalars (same value for every name at t).
    dispersion = r1.rolling(21).sum().std(axis=1)
    breadth = (close > close.rolling(200).mean()).mean(axis=1)

    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        f = pd.DataFrame(index=panel[t].index)
        for name, frame in ranks.items():
            f[name] = frame[t].reindex(panel[t].index)
        f["cs_dispersion"] = dispersion.reindex(panel[t].index)
        f["cs_breadth_200"] = breadth.reindex(panel[t].index)
        out[t] = f
    return out
