"""Survival book — conservative, no-alpha, inverse-vol risk-parity, vol-targeted
multi-asset portfolio.

This is RISK ENGINEERING, not alpha. It does NOT predict direction; it minimizes
ruin and stabilizes a short-horizon Sharpe by VARIANCE CONTROL. The design is
literature-grounded:

  * Naive risk parity (inverse-vol) over mean-variance optimization, because
    estimation error makes MVO worse than 1/N out-of-sample (DeMiguel, Garlappi
    & Uppal 2009). We therefore deliberately AVOID min-var / max-Sharpe weights.
  * Inverse-vol rather than equal-NOTIONAL so that high-vol crypto cannot
    dominate the book by size alone.
  * Whole-book vol targeting to a LOW target (default 3% ann. — the book's
    HONEST realized vol; the prior 6% label was decorative because the 0.5
    leverage cap bound ~100% of rebalances, B0108 mandate 2026-05-31) with a hard
    leverage cap (default <= 0.5). For a no-alpha book leverage only scales
    variance and ruin probability, so it is capped tightly.
  * A combined-crypto variance-budget cap (default ~30%), treating the 3 crypto
    as far fewer than 3 independent bets.

Causality / no-look-ahead (CLAUDE.md invariant)
-----------------------------------------------
Every volatility / covariance / trend input used to decide the weights for the
holding period starting at date `t` is computed from returns strictly BEFORE `t`
(`<= t-1`). `realized_vol` shifts by one bar before rolling. The backtest applies
weights decided on `t-1`'s information to `t`'s realized return. There is no
walk-forward purge/embargo here because, unlike the meta-labeling backtest, a
continuously-held portfolio has no overlapping triple-barrier outcome windows to
leak across — causality is enforced purely by the bar-close shift.

Sharpe annualization distinction (CLAUDE.md)
--------------------------------------------
pipeline.metrics.strategy_metrics annualizes SPARSE per-trade pnl with
sqrt(trades_per_year) (metrics.py lines 47-54, 90-91). That convention is for
event-driven meta-labeling where pnl is non-zero only on filtered events. THIS
book is a CONTINUOUSLY-HELD portfolio producing a daily return every bar, so the
correct annualization is sqrt(252) on the daily portfolio-return series. The two
are not interchangeable; the backtest script documents which it uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# --------------------------------------------------------------------------- #
# PERMANENT covariance-exclusion set (FINAL HARDENING, 2026-06-17)
# --------------------------------------------------------------------------- #
# Symbols that must NEVER enter the survival book's covariance estimate, REGARDLESS
# of how much history they accrue. This is stronger than the cov-ADMISSION DATA
# threshold (>= cov_window bars) and the 0.00 per-asset weight cap: even if such a
# symbol crossed the bar-count threshold AND a cap were (accidentally) loosened, it
# is structurally barred from the cov matrix, so it can never earn an inverse-vol
# weight, never distort the vol-target scale, and never inject a spurious
# correlation into the matrix that would mis-weight the deep book.
#
# BARUSD (FC-Barcelona fan token) is the canonical member: illiquid, extreme-vol,
# severe market-impact/fill risk -> any correlation estimated against it is
# spurious. The CEO-delegated advisor (conf 0.9) ruled the "8 effectively traded"
# guarantee must be STRUCTURAL; this set is the cov-side half of that lock (the
# per-asset 0.00 weight cap in scripts.run_survival_live is the weight-side half).
# It is a module-level constant (not buried in a config) so it is explicit,
# greppable, and impossible to disable by tuning a numeric threshold.
COV_PERMANENT_EXCLUDE: frozenset[str] = frozenset({"BARUSD"})


# --------------------------------------------------------------------------- #
# Persistent risk state (FIX 1 / FIX 3) — latches survive across bars
# --------------------------------------------------------------------------- #
@dataclass
class RiskState:
    """Mutable, persistent risk-control state threaded through the backtest /
    live loop. The max-drawdown circuit breaker is a MANUAL-RESET LATCH: once it
    trips it stays armed (book flat) on every subsequent bar until an explicit
    manual reset is supplied. The daily-loss kill is intentionally NOT latched
    here — it models an intraday budget and may auto-reset the next bar.

    `discipline_breach_since` (B0189) is a PURE DIAGNOSTIC, OBSERVABILITY-ONLY
    field: the epoch-seconds timestamp at which the CURRENT §13 RiskDiscipline
    concentration / net-directional breach began (single-instrument share > 0.90
    OR net-directional share > 0.95), or None when not breaching. The live loop
    SETs it on the first breaching cycle, KEEPs it stable while the breach
    persists (so the watchdog's >= 30-min "sustained" check can measure the real
    elapsed duration), and CLEARs it once both shares are back within limits. It
    NEVER feeds a trading/sizing/throttle/kill-switch decision — it exists solely
    so the alert-only DISCIPLINE_CONCENTRATION watchdog rule can fire.
    """
    dd_latched: bool = False
    discipline_breach_since: float | None = None


@dataclass
class SleeveState:
    """Persistent state for the ISOLATED lottery sleeve's hard envelope cap
    (FIX 3). `envelope` is the sleeve's allocated risk budget (gross fraction of
    equity). The cap tracks the sleeve's cumulative P&L and its HIGH-WATER MARK;
    once the DRAWDOWN from that high-water exceeds a fraction of the envelope, the
    sleeve LATCHES flat permanently. A high-water drawdown cap (vs a raw
    cumulative-loss cap) genuinely BOUNDS the sleeve's contribution to the
    combined (book+sleeve) drawdown even when the sleeve rallied first and then
    bled back.
    """
    envelope: float
    cum_pnl: float = 0.0
    hwm_pnl: float = 0.0
    latched: bool = False


# --------------------------------------------------------------------------- #
# Volatility
# --------------------------------------------------------------------------- #
def realized_vol(
    returns: pd.Series,
    window: int,
    *,
    annualize: bool = True,
) -> pd.Series:
    """Trailing realized volatility, CAUSAL.

    sigma[t] is the std (ddof=1) of the `window` returns ending at t-1 — the
    return realized ON bar t is NOT included, so the estimate available at the
    open of bar t (when we set weights) never peeks at bar t's outcome. NaN
    until `window` causal observations exist.

    Annualized by sqrt(252) when `annualize` (returns are daily).
    """
    shifted = returns.shift(1)
    sig = shifted.rolling(window, min_periods=window).std(ddof=1)
    if annualize:
        sig = sig * np.sqrt(TRADING_DAYS)
    return sig


# --------------------------------------------------------------------------- #
# HAR volatility forecast (B0101) — direction is unforecastable for us, but
# volatility is (Corsi 2009). A better next-period vol estimate tightens the
# vol-target denominator -> lower realized vol -> higher Sharpe. Pure + causal.
# --------------------------------------------------------------------------- #
def daily_realized_variance(intraday_close: pd.Series) -> pd.Series:
    """Daily realized variance from intraday closes: the sum of squared intraday
    log-returns within each UTC day. Causal building block for the HAR forecast —
    feed CLOSED intraday bars only. Drops non-positive (empty) days.
    """
    c = pd.Series(intraday_close).dropna()
    r = np.log(c / c.shift(1)).dropna()
    if r.empty:
        return pd.Series(dtype="float64")
    idx = r.index
    day = idx.tz_convert("UTC").normalize() if getattr(idx, "tz", None) is not None \
        else pd.DatetimeIndex(idx).normalize()
    rv = (r ** 2).groupby(day).sum()
    return rv[rv > 0]


def har_vol_forecast(
    rv_daily: pd.Series, *, burn_in: int = 252, annualize: bool = True
) -> float:
    """One-step-ahead volatility forecast from a HAR(RV) model (Corsi 2009) fit on
    the GIVEN daily realized-variance history (causal — pass RV up to the last
    CLOSED day). Returns the ANNUALIZED vol forecast (sqrt of forecast daily
    variance, x sqrt(252)), or NaN with < `burn_in` usable observations.

    HAR: RV_t = c + b1 RV_{t-1} + b5 mean(RV_{t-5..t-1}) + b22 mean(RV_{t-22..t-1}).
    Fit by ordinary least squares on all available rows (deterministic, so the
    forecast is reproducible -> backtest/live parity-safe), then predict the next
    day from the most recent lags.
    """
    rv = pd.Series(rv_daily).dropna()
    rv = rv[rv > 0]
    if rv.shape[0] < burn_in:
        return float("nan")
    rv1 = rv.shift(1)
    rv5 = rv.shift(1).rolling(5).mean()
    rv22 = rv.shift(1).rolling(22).mean()
    X = pd.concat([rv1, rv5, rv22], axis=1).dropna()
    y = rv.reindex(X.index)
    if X.shape[0] < 30:
        return float("nan")
    A = np.column_stack([np.ones(len(X)), X.values])
    coef, *_ = np.linalg.lstsq(A, y.values, rcond=None)
    last1 = float(rv.iloc[-1])
    last5 = float(rv.iloc[-5:].mean())
    last22 = float(rv.iloc[-22:].mean())
    pred_var = coef[0] + coef[1] * last1 + coef[2] * last5 + coef[3] * last22
    pred_var = max(float(pred_var), 1e-12)
    vol = np.sqrt(pred_var)
    return float(vol * np.sqrt(TRADING_DAYS)) if annualize else float(vol)


def _rescale_cov_to_vols(cov: pd.DataFrame, vol_forecast: pd.Series) -> pd.DataFrame:
    """Return a copy of `cov` whose per-asset volatilities (the diagonal) are
    replaced by `vol_forecast` while the sample CORRELATIONS are kept:
    cov' = D cov D with D = diag(forecast_i / sample_i). Assets with no usable
    forecast are left untouched (ratio 1). This makes the vol-target scale and the
    variance shares react to the forecast without disturbing co-movement.
    """
    idx = list(cov.index)
    s_old = np.sqrt(np.clip(np.diag(cov.values), 1e-24, None))
    ratio = np.ones(len(idx))
    for i, a in enumerate(idx):
        f = vol_forecast.get(a, np.nan) if vol_forecast is not None else np.nan
        if np.isfinite(f) and f > 0 and s_old[i] > 0:
            ratio[i] = float(f) / float(s_old[i])
    out = cov.values * np.outer(ratio, ratio)
    return pd.DataFrame(out, index=idx, columns=idx)


# --------------------------------------------------------------------------- #
# Inverse-vol (naive risk parity) weights
# --------------------------------------------------------------------------- #
def inverse_vol_weights(sigmas: pd.Series) -> pd.Series:
    """Naive risk parity weights: w_i proportional to 1/sigma_i, summing to 1.

    Assets with a NaN or non-positive vol estimate (e.g. before they have
    history) receive zero weight and are excluded from the normalization.
    Returns an all-zero series if no asset has a usable estimate.
    """
    inv = pd.Series(0.0, index=sigmas.index, dtype="float64")
    usable = sigmas.notna() & (sigmas > 0)
    inv[usable] = 1.0 / sigmas[usable]
    total = inv.sum()
    if total <= 0:
        return pd.Series(0.0, index=sigmas.index, dtype="float64")
    return inv / total


# --------------------------------------------------------------------------- #
# Vol targeting
# --------------------------------------------------------------------------- #
def _portfolio_vol(weights: pd.Series, cov: pd.DataFrame) -> float:
    w = weights.reindex(cov.index).fillna(0.0).values
    var = float(w @ cov.values @ w)
    return float(np.sqrt(max(var, 0.0)))


def vol_target_scale(
    weights: pd.Series,
    cov: pd.DataFrame,
    target_vol: float,
    max_leverage: float,
    min_book_vol: float = 0.005,
) -> float:
    """Scalar k so that k * ex_ante_vol(weights) == target_vol, clamped to
    [0, max_leverage].

    For a no-alpha book leverage only buys variance, so k is hard-capped at
    `max_leverage` (default caller value <= 0.5). If the book is already too hot
    for the target, k < 1 scales it down.

    Vol-estimate floor (B0114) — the denominator is floored at `min_book_vol`
    (default 0.5% annualized) so that a collapsing ex-ante vol estimate cannot
    blow k up without bound: `k = target_vol / max(book_vol, min_book_vol)`. This
    is the lever-into-calm-then-spike failure mode of vol targeting; before the
    floor, k was bounded ONLY by `max_leverage`, so raising the cap directly
    widened the hole. The default floor is low enough that a normal-vol book is
    unaffected (the floor never binds at the production cap), so this is a safety
    rail, not a behavior change — it is the prerequisite for any future cap raise.
    """
    book_vol = _portfolio_vol(weights, cov)
    if book_vol <= 0:
        return 0.0
    k = target_vol / max(book_vol, min_book_vol)
    return float(min(k, max_leverage))


# --------------------------------------------------------------------------- #
# Crypto variance-budget cap
# --------------------------------------------------------------------------- #
def _variance_shares(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Marginal variance-contribution share per asset: w_i (Σw)_i / (w'Σw)."""
    w = weights.reindex(cov.index).fillna(0.0).values
    port_var = float(w @ cov.values @ w)
    if port_var <= 0:
        return pd.Series(0.0, index=cov.index)
    sigma_w = cov.values @ w
    contrib = w * sigma_w / port_var
    return pd.Series(contrib, index=cov.index)


def crypto_cap(
    weights: pd.Series,
    cov: pd.DataFrame,
    crypto_assets: list[str],
    max_crypto_var_share: float,
) -> pd.Series:
    """Cap the COMBINED crypto contribution to portfolio variance at
    `max_crypto_var_share`, then renormalize weights to sum to 1.

    No-op (up to renormalization) when crypto is already under budget. When it
    binds, the crypto block is scaled by a single factor `a` solved so that the
    post-scaling crypto variance share equals the cap exactly. Because variance
    is quadratic, the share is a monotone function of `a`; we solve it in closed
    form on the 2x2 (crypto-block vs rest) decomposition so the result is exact
    regardless of cross-correlations.
    """
    w = weights.reindex(cov.index).fillna(0.0).copy()
    present_crypto = [a for a in crypto_assets if a in w.index]
    w_norm = w / w.sum() if w.sum() != 0 else w

    if not present_crypto:
        return w_norm

    shares = _variance_shares(w_norm, cov)
    cur = float(shares[present_crypto].sum())
    if cur <= max_crypto_var_share + 1e-12:
        return w_norm

    # Decompose total variance V(a) for crypto block scaled by `a`:
    #   V(a) = a^2 * Vc + 2a * Vcr + Vr
    # where Vc = crypto-block variance, Vr = non-crypto variance,
    # Vcr = cross covariance term. Crypto variance contribution (by the same
    # marginal accounting) is the crypto rows of (cov @ w), i.e.
    #   contrib_c(a) = a^2 * Vc + a * Vcr     (crypto's half of the cross term)
    # We want contrib_c(a) / V(a) = cap.
    idx = list(cov.index)
    is_crypto = np.array([a in present_crypto for a in idx])
    wv = w_norm.values
    wc = wv * is_crypto          # crypto-only weight vector
    wr = wv * (~is_crypto)       # rest-only weight vector
    C = cov.values

    Vc = float(wc @ C @ wc)
    Vr = float(wr @ C @ wr)
    Vcr = float(wc @ C @ wr)     # cross term (single, symmetric)

    cap = max_crypto_var_share
    # contrib_c(a) = a^2 Vc + a Vcr ; V(a) = a^2 Vc + 2 a Vcr + Vr
    # (a^2 Vc + a Vcr) = cap (a^2 Vc + 2 a Vcr + Vr)
    # => a^2 Vc (1-cap) + a Vcr (1 - 2cap) - cap Vr = 0
    A = Vc * (1.0 - cap)
    B = Vcr * (1.0 - 2.0 * cap)
    Cc = -cap * Vr
    if abs(A) < 1e-18:
        a = -Cc / B if abs(B) > 1e-18 else 1.0
    else:
        disc = B * B - 4 * A * Cc
        disc = max(disc, 0.0)
        a = (-B + np.sqrt(disc)) / (2 * A)
    a = float(np.clip(a, 0.0, 1.0))

    out = w_norm.copy()
    out[present_crypto] = w_norm[present_crypto] * a
    s = out.sum()
    if s != 0:
        out = out / s
    return out


# --------------------------------------------------------------------------- #
# Effective number of bets
# --------------------------------------------------------------------------- #
def effective_bets(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Effective number of bets via the squared diversification ratio
    (Choueifaty & Coignard 2008):

        ENB = (sum_i w_i * sigma_i)^2 / (w' Σ w)

    where sigma_i = sqrt(diag(Σ))_i. This equals N for N equal-weight,
    equal-vol, UNCORRELATED assets, and collapses toward 1 as assets become
    perfectly correlated — unlike the inverse-HHI of marginal risk
    contributions, which returns N for equal weights regardless of correlation
    and therefore does NOT measure correlation-driven diversification loss.

    Computed from the LIVE covariance matrix, so it reflects realized
    co-movement, not nominal asset count.
    """
    w = weights.reindex(cov.index).fillna(0.0).values
    port_var = float(w @ cov.values @ w)
    if port_var <= 0:
        return 0.0
    sigma = np.sqrt(np.clip(np.diag(cov.values), 0.0, None))
    weighted_vol_sum = float(np.abs(w) @ sigma)
    return (weighted_vol_sum**2) / port_var


# --------------------------------------------------------------------------- #
# Rebalance schedule — trade rarely
# --------------------------------------------------------------------------- #
def rebalance_schedule(dates: pd.DatetimeIndex, freq: str = "weekly") -> pd.Series:
    """Boolean mask over `dates` marking rebalance days.

    A no-alpha book bleeds to turnover/costs, so it rebalances rarely. The first
    eligible date is always a rebalance (the book must be initialized).

      * "weekly": one rebalance per ISO calendar week.
      * "daily":  every bar (diagnostic / stress only).
    """
    mask = pd.Series(False, index=dates)
    if len(dates) == 0:
        return mask
    if freq == "daily":
        mask[:] = True
        return mask
    if freq == "weekly":
        iso = dates.isocalendar()
        week_id = iso["year"].astype(int) * 100 + iso["week"].astype(int)
        week_id.index = dates
        first_of_week = ~week_id.duplicated(keep="first")
        mask[first_of_week.values] = True
        mask.iloc[0] = True
        return mask
    raise ValueError(f"unknown rebalance freq: {freq!r}")


# --------------------------------------------------------------------------- #
# Trend-tilt lottery sleeve (ISOLATED positive-skew sleeve)
# --------------------------------------------------------------------------- #
def trend_tilt_sleeve(
    returns: pd.DataFrame,
    sigmas: pd.Series,
    lookback: int,
    top_n: int,
) -> pd.Series:
    """A small, isolated positive-skew LOTTERY sleeve. NOT validated alpha — a
    tournament-theory variance-buy that converts a slice of the risk budget into
    a positive-skew P&L-prize bet.

    Direction = sign of the trailing `lookback`-day cumulative return per asset
    (causal: uses the last `lookback` returns up to the most recent bar, which in
    the backtest are returns <= t-1). Magnitude = inverse-vol scaled. The sleeve
    is CONCENTRATED in the `top_n` assets with the strongest |trend|; all other
    assets get zero. Output weights are L1-normalized so the sleeve's gross
    exposure is 1 unit (the backtest scales it by a tiny risk-budget fraction and
    keeps it isolated from the survival book's kill-switch).
    """
    if returns.shape[0] < lookback:
        return pd.Series(0.0, index=returns.columns, dtype="float64")

    trail = returns.iloc[-lookback:]
    cum = (1.0 + trail).prod() - 1.0       # trailing cumulative return per asset
    cum = cum.reindex(returns.columns)

    strength = cum.abs()
    strength = strength[sigmas.reindex(strength.index).notna() & (sigmas.reindex(strength.index) > 0)]
    strength = strength.dropna()
    if strength.empty:
        return pd.Series(0.0, index=returns.columns, dtype="float64")

    winners = strength.sort_values(ascending=False).head(top_n).index

    tilt = pd.Series(0.0, index=returns.columns, dtype="float64")
    for a in winners:
        sgn = np.sign(cum[a])
        if sgn == 0:
            continue
        tilt[a] = sgn / sigmas[a]          # inverse-vol scaled, signed

    gross = tilt.abs().sum()
    if gross <= 0:
        return pd.Series(0.0, index=returns.columns, dtype="float64")
    return tilt / gross


# --------------------------------------------------------------------------- #
# Hard risk controls — kill-switches (B0001 sizing / B0002 shut-off gates)
# --------------------------------------------------------------------------- #
def apply_risk_controls(
    weights: pd.Series,
    asset_returns: pd.Series,
    *,
    per_asset_loss_cap: float,
    portfolio_kill: float,
    max_dd_stop: float,
    equity_high_water: float,
    equity_now: float,
    state: RiskState | None = None,
    manual_reset: bool = False,
    gap_assets: list[str] | None = None,
) -> dict:
    """Apply the hard risk controls to one day's realized return and report which
    switches tripped. Order matters and mirrors a real risk desk:

      1. max-drawdown stop (state level, MANUAL-RESET LATCH — FIX 1): if current
         equity is more than `max_dd_stop` below its high-water mark, the book is
         FORCED FLAT and the breaker LATCHES. Once latched it stays flat on EVERY
         subsequent bar — including bars where equity has recovered above the
         threshold — until an explicit `manual_reset=True` is supplied. The old
         stateless recompute auto-re-armed: the first recovery bar traded full
         size again. The latch lives on `state` (a `RiskState`); pass the same
         instance every bar. With no `state` the breaker is one-shot per call
         (legacy behaviour, used by the stateless unit tests).
      2. per-asset daily-loss cap (leg level): each asset's daily return is
         floored at `-per_asset_loss_cap` before aggregation, so one cratering
         leg cannot sink the book (a single asset gap can exceed any portfolio
         budget).
      3. portfolio daily-loss kill-switch (book level, GAP-AWARE — FIX 2): the
         loss is split into an INTRADAY-STOPPABLE block (FX/metals, where a stop
         can plausibly fill) and a GAP-THROUGH block (`gap_assets`, the 7-day
         crypto legs that gap over weekends). Only the intraday block is clamped
         at `-portfolio_kill` (the modelled stop fill); the crypto block realizes
         its FULL per-asset loss because a real gap jumps THROUGH the stop. When
         `gap_assets` is None every leg is intraday-stoppable (legacy behaviour).

    The daily kill is intentionally NOT latched (it auto-resets next bar); only
    the drawdown breaker latches. All thresholds are POSITIVE magnitudes.

    Returns a dict with the realized `port_return` and boolean flags `killed`,
    `dd_stopped`, `dd_latched`, `per_asset_capped`, `gap_through`.
    """
    w = weights.copy()
    r = asset_returns.reindex(w.index).fillna(0.0)

    if manual_reset and state is not None:
        state.dd_latched = False

    # 1. max-drawdown stop — MANUAL-RESET LATCH.
    if equity_high_water > 0:
        dd = 1.0 - equity_now / equity_high_water
    else:
        dd = 0.0
    latched_now = bool(state.dd_latched) if state is not None else False
    if dd >= max_dd_stop or latched_now:
        if state is not None:
            state.dd_latched = True
        return {
            "port_return": 0.0,
            "killed": False,
            "dd_stopped": True,
            "dd_latched": True if state is not None else False,
            "per_asset_capped": False,
            "gap_through": False,
        }

    # 2. per-asset daily-loss cap.
    capped = r.clip(lower=-per_asset_loss_cap)
    per_asset_capped = bool((capped > r).any())

    # 3. portfolio daily-loss kill-switch — GAP-AWARE.
    gaps = set(gap_assets) if gap_assets is not None else set()
    is_gap = w.index.isin(gaps) if gaps else np.zeros(len(w), dtype=bool)

    contrib = w * capped
    intraday_return = float(contrib[~is_gap].sum())
    gap_return = float(contrib[is_gap].sum())

    killed = False
    # The intraday-stoppable block can be clamped — a stop can fill there.
    if intraday_return <= -portfolio_kill:
        intraday_return = -portfolio_kill
        killed = True

    # The gap-through (crypto) block realizes its full loss: a weekend gap jumps
    # past the intraday stop and fills at the gap. No clamp is applied to it.
    gap_through = bool(gap_return < -portfolio_kill + 1e-12) and bool(gaps)

    port_return = intraday_return + gap_return

    return {
        "port_return": port_return,
        "killed": killed,
        "dd_stopped": False,
        "dd_latched": bool(state.dd_latched) if state is not None else False,
        "per_asset_capped": per_asset_capped,
        "gap_through": gap_through,
    }


# --------------------------------------------------------------------------- #
# FIX 3 — hard capital-isolation cap for the lottery sleeve
# --------------------------------------------------------------------------- #
def apply_sleeve_cap(
    sleeve_return: float,
    state: SleeveState,
    *,
    max_loss_frac_of_envelope: float,
) -> float:
    """Capital-isolate the lottery sleeve with a hard, latching DRAWDOWN cap
    (FIX 3).

    The sleeve is accounting-isolated from the survival book's kill-switch, but
    it still shares account equity — so an unbounded sleeve drawdown inflated the
    COMBINED (book+sleeve) maxDD beyond the book-only figure. This cap bounds it:
    the sleeve accumulates realized P&L and tracks its high-water mark; once the
    DRAWDOWN from that high-water exceeds `max_loss_frac_of_envelope * envelope`
    of equity, the sleeve LATCHES FLAT permanently (returns 0 thereafter). The
    breaching day's loss is still realized (the stop fills at the bar), so the
    sleeve's worst-case contribution to combined drawdown is bounded by the cap
    plus one bar's move.

    Returns the realized sleeve return for the day (0.0 once latched).
    """
    if state.latched:
        return 0.0
    state.cum_pnl += sleeve_return
    state.hwm_pnl = max(state.hwm_pnl, state.cum_pnl)
    drawdown = state.hwm_pnl - state.cum_pnl
    dd_cap = abs(max_loss_frac_of_envelope) * abs(state.envelope)
    if drawdown >= dd_cap:
        state.latched = True
    return float(sleeve_return)


# --------------------------------------------------------------------------- #
# FIX 4 — per-asset notional sub-cap (general + tighter SOL cap)
# --------------------------------------------------------------------------- #
def cap_asset_weights(
    weights: pd.Series,
    *,
    per_asset_cap: float,
    asset_caps: dict[str, float] | None = None,
) -> pd.Series:
    """Clamp each asset's weight to a hard per-asset cap, then renormalize to
    sum 1 (FIX 4).

    `per_asset_cap` is the general ceiling; `asset_caps` overrides it per asset
    (e.g. a tighter SOL sub-cap of 0.05 because SOL has only ~1.4y of history and
    must not dominate the crypto block in the long backtest). Because clamping
    one asset and renormalizing can push another back over its cap, we iterate to
    a fixed point (bounded iterations). Operates on the (already crypto-capped)
    inverse-vol weights, BEFORE vol-target scaling.

    If the caps are jointly INFEASIBLE (their sum < 1), the hard cap dominates
    the sum-to-1 convention: the book is left UNDER-allocated (weights sum < 1,
    i.e. it holds cash) rather than renormalizing a pinned weight back over its
    cap. The downstream vol-target step scales whatever gross remains.
    """
    asset_caps = asset_caps or {}
    w = weights.copy().astype("float64")
    s = w.sum()
    if s <= 0:
        return w
    w = w / s

    def cap_for(a: str) -> float:
        return float(asset_caps.get(a, per_asset_cap))

    # Water-filling with a CUMULATIVE pinned set: once an asset is pinned at its
    # cap it stays pinned, so redistributing freed weight onto the remaining
    # assets can never push a previously-pinned asset back over its cap (the bug
    # in a per-iteration recompute, where SOL and BTC oscillate).
    pinned: set[str] = set()
    for _ in range(len(w) + 2):
        free = [a for a in w.index if a not in pinned]
        pinned_total = float(sum(cap_for(a) for a in pinned))
        remaining = 1.0 - pinned_total
        free_total = float(w[free].sum()) if free else 0.0
        if not free or free_total <= 0 or remaining <= 0:
            break
        # Tentatively fill the free assets proportionally to current weight.
        scale = remaining / free_total
        newly_over = [a for a in free if w[a] * scale > cap_for(a) + 1e-15]
        if not newly_over:
            for a in free:
                w[a] = w[a] * scale
            break
        # Pin the assets that would exceed their cap and repeat.
        pinned.update(newly_over)

    for a in pinned:
        w[a] = cap_for(a)
    total = w.sum()
    # Renormalize DOWN only. Never scale up past the caps: if the caps are
    # jointly infeasible (total < 1) the book holds cash rather than breaching a
    # hard cap. Floating-point overshoot (total slightly > 1) is corrected down.
    if total > 1.0 + 1e-12:
        w = w / total
    return w


# --------------------------------------------------------------------------- #
# FIX 5 — stressed (high-corr) covariance for the crypto variance cap
# --------------------------------------------------------------------------- #
def stress_covariance(
    cov: pd.DataFrame,
    crypto_assets: list[str],
    *,
    corr_floor: float,
) -> pd.DataFrame:
    """Return a STRESSED copy of `cov` in which every CRYPTO-CRYPTO pairwise
    correlation is floored at `corr_floor` (FIX 5).

    The 30% crypto variance cap is solved on the rebalance-day covariance and the
    resulting weights are HELD up to a week. If live crypto correlations spike
    toward 1 within that week, the held crypto variance share overshoots the
    budget (~36.5% in the review). Solving the cap on this stressed matrix — with
    crypto correlations pre-floored near 1 — makes the held weights robust to a
    correlation blow-out: they stay at-or-under budget even when realized corr
    hits 1.

    Only crypto-crypto OFF-DIAGONAL entries are raised; variances (the diagonal)
    and all non-crypto / cross-block entries are left untouched. If a crypto pair
    is already MORE correlated than the floor, it is left as-is (we never reduce
    correlation — that would understate risk).
    """
    idx = list(cov.index)
    present = [a for a in crypto_assets if a in idx]
    if len(present) < 2:
        return cov.copy()

    out = cov.copy().astype("float64")
    sig = np.sqrt(np.clip(np.diag(cov.values), 0.0, None))
    pos = {a: idx.index(a) for a in idx}

    for i, a in enumerate(present):
        for b in present[i + 1:]:
            ia, ib = pos[a], pos[b]
            sa, sb = sig[ia], sig[ib]
            if sa <= 0 or sb <= 0:
                continue
            cur = out.iloc[ia, ib]
            cur_corr = cur / (sa * sb)
            new_corr = max(cur_corr, corr_floor)
            new_cov = new_corr * sa * sb
            out.iloc[ia, ib] = new_cov
            out.iloc[ib, ia] = new_cov
    return out


# --------------------------------------------------------------------------- #
# Variance-share diagnostics (named, with combined-crypto roll-up)
# --------------------------------------------------------------------------- #
def named_var_shares(
    weights: pd.Series, cov: pd.DataFrame, crypto_assets: list[str]
) -> dict:
    """Per-asset marginal variance-contribution share, plus a `_crypto_combined`
    roll-up. Identical math to scripts.backtest_survival_book._named_var_shares;
    extracted here so backtest + live share one implementation."""
    w = weights.reindex(cov.index).fillna(0.0).values
    port_var = float(w @ cov.values @ w)
    if port_var <= 0:
        return {}
    sigma_w = cov.values @ w
    contrib = w * sigma_w / port_var
    out = {a: float(c) for a, c in zip(cov.index, contrib)}
    out["_crypto_combined"] = float(sum(out.get(a, 0.0) for a in crypto_assets))
    return out


def held_book_cap_breach(
    rets: pd.DataFrame,
    weights: pd.Series,
    *,
    universe: list[str],
    crypto: list[str],
    cov_window: int,
    max_leverage: float,
    per_asset_weight_cap: float,
    asset_weight_caps: dict | None,
    crypto_var_cap: float,
    crypto_corr_stress_floor: float,
    eps: float = 1e-6,
) -> dict:
    """Does the HELD weight vector breach any survival-book cap? SAFETY guard for the
    live no-trade band (B0118, risk-officer must-fix 2026-05-31): the band must NEVER
    skip a rebalance while the HELD book is over-cap. The SSOT target is always
    cap-compliant by construction, so a target-side flag can never detect a held breach
    caused by price drift since the last trade — this measures the held book directly.

    Uses an INDEPENDENT, slightly-conservative trailing covariance (NOT the SSOT cov):
    this is a guard, so over-acting is the safe error and `compute_survival_target` is
    left untouched. Checks per-asset weight cap, gross vs `max_leverage`, and the
    STRESSED crypto variance share vs `crypto_var_cap`.
    """
    w = weights.reindex(universe).fillna(0.0)
    gross = float(w.abs().sum())
    gross_breach = gross > max_leverage + eps
    caps = dict(asset_weight_caps or {})
    per_asset_breach = any(
        abs(float(w.get(a, 0.0))) > caps.get(a, per_asset_weight_cap) + eps
        for a in universe
    )
    crypto_share = float("nan")
    crypto_breach = False
    cov_hist = rets[[a for a in universe if a in rets.columns]].tail(cov_window).dropna(how="any")
    present = [a for a in cov_hist.columns if cov_hist[a].notna().sum() > 2]
    if len(present) >= 2 and cov_hist.shape[0] >= 20:
        cov_m = cov_hist[present].cov() * TRADING_DAYS
        cov_stress = stress_covariance(cov_m, crypto, corr_floor=crypto_corr_stress_floor)
        shares = named_var_shares(w.reindex(present).fillna(0.0), cov_stress, crypto)
        crypto_share = shares.get("_crypto_combined", float("nan"))
        if np.isfinite(crypto_share):
            crypto_breach = crypto_share > crypto_var_cap + eps
    return {
        "breach": bool(gross_breach or per_asset_breach or crypto_breach),
        "gross": gross, "gross_breach": bool(gross_breach),
        "per_asset_breach": bool(per_asset_breach),
        "crypto_var_share_stressed": crypto_share, "crypto_breach": bool(crypto_breach),
    }


# --------------------------------------------------------------------------- #
# Rebalance-day target weights — SINGLE SOURCE OF TRUTH (parity seam)
# --------------------------------------------------------------------------- #
@dataclass
class SurvivalTarget:
    """Result of one rebalance-day weight decision.

    `weights` are the FINAL scaled (post-leverage) target weights over the full
    universe, summing to <= max_leverage. The diagnostics mirror what the
    backtest reports per rebalance so a live caller can log/persist the same
    numbers. `book_active=False` means the DD breaker has latched: the book is
    held flat and `weights` are all zero.
    """
    weights: pd.Series
    book_active: bool
    gross_leverage: float
    leverage_scale: float
    leverage_capped: bool
    unscaled_exante_vol: float
    scaled_exante_vol: float
    effective_bets: float
    variance_shares: dict
    held_crypto_var_share_stressed: float


def compute_survival_target(
    rets_history: pd.DataFrame,
    *,
    universe: list[str],
    crypto: list[str],
    vol_window: int,
    cov_window: int,
    target_vol: float,
    max_leverage: float,
    crypto_var_cap: float,
    per_asset_weight_cap: float,
    asset_weight_caps: dict[str, float] | None = None,
    crypto_corr_stress_floor: float,
    risk_state: RiskState | None = None,
    vol_forecast: pd.Series | None = None,
    cov_exclude: frozenset[str] | set[str] | None = None,
) -> SurvivalTarget:
    """Compute the survival book's FINAL scaled target weights for the holding
    period beginning at the bar AFTER the last row of `rets_history`.

    This is the SINGLE SOURCE OF TRUTH for the rebalance-day weight decision: it
    is the verbatim lift of the inline rebalance block formerly in
    scripts.backtest_survival_book.run_backtest. The backtest and the live loop
    both call it, so "live decisions are identical to backtest decisions on the
    same data" is structural.

    Causality / no-look-ahead (CLAUDE.md)
    -------------------------------------
    `rets_history` is the trailing window of realized returns INCLUDING the
    decision bar's own row (mirroring the backtest, which passes `rets[:i+1]`).
    The decision bar's return never leaks because:
      * `realized_vol` shifts by one bar internally -> vol uses returns <= the
        bar before the decision bar;
      * the covariance window EXCLUDES the decision bar (`hist[:-1]`).
    There is no walk-forward purge/embargo: a continuously-held portfolio has no
    overlapping triple-barrier outcome windows, and live trading is inherently
    causal (we fire on bar close, feeding only closed bars). Purge/embargo is a
    backtest-only meta-labeling concern and is bypassed here by design.

    `risk_state` carries the persistent DD latch. When `risk_state.dd_latched`
    is True the book is FORCED FLAT (all-zero weights, `book_active=False`); the
    live loop must never place orders in that state until a manual reset clears
    the latch on the state.

    `cov_exclude` is the PERMANENT covariance-exclusion set (default
    COV_PERMANENT_EXCLUDE = {"BARUSD"}): listed symbols never enter the cov matrix
    regardless of bar count, so they can never earn weight or inject a spurious
    correlation. Pass an explicit `frozenset()` to disable (tests only); the
    deployed callers leave it None to inherit the documented default.
    """
    asset_weight_caps = asset_weight_caps or {}
    book_active = not (risk_state is not None and risk_state.dd_latched)

    flat = pd.Series(0.0, index=universe, dtype="float64")
    if not book_active:
        return SurvivalTarget(
            weights=flat, book_active=False, gross_leverage=0.0,
            leverage_scale=0.0, leverage_capped=False,
            unscaled_exante_vol=0.0, scaled_exante_vol=0.0,
            effective_bets=float("nan"), variance_shares={},
            held_crypto_var_share_stressed=float("nan"),
        )

    hist = rets_history
    n = hist.shape[0]

    # Per-asset trailing realized vol (causal: realized_vol shifts by 1).
    sig = pd.Series(
        {a: realized_vol(hist[a].dropna(), vol_window, annualize=True).iloc[-1]
         if hist[a].dropna().shape[0] > vol_window else np.nan
         for a in universe}
    )
    # B0101 — optional HAR vol forecast REPLACES the per-asset vol used for the
    # inverse-vol weights (and, below, the covariance diagonal), keeping sample
    # correlations. vol_forecast=None -> sig_used == sig -> behavior UNCHANGED.
    if vol_forecast is not None:
        vf = vol_forecast.reindex(universe)
        sig_used = vf.where(vf.notna() & (vf > 0), sig)
    else:
        sig_used = sig
    base_w = inverse_vol_weights(sig_used)

    # Covariance window EXCLUDES the decision bar (the final row) -> causal.
    cov_hist = hist.iloc[max(0, n - 1 - cov_window): n - 1] if n >= 1 else hist.iloc[0:0]
    # WP2 (2026-06-17) — cov-ADMISSION threshold raised from `> cov_window // 2`
    # (=60) to a FULL `cov_window` (=120) of TOTAL available history per asset.
    #
    # Rationale (quant-advisor verdict 2026-06-17, DeMiguel/Garlappi/Uppal 2009 +
    # LdP MLfAM §2.2): a half-window row (~61 obs) on a newly-listed symbol is
    # dominated by Marcenko-Pastur noise (band ~(1±sqrt(N/T))^2), ill-conditioning
    # the matrix and — via the equally-noisy ~61-bar vol estimate — earning an
    # OUTSIZED inverse-vol weight on the most-fragile asset (the exact §13/§14
    # concentration failure the survival book exists to avoid). Requiring a full
    # window means the 7 thin 15-symbol-expansion entrants (<= ~11 bars over the
    # ~5-day contest) are never admitted, so the live cov matrix is the
    # deep-history incumbents only — which is correct.
    #
    # We count an asset's TOTAL non-NaN returns in `hist` (its whole available
    # history up to the decision bar), NOT its presence in the trailing
    # `cov_hist` slice. This is deliberate and load-bearing: on a RAGGED calendar
    # (crypto trades 24/7 while FX/metals are closed -> the union calendar
    # extends past a closed market's last bar), a deep asset has a few trailing
    # NaNs in `cov_hist` but thousands of bars overall. Gating on the trailing
    # slice would WRONGLY starve the whole book on weekends; gating on total
    # history admits the deep asset (its stale tail rows are then dropped by the
    # `dropna(how="any")` below, identical to the prior behaviour) while still
    # excluding a genuine newcomer that has only a handful of bars TOTAL.
    hist_counts = hist[universe].notna().sum()
    # PERMANENT covariance-exclusion (FINAL HARDENING, 2026-06-17): a symbol in
    # `cov_exclude` (default COV_PERMANENT_EXCLUDE = {"BARUSD"}) is barred from the
    # cov matrix REGARDLESS of bar count. Because the inverse-vol weights are
    # renormalized over `present` and every downstream step (crypto cap, per-asset
    # cap, vol-target, variance shares) operates on the cov index, a symbol that is
    # never `present` can never earn weight and never appears in the diagnostics —
    # so the illiquid fan token can never inject a spurious correlation even if it
    # somehow accrued >= cov_window bars. This is the cov-side half of the
    # structural "8 effectively traded" lock; the 0.00 per-asset weight cap is the
    # weight-side half.
    excluded = COV_PERMANENT_EXCLUDE if cov_exclude is None else frozenset(cov_exclude)
    present = [
        a for a in universe
        if int(hist_counts[a]) >= cov_window and a not in excluded
    ]

    if not (len(present) >= 2 and base_w[present].sum() > 0):
        return SurvivalTarget(
            weights=flat.copy(), book_active=True, gross_leverage=0.0,
            leverage_scale=0.0, leverage_capped=False,
            unscaled_exante_vol=0.0, scaled_exante_vol=0.0,
            effective_bets=float("nan"), variance_shares={},
            held_crypto_var_share_stressed=float("nan"),
        )

    cov = cov_hist[present].dropna(how="any")
    if cov.shape[0] >= max(20, cov_window // 4):
        cov_m = cov.cov() * TRADING_DAYS
    else:
        cov_m = pd.DataFrame(
            np.diag((sig[present].fillna(sig[present].mean())) ** 2),
            index=present, columns=present,
        )
    # B0101 — rescale the covariance diagonal to the HAR forecast vols (keeping
    # sample correlations) so the vol-target scale + variance shares react to the
    # forecast. No-op (ratio 1 everywhere) when vol_forecast is None.
    if vol_forecast is not None:
        cov_m = _rescale_cov_to_vols(cov_m, vol_forecast)
    w_present = base_w[present] / base_w[present].sum()

    # FIX 5 — solve the crypto var cap on a STRESSED covariance.
    cov_stress = stress_covariance(cov_m, crypto, corr_floor=crypto_corr_stress_floor)
    w_capped = crypto_cap(w_present, cov_stress, crypto, crypto_var_cap)

    # FIX 4 — hard per-asset notional sub-caps.
    w_capped = cap_asset_weights(
        w_capped, per_asset_cap=per_asset_weight_cap, asset_caps=asset_weight_caps,
    )

    stressed_shares = named_var_shares(
        w_capped.reindex(present).fillna(0.0), cov_stress, crypto)
    held_crypto_share_stressed = stressed_shares.get("_crypto_combined", float("nan"))

    book_vol_unscaled = float(np.sqrt(
        w_capped.values @ cov_m.values @ w_capped.values))
    k = vol_target_scale(w_capped, cov_m, target_vol, max_leverage)
    scaled = (w_capped * k).reindex(universe).fillna(0.0)

    return SurvivalTarget(
        weights=scaled,
        book_active=True,
        gross_leverage=float(scaled.abs().sum()),
        leverage_scale=float(k),
        leverage_capped=bool(k >= max_leverage - 1e-9),
        unscaled_exante_vol=book_vol_unscaled,
        scaled_exante_vol=book_vol_unscaled * k,
        effective_bets=effective_bets(w_capped, cov_m),
        variance_shares=named_var_shares(w_capped, cov_m, crypto),
        held_crypto_var_share_stressed=held_crypto_share_stressed,
    )
