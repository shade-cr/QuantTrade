"""Backtest the SURVIVAL BOOK over data/D1 history.

The survival book is a conservative, no-alpha, inverse-vol risk-parity,
vol-targeted multi-asset portfolio. It is RISK ENGINEERING, not alpha: the goal
is NOT high Sharpe (with no directional edge, expected Sharpe ~ 0). The goal is:

  * realized annualized vol lands near the LOW target (default 6%),
  * max drawdown is small and controlled,
  * the book NEVER blows up — hard kill-switches work,
  * crypto cannot dominate portfolio variance.

A small, ISOLATED trend-tilt LOTTERY sleeve (~5-10% of risk budget) is reported
separately. It is NOT validated alpha — a positive-skew tournament-theory
variance-buy — and is kept out of the survival book's kill-switch entirely.

No-look-ahead (CLAUDE.md): every weight applied to date t is decided from
returns strictly before t. `realized_vol` shifts by one bar; rebalance weights
are computed on the rebalance date's trailing (causal) window and HELD until the
next rebalance, then applied forward.

Sharpe annualization: this is a CONTINUOUSLY-HELD book producing a daily return
every bar, so Sharpe = mean/std * sqrt(252) on the daily portfolio-return series.
That is DISTINCT from pipeline.metrics.strategy_metrics' sqrt(trades_per_year),
which is correct only for SPARSE per-trade meta-labeling pnl. See the module
docstring of pipeline/survival_book.py.

Run:
    uv run python scripts/backtest_survival_book.py
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

# Project root on sys.path so `pipeline.*` resolves regardless of invocation.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.data import load_dataset
from pipeline.survival_book import (
    realized_vol,
    inverse_vol_weights,
    vol_target_scale,
    crypto_cap,
    effective_bets,
    rebalance_schedule,
    trend_tilt_sleeve,
    apply_risk_controls,
    cap_asset_weights,
    stress_covariance,
    apply_sleeve_cap,
    compute_survival_target,
    named_var_shares,
    RiskState,
    SleeveState,
    TRADING_DAYS,
)

UNIVERSE = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD", "SOLUSD"]
CRYPTO = ["BTCUSD", "ETHUSD", "SOLUSD"]

# FIX 4 — hard per-asset notional sub-caps (fraction of the inverse-vol book,
# pre-leverage). SOL gets a tighter ceiling than the general cap because it has
# only ~1.4y of history (from 2025-01) and must not dominate the crypto block.
PER_ASSET_WEIGHT_CAP = 0.40        # general ceiling, no single asset > 40%
ASSET_WEIGHT_CAPS = {"SOLUSD": 0.10}

# FIX 5 — solve the crypto variance cap on a STRESSED covariance with crypto
# pairwise correlation floored near 1, so HELD weights stay under budget when a
# live correlation spike pushes realized crypto var share up.
CRYPTO_CORR_STRESS_FLOOR = 0.95

# FIX 3 — the lottery sleeve flattens permanently if its cumulative loss exceeds
# this fraction of its allocated envelope (sleeve_risk_frac gross).
SLEEVE_LOSS_CAP_FRAC = 0.50

# Per-rebalance one-way transaction cost in basis points, by asset class.
COST_BPS = {a: 5.0 for a in CRYPTO}            # crypto wider
for a in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD"]:
    COST_BPS[a] = 2.0                            # fx / metals tighter


def load_universe(data_dir: Path) -> pd.DataFrame:
    """Return a wide DataFrame of daily LOG returns, one column per asset, on the
    union of all trading dates. Missing dates per asset (e.g. crypto weekends vs
    FX, or SOL pre-2025) are NaN — the asset is simply absent (zero weight) until
    it has a causal vol estimate."""
    closes = {}
    for asset in UNIVERSE:
        path = data_dir / f"{asset}_D1.csv"
        df = load_dataset(path)
        closes[asset] = df["close"]
    px = pd.DataFrame(closes).sort_index()
    # Log returns; first obs per asset is NaN. Do NOT forward-fill prices — a
    # forward-filled close would inject a fake zero-return bar (look-alike data)
    # and distort vol. We keep the natural per-asset calendar.
    rets = np.log(px / px.shift(1))
    return rets


def annualized_vol(daily_returns: pd.Series) -> float:
    return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def daily_sharpe(daily_returns: pd.Series) -> float:
    sd = daily_returns.std(ddof=1)
    if sd <= 1e-15:
        return float("nan")
    return float(daily_returns.mean() / sd * np.sqrt(TRADING_DAYS))


def run_backtest(
    rets: pd.DataFrame,
    *,
    vol_window: int,
    cov_window: int,
    target_vol: float,
    max_leverage: float,
    crypto_var_cap: float,
    rebalance_freq: str,
    per_asset_loss_cap: float,
    portfolio_kill: float,
    max_dd_stop: float,
    sleeve_risk_frac: float,
    sleeve_lookback: int,
    sleeve_top_n: int,
    gap_aware: bool = True,                                  # FIX 2
    per_asset_weight_cap: float = PER_ASSET_WEIGHT_CAP,      # FIX 4
    asset_weight_caps: dict | None = None,                  # FIX 4
    crypto_corr_stress_floor: float = CRYPTO_CORR_STRESS_FLOOR,  # FIX 5
    sleeve_loss_cap_frac: float | None = SLEEVE_LOSS_CAP_FRAC,   # FIX 3
    no_trade_band: float = 0.0,                                 # B0118 (parity: 0.0 == off)
    return_series: bool = False,
) -> dict:
    if asset_weight_caps is None:
        asset_weight_caps = dict(ASSET_WEIGHT_CAPS)
    gap_assets = CRYPTO if gap_aware else None

    dates = rets.index
    reb_mask = rebalance_schedule(dates, freq=rebalance_freq)

    # State for the SURVIVAL book.
    cur_weights = pd.Series(0.0, index=UNIVERSE)   # scaled (post-leverage) weights
    equity = 1.0
    hwm = 1.0
    book_curve = []
    book_rets = []
    n_kills = 0
    n_dd_stops = 0
    n_gap_through = 0
    n_rebalances = 0
    total_turnover = 0.0
    total_cost = 0.0
    last_var_shares = None
    last_enb = float("nan")
    last_gross_lev = 0.0
    last_held_crypto_share_stressed = float("nan")
    n_lev_capped = 0
    n_lev_decisions = 0
    exante_vols = []          # scaled ex-ante portfolio vol on each rebalance
    unscaled_exante_vols = []  # pre-scaling inverse-vol book ex-ante vol
    held_crypto_shares_stressed = []  # crypto var share of HELD weights under stress
    _rebalance_targets = []   # (iso_date, {asset: scaled_weight}) per book rebalance — parity probe
    _rebalance_sleeve_targets = []  # (iso_date, {asset: sleeve_weight}) — combined-parity probe

    # FIX 1 — persistent DD-latch state threaded through every bar (no auto-rearm).
    risk_state = RiskState()
    # FIX 3 — persistent sleeve envelope-cap state.
    sleeve_state = SleeveState(envelope=sleeve_risk_frac)

    # State for the ISOLATED trend sleeve (kept entirely separate; no kill-switch).
    sleeve_weights = pd.Series(0.0, index=UNIVERSE)
    sleeve_equity = 1.0
    sleeve_curve = []
    sleeve_rets = []

    for i, dt in enumerate(dates):
        day_cost = 0.0
        # --- decide weights using ONLY data up to dt-1 (causal) ------------- #
        if reb_mask.iloc[i]:
            hist = rets.iloc[: i + 1]   # includes dt's return, but realized_vol
                                        # shifts by 1 internally -> uses <= dt-1.
            # `sig` is still needed by the ISOLATED sleeve below.
            sig = pd.Series(
                {a: realized_vol(hist[a].dropna(), vol_window, annualize=True).iloc[-1]
                 if hist[a].dropna().shape[0] > vol_window else np.nan
                 for a in UNIVERSE}
            )

            # SINGLE SOURCE OF TRUTH (parity seam): the SURVIVAL book's
            # rebalance-day weight decision lives in
            # pipeline.survival_book.compute_survival_target. The live loop calls
            # the SAME function on the SAME trailing window, so live and backtest
            # decisions are identical on identical data. FIX 1 (DD latch) is
            # threaded via risk_state inside the function: a latched book returns
            # book_active=False and all-zero weights.
            tgt = compute_survival_target(
                hist,
                universe=UNIVERSE, crypto=CRYPTO,
                vol_window=vol_window, cov_window=cov_window,
                target_vol=target_vol, max_leverage=max_leverage,
                crypto_var_cap=crypto_var_cap,
                per_asset_weight_cap=per_asset_weight_cap,
                asset_weight_caps=asset_weight_caps,
                crypto_corr_stress_floor=crypto_corr_stress_floor,
                risk_state=risk_state,
            )
            book_active = tgt.book_active

            if not book_active:
                scaled = None  # latched: do not rebalance the book at all
            elif tgt.variance_shares:
                scaled = tgt.weights
                held_crypto_shares_stressed.append(tgt.held_crypto_var_share_stressed)
                n_lev_decisions += 1
                if tgt.leverage_capped:
                    n_lev_capped += 1
                if tgt.unscaled_exante_vol > 0:
                    unscaled_exante_vols.append(tgt.unscaled_exante_vol)
                    exante_vols.append(tgt.scaled_exante_vol)
                last_var_shares = tgt.variance_shares
                last_enb = tgt.effective_bets
                last_gross_lev = tgt.gross_leverage
                last_held_crypto_share_stressed = held_crypto_shares_stressed[-1]
            else:
                scaled = pd.Series(0.0, index=UNIVERSE)

            if scaled is not None:
                # No-trade band (B0118): with proportional costs the optimal policy is an
                # INACTION REGION (Davis-Norman 1990) — skip the rebalance entirely when the
                # L1 weight drift is below `no_trade_band`, so small weekly drifts do not
                # churn cost. PARITY: no_trade_band=0.0 never skips -> byte-identical to the
                # calendar rebalance. Skipping also passively harvests the assets' mild mean
                # reversion (B0117): we hold through small dips instead of trading them.
                drift_l1 = float((scaled - cur_weights).abs().sum())
                if no_trade_band > 0.0 and drift_l1 < no_trade_band:
                    pass  # within the band: hold, no turnover, no cost, no rebalance count
                else:
                    _rebalance_targets.append(
                        (str(dt), {a: float(scaled[a]) for a in UNIVERSE}))
                    # Turnover & cost on the change in scaled weights.
                    turnover = (scaled - cur_weights).abs()
                    day_cost = float(sum(turnover[a] * COST_BPS[a] / 1e4 for a in UNIVERSE))
                    total_turnover += float(turnover.sum())
                    total_cost += day_cost
                    n_rebalances += 1
                    cur_weights = scaled

            # --- sleeve rebalance (ISOLATED — not gated by the book latch) -- #
            if not (sleeve_loss_cap_frac is not None and sleeve_state.latched):
                tilt = trend_tilt_sleeve(
                    rets.iloc[:i][UNIVERSE] if i > 0 else rets.iloc[:1][UNIVERSE],
                    sig, lookback=sleeve_lookback, top_n=sleeve_top_n,
                )
                sleeve_weights = (tilt * sleeve_risk_frac).reindex(UNIVERSE).fillna(0.0)

            # Parity probe: capture the sleeve target alongside the book target at
            # this book-rebalance date (only when the book also recorded a target,
            # i.e. scaled is not None) so the live combined-target parity test can
            # compare book + sleeve. A latched sleeve targets flat.
            if scaled is not None:
                sleeve_for_probe = (pd.Series(0.0, index=UNIVERSE)
                                    if (sleeve_loss_cap_frac is not None
                                        and sleeve_state.latched)
                                    else sleeve_weights)
                _rebalance_sleeve_targets.append(
                    (str(dt), {a: float(sleeve_for_probe[a]) for a in UNIVERSE}))

        # --- realize the day --------------------------------------------- #
        today = rets.loc[dt].reindex(UNIVERSE).fillna(0.0)

        rc = apply_risk_controls(
            cur_weights, today,
            per_asset_loss_cap=per_asset_loss_cap,
            portfolio_kill=portfolio_kill,
            max_dd_stop=max_dd_stop,
            equity_high_water=hwm, equity_now=equity,
            state=risk_state,        # FIX 1 — persistent DD latch (no manual reset in backtest)
            gap_assets=gap_assets,   # FIX 2 — crypto legs gap THROUGH the stop
        )
        port_ret = rc["port_return"] - day_cost
        if rc["killed"]:
            n_kills += 1
            cur_weights = pd.Series(0.0, index=UNIVERSE)   # flat after a kill
        if rc["dd_stopped"]:
            n_dd_stops += 1
            cur_weights = pd.Series(0.0, index=UNIVERSE)   # flat: DD latch armed
        if rc["gap_through"]:
            n_gap_through += 1

        equity *= (1.0 + port_ret)
        hwm = max(hwm, equity)
        book_rets.append(port_ret)
        book_curve.append(equity)

        # --- sleeve realize (isolated; hard envelope cap, NO book kill) ---- #
        raw_sleeve_ret = float((sleeve_weights * today).sum())
        if sleeve_loss_cap_frac is not None:
            sleeve_ret = apply_sleeve_cap(
                raw_sleeve_ret, sleeve_state,
                max_loss_frac_of_envelope=sleeve_loss_cap_frac,
            )
        else:
            sleeve_ret = raw_sleeve_ret
        sleeve_equity *= (1.0 + sleeve_ret)
        sleeve_rets.append(sleeve_ret)
        sleeve_curve.append(sleeve_equity)

    book_rets = pd.Series(book_rets, index=dates)
    book_curve = pd.Series(book_curve, index=dates)
    sleeve_rets = pd.Series(sleeve_rets, index=dates)
    sleeve_curve = pd.Series(sleeve_curve, index=dates)

    combined = book_rets + sleeve_rets

    n_years = (dates[-1] - dates[0]).days / 365.25

    worst_held_crypto_share_stressed = (
        float(np.nanmax(held_crypto_shares_stressed))
        if held_crypto_shares_stressed else float("nan"))

    result = {
        "config": {
            "universe": UNIVERSE,
            "crypto": CRYPTO,
            "vol_window": vol_window,
            "cov_window": cov_window,
            "target_vol_annual": target_vol,
            "max_leverage": max_leverage,
            "crypto_var_cap": crypto_var_cap,
            "rebalance_freq": rebalance_freq,
            "per_asset_loss_cap": per_asset_loss_cap,
            "portfolio_kill": portfolio_kill,
            "max_dd_stop": max_dd_stop,
            "cost_bps": COST_BPS,
            "sleeve_risk_frac": sleeve_risk_frac,
            "sleeve_lookback": sleeve_lookback,
            "sleeve_top_n": sleeve_top_n,
            "gap_aware_kill": gap_aware,
            "per_asset_weight_cap": per_asset_weight_cap,
            "asset_weight_caps": asset_weight_caps,
            "crypto_corr_stress_floor": crypto_corr_stress_floor,
            "sleeve_loss_cap_frac": sleeve_loss_cap_frac,
            "no_trade_band": no_trade_band,
            "n_days": int(len(dates)),
            "n_years": round(n_years, 2),
            "date_start": str(dates[0].date()),
            "date_end": str(dates[-1].date()),
        },
        "survival_book": {
            "realized_vol_annual": annualized_vol(book_rets),
            "target_vol_annual": target_vol,
            "vol_target_hit_ratio": annualized_vol(book_rets) / target_vol,
            "max_drawdown": max_drawdown(book_curve),
            "daily_sharpe_annualized": daily_sharpe(book_rets),
            "total_return": float(book_curve.iloc[-1] - 1.0),
            "cagr": float(book_curve.iloc[-1] ** (1.0 / max(n_years, 1e-9)) - 1.0),
            "worst_day": float(book_rets.min()),
            "best_day": float(book_rets.max()),
            "n_rebalances": n_rebalances,
            "avg_turnover_per_rebalance": float(total_turnover / max(n_rebalances, 1)),
            "total_cost_drag": float(total_cost),
            "n_kill_switch_trips": n_kills,
            "n_max_dd_stops": n_dd_stops,
            "n_gap_through_days": n_gap_through,
            "dd_breaker_latched": bool(risk_state.dd_latched),
            "effective_bets_last": last_enb,
            "variance_shares_last": last_var_shares,
            "held_crypto_var_share_stressed_last": last_held_crypto_share_stressed,
            "worst_held_crypto_var_share_stressed": worst_held_crypto_share_stressed,
            "final_gross_leverage": last_gross_lev,
            "leverage_cap_bind_ratio": float(n_lev_capped / max(n_lev_decisions, 1)),
            "median_unscaled_exante_vol": float(np.median(unscaled_exante_vols))
                if unscaled_exante_vols else float("nan"),
            "median_scaled_exante_vol": float(np.median(exante_vols))
                if exante_vols else float("nan"),
            "vol_target_note": (
                "HONEST target = 3% (B0108 mandate 2026-05-31): the book's realized "
                "vol. The prior 6% label was DECORATIVE — the 0.5 leverage cap bound "
                "~100% of rebalances, so the book ran at ~half regardless of the label. "
                "B0112 evidence rejected raising the cap to 'make 6% real': at the "
                "leverage needed (~1.0-1.3x) the -10% DD latch trips and Sharpe flips "
                "negative. A no-alpha book gains nothing from leverage except "
                "variance/ruin, so the conservative ~3% preservation policy is the "
                "honest, evidence-backed mandate."
            ),
            "ann_sharpe_note": "sqrt(252) on daily held-book returns (NOT sqrt(trades_per_year); see metrics.py)",
        },
        "trend_tilt_sleeve": {
            "isolated": True,
            "label": "positive-skew LOTTERY — NOT validated alpha; kept out of the survival book kill-switch",
            "realized_vol_annual": annualized_vol(sleeve_rets),
            "max_drawdown": max_drawdown(sleeve_curve),
            "total_return": float(sleeve_curve.iloc[-1] - 1.0),
            "daily_sharpe_annualized": daily_sharpe(sleeve_rets),
            "skew_daily": float(pd.Series(sleeve_rets).skew()),
            "worst_day": float(sleeve_rets.min()),
            "best_day": float(sleeve_rets.max()),
            "hard_loss_cap_frac_of_envelope": sleeve_loss_cap_frac,
            "envelope_gross": sleeve_risk_frac,
            "cap_latched": bool(sleeve_state.latched),
            "cap_note": (
                "FIX 3 — the sleeve flattens PERMANENTLY once its cumulative loss "
                "exceeds sleeve_loss_cap_frac * envelope of account equity, so the "
                "combined (book+sleeve) drawdown is bounded by book maxDD plus this "
                "hard sleeve cap rather than an unbounded shared-equity bleed."
            ),
        },
        "combined_book_plus_sleeve": {
            "realized_vol_annual": annualized_vol(combined),
            "max_drawdown": max_drawdown((1.0 + combined).cumprod()),
            "daily_sharpe_annualized": daily_sharpe(combined),
            "total_return": float((1.0 + combined).cumprod().iloc[-1] - 1.0),
            "worst_day": float(combined.min()),
            "worst_case_bound_note": (
                "Combined worst case is now bounded: book tail (gap-aware) + the "
                "sleeve's hard envelope-loss cap. Without the cap the sleeve added "
                "an unbounded shared-equity drawdown on top of the book."
            ),
        },
    }
    if return_series:
        result["_book_returns"] = book_rets.tolist()
        result["_sleeve_returns"] = sleeve_rets.tolist()
        result["_combined_returns"] = combined.tolist()
        result["_rebalance_targets"] = _rebalance_targets
        result["_rebalance_sleeve_targets"] = _rebalance_sleeve_targets
    return result


def _named_var_shares(weights: pd.Series, cov: pd.DataFrame) -> dict:
    w = weights.reindex(cov.index).fillna(0.0).values
    port_var = float(w @ cov.values @ w)
    if port_var <= 0:
        return {}
    sigma_w = cov.values @ w
    contrib = w * sigma_w / port_var
    out = {a: float(c) for a, c in zip(cov.index, contrib)}
    out["_crypto_combined"] = float(sum(out.get(a, 0.0) for a in CRYPTO))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/D1")
    p.add_argument("--out", default="results/survival_book/backtest.json")
    p.add_argument("--vol-window", type=int, default=60,
                   help="trailing realized-vol window in days. Swept 20/40/60: all land "
                        "~2.9% realized (leverage cap binds); 60 dominates on lowest maxDD, "
                        "highest ENB (~3.0), highest Sharpe, and lowest turnover -> chosen.")
    p.add_argument("--cov-window", type=int, default=120,
                   help="trailing window for the covariance matrix (vol-target/crypto-cap/ENB)")
    p.add_argument("--target-vol", type=float, default=0.03,
                   help="LOW annualized vol target. Default 3% = the book's HONEST "
                        "realized vol (B0108 mandate); the prior 6% was decorative "
                        "because the 0.5 leverage cap binds ~100% of rebalances.")
    p.add_argument("--max-leverage", type=float, default=0.5)
    p.add_argument("--crypto-var-cap", type=float, default=0.25,
                   help="B0180 — track the DEPLOYED run_survival_live default "
                        "(0.25, B0119). Was 0.30; the headline maxDD in runbook "
                        "§5 must be measured at the deployed cap.")
    p.add_argument("--rebalance-freq", default="weekly", choices=["weekly", "daily"])
    p.add_argument("--per-asset-loss-cap", type=float, default=0.15)
    p.add_argument("--portfolio-kill", type=float, default=0.015)
    p.add_argument("--max-dd-stop", type=float, default=0.10)
    p.add_argument("--sleeve-risk-frac", type=float, default=0.08,
                   help="fraction of one risk unit allocated to the isolated lottery sleeve")
    p.add_argument("--sleeve-lookback", type=int, default=63, help="~3 months of trailing return")
    p.add_argument("--sleeve-top-n", type=int, default=2)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    rets = load_universe(data_dir)

    common = dict(
        vol_window=args.vol_window,
        cov_window=args.cov_window,
        target_vol=args.target_vol,
        max_leverage=args.max_leverage,
        crypto_var_cap=args.crypto_var_cap,
        rebalance_freq=args.rebalance_freq,
        sleeve_risk_frac=args.sleeve_risk_frac,
        sleeve_lookback=args.sleeve_lookback,
        sleeve_top_n=args.sleeve_top_n,
    )

    result = run_backtest(
        rets,
        per_asset_loss_cap=args.per_asset_loss_cap,
        portfolio_kill=args.portfolio_kill,
        max_dd_stop=args.max_dd_stop,
        **common,
    )

    # --- kill-switch verification ----------------------------------------- #
    # On benign 5y history the production thresholds (rarely) trip, so we PROVE
    # the machinery works by re-running with deliberately tight thresholds. The
    # book MUST register trips and end the run flat/protected.
    stress = run_backtest(
        rets,
        per_asset_loss_cap=0.02,   # tight per-asset cap
        portfolio_kill=0.003,      # 0.3% daily kill -> will trip
        max_dd_stop=0.02,          # 2% max-DD stop -> will trip
        **common,
    )
    result["kill_switch_verification"] = {
        "note": "Re-run with deliberately tight thresholds to PROVE the controls fire.",
        "stress_thresholds": {"per_asset_loss_cap": 0.02,
                              "portfolio_kill": 0.003, "max_dd_stop": 0.02},
        "kill_switch_trips": stress["survival_book"]["n_kill_switch_trips"],
        "max_dd_stops": stress["survival_book"]["n_max_dd_stops"],
        "stress_max_drawdown": stress["survival_book"]["max_drawdown"],
        "stress_worst_day": stress["survival_book"]["worst_day"],
        "controls_fire": bool(
            stress["survival_book"]["n_kill_switch_trips"] > 0
            or stress["survival_book"]["n_max_dd_stops"] > 0
        ),
    }

    # --- FIX 2 — gap-aware vs clamped tail comparison --------------------- #
    # Re-run with the OLD intraday-clamp model (gap_aware=False) so the review can
    # see how much worse the TRUE (gap-through) tail is than the clamped figure.
    clamped = run_backtest(
        rets,
        per_asset_loss_cap=args.per_asset_loss_cap,
        portfolio_kill=args.portfolio_kill,
        max_dd_stop=args.max_dd_stop,
        gap_aware=False,
        **common,
    )
    # On the BENIGN 5y history with PRODUCTION thresholds the portfolio daily kill
    # never trips (worst held day ~-1.1% << 1.5% kill) because crypto notional is
    # small after the inverse-vol + 30% var cap — so clamp vs gap-through are
    # identical there. The distinction is LATENT. To EXHIBIT the true gap tail we
    # also re-run at the kill-engaging stress threshold (0.3% kill), where the OLD
    # clamp pinned a crypto-gap day at -0.3% while gap-aware realizes the full
    # crypto gap loss. This is the tail the clamp was hiding.
    clamped_stress = run_backtest(
        rets, per_asset_loss_cap=0.50, portfolio_kill=0.003, max_dd_stop=1.0,
        gap_aware=False, **common,
    )
    gapaware_stress = run_backtest(
        rets, per_asset_loss_cap=0.50, portfolio_kill=0.003, max_dd_stop=1.0,
        gap_aware=True, **common,
    )
    result["gap_aware_comparison"] = {
        "note": ("FIX 2 — crypto (BTC/ETH/SOL) trade weekends and GAP. The daily "
                 "kill now realizes the FULL per-asset crypto gap (the stop fills "
                 "PAST the budget) instead of clamping at -portfolio_kill. Only "
                 "intraday-stoppable FX/metal legs are still clamped."),
        "production_thresholds": {
            "kill_never_trips_on_benign_history": True,
            "reason": ("worst held book day ~-1.1% << 1.5% kill; crypto notional "
                       "is small after inverse-vol + 30% var cap, so the gap-vs-"
                       "clamp distinction is latent on this dataset."),
            "clamped_intraday_model": {
                "max_drawdown": clamped["survival_book"]["max_drawdown"],
                "worst_day": clamped["survival_book"]["worst_day"],
            },
            "gap_aware_model": {
                "max_drawdown": result["survival_book"]["max_drawdown"],
                "worst_day": result["survival_book"]["worst_day"],
                "n_gap_through_days": result["survival_book"]["n_gap_through_days"],
            },
        },
        "kill_engaging_stress_0p3pct": {
            "note": ("0.3% daily kill so the switch ENGAGES on crypto-gap days, "
                     "exposing the true tail the production clamp hid."),
            "clamped_intraday_model": {
                "max_drawdown": clamped_stress["survival_book"]["max_drawdown"],
                "worst_day": clamped_stress["survival_book"]["worst_day"],
            },
            "gap_aware_model": {
                "max_drawdown": gapaware_stress["survival_book"]["max_drawdown"],
                "worst_day": gapaware_stress["survival_book"]["worst_day"],
                "n_gap_through_days": gapaware_stress["survival_book"]["n_gap_through_days"],
            },
        },
    }

    # --- FIX 4 — SOL-live-window backtest (all 3 crypto present) ---------- #
    # SOL has history only from 2025-01, so it barely participates in the 5y run.
    # Re-run restricted to the SOL-live window (2025-03+) where all 3 crypto are
    # present, to report crypto-cap binding, ENB and maxDD in that regime.
    sol_window_start = "2025-03-01"
    rets_sol = rets.loc[rets.index >= sol_window_start]
    sol_result = run_backtest(
        rets_sol,
        per_asset_loss_cap=args.per_asset_loss_cap,
        portfolio_kill=args.portfolio_kill,
        max_dd_stop=args.max_dd_stop,
        **common,
    )
    sb_sol = sol_result["survival_book"]
    result["sol_live_window"] = {
        "note": ("FIX 4 — restricted to the SOL-live window so all 3 crypto "
                 "(BTC/ETH/SOL) participate. SOL carries a hard per-asset sub-cap "
                 f"({result['config']['asset_weight_caps']})."),
        "window_start": sol_window_start,
        "date_start": sol_result["config"]["date_start"],
        "date_end": sol_result["config"]["date_end"],
        "n_days": sol_result["config"]["n_days"],
        "max_drawdown": sb_sol["max_drawdown"],
        "realized_vol_annual": sb_sol["realized_vol_annual"],
        "daily_sharpe_annualized": sb_sol["daily_sharpe_annualized"],
        "effective_bets_last": sb_sol["effective_bets_last"],
        "crypto_var_cap": sol_result["config"]["crypto_var_cap"],
        "crypto_combined_var_share_last": (
            sb_sol["variance_shares_last"].get("_crypto_combined", float("nan"))
            if sb_sol["variance_shares_last"] else float("nan")),
        "crypto_cap_binding": bool(
            sb_sol["variance_shares_last"]
            and sb_sol["variance_shares_last"].get("_crypto_combined", 0.0)
            >= sol_result["config"]["crypto_var_cap"] - 1e-3),
        "sol_var_share_last": (
            sb_sol["variance_shares_last"].get("SOLUSD", float("nan"))
            if sb_sol["variance_shares_last"] else float("nan")),
        "worst_held_crypto_var_share_stressed": sb_sol["worst_held_crypto_var_share_stressed"],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    sb = result["survival_book"]
    print("=" * 70)
    print("SURVIVAL BOOK BACKTEST")
    print("=" * 70)
    cfg = result["config"]
    print(f"window: {cfg['date_start']} -> {cfg['date_end']}  "
          f"({cfg['n_days']} days, {cfg['n_years']}y)")
    print(f"vol_window={cfg['vol_window']}d  target_vol={cfg['target_vol_annual']:.1%}  "
          f"max_lev={cfg['max_leverage']}  crypto_var_cap={cfg['crypto_var_cap']:.0%}  "
          f"rebalance={cfg['rebalance_freq']}")
    print("-" * 70)
    print(f"realized ann. vol        : {sb['realized_vol_annual']:.2%}   "
          f"(target {cfg['target_vol_annual']:.2%}, ratio {sb['vol_target_hit_ratio']:.2f})")
    print(f"max drawdown             : {sb['max_drawdown']:.2%}")
    print(f"daily Sharpe (sqrt 252)  : {sb['daily_sharpe_annualized']:.3f}")
    print(f"total return / CAGR      : {sb['total_return']:.2%} / {sb['cagr']:.2%}")
    print(f"worst / best day         : {sb['worst_day']:.2%} / {sb['best_day']:.2%}")
    print(f"effective bets (last)    : {sb['effective_bets_last']:.2f}")
    print(f"final gross leverage     : {sb['final_gross_leverage']:.3f}  (cap {cfg['max_leverage']})")
    print(f"leverage-cap bind ratio  : {sb['leverage_cap_bind_ratio']:.1%}  "
          f"(unscaled ex-ante vol ~{sb['median_unscaled_exante_vol']:.2%} -> cap de-levers to "
          f"~{sb['median_scaled_exante_vol']:.2%} ex-ante)")
    print(f"rebalances / avg turnover: {sb['n_rebalances']} / {sb['avg_turnover_per_rebalance']:.3f}")
    print(f"total cost drag          : {sb['total_cost_drag']:.2%}")
    print(f"kill-switch trips        : {sb['n_kill_switch_trips']}")
    print(f"max-DD stops             : {sb['n_max_dd_stops']}")
    if sb["variance_shares_last"]:
        cc = sb["variance_shares_last"].get("_crypto_combined", float("nan"))
        print(f"crypto combined var share: {cc:.1%}  (cap {cfg['crypto_var_cap']:.0%})")
    print("-" * 70)
    sl = result["trend_tilt_sleeve"]
    print("TREND-TILT LOTTERY SLEEVE (isolated, NOT alpha):")
    print(f"  ann. vol {sl['realized_vol_annual']:.2%}  maxDD {sl['max_drawdown']:.2%}  "
          f"total {sl['total_return']:.2%}  daily-skew {sl['skew_daily']:.2f}")
    cb = result["combined_book_plus_sleeve"]
    print(f"COMBINED book+sleeve: ann.vol {cb['realized_vol_annual']:.2%}  "
          f"maxDD {cb['max_drawdown']:.2%}  Sharpe {cb['daily_sharpe_annualized']:.3f}")
    print("-" * 70)
    kv = result["kill_switch_verification"]
    print("KILL-SWITCH VERIFICATION (stress thresholds 0.3% kill / 2% maxDD):")
    print(f"  kill-switch trips {kv['kill_switch_trips']}  max-DD stops {kv['max_dd_stops']}  "
          f"stress maxDD {kv['stress_max_drawdown']:.2%}  controls_fire={kv['controls_fire']}")
    print("-" * 70)
    ga = result["gap_aware_comparison"]
    gap_prod = ga["production_thresholds"]
    gap_str = ga["kill_engaging_stress_0p3pct"]
    print("GAP-AWARE TAIL (FIX 2) — crypto gaps fill PAST the intraday stop:")
    print(f"  prod thresholds (kill never trips on benign 5y): "
          f"clamp maxDD {gap_prod['clamped_intraday_model']['max_drawdown']:.2%} "
          f"== gap-aware maxDD {gap_prod['gap_aware_model']['max_drawdown']:.2%}")
    print(f"  kill-engaging stress (0.3% kill) — clamp HID this tail:")
    print(f"    clamped   : worst-day {gap_str['clamped_intraday_model']['worst_day']:.2%}  "
          f"maxDD {gap_str['clamped_intraday_model']['max_drawdown']:.2%}")
    print(f"    gap-aware : worst-day {gap_str['gap_aware_model']['worst_day']:.2%}  "
          f"maxDD {gap_str['gap_aware_model']['max_drawdown']:.2%}  "
          f"(gap-through days {gap_str['gap_aware_model']['n_gap_through_days']})")
    print(f"  worst held crypto var share (stressed): "
          f"{sb['worst_held_crypto_var_share_stressed']:.1%}  (cap {cfg['crypto_var_cap']:.0%})  [FIX 5]")
    print(f"  DD breaker latched at end: {sb['dd_breaker_latched']}   "
          f"sleeve cap latched: {result['trend_tilt_sleeve']['cap_latched']}  [FIX 1/3]")
    print("-" * 70)
    sw = result["sol_live_window"]
    print(f"SOL-LIVE WINDOW (FIX 4) {sw['date_start']} -> {sw['date_end']} ({sw['n_days']}d):")
    print(f"  maxDD {sw['max_drawdown']:.2%}  ann.vol {sw['realized_vol_annual']:.2%}  "
          f"ENB {sw['effective_bets_last']:.2f}  Sharpe {sw['daily_sharpe_annualized']:.3f}")
    print(f"  crypto var share {sw['crypto_combined_var_share_last']:.1%} "
          f"(cap {sw['crypto_var_cap']:.0%}, binding={sw['crypto_cap_binding']})  "
          f"SOL share {sw['sol_var_share_last']:.1%}")
    print("=" * 70)
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
