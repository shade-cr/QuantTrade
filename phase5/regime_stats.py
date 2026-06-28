"""Per-regime aggregate dossier builder (quantile-encoded, no dates).

For an asset's regime parquet and feature frame, build one JSON dossier per
regime that the phase5-hypothesizer agent receives as its sole input. The
dossier is the structural firewall: it exposes regime-conditional aggregate
statistics in quantile-encoded form (no absolute values, no calendar dates)
so that the LLM hypothesizer cannot pattern-match a specific historical
episode.

Output: signals/regime_stats/<asset>/<regime_id>.json

The schema is defined in .claude/skills/phase5-regime-methodology/SKILL.md
§ "Regime stats dossier schema". Key fields:
  - sample_sufficient: True iff (n_bars>=200) AND (n_episodes>=3) AND
    (fraction_of_total_bars>=0.05). When False, the orchestrator tags any
    proposal targeting this regime as diagnostic_only.
  - features_quantile_summary: per-feature quantile rank IN this regime,
    plus a categorical comparison to other regimes' median.
  - regime_episode_ordinals: list of episode INDICES in this regime
    (NOT dates) — for the hypothesizer's mandatory shape attestation.
  - n_unlabeled_bars: bars where regime_id is NaN (burn-in window).

CLI:
  uv run python -m phase5.regime_stats --asset XAUUSD --frequency D1 \\
      --features-source config --out signals/regime_stats/
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pipeline.regimes import regime_episodes, REGIMES, FREQ_BARS_PER_DAY, FREQ_BARS_PER_YEAR
from pipeline.features import _atr
from pipeline.labels import (
    ema_crossover_signal,
    momentum_zscore_signal,
    bollinger_meanrev_signal,
    cusum_filter_signal,
    triple_barrier_labels,
)


SAMPLE_SUFFICIENT_BARS_MIN = 200
SAMPLE_SUFFICIENT_EPISODES_MIN = 3
SAMPLE_SUFFICIENT_FRACTION_MIN = 0.05

# B0036: regime-defining features (kept in sync with anti-circularity rule
# in phase5/lookahead_lint.py REGIME_DEFINING_FEATURES). Used to populate
# the dossier's `orthogonal_features` list and to compute
# `correlation_with_regime_features` for new features.
REGIME_DEFINING_FEATURES = ("roc_63", "ma_50", "ma_200", "rv_20")

# B0068 Tier-1: macro-exogenous feature names injected into the dossier for non-fx assets.
_MACRO_MEMBERS = (
    "vix_level", "vix_chg_5", "dxy_z252",
    "real_yield_5y_z252d", "breakeven_5y_chg5", "us_5y2y_z252",
)

# B0047: |spearman rho| above which a feature is "substantially collinear" with a
# regime-defining feature -> quasi_circular=True AND excluded from orthogonal_features.
# 0.35 catches the |0.485| real_yield case with margin without over-flagging mild
# (0.2-0.3) correlations. Was an inline 0.5, which let |0.485| pass as orthogonal.
QUASI_CIRCULAR_RHO = 0.35

# B0036 MVP scope: orthogonal alt-data features added to the dossier.
# Each entry maps feature_name → (cadence_label, effective_n_divisor).
# effective_n_divisor: D1 bar count is divided by this to estimate
# independent observations (1 = daily, 5 = weekly, 21 = monthly).
B0036_MVP_FEATURES = {
    "cot_net_noncomm_z52w": ("weekly", 5),
    "real_yield_5y_z252d": ("daily", 1),
    "vix_level": ("daily", 1),
    "vix_chg_5": ("daily", 1),
    "dxy_z252": ("daily", 1),
    "breakeven_5y_chg5": ("daily", 1),
    "us_5y2y_z252": ("daily", 1),
    # B0135: per-bar estimate but beta is smoothed over a 21-bar window ->
    # neighbouring values are heavily dependent; divisor 5 is the conservative
    # effective-n call (same as weekly-cadence COT).
    "cs_spread_21": ("smoothed_21bar", 5),
    # B0147: GLD real-volume block (metals packs only). dvol z has a fresh
    # daily innovation (divisor 1); the amihud z carries a 21d inner mean.
    "gld_dvol_z42": ("daily", 1),
    "gld_amihud_z252": ("smoothed_21d", 5),
}


# --- B0057 primary baseline ---------------------------------------------------
# A primary is rule-based and deterministic, so its UNFILTERED in-regime
# performance is measurable at dossier-build time (no OOF). These four OHLCV-only
# primaries form a canonical, asset-agnostic baseline panel. Each adapter maps the
# (ohlc, atr) call convention onto the primary's own signature.
BASELINE_TP_MULT = 3.0            # matches the proposal/audit default barrier (B0046 like-for-like)
BASELINE_SL_MULT = 1.0
BASELINE_HORIZON_D1 = 20          # holding horizon in D1 bars; scaled by FREQ_BARS_PER_DAY
MIN_BASELINE_EVENTS = 30          # project-wide competence floor (== min_trades_per_fold; B0036 effective_n<30)

BASELINE_PANEL = (
    ("ema_crossover", lambda ohlc, atr: ema_crossover_signal(ohlc["close"], atr)),
    ("momentum_zscore", lambda ohlc, atr: momentum_zscore_signal(ohlc["close"])),
    ("bollinger_meanrev", lambda ohlc, atr: bollinger_meanrev_signal(ohlc["close"])),
    ("cusum_filter", lambda ohlc, atr: cusum_filter_signal(ohlc["close"], atr)),
)


def _primary_raw_metrics(
    ohlc: pd.DataFrame,
    atr: pd.Series,
    regimes_df: pd.DataFrame,
    sig: pd.Series,
    frequency: str,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
) -> dict[str, dict]:
    """Raw (un-encoded) baseline metrics per regime for ONE primary's signal.

    Runs the signal's non-zero entries through the triple barrier, buckets each
    event to the regime of its ENTRY bar (PIT: the entry-bar label is known at
    entry), and returns {regime_id: {n_events, trade_count_per_year, hit_rate,
    median_per_trade_return}}. Metrics are None for a regime with zero events.
    Events entered on an unlabeled (NaN-regime, burn-in) bar are dropped — not
    attributed to any regime — so per-regime n_events sums to <= total signals.
    """
    bars_per_year = FREQ_BARS_PER_YEAR[frequency]
    out = {
        r: {"n_events": 0, "trade_count_per_year": None,
            "hit_rate": None, "median_per_trade_return": None}
        for r in REGIMES
    }
    entries = sig[sig != 0]
    if entries.empty:
        return out
    events = pd.DataFrame({"side": entries.astype(int)})
    labels = triple_barrier_labels(ohlc, events, atr, horizon, tp_mult, sl_mult)
    entry_close = ohlc["close"].reindex(labels.index)
    per_trade_ret = np.log(labels["exit_price"] / entry_close) * labels["side"]
    entry_regime = regimes_df["regime_id"].reindex(labels.index)
    for r in REGIMES:
        mask = (entry_regime == r).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        in_regime_bars = int((regimes_df["regime_id"] == r).sum())
        years = max(in_regime_bars / bars_per_year, 1e-9)
        out[r] = {
            "n_events": n,
            "trade_count_per_year": float(n / years),
            "hit_rate": float((labels["label"].to_numpy()[mask] == 1).mean()),
            "median_per_trade_return": float(np.median(per_trade_ret.to_numpy()[mask])),
        }
    return out


def _encode_primary_across_regimes(raw_by_regime: dict[str, dict]) -> dict[str, dict]:
    """Encode one primary's raw per-regime metrics into firewall-safe ranks.

    Each metric is quantile-ranked across the regimes that have a non-null value;
    if fewer than 2 qualify the rank is None (unrankable) and rankable=False. B1:
    low_confidence reflects ONLY n_events < MIN_BASELINE_EVENTS — unrankable does
    NOT make a regime low-confidence (a regime-specialist primary is measured).
    """
    metric_keys = ("trade_count_per_year", "hit_rate", "median_per_trade_return")
    q_names = {
        "trade_count_per_year": "trade_count_per_year_q",
        "hit_rate": "hit_rate_q",
        "median_per_trade_return": "median_per_trade_return_q",
    }
    value_sets = {
        m: {r: raw_by_regime[r][m] for r in REGIMES if raw_by_regime[r][m] is not None}
        for m in metric_keys
    }

    def _tag(metric: str, regime: str) -> str:
        vals = value_sets[metric]
        my = raw_by_regime[regime][metric]
        others = [v for k, v in vals.items() if k != regime]
        if my is None or not others:
            return "unknown"
        if my > max(others):
            return "higher"
        if my < min(others):
            return "lower"
        return "similar"

    out: dict[str, dict] = {}
    for r in REGIMES:
        enc: dict = {}
        rankable_any_missing = False
        for m in metric_keys:
            vals = value_sets[m]
            my = raw_by_regime[r][m]
            if my is not None and len(vals) >= 2:
                enc[q_names[m]] = round(_quantile_rank_within(pd.Series(list(vals.values())), my), 4)
            else:
                enc[q_names[m]] = None
                rankable_any_missing = True
        enc["hit_rate_vs_other_regimes"] = _tag("hit_rate", r)
        enc["return_vs_other_regimes"] = _tag("median_per_trade_return", r)
        enc["n_events"] = int(raw_by_regime[r]["n_events"])
        # B1: rankability is reported separately and does NOT taint low_confidence.
        enc["rankable"] = not rankable_any_missing
        enc["low_confidence"] = bool(raw_by_regime[r]["n_events"] < MIN_BASELINE_EVENTS)
        out[r] = enc
    return out


def build_primary_baselines(
    ohlc: pd.DataFrame,
    atr: pd.Series,
    regimes_df: pd.DataFrame,
    frequency: str,
    *,
    panel: tuple = BASELINE_PANEL,
    tp_mult: float = BASELINE_TP_MULT,
    sl_mult: float = BASELINE_SL_MULT,
) -> dict[str, dict]:
    """Per-regime primary baseline summary, build-time deterministic.

    Returns {regime_id: {primary_name: encoded_metrics}} ready to drop into each
    dossier's primary_baseline_summary. Firewall-safe: only quantile ranks, raw
    counts, bools, and categorical tags — no absolute prices/returns/dates.
    """
    horizon = BASELINE_HORIZON_D1 * FREQ_BARS_PER_DAY[frequency]
    result: dict[str, dict] = {r: {} for r in REGIMES}
    for name, adapter in panel:
        sig = adapter(ohlc, atr)
        raw = _primary_raw_metrics(
            ohlc, atr, regimes_df, sig, frequency, tp_mult, sl_mult, horizon
        )
        encoded = _encode_primary_across_regimes(raw)
        for r in REGIMES:
            result[r][name] = encoded[r]
    return result


def _quantile_rank_within(series: pd.Series, value: float) -> float:
    """Return the quantile rank (0-1) of `value` within `series`."""
    s = series.dropna()
    if s.empty or not np.isfinite(value):
        return float("nan")
    return float((s <= value).mean())


def _compute_sample_sufficient(n_bars: int, n_episodes: int, frac: float) -> tuple[bool, Optional[str]]:
    reasons = []
    if n_bars < SAMPLE_SUFFICIENT_BARS_MIN:
        reasons.append(f"n_bars={n_bars} < {SAMPLE_SUFFICIENT_BARS_MIN}")
    if n_episodes < SAMPLE_SUFFICIENT_EPISODES_MIN:
        reasons.append(f"n_episodes={n_episodes} < {SAMPLE_SUFFICIENT_EPISODES_MIN}")
    if frac < SAMPLE_SUFFICIENT_FRACTION_MIN:
        reasons.append(f"fraction_of_total_bars={frac:.4f} < {SAMPLE_SUFFICIENT_FRACTION_MIN}")
    if reasons:
        return False, "; ".join(reasons)
    return True, None


def load_cot_net_noncomm_z(
    asset: str,
    target_index: pd.DatetimeIndex,
    z_window_weeks: int = 52,
    publication_lag_days: int = 5,
) -> pd.Series:
    """Load CFTC COT non-commercial net positioning z-score, aligned to D1.

    Publication-lag handling: CFTC publishes Tuesday's data the following
    Friday. To guarantee no lookahead at D1 bar `t`, we treat each row's
    `report_date` as available only from `report_date + publication_lag_days`
    onwards (default 5 days = Sat-Mon after Fri publication, conservative).

    Z-score is computed on the WEEKLY series (52 weekly observations) then
    forward-filled to D1. Computing z on D1 forward-filled values would
    bias the std downward because of within-week zero-variance repeats.

    Returns a series indexed by target_index (UTC), name='cot_net_noncomm_z52w'.
    """
    cache = Path("cache/cot") / f"cot_{asset}.parquet"
    if not cache.exists():
        return pd.Series(np.nan, index=target_index, name="cot_net_noncomm_z52w")
    raw = pd.read_parquet(cache)
    if "net_noncomm" not in raw.columns or "report_date" not in raw.columns:
        return pd.Series(np.nan, index=target_index, name="cot_net_noncomm_z52w")
    raw = raw.copy()
    raw["available_from"] = raw["report_date"] + pd.Timedelta(days=publication_lag_days)
    raw = raw.sort_values("available_from").set_index("available_from")
    weekly_z = (
        (raw["net_noncomm"] - raw["net_noncomm"].rolling(z_window_weeks, min_periods=z_window_weeks).mean())
        / raw["net_noncomm"].rolling(z_window_weeks, min_periods=z_window_weeks).std()
    )
    if weekly_z.index.tz is None:
        weekly_z.index = weekly_z.index.tz_localize("UTC")
    out = weekly_z.reindex(target_index, method="ffill")
    out.name = "cot_net_noncomm_z52w"
    return out


def load_real_yield_z(
    target_index: pd.DatetimeIndex,
    z_window_days: int = 252,
    fred_cache_dir: Path | str = "cache/fred",
) -> pd.Series:
    """Load 5y TIPS real yield (DFII5) z-score, aligned to D1.

    Publication-lag handling: pipeline.macro_fetch.build_macro_frame already
    applies a 1-day shift (FRED stamps date t; published t+1). We mirror
    that here by loading the cached parquet and applying .shift(1) on
    the daily-calendar reindex.

    Returns a series indexed by target_index (UTC), name='real_yield_5y_z252d'.
    """
    cache = Path(fred_cache_dir) / "DFII5.parquet"
    if not cache.exists():
        return pd.Series(np.nan, index=target_index, name="real_yield_5y_z252d")
    s = pd.read_parquet(cache)["DFII5"]
    if s.index.tz is None:
        s.index = pd.to_datetime(s.index).tz_localize("UTC")
    # Reindex to daily calendar matching target_index range, ffill, then shift(1)
    cal_start = target_index.min()
    cal_end = target_index.max()
    cal = pd.date_range(cal_start, cal_end, freq="D", tz="UTC")
    on_cal = s.reindex(cal, method="ffill").shift(1)
    z = (
        (on_cal - on_cal.rolling(z_window_days, min_periods=z_window_days).mean())
        / on_cal.rolling(z_window_days, min_periods=z_window_days).std()
    )
    out = z.reindex(target_index, method="ffill")
    out.name = "real_yield_5y_z252d"
    return out


def _fred_on_daily_cal(code, fred_cache_dir, target_index):
    """Load a cached FRED series onto a daily calendar spanning target_index,
    ffilled and shift(1) for the 1-day publication lag. Returns None if no cache."""
    cache = Path(fred_cache_dir) / f"{code}.parquet"
    if not cache.exists():
        return None
    s = pd.read_parquet(cache)[code]
    if s.index.tz is None:
        s.index = pd.to_datetime(s.index).tz_localize("UTC")
    cal = pd.date_range(target_index.min().normalize(), target_index.max(), freq="D", tz="UTC")
    return s.reindex(cal, method="ffill").shift(1)


def _zscore(s, window):
    return (s - s.rolling(window, min_periods=window).mean()) / s.rolling(window, min_periods=window).std()


def load_macro_pack(target_index, members, fred_cache_dir="cache/fred"):
    """Cache-only, PIT macro feature pack. Each member's transform is computed on a
    daily calendar (so '252' is 252 calendar days regardless of D1/H4), then reindexed
    to target_index. A member whose FRED cache is absent is silently skipped (no NaN
    column). Mirrors load_real_yield_z. NO network."""
    out = pd.DataFrame(index=target_index)

    def daily(code):
        return _fred_on_daily_cal(code, fred_cache_dir, target_index)

    for m in members:
        col = None
        if m == "real_yield_5y_z252d":
            s = load_real_yield_z(target_index, fred_cache_dir=fred_cache_dir)
            col = s if s.notna().any() else None
        elif m == "vix_level":
            d = daily("VIXCLS"); col = None if d is None else d.reindex(target_index, method="ffill")
        elif m == "vix_chg_5":
            d = daily("VIXCLS"); col = None if d is None else d.pct_change(5, fill_method=None).reindex(target_index, method="ffill")
        elif m == "dxy_z252":
            d = daily("DTWEXBGS"); col = None if d is None else _zscore(d, 252).reindex(target_index, method="ffill")
        elif m == "breakeven_5y_chg5":
            d = daily("T5YIE"); col = None if d is None else d.diff(5).reindex(target_index, method="ffill")
        elif m == "us_5y2y_z252":
            a, b = daily("DGS5"), daily("DGS2")
            col = None if (a is None or b is None) else _zscore(a - b, 252).reindex(target_index, method="ffill")
        if col is not None:
            out[m] = col
    return out


def build_dossier_features(
    df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    asset: str,
    feature_pack: tuple[str, ...],
) -> pd.DataFrame:
    """Assemble the feature frame for the dossier: OHLCV + regime indicators +
    the asset's declared alt-feature pack. A declared feature is skipped if its
    cache is absent (no phantom NaN columns). Crypto/fx packs are typically empty.
    """
    features_df = df.copy()
    for col in ("rv_20", "ma_50", "ma_200", "roc_63"):
        if col in regimes_df.columns:
            features_df[col] = regimes_df[col]
    target_idx = features_df.index
    if "cot_net_noncomm_z52w" in feature_pack:
        cot_path = Path("cache/cot") / f"cot_{asset}.parquet"
        if cot_path.exists():
            features_df["cot_net_noncomm_z52w"] = load_cot_net_noncomm_z(asset, target_idx)
    if "real_yield_5y_z252d" in feature_pack:
        if (Path("cache/fred") / "DFII5.parquet").exists():
            features_df["real_yield_5y_z252d"] = load_real_yield_z(target_idx)
    macro_members = [m for m in feature_pack if m in _MACRO_MEMBERS]
    if macro_members:
        mp = load_macro_pack(target_idx, macro_members)
        for col in mp.columns:
            features_df[col] = mp[col]
    # B0135: Corwin-Schultz spread — computed from the asset's OWN high/low
    # (trailing-only), so it needs no cache and works for every asset class.
    if "cs_spread_21" in feature_pack:
        from pipeline.microstructure import corwin_schultz_spread
        features_df["cs_spread_21"] = corwin_schultz_spread(
            df["high"], df["low"], beta_window=21)
    # B0147: GLD real-volume block (declared in metals packs only). Loader
    # applies the PIT calendar shift; absent cache -> silently skipped per
    # this builder's contract (no phantom NaN columns).
    gld_members = [m for m in feature_pack
                   if m in ("gld_dvol_z42", "gld_amihud_z252")]
    if gld_members:
        from pipeline.alt_data.gld_volume import (
            DEFAULT_CACHE_PATH as _GLD_CACHE,
            load_gld_volume_features,
        )
        if _GLD_CACHE.exists():
            gld = load_gld_volume_features(target_idx)
            for m in gld_members:
                features_df[m] = gld[m]
    return features_df


def _compute_correlation_with_regime_features(
    feature_series: pd.Series,
    regime_features_df: pd.DataFrame,
) -> dict[str, float]:
    """Spearman rho of feature_series vs each regime-defining feature.

    Drops NaN rows pairwise. Used to flag semantic circularity (DA high #1).
    """
    out: dict[str, float] = {}
    for col in REGIME_DEFINING_FEATURES:
        if col not in regime_features_df.columns:
            out[col] = float("nan")
            continue
        paired = pd.concat([feature_series, regime_features_df[col]], axis=1).dropna()
        if len(paired) < 30:
            out[col] = float("nan")
            continue
        out[col] = float(paired.iloc[:, 0].corr(paired.iloc[:, 1], method="spearman"))
    return out


def build_regime_dossiers(
    regimes_df: pd.DataFrame,
    features_df: pd.DataFrame,
    asset_class: str,
    primary_baselines: dict[str, dict] | None = None,
    frequency: str = "D1",
) -> dict[str, dict]:
    """Build one dossier per regime for an asset.

    Args:
      regimes_df: output of pipeline.regimes.label_regimes (DatetimeIndex,
        contains 'regime_id' column among others).
      features_df: feature frame aligned to regimes_df.index (DatetimeIndex).
        Each numeric column becomes one entry in features_quantile_summary.
      asset_class: 'fx' | 'metal' | 'crypto' | 'commodity' | 'equity_index'.

    Returns: {regime_id: dossier_dict}.
    """
    # Align frames on the intersection of indices that have a labeled regime
    regimes = regimes_df["regime_id"]
    n_total_labeled = int(regimes.notna().sum())
    n_total = int(len(regimes))
    n_unlabeled = n_total - n_total_labeled

    # All-bars quantile basis: each feature's distribution across labeled bars.
    feat_cols = [c for c in features_df.columns if pd.api.types.is_numeric_dtype(features_df[c])]
    full_idx = regimes.index.intersection(features_df.index)
    feat_full = features_df.loc[full_idx, feat_cols]

    # Episodes for ordinals
    episodes = regime_episodes(regimes)

    dossiers: dict[str, dict] = {}
    # Compute global cross-regime medians for the vs_other_regimes_rank tag
    medians_per_regime: dict[str, pd.Series] = {}
    for r in REGIMES:
        mask_r = regimes.reindex(full_idx) == r
        if mask_r.any():
            medians_per_regime[r] = feat_full.loc[mask_r].median()

    for r in REGIMES:
        mask = regimes.reindex(full_idx) == r
        n_bars = int(mask.sum())
        regime_eps = episodes[episodes["regime_id"] == r].reset_index(drop=True)
        n_episodes = int(len(regime_eps))
        frac = n_bars / max(n_total_labeled, 1)
        sufficient, reason = _compute_sample_sufficient(n_bars, n_episodes, frac)

        # Quantile-encoded feature summary within this regime, compared to all regimes
        features_quantile_summary: dict[str, dict] = {}
        feat_this_regime = feat_full.loc[mask]
        if not feat_this_regime.empty:
            this_median = feat_this_regime.median()
            this_q25 = feat_this_regime.quantile(0.25)
            this_q75 = feat_this_regime.quantile(0.75)
            for col in feat_cols:
                full_col = feat_full[col]
                this_med_q = _quantile_rank_within(full_col, this_median.get(col, float("nan")))
                this_q25_q = _quantile_rank_within(full_col, this_q25.get(col, float("nan")))
                this_q75_q = _quantile_rank_within(full_col, this_q75.get(col, float("nan")))
                # Compare this regime's median to other regimes' medians
                other_medians = [m.get(col) for k, m in medians_per_regime.items()
                                 if k != r and pd.notna(m.get(col))]
                my_median = this_median.get(col, float("nan"))
                if other_medians and pd.notna(my_median):
                    om = pd.Series(other_medians)
                    if my_median > om.max():
                        vs_tag = "higher"
                    elif my_median < om.min():
                        vs_tag = "lower"
                    else:
                        vs_tag = "similar"
                else:
                    vs_tag = "unknown"
                summary = {
                    "median_quantile": this_med_q,
                    "iqr_quantile_low": this_q25_q,
                    "iqr_quantile_high": this_q75_q,
                    "vs_other_regimes_rank": vs_tag,
                }
                # B0036 — for MVP orthogonal features add effective_n + correlation
                if col in B0036_MVP_FEATURES:
                    cadence_label, base_divisor = B0036_MVP_FEATURES[col]
                    eff_divisor = base_divisor * FREQ_BARS_PER_DAY.get(frequency, 1)
                    n_obs_in_regime = int(feat_this_regime[col].notna().sum())
                    effective_n = max(1, n_obs_in_regime // eff_divisor)
                    summary["cadence"] = cadence_label
                    summary["effective_n"] = effective_n
                    summary["low_confidence"] = effective_n < 30
                    # Correlation vs regime-defining features (full sample)
                    corr = _compute_correlation_with_regime_features(feat_full[col], feat_full)
                    summary["correlation_with_regime_features"] = {
                        k: round(v, 3) if pd.notna(v) else None for k, v in corr.items()
                    }
                    summary["quasi_circular"] = any(
                        v is not None and abs(v) > QUASI_CIRCULAR_RHO
                        for v in summary["correlation_with_regime_features"].values()
                    )
                    corr_vals = summary["correlation_with_regime_features"].values()
                    summary["vetting_inconclusive"] = all(v is None for v in corr_vals)
                features_quantile_summary[col] = summary

        # Return distribution (log returns) quantile summary within this regime
        if "close" in features_df.columns:
            log_ret = np.log(features_df["close"]).diff()
            full_ret = log_ret.reindex(full_idx)
            this_ret = full_ret.loc[mask].dropna()
            if not this_ret.empty:
                ret_median_q = _quantile_rank_within(full_ret.dropna(), float(this_ret.median()))
                ret_q25_q = _quantile_rank_within(full_ret.dropna(), float(this_ret.quantile(0.25)))
                ret_q75_q = _quantile_rank_within(full_ret.dropna(), float(this_ret.quantile(0.75)))
                ret_summary = {
                    "median_quantile": ret_median_q,
                    "iqr_low": ret_q25_q,
                    "iqr_high": ret_q75_q,
                }
            else:
                ret_summary = None
        else:
            ret_summary = None

        # B0036 — list features not in REGIME_DEFINING_FEATURES (and not raw OHLC level)
        # as orthogonal hints for the hypothesizer (anti-circularity affordance).
        # B0047: a feature flagged quasi_circular (|rho| > QUASI_CIRCULAR_RHO vs any
        # regime-defining feature) is NOT orthogonal — exclude it from the hint list.
        orthogonal_features = [
            c for c in feat_cols
            if c not in REGIME_DEFINING_FEATURES and c not in ("open", "high", "low", "close")
            and not features_quantile_summary.get(c, {}).get("quasi_circular", False)
            and not features_quantile_summary.get(c, {}).get("vetting_inconclusive", False)
        ]

        dominant_episode_fraction = (
            float(regime_eps["n_bars"].max()) / n_bars if n_bars > 0 and n_episodes > 0 else 0.0
        )

        dossiers[r] = {
            "asset_class": asset_class,
            "regime_id": r,
            "n_bars": n_bars,
            "n_episodes": n_episodes,
            "median_episode_len_bars": int(regime_eps["n_bars"].median()) if n_episodes > 0 else 0,
            "fraction_of_total_bars": round(frac, 4),
            "dominant_episode_fraction": round(dominant_episode_fraction, 4),
            "n_unlabeled_bars": n_unlabeled,
            "sample_sufficient": sufficient,
            "sample_insufficient_reason": reason,
            "features_quantile_summary": features_quantile_summary,
            "orthogonal_features": orthogonal_features,
            "regime_defining_features": list(REGIME_DEFINING_FEATURES),
            "return_distribution_quantile": ret_summary,
            # Per-regime episode ordinals: the global ordinal in `episodes`. The
            # hypothesizer must cite >=2 of these in lookahead_shape_attestation.
            "regime_episode_ordinals": [
                int(episodes[episodes["regime_id"].notna()].index[i])
                for i in range(len(episodes))
                if episodes.iloc[i]["regime_id"] == r
            ],
            # B0057: per-regime primary baseline; {} when not supplied (back-compatible).
            "primary_baseline_summary": (primary_baselines or {}).get(r, {}),
        }

    return dossiers


def main() -> int:
    from pipeline.data import load_dataset
    from pipeline.regimes import _resolve_data_path

    ap = argparse.ArgumentParser(description="Build per-regime dossiers for a labeled asset")
    ap.add_argument("--asset", required=True)
    ap.add_argument("--frequency", choices=("D1", "H4"), default="D1")
    ap.add_argument("--regimes-path", default=None,
                    help="path to <asset>_<freq>_regimes.parquet (defaults to data/regimes/)")
    ap.add_argument("--data-path", default=None,
                    help="path to OHLCV CSV (defaults to data/D1_22y/<asset>.csv or data/D1/<asset>.csv)")
    ap.add_argument("--asset-class", required=True,
                    choices=("fx", "metal", "crypto", "commodity", "equity", "equity_index"))
    ap.add_argument("--out", default="signals/regime_stats/")
    args = ap.parse_args()

    regimes_path = (
        Path(args.regimes_path)
        if args.regimes_path
        else Path("data/regimes") / f"{args.asset}_{args.frequency.lower()}_regimes.parquet"
    )
    if not regimes_path.exists():
        print(f"ERROR: regimes parquet not found at {regimes_path}", flush=True)
        print(f"Run: uv run python -m pipeline.regimes --asset {args.asset} --frequency {args.frequency}", flush=True)
        return 1

    data_path = _resolve_data_path(args.asset, args.frequency, args.data_path)
    print(f"Loading OHLCV from {data_path}", flush=True)
    df = load_dataset(data_path)
    print(f"Loading regimes from {regimes_path}", flush=True)
    regimes_df = pd.read_parquet(regimes_path)
    # Build a minimal features frame: include the OHLCV close so we can
    # compute return_distribution_quantile, plus the regime indicators
    # themselves (rv_20, ma_50, ma_200, roc_63) for the features_quantile_summary.
    from phase5.asset_registry import ASSET_REGISTRY
    spec = ASSET_REGISTRY.get(args.asset)
    # B0154: use the WIDENED pack (base + macro members for non-fx; fx pack now
    # declared in the registry). The CLI previously read spec.feature_pack raw,
    # so dossier_feature_pack()'s macro widening was never applied — dossiers
    # carried only COT + real-yield and the hypothesizer was starved of
    # orthogonal drivers (the documented persona-ceiling bottleneck).
    feature_pack = spec.dossier_feature_pack() if spec else ("cot_net_noncomm_z52w", "real_yield_5y_z252d")
    features_df = build_dossier_features(df, regimes_df, asset=args.asset, feature_pack=feature_pack)

    atr = _atr(df["high"], df["low"], df["close"])
    primary_baselines = build_primary_baselines(df, atr, regimes_df, frequency=args.frequency)
    dossiers = build_regime_dossiers(
        regimes_df, features_df, asset_class=args.asset_class,
        primary_baselines=primary_baselines, frequency=args.frequency,
    )
    from phase5.asset_registry import dossier_dirname
    out_dir = Path(args.out) / dossier_dirname(args.asset, args.frequency)
    out_dir.mkdir(parents=True, exist_ok=True)
    for regime_id, dossier in dossiers.items():
        out_path = out_dir / f"{regime_id}.json"
        out_path.write_text(json.dumps(dossier, indent=2, default=str), encoding="utf-8")
        print(
            f"  {regime_id}: n_bars={dossier['n_bars']}  n_episodes={dossier['n_episodes']}  "
            f"sufficient={dossier['sample_sufficient']}  -> {out_path}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
