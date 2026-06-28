"""Primary signals + primary state + triple-barrier labels.

Two stages of the López-de-Prado pipeline live here:
 - Stage 1: primary signals (ema_crossover_signal, momentum_zscore_signal,
            vix_regime_riskflow_signal, bollinger_meanrev_signal,
            cusum_filter_signal) and their derived state (compute_primary_state).
 - Stage 2: triple_barrier_labels — the meta-label producer.

Primary contract (CLAUDE.md invariant)
--------------------------------------
Primaries are **deterministic, rule-based** functions returning a
pd.Series in {-1, 0, +1}. They are NOT fitted classifiers — no `.fit()`
step, no ML training on the same `tier2_features` the meta sees
downstream. This avoids the "squeeze the orange twice" failure mode
documented in the QuantConnect meta-labeling thread (cached at
cache/blogs/qc_meta_labeling.txt): an ML primary trained on the same
data as the meta cannot, in expectation, benefit from meta-labeling.
The only validated regime is `rule-based primary + ML meta on richer
features` — exactly what H&T's reference architecture uses.

Adding a new primary:
  1. Write a deterministic function. Inputs are typically `(close, atr)`
     plus optional macro series. The dispatched signature in
     `scripts/run_xau_d1.py::_select_primary` is `(ohlcv, features, cfg)
     -> pd.Series` for `phase5_*` customs.
  2. Use only deterministic operations (rolling/EMA/comparison/sign).
     No `.fit()`, no `import sklearn`, no `np.random` without a fixed seed.
  3. Register it in `tests/test_primary_input_contract.py::REGISTERED_PRIMARIES`
     so the determinism + no-fit invariant is checked.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


def ema_crossover_signal(
    close: pd.Series,
    atr: pd.Series,
    fast: int = 20,
    slow: int = 50,
    dead_zone_atr: float = 0.25,
) -> pd.Series:
    """Return primary signal in {-1, 0, +1}: +1 if EMA_fast > EMA_slow by > dead_zone_atr * ATR, etc."""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    spread_in_atr = (ema_f - ema_s) / atr.replace(0, np.nan)
    sig = pd.Series(0.0, index=close.index)
    sig[spread_in_atr >= dead_zone_atr] = 1.0
    sig[spread_in_atr <= -dead_zone_atr] = -1.0
    return sig


def momentum_zscore_signal(close: pd.Series, lookback: int = 20, threshold: float = 0.3) -> pd.Series:
    """Return primary signal in {-1, 0, +1} based on z-score of the rolling log return."""
    r = np.log(close / close.shift(lookback))
    mu = r.rolling(252).mean()
    sd = r.rolling(252).std()
    z = (r - mu) / sd
    sig = pd.Series(0.0, index=close.index)
    sig[z > threshold] = 1.0
    sig[z < -threshold] = -1.0
    return sig


# Default USD-beta lookup for symbols in the hackathon universe.
# Positive beta = symbol moves WITH USD (USD strength → symbol up).
# Negative beta = symbol moves AGAINST USD.
_DEFAULT_USD_BETA: dict[str, int] = {
    # FX where USD is the quote currency (XXX/USD) — opposite to USD
    "EURUSD": -1, "GBPUSD": -1, "AUDUSD": -1, "NZDUSD": -1,
    # FX where USD is the base currency (USD/XXX) — with USD
    "USDJPY": +1, "USDCHF": +1, "USDCAD": +1,
    # Precious metals — anti-USD (gold/silver inverse to USD)
    "XAUUSD": -1, "XAGUSD": -1,
    # Crypto — risk assets, anti-USD in risk-on regimes
    "BTCUSD": -1, "ETHUSD": -1, "SOLUSD": -1,
}


def vix_regime_riskflow_signal(
    target_close: pd.Series,
    vix_daily: pd.Series,
    dxy_daily: pd.Series,
    target_symbol: str,
    vix_lookback: int = 252,
    vix_low_pct: float = 0.25,
    vix_high_pct: float = 0.75,
    dxy_ema_fast: int = 5,
    dxy_ema_slow: int = 20,
    usd_beta_lookup: dict[str, int] | None = None,
) -> pd.Series:
    """VIX-regime + DXY USD-funding-flow primary signal.

    Information axis: macro volatility regime (VIX rolling percentile) + USD
    funding direction (DXY EMA cross). ZERO target-price autocorrelation in
    the trigger — the only target-asset dependence is via the symbol's USD-beta
    sign that maps the directional bias.

    Returns a pd.Series in {-1.0, 0.0, +1.0} with same index as `target_close`.

    Mechanism:
      1. Reindex vix_daily and dxy_daily to target_close's index with ffill,
         then shift(1) bar to prevent look-ahead (the daily value stamped at
         date t can only be used by H4 bars on date t+1 onwards).
      2. vix_pct = vix.rolling(vix_lookback).rank(pct=True).shift(1).
         Classify regime: LOW if < vix_low_pct, HIGH if > vix_high_pct,
         else NEUTRAL.
      3. dxy_dir = sign(EMA(dxy, fast) - EMA(dxy, slow)).shift(1).
      4. In NEUTRAL regime: emit 0 (sparsity by design — quiescent in the
         middle-of-distribution VIX state where macro signals are weakest).
      5. In LOW regime: side = -dxy_dir * usd_beta (risk-on; typical USD
         relationships hold).
      6. In HIGH regime: side = +dxy_dir * usd_beta (risk-off sign flip;
         flight-to-quality flips the usual cross-asset reactions).

    Args:
      target_close: H4 close series for the target asset. Defines output index.
      vix_daily: VIX daily values (FRED VIXCLS). Index ⊆ daily timestamps,
        any tz handled internally.
      dxy_daily: DXY daily values (FRED DTWEXBGS). Same index expectation.
      target_symbol: must be a key in usd_beta_lookup (or _DEFAULT_USD_BETA).
      vix_lookback: rolling window (daily bars) for percentile.
      vix_low_pct, vix_high_pct: thresholds in [0, 1].
      dxy_ema_fast, dxy_ema_slow: EMA spans for DXY direction.
      usd_beta_lookup: optional override of the default {symbol: beta} table.

    Raises:
      KeyError if target_symbol is not in the lookup.
    """
    beta_lookup = usd_beta_lookup if usd_beta_lookup is not None else _DEFAULT_USD_BETA
    if target_symbol not in beta_lookup:
        raise KeyError(
            f"target_symbol={target_symbol!r} not in USD-beta lookup. "
            f"Known symbols: {sorted(beta_lookup.keys())}"
        )
    usd_beta = beta_lookup[target_symbol]

    target_idx = target_close.index

    # Normalize daily series tz: make tz-aware UTC if naive.
    def _to_utc(series: pd.Series) -> pd.Series:
        s = series.copy()
        if s.index.tz is None:
            s.index = pd.to_datetime(s.index, utc=True)
        else:
            s.index = s.index.tz_convert("UTC")
        return s

    vix = _to_utc(vix_daily.dropna())
    dxy = _to_utc(dxy_daily.dropna())

    # Compute rolling percentile rank of VIX on its NATIVE daily index.
    if len(vix) >= 2:
        vix_pct_daily = vix.rolling(vix_lookback).rank(pct=True).shift(1)
    else:
        vix_pct_daily = pd.Series(np.nan, index=vix.index)

    # Compute DXY EMA cross direction on its NATIVE daily index.
    if len(dxy) >= max(dxy_ema_fast, dxy_ema_slow):
        ema_fast = dxy.ewm(span=dxy_ema_fast, adjust=False).mean()
        ema_slow = dxy.ewm(span=dxy_ema_slow, adjust=False).mean()
        dxy_dir_daily = np.sign(ema_fast - ema_slow).shift(1)
    else:
        dxy_dir_daily = pd.Series(np.nan, index=dxy.index)

    # Reindex BOTH to target H4 index via ffill (no look-ahead since they were
    # already shifted on their daily timeline above).
    vix_pct = vix_pct_daily.reindex(target_idx, method="ffill")
    dxy_dir = dxy_dir_daily.reindex(target_idx, method="ffill")

    # Classify regime
    regime = pd.Series(0, index=target_idx, dtype=float)   # 0 = NEUTRAL
    valid = vix_pct.notna()
    regime[valid & (vix_pct < vix_low_pct)] = +1   # LOW
    regime[valid & (vix_pct > vix_high_pct)] = -1  # HIGH

    # Build signal
    sig = pd.Series(0.0, index=target_idx)
    valid_signal = (regime != 0) & dxy_dir.notna() & (dxy_dir != 0)
    if valid_signal.any():
        # For LOW (regime=+1): side = -dxy_dir * usd_beta
        # For HIGH (regime=-1): side = +dxy_dir * usd_beta
        # Combined: side = -regime * dxy_dir * usd_beta
        raw = -regime * dxy_dir * usd_beta
        sig[valid_signal] = np.sign(raw[valid_signal])
    return sig


def bollinger_meanrev_signal(
    close: pd.Series,
    period: int = 20,
    k_stdev: float = 2.0,
) -> pd.Series:
    """Bollinger-band mean-reversion primary.

    Returns +1 when close <= lower_band (oversold, expect mean reversion up),
    -1 when close >= upper_band (overbought, expect mean reversion down),
    0 otherwise. ORTHOGONAL to trend-following primaries: this engine expects
    price to revert, designed for choppy/range regimes (typical of liquid FX).

    Returns a pd.Series in {-1.0, 0.0, +1.0} with same index as `close`.
    """
    middle = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = middle + k_stdev * sd
    lower = middle - k_stdev * sd
    sig = pd.Series(0.0, index=close.index)
    # Only fire where the bands are defined and non-degenerate
    valid = sd.notna() & (sd > 0)
    sig[valid & (close <= lower)] = 1.0
    sig[valid & (close >= upper)] = -1.0
    return sig


def cusum_filter_signal(
    close: pd.Series,
    atr: pd.Series,
    threshold_atr: float = 2.0,
) -> pd.Series:
    """CUSUM filter primary (López de Prado, AFML §3.3) — sparse event detector.

    Accumulates positive and negative log-returns separately. When the positive
    accumulator exceeds the threshold, emits +1 and resets it; symmetric for
    the negative side. Naturally produces sparse, regime-aware events:
    quiescent during low-vol/range periods, alive during directional moves.

    Threshold is volatility-adaptive: `threshold_atr * ATR[t-1] / close[t-1]`
    (interpreted as a fraction of price). A fixed threshold misbehaves across
    regimes — too sensitive in high vol, too quiet in low vol.

    Returns a pd.Series in {-1.0, 0.0, +1.0} with the same index as `close`.
    """
    n = len(close)
    close_v = close.values.astype(float)
    atr_v = atr.values.astype(float)
    # log-return at t: log(close[t] / close[t-1])
    log_ret = np.zeros(n, dtype=float)
    log_ret[1:] = np.log(close_v[1:] / close_v[:-1])
    # Threshold at t: threshold_atr * ATR[t-1] / close[t-1], else NaN
    threshold = np.full(n, np.nan, dtype=float)
    threshold[1:] = threshold_atr * atr_v[:-1] / close_v[:-1]

    sig = np.zeros(n, dtype=float)
    s_pos = 0.0
    s_neg = 0.0
    for t in range(1, n):
        r = log_ret[t]
        thr = threshold[t]
        if not np.isfinite(r) or not np.isfinite(thr) or thr <= 0.0:
            continue
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if s_pos >= thr:
            sig[t] = 1.0
            s_pos = 0.0
        elif s_neg <= -thr:
            sig[t] = -1.0
            s_neg = 0.0
    return pd.Series(sig, index=close.index)


def compute_primary_state(side: pd.Series, cap: int = 60) -> pd.DataFrame:
    """Given a primary side series (already filtered to side != 0), compute state features.

    Returns a DataFrame with columns:
      - primary_side: identical to input
      - bars_since_signal: integer count of bars since side last changed, capped at `cap`
    """
    if side.empty:
        return pd.DataFrame(columns=["primary_side", "bars_since_signal"], index=side.index)
    group = (side.diff().fillna(side.iloc[0]) != 0).cumsum()
    bars_since = side.groupby(group).cumcount()
    return pd.DataFrame(
        {"primary_side": side.astype(int), "bars_since_signal": bars_since.clip(upper=cap).astype(int)},
        index=side.index,
    )


@dataclass(frozen=True)
class TripleBarrierEvent:
    t_idx: int           # integer position of event in the OHLC frame
    t_end_idx: int       # integer position where outcome was resolved (or horizon expiry)
    side: int            # +1 (long) or -1 (short)
    label: int           # 1 = TP first; 0 = SL first OR timeout
    exit_price: float    # realized fill price (gap-aware)


def triple_barrier_labels(
    ohlc: pd.DataFrame,
    events: pd.DataFrame,   # indexed by timestamp, column "side" ∈ {-1, +1}
    atr: pd.Series,
    horizon: int,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
) -> pd.DataFrame:
    """Compute triple-barrier labels with explicit long/short asymmetry and gap-aware exit_price.

    Returns a DataFrame indexed by event timestamp with columns:
      side, label, t_end_idx, exit_price.

    exit_price semantics:
      - Intrabar TP/SL touch (no gap): exit_price = TP (label=1) or SL (label=0).
      - Gap through TP/SL: exit_price = open[t_end_idx] (slippage realism).
      - Timeout: exit_price = close[t_end_idx].
    """
    high = ohlc["high"].values
    low = ohlc["low"].values
    close = ohlc["close"].values
    open_ = ohlc["open"].values
    atr_v = atr.reindex(ohlc.index).values
    idx_pos = {ts: i for i, ts in enumerate(ohlc.index)}

    out_rows = []
    n = len(ohlc)
    for ts, row in events.iterrows():
        side = int(row["side"])
        i = idx_pos[ts]
        entry = close[i]
        # TP/SL formulas (long → TP above, SL below; short → TP below, SL above):
        tp = entry + side * tp_mult * atr_v[i]
        sl = entry - side * sl_mult * atr_v[i]
        end_cap = min(i + horizon, n - 1)
        label = 0
        t_end_idx = end_cap
        exit_price = close[end_cap]  # default: timeout exits at the close
        for k in range(i + 1, end_cap + 1):
            if side == 1:
                if open_[k] >= tp:    # gap-up beyond TP
                    label, t_end_idx, exit_price = 1, k, open_[k]; break
                if open_[k] <= sl:    # gap-down beyond SL
                    label, t_end_idx, exit_price = 0, k, open_[k]; break
                hit_tp = high[k] >= tp
                hit_sl = low[k] <= sl
            else:  # short
                if open_[k] <= tp:    # gap-down beyond short-TP
                    label, t_end_idx, exit_price = 1, k, open_[k]; break
                if open_[k] >= sl:    # gap-up beyond short-SL
                    label, t_end_idx, exit_price = 0, k, open_[k]; break
                hit_tp = low[k] <= tp
                hit_sl = high[k] >= sl
            if hit_tp and hit_sl:
                # Conservative tie: SL wins, exit at SL level.
                label, t_end_idx, exit_price = 0, k, sl; break
            if hit_tp:
                label, t_end_idx, exit_price = 1, k, tp; break
            if hit_sl:
                label, t_end_idx, exit_price = 0, k, sl; break
        out_rows.append({
            "side": side,
            "label": label,
            "t_end_idx": t_end_idx,
            "exit_price": exit_price,
        })

    return pd.DataFrame(out_rows, index=events.index)
