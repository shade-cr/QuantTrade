"""Phase 5 deterministic regime labeler.

Produces a 4-class regime label per bar for a given asset+frequency. The
classifier is rule-based, point-in-time, asset-agnostic.

Two axes, binarized:

  trend axis: +1 if (63-bar ROC > 0) AND (50-bar MA > 200-bar MA)
              -1 if both inverted
               0 otherwise (sticky — reassigned to prior label)
  vol axis: HIGH if realized 20-bar log-return vol >= 75th pct of trailing
            5-year window; LOW otherwise.

Cross-product -> 4 regimes:
  BULL_QUIET, BULL_STRESSED, BEAR_QUIET, BEAR_STRESSED.

Frequency-aware: bar counts scale by `bars_per_day` (D1: 1, H4: 6).
The 5-year vol-percentile window is computed as `bars_per_year * 5`.

PIT discipline: every rolling stat uses `min_periods=window` (no warmup
NaN-fill), and the resulting label series has NaN before the trailing
windows are full. Sticky transitions look ONLY backward.

CLI:
  uv run python -m pipeline.regimes --asset XAUUSD --frequency D1 --out data/regimes/

The output parquet is keyed on the timestamp and has columns:
  regime_id, trend_axis, vol_axis, roc_63, ma_50, ma_200, rv_20, rv_75pct_5y
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REGIMES = ("BULL_QUIET", "BULL_STRESSED", "BEAR_QUIET", "BEAR_STRESSED")

# Bar multipliers per frequency relative to D1
FREQ_BARS_PER_DAY = {"D1": 1, "H4": 6}
FREQ_BARS_PER_YEAR = {"D1": 252, "H4": 252 * 6}

# Asset-class to default data dir
DEFAULT_DATA_DIRS = {
    "D1": "data/D1_22y",   # prefer 22y CSVs when present, else fall back
    "H4": "data/H4",
}
DEFAULT_DATA_DIRS_FALLBACK = {
    "D1": "data/D1",
}


def _scale_bars(d1_bars: int, frequency: str) -> int:
    """Scale a D1-bar window to the given frequency."""
    return d1_bars * FREQ_BARS_PER_DAY[frequency]


def label_regimes(
    close: pd.Series,
    frequency: str = "D1",
    *,
    trend_roc_lookback_d1: int = 63,
    trend_ma_fast_d1: int = 50,
    trend_ma_slow_d1: int = 200,
    vol_lookback_d1: int = 20,
    vol_pct_window_years: float = 5,
    vol_enter_high_pct: float = 0.80,
    vol_exit_high_pct: float = 0.70,
    min_dwell_d1: int = 40,
) -> pd.DataFrame:
    """Compute per-bar regime labels for a close-price series.

    Returns a DataFrame indexed identically to ``close`` with columns:
      regime_id        — string in REGIMES (NaN until trailing windows full)
      trend_axis       — int in {-1, 0, +1} (raw, before sticky reassignment)
      vol_axis         — string in {"HIGH", "LOW"} (NaN until vol-pct window full)
      roc_63           — rolling ROC at the configured lookback
      ma_50, ma_200    — rolling means (at configured lookbacks)
      rv_20            — rolling realized vol (annualized)
      rv_enter_pct     — trailing percentile threshold for entering HIGH (default 80th pct)
      rv_exit_pct      — trailing percentile threshold for exiting HIGH (default 70th pct)

    Hysteresis on the vol axis: the system enters HIGH when rv crosses
    above ``rv_enter_pct``, and exits HIGH only when rv falls below
    ``rv_exit_pct``. This prevents single-bar noise around the threshold
    from flipping the regime label.

    Minimum-dwell on the regime label: any new candidate label must persist
    for ``min_dwell_d1`` bars (scaled to frequency) before being accepted.
    Sub-dwell flips are kept at the prior stable label. Together with
    hysteresis, this targets the sanity bound "no episode <60 trading days".

    PIT: all rolling stats use the SAME timestamp as the right edge.
    Sticky transitions are forward-only and look only at previously labeled bars.
    """
    if frequency not in FREQ_BARS_PER_DAY:
        raise ValueError(f"Unsupported frequency {frequency!r}; want one of {list(FREQ_BARS_PER_DAY)}")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must have a DatetimeIndex")
    if not (0 < vol_exit_high_pct < vol_enter_high_pct < 1):
        raise ValueError(
            f"Need 0 < vol_exit_high_pct < vol_enter_high_pct < 1; "
            f"got exit={vol_exit_high_pct}, enter={vol_enter_high_pct}"
        )

    roc_n = _scale_bars(trend_roc_lookback_d1, frequency)
    ma_fast_n = _scale_bars(trend_ma_fast_d1, frequency)
    ma_slow_n = _scale_bars(trend_ma_slow_d1, frequency)
    vol_n = _scale_bars(vol_lookback_d1, frequency)
    vol_pct_n = int(FREQ_BARS_PER_YEAR[frequency] * vol_pct_window_years)
    min_dwell_n = _scale_bars(min_dwell_d1, frequency)

    log_close = np.log(close.astype(float))
    log_ret = log_close.diff()

    # Trend axis components
    roc = log_close.diff(roc_n)
    ma_fast = close.rolling(ma_fast_n, min_periods=ma_fast_n).mean()
    ma_slow = close.rolling(ma_slow_n, min_periods=ma_slow_n).mean()

    trend_up = (roc > 0) & (ma_fast > ma_slow)
    trend_dn = (roc < 0) & (ma_fast < ma_slow)
    trend_axis = pd.Series(0, index=close.index, dtype="int8")
    trend_axis = trend_axis.mask(trend_up, 1).mask(trend_dn, -1)
    not_computable = roc.isna() | ma_fast.isna() | ma_slow.isna()
    trend_axis = trend_axis.where(~not_computable, other=np.nan)

    # Vol axis with hysteresis: rolling pct-thresholds, two-state machine
    bars_per_year = FREQ_BARS_PER_YEAR[frequency]
    rv = log_ret.rolling(vol_n, min_periods=vol_n).std(ddof=0) * np.sqrt(bars_per_year)
    rv_enter = rv.rolling(vol_pct_n, min_periods=vol_pct_n).quantile(vol_enter_high_pct)
    rv_exit = rv.rolling(vol_pct_n, min_periods=vol_pct_n).quantile(vol_exit_high_pct)

    vol_axis = pd.Series(np.nan, index=close.index, dtype=object)
    state: str | None = None
    for ts, rv_val, enter_v, exit_v in zip(close.index, rv.values, rv_enter.values, rv_exit.values):
        if pd.isna(rv_val) or pd.isna(enter_v) or pd.isna(exit_v):
            continue
        if state is None:
            # Initialize from first computable bar using the symmetric pct (midpoint)
            state = "HIGH" if rv_val >= (enter_v + exit_v) / 2 else "LOW"
        elif state == "LOW" and rv_val >= enter_v:
            state = "HIGH"
        elif state == "HIGH" and rv_val <= exit_v:
            state = "LOW"
        vol_axis.loc[ts] = state

    # Combine into 4-class label with sticky transitions on trend_axis=0
    raw_regime = pd.Series(np.nan, index=close.index, dtype=object)
    last_label: str | None = None
    for ts, tval, vval in zip(close.index, trend_axis.values, vol_axis.values):
        if pd.isna(tval) or vval is None or (isinstance(vval, float) and np.isnan(vval)):
            continue
        if tval == 1:
            label = "BULL_STRESSED" if vval == "HIGH" else "BULL_QUIET"
        elif tval == -1:
            label = "BEAR_STRESSED" if vval == "HIGH" else "BEAR_QUIET"
        else:  # tval == 0: sticky — reuse prior label, else stay NaN until first signal
            if last_label is None:
                continue
            label = last_label
        raw_regime.loc[ts] = label
        last_label = label

    # Minimum-dwell smoothing: any new candidate label must persist for >=
    # min_dwell_n consecutive bars before being accepted. Sub-dwell flips
    # are absorbed by holding the prior stable label.
    regime_id = pd.Series(np.nan, index=close.index, dtype=object)
    confirmed: str | None = None
    pending: str | None = None
    pending_count = 0
    raw_values = raw_regime.values
    for i, ts in enumerate(close.index):
        candidate = raw_values[i]
        if candidate is None or (isinstance(candidate, float) and np.isnan(candidate)):
            continue
        if confirmed is None:
            # Bootstrap: first candidate label that persists for >= min_dwell_n bars
            if candidate == pending:
                pending_count += 1
            else:
                pending = candidate
                pending_count = 1
            if pending_count >= min_dwell_n:
                confirmed = pending
                # Backfill: assign the dwell window itself
                start = max(0, i - pending_count + 1)
                for j in range(start, i + 1):
                    regime_id.iloc[j] = confirmed
                pending = None
                pending_count = 0
            continue
        if candidate == confirmed:
            regime_id.iloc[i] = confirmed
            pending = None
            pending_count = 0
        else:
            if candidate == pending:
                pending_count += 1
            else:
                pending = candidate
                pending_count = 1
            if pending_count >= min_dwell_n:
                # Promote pending to confirmed; backfill the dwell window
                confirmed = pending
                start = max(0, i - pending_count + 1)
                for j in range(start, i + 1):
                    regime_id.iloc[j] = confirmed
                pending = None
                pending_count = 0
            else:
                # Hold the prior confirmed label through the sub-dwell flip
                regime_id.iloc[i] = confirmed

    out = pd.DataFrame(
        {
            "regime_id": regime_id,
            "trend_axis": trend_axis,
            "vol_axis": vol_axis,
            "roc_63": roc,
            "ma_50": ma_fast,
            "ma_200": ma_slow,
            "rv_20": rv,
            "rv_enter_pct": rv_enter,
            "rv_exit_pct": rv_exit,
        },
        index=close.index,
    )
    return out


def regime_episodes(regimes: pd.Series) -> pd.DataFrame:
    """Collapse a per-bar regime series into episode boundaries.

    Returns a DataFrame with columns: regime_id, start_ts, end_ts, n_bars.
    NaN regions split episodes (so a NaN gap produces two distinct episodes
    of the same regime).
    """
    s = regimes.dropna()
    if s.empty:
        return pd.DataFrame(columns=["regime_id", "start_ts", "end_ts", "n_bars"])
    # Episode = contiguous run of identical regime_id
    grp = (s != s.shift()).cumsum()
    rows = []
    for _, chunk in s.groupby(grp):
        rows.append(
            {
                "regime_id": chunk.iloc[0],
                "start_ts": chunk.index[0],
                "end_ts": chunk.index[-1],
                "n_bars": len(chunk),
            }
        )
    return pd.DataFrame(rows)


def sanity_report(df: pd.DataFrame) -> dict:
    """Compute the sanity checks from SKILL.md.

    Returns: {
        "regime_counts": {regime: n_bars},
        "regime_fractions": {regime: n_bars / total_labeled},
        "n_episodes": int,
        "min_episode_bars": int,
        "regimes_below_5pct": [regime, ...],
        "episodes_below_60_bars": int,
    }
    """
    s = df["regime_id"].dropna()
    eps = regime_episodes(s)
    counts = s.value_counts().to_dict()
    total = int(s.shape[0])
    # B0152: 0 labeled bars (history shorter than the warmup windows) must be
    # an explicit, readable verdict — not a ZeroDivisionError.
    if total == 0:
        return {
            "regime_counts": {r: 0 for r in REGIMES},
            "regime_fractions": {r: float("nan") for r in REGIMES},
            "n_episodes": 0,
            "min_episode_bars": 0,
            "regimes_below_5pct": list(REGIMES),
            "episodes_below_60_bars": 0,
            "insufficient_history": (
                "0 labeled bars: the input series is shorter than the regime "
                "warmup windows (ma_200 + rv 5y percentile). Check the data "
                "path — this is the stale-stub signature (B0152)."
            ),
        }
    fractions = {r: counts.get(r, 0) / total for r in REGIMES}
    return {
        "regime_counts": {r: int(counts.get(r, 0)) for r in REGIMES},
        "regime_fractions": fractions,
        "n_episodes": int(len(eps)),
        "min_episode_bars": int(eps["n_bars"].min()) if not eps.empty else 0,
        "regimes_below_5pct": [r for r, f in fractions.items() if f < 0.05],
        "episodes_below_60_bars": int((eps["n_bars"] < 60).sum()) if not eps.empty else 0,
    }


def _count_csv_rows(path: Path) -> int:
    """Cheap line count (header included) — enough to compare archive depth."""
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def _resolve_data_path(asset: str, frequency: str, explicit: str | None) -> Path:
    """Resolve the input CSV, preferring the file with MORE bars.

    B0152: data/D1_22y can contain stale short stubs (GBPUSD_D1.csv was a
    1310-bar 5y stub next to the 27-year data/D1 file). A directory-priority
    rule silently picks the stub, labels 0 bars, and crashes sanity_report.
    When both candidates exist, the deeper archive wins (ties -> primary dir,
    preserving the historical 22y preference for metals).
    """
    if explicit:
        return Path(explicit)
    primary = Path(DEFAULT_DATA_DIRS[frequency]) / f"{asset}_{frequency}.csv"
    fallback_dir = DEFAULT_DATA_DIRS_FALLBACK.get(frequency)
    fallback = Path(fallback_dir) / f"{asset}_{frequency}.csv" if fallback_dir else None
    if primary.exists() and fallback is not None and fallback.exists():
        chosen = primary if _count_csv_rows(primary) >= _count_csv_rows(fallback) else fallback
        if chosen != primary:
            print(f"NOTE: {primary} exists but {fallback} has more bars; using {fallback}",
                  flush=True)
        return chosen
    if primary.exists():
        return primary
    if fallback is not None and fallback.exists():
        return fallback
    raise FileNotFoundError(f"No data CSV found for {asset} {frequency} (looked in {DEFAULT_DATA_DIRS[frequency]}, {DEFAULT_DATA_DIRS_FALLBACK.get(frequency)})")


def main() -> int:
    from pipeline.data import load_dataset  # local import: avoids cycle at import time

    ap = argparse.ArgumentParser(description="Phase 5 regime labeler")
    ap.add_argument("--asset", required=True, help="e.g. XAUUSD, XAGUSD, BTCUSD")
    ap.add_argument("--frequency", choices=list(FREQ_BARS_PER_DAY), default="D1")
    ap.add_argument("--data-path", default=None, help="explicit CSV path; else infer from data/D1_22y or data/D1 or data/H4")
    ap.add_argument("--out", default="data/regimes/", help="output directory for the regime parquet")
    ap.add_argument("--print-sanity", action="store_true", help="print the sanity report to stdout")
    args = ap.parse_args()

    data_path = _resolve_data_path(args.asset, args.frequency, args.data_path)
    print(f"Loading {data_path}", flush=True)
    df = load_dataset(data_path)
    print(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}", flush=True)

    regimes = label_regimes(df["close"], frequency=args.frequency)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.asset}_{args.frequency.lower()}_regimes.parquet"
    regimes.to_parquet(out_path)
    print(f"Wrote {out_path} ({regimes['regime_id'].notna().sum()} labeled bars / {len(regimes)} total)", flush=True)

    rep = sanity_report(regimes)
    print(f"Sanity report:", flush=True)
    print(f"  regime_counts:        {rep['regime_counts']}", flush=True)
    print(f"  regime_fractions:     {{ {', '.join(f'{r}: {f:.3f}' for r, f in rep['regime_fractions'].items())} }}", flush=True)
    print(f"  n_episodes:           {rep['n_episodes']}", flush=True)
    print(f"  min_episode_bars:     {rep['min_episode_bars']}", flush=True)
    print(f"  regimes_below_5pct:   {rep['regimes_below_5pct']}", flush=True)
    print(f"  episodes_below_60:    {rep['episodes_below_60_bars']}", flush=True)

    # B0152: 0 labeled bars is a hard failure for refresh scripts — the old
    # behavior wrote an all-NaN parquet and crashed only at sanity print.
    if rep.get("insufficient_history"):
        print(f"  INSUFFICIENT HISTORY: {rep['insufficient_history']}", flush=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
