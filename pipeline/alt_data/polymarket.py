"""Polymarket stitched-series builder + PIT-aligned loader (B0170).

Reads cache/polymarket/ (populated by scripts/ingest_polymarket.py) and builds:

  * build_fomc_stitched  — a continuous daily series from episodic per-meeting
    FOMC markets, rolled futures-style: event k's active (front) window is
    (prev_event_end, event_end]; pm_roll tags the first covered day of every
    non-first event so downstream diffs can mask the mechanical roll jump.
  * build_event_series   — a single standing bucket event (e.g. "how many Fed
    rate cuts in 2026") -> expected-cuts mean + entropy; no roll.
  * load_polymarket_features — PIT-shifted alignment to a target bar index.

PIT discipline (same convention as pipeline/alt_data/gld_holdings.py): a
probability stamped at UTC calendar date t is visible to market bars at
date >= t+1 only — calendar-day shift BEFORE the reindex. One documented
deviation from the GLD loader: the forward-fill is BOUNDED (ffill_limit
calendar days, default 5) because Polymarket gaps are dead zones between
meetings or data outages, not weekends — unbounded ffill would bridge
multi-week gaps with stale probabilities.

Honesty conventions: bucket aggregates (cut/nochange/hike) are NaN on any day
where one of that bucket's legs is missing (never a silent partial sum);
entropy/prob_sum/exp_cuts require ALL legs present; pm_coverage reports the
per-day fraction of legs present; entropy is computed on the NORMALIZED
outcome vector (book mids need not sum to 1) with the raw sum emitted as
prob_sum for data-quality inspection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CACHE_DIR = Path("cache/polymarket")
AGG_BUCKETS = ("cut", "nochange", "hike")


class PolymarketCacheMissing(FileNotFoundError):
    """Raised when the cache/manifest is absent. Run scripts/ingest_polymarket.py."""


# ---------------------------------------------------------------- cache access
def _load_records(cache_dir: Path, name: str) -> list[dict]:
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise PolymarketCacheMissing(
            f"Polymarket manifest not found at {manifest_path}. "
            "Run: uv run python scripts/ingest_polymarket.py --config configs/polymarket.yaml"
        )
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = [r for r in records if r.get("name") == name and r.get("file")]
    if not out:
        raise PolymarketCacheMissing(
            f"No cached markets for target {name!r} in {manifest_path}. "
            "Run: uv run python scripts/ingest_polymarket.py --config configs/polymarket.yaml"
        )
    return out


def _daily_last(parquet_path: Path) -> pd.Series:
    """One market's history resampled to daily (last obs of each UTC day).

    This is the fidelity normalizer: 60-min live legs and 720-min resolved
    legs become comparable daily observations."""
    df = pd.read_parquet(parquet_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df["p"].sort_index().resample("1D").last().dropna()


def _event_end(record: dict) -> pd.Timestamp:
    return pd.to_datetime(record["event_end_date"], utc=True).normalize()


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy of a normalized probability vector; 0*log(0) := 0."""
    total = p.sum()
    if total <= 0:
        return float("nan")
    q = p / total
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(q > 0, q * np.log(q), 0.0)
    return float(-terms.sum())


def _legs_frame(records: list[dict], cache_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """days x token_id frame of daily-last probs + token -> bucket map."""
    series = {}
    buckets = {}
    for r in records:
        series[r["token_id"]] = _daily_last(Path(cache_dir) / r["file"])
        buckets[r["token_id"]] = r["outcome_bucket"]
    return pd.concat(series, axis=1), buckets


def _aggregate_day_rows(legs: pd.DataFrame, buckets: dict[str, str], prefix: str) -> pd.DataFrame:
    """Per-day bucket sums, entropy, prob_sum, coverage for one event's legs."""
    n_legs = legs.shape[1]
    present = legs.notna()
    out = pd.DataFrame(index=legs.index)
    out["pm_coverage"] = present.sum(axis=1) / n_legs

    for bucket in AGG_BUCKETS:
        cols = [t for t, b in buckets.items() if b == bucket]
        col_name = f"{prefix}_{bucket}_prob"
        if not cols:
            out[col_name] = np.nan
            continue
        all_present = present[cols].all(axis=1)
        sums = legs[cols].sum(axis=1)
        out[col_name] = sums.where(all_present)

    all_legs_present = present.all(axis=1)
    raw_sum = legs.sum(axis=1).where(all_legs_present)
    out["prob_sum"] = raw_sum
    ent = pd.Series(np.nan, index=legs.index)
    full_rows = legs[all_legs_present]
    if not full_rows.empty:
        ent.loc[full_rows.index] = [_entropy(row) for row in full_rows.to_numpy(dtype=float)]
    out[f"{prefix}_entropy"] = ent
    return out


# ---------------------------------------------------------------- stitcher
def build_fomc_stitched(cache_dir: Path | str = DEFAULT_CACHE_DIR, name: str = "fomc") -> pd.DataFrame:
    """Continuous daily series from the episodic per-meeting markets.

    Columns: {prefix}_cut_prob, {prefix}_nochange_prob, {prefix}_hike_prob,
    {prefix}_entropy, {prefix}_days_to_meeting, pm_roll, pm_coverage, prob_sum
    (prefix = pm_<name>). Daily UTC index spanning first to last covered day;
    days no front-event market traded stay NaN — never forward-filled here."""
    cache_dir = Path(cache_dir)
    records = _load_records(cache_dir, name)
    prefix = f"pm_{name}"

    by_event: dict[str, list[dict]] = {}
    for r in records:
        by_event.setdefault(r["event_slug"], []).append(r)
    events = sorted(by_event.items(), key=lambda kv: _event_end(kv[1][0]))

    pieces: list[pd.DataFrame] = []
    prev_end: pd.Timestamp | None = None
    for k, (event_slug, event_records) in enumerate(events):
        end = _event_end(event_records[0])
        legs, buckets = _legs_frame(event_records, cache_dir)
        # Front window: (prev_end, end]. Pre-window data (the next meeting's
        # market already trading) and post-resolution stamps are excluded.
        mask = legs.index <= end
        if prev_end is not None:
            mask &= legs.index > prev_end
        legs = legs.loc[mask]
        prev_end = end
        if legs.empty:
            continue
        piece = _aggregate_day_rows(legs, buckets, prefix)
        piece[f"{prefix}_days_to_meeting"] = (end - piece.index).days
        # Float 0/1 here (cast to bool after the final reindex) so the
        # NaN-introducing reindex never coerces the column to object dtype.
        piece["pm_roll"] = 0.0
        if k > 0:
            piece.iloc[0, piece.columns.get_loc("pm_roll")] = 1.0
        pieces.append(piece)

    if not pieces:
        raise PolymarketCacheMissing(f"target {name!r} produced no in-window data")
    stitched = pd.concat(pieces).sort_index()
    full_idx = pd.date_range(stitched.index.min(), stitched.index.max(), freq="D", tz="UTC")
    stitched = stitched.reindex(full_idx)
    stitched["pm_roll"] = stitched["pm_roll"].fillna(0.0).astype(bool)
    return stitched


# ---------------------------------------------------------------- standing event
def build_event_series(cache_dir: Path | str = DEFAULT_CACHE_DIR, name: str = "cuts2026") -> pd.DataFrame:
    """Daily series for a standing bucket event (no roll).

    Bucket labels are expected to start with the bucket's integer count
    (e.g. '3-75-bps' -> k=3). Columns: pm_<name>_exp_cuts (sum k * normalized
    prob, all legs required), pm_<name>_entropy, prob_sum, pm_coverage."""
    cache_dir = Path(cache_dir)
    records = _load_records(cache_dir, name)
    prefix = f"pm_{name}"
    legs, buckets = _legs_frame(records, cache_dir)

    ks = {}
    for token, bucket in buckets.items():
        try:
            ks[token] = int(str(bucket).split("-")[0])
        except ValueError as e:
            raise ValueError(f"bucket {bucket!r} does not start with an integer count") from e

    present = legs.notna()
    all_present = present.all(axis=1)
    out = pd.DataFrame(index=legs.index)
    out["pm_coverage"] = present.sum(axis=1) / legs.shape[1]
    out["prob_sum"] = legs.sum(axis=1).where(all_present)

    k_vec = np.array([ks[t] for t in legs.columns], dtype=float)
    vals = legs.to_numpy(dtype=float)
    totals = np.nansum(vals, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        exp_cuts = (vals * k_vec).sum(axis=1) / totals
    out[f"{prefix}_exp_cuts"] = pd.Series(exp_cuts, index=legs.index).where(all_present)

    ent = pd.Series(np.nan, index=legs.index)
    full_rows = legs[all_present]
    if not full_rows.empty:
        ent.loc[full_rows.index] = [_entropy(row) for row in full_rows.to_numpy(dtype=float)]
    out[f"{prefix}_entropy"] = ent
    return out


# ---------------------------------------------------------------- PIT loader
def load_polymarket_features(
    target_index: pd.DatetimeIndex,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    name: str = "fomc",
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """Stitched series aligned to target_index with PIT shift applied.

    Calendar-day shift (value stamped UTC date t visible at bars >= t+1) BEFORE
    the alignment, exactly like pipeline/alt_data/gld_holdings.py. Forward-fill
    is bounded at `ffill_limit` CALENDAR days regardless of target frequency
    (the daily fill happens on a daily calendar, then maps to target bars), so
    H4 callers get every bar of day t+1 seeing day t's stamp — conservative."""
    target_index = pd.DatetimeIndex(target_index)
    daily = build_fomc_stitched(cache_dir, name=name)
    daily = daily.astype({"pm_roll": "float64"})

    if len(target_index) == 0:
        return pd.DataFrame(columns=daily.columns, index=target_index)

    shifted = daily.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)

    cal_start = min(shifted.index.min(), target_index.min().normalize())
    cal_end = max(shifted.index.max(), target_index.max().normalize())
    calendar = pd.date_range(cal_start, cal_end, freq="D", tz="UTC")
    on_calendar = shifted.reindex(calendar).ffill(limit=ffill_limit)

    aligned = on_calendar.reindex(target_index.normalize())
    aligned.index = target_index
    return aligned


# ---------------------------------------------------------------- helpers
def roll_masked_diff(series: pd.Series, roll: pd.Series) -> pd.Series:
    """First difference with roll-day diffs masked to NaN.

    A diff that spans a roll boundary (prev front event -> next) is a
    mechanical level jump, not information; the spike's lead-lag panel must
    never treat it as a signal change."""
    d = series.diff()
    d[roll.reindex(series.index).fillna(False).astype(bool)] = np.nan
    return d
