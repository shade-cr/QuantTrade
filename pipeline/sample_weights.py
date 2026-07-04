"""Compute López-de-Prado average-uniqueness sample weights.

References:
  Marcos López de Prado, *Advances in Financial Machine Learning*, Wiley 2018,
  Chapter 4 (Sample Weights). The function mp_sample_tw computes, for each
  event i, the mean of (1 / num_co_events(t)) over the outcome bars
  t ∈ [t_start_i, t_end_i].
"""
from __future__ import annotations
import warnings

import numpy as np
import pandas as pd


def avg_uniqueness(t_starts: np.ndarray, t_ends: np.ndarray, n_bars: int) -> np.ndarray:
    """Return per-event sample weights ∈ (0, 1] via AFML §4 average-uniqueness.

    Args:
      t_starts: int array, position of each event's entry bar in the ohlcv frame.
      t_ends:   int array, position of each event's outcome resolution bar.
      n_bars:   total number of bars (sets the size of the concurrency count vector).
    """
    t_starts = np.asarray(t_starts, dtype=int)
    t_ends = np.asarray(t_ends, dtype=int)
    if t_starts.shape != t_ends.shape:
        raise ValueError("t_starts and t_ends must have the same shape")
    if (t_ends < t_starts).any():
        raise ValueError("t_ends must be >= t_starts")

    # Concurrency: how many open events cover each bar.
    co_events = np.zeros(n_bars, dtype=int)
    for s, e in zip(t_starts, t_ends):
        co_events[s : e + 1] += 1

    weights = np.empty(len(t_starts), dtype=float)
    for k, (s, e) in enumerate(zip(t_starts, t_ends)):
        weights[k] = (1.0 / co_events[s : e + 1]).mean()
    return weights


def pooled_avg_uniqueness(
    event_time, label_end_time
) -> np.ndarray:
    """Cross-asset wall-clock average-uniqueness weights ∈ (0, 1] (AFML §4.3, B0148).

    Generalizes :func:`avg_uniqueness` from single-asset bar positions to a SHARED
    wall-clock timeline built from the union of every event's ``[event_time,
    label_end_time]`` span across the whole pool. This is the load-bearing
    correctness fix for cross-asset pooling (spec blocker B1): concatenating
    per-asset weight vectors does NOT down-weight contemporaneous events on
    different assets, which inflates the effective-N premise behind the pooled
    DSR/MinBTL read.

    Concurrency model (the ρ=1 conservative span-overlap bound)
    -----------------------------------------------------------
    The union of all span endpoints partitions the timeline into atomic
    sub-intervals. Concurrency on each sub-interval = number of events whose span
    covers it. Each event's weight = the DURATION-WEIGHTED mean of ``1/concurrency``
    over the sub-intervals it covers.

    AFML §4.3 defines concurrency as labels sharing a common return ``r_{t-1,t}``;
    within one asset overlapping labels literally share the draw. Across assets,
    contemporaneous events have *correlated but distinct* returns (ρ≈0.8, not 1.0),
    so pure span-overlap counting (two contemporaneous events → u≈0.5) treats them
    as fully redundant — the ρ=1 limit. This OVER-penalizes, which is the SAFE
    direction for a validation gate (shrinks effective-N → harder to clear DSR,
    never easier → no leak). The faithful correlation-weighted generalization is
    deferred (spec [R2 — corpus caveat]).

    Args:
      event_time:     array-like of pandas Timestamps — each event's entry-bar time.
      label_end_time: array-like of pandas Timestamps — each event's triple-barrier
                      resolution time (its own asset's ``t_end_idx`` bar timestamp).
                      Must satisfy ``label_end_time[i] >= event_time[i]``.

    Returns:
      np.ndarray of per-event weights in the ORIGINAL input order. Empty input
      returns an empty float array.
    """
    starts = pd.DatetimeIndex(pd.to_datetime(list(event_time)))
    ends = pd.DatetimeIndex(pd.to_datetime(list(label_end_time)))
    if len(starts) != len(ends):
        raise ValueError("event_time and label_end_time must have the same length")
    n = len(starts)
    if n == 0:
        return np.empty(0, dtype=float)
    if (ends < starts).any():
        raise ValueError("label_end_time must be >= event_time for every event")

    s = starts.asi8.astype(np.int64)   # ns since epoch
    e = ends.asi8.astype(np.int64)

    # Atomic sub-interval boundaries: the sorted union of all span endpoints.
    boundaries = np.unique(np.concatenate([s, e]))
    # Sub-interval i spans [boundaries[i], boundaries[i+1]); its duration in ns.
    seg_lo = boundaries[:-1]
    seg_hi = boundaries[1:]
    seg_dur = (seg_hi - seg_lo).astype(np.float64)
    n_seg = len(seg_lo)
    if n_seg == 0:
        # All spans are zero-duration single instants → treat each as fully unique
        # unless they coincide. Fall back to instant-point concurrency.
        return _instant_point_uniqueness(s, e)

    # Concurrency per sub-interval: an event covers sub-interval i iff its span
    # [s_k, e_k] overlaps [seg_lo[i], seg_hi[i]). Using half-open segments with
    # an inclusive event end means coverage = (s_k <= seg_lo[i]) & (e_k >= seg_hi[i]).
    concurrency = np.zeros(n_seg, dtype=np.int64)
    covers = np.empty((n, n_seg), dtype=bool)
    for k in range(n):
        c = (s[k] <= seg_lo) & (e[k] >= seg_hi)
        covers[k] = c
        concurrency += c.astype(np.int64)

    # Guard against zero concurrency on a covered segment (shouldn't happen since
    # every covered segment is covered by >=1 event, but be safe).
    inv_conc = np.where(concurrency > 0, 1.0 / np.maximum(concurrency, 1), 0.0)

    weights = np.empty(n, dtype=float)
    for k in range(n):
        cov = covers[k]
        dur = seg_dur[cov]
        total = dur.sum()
        if total <= 0:
            # zero-duration event (s_k == e_k): single instant → unique
            weights[k] = 1.0
        else:
            weights[k] = float((inv_conc[cov] * dur).sum() / total)
    return weights


def _instant_point_uniqueness(s: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Degenerate fallback: all spans are single instants (s == e). Concurrency at
    each instant = number of events sharing that exact timestamp."""
    n = len(s)
    weights = np.empty(n, dtype=float)
    for k in range(n):
        same = int((s == s[k]).sum())
        weights[k] = 1.0 / same
    return weights


# --- B0012 v2: fit-weight/inference decoupling -------------------------------
# Spec: docs/superpowers/specs/2026-07-04-b0012-uniqueness-v2-decoupling.md.
# These constants are FROZEN pre-registration values (spec §2) — do not tune
# against results. The functions below feed FIT WEIGHTS ONLY; every gate/floor
# keeps consuming the rho=1 `pooled_avg_uniqueness` above (the firewall, §1).
RHO_WINDOW = 252
RHO_REFRESH = 21
RHO_SHRINK_LAMBDA = 0.5
RHO_FLOOR = 0.15


def rolling_panel_rho(
    close: "pd.DataFrame",
    window: int = RHO_WINDOW,
    refresh: int = RHO_REFRESH,
    shrink_lambda: float = RHO_SHRINK_LAMBDA,
    rho_floor: float = RHO_FLOOR,
    min_periods: int = 126,
) -> list:
    """Point-in-time shrunk correlation schedule for the pooled panel.

    Returns [(effective_from, rho_star), ...] where each rho_star (assets x
    assets, diag 1) is estimated from log returns STRICTLY BEFORE
    effective_from (rows [k-window, k) feed the matrix effective at index[k]),
    shrunk toward the panel-mean correlation (constant-correlation target,
    lambda fixed) and clipped to [rho_floor, 1]. Refreshed every `refresh`
    bars; consumers hold each matrix constant until the next effective_from.
    Pairs with insufficient overlap fall back to the panel mean, then the floor.
    """
    import pandas as pd  # local: keep module import surface unchanged

    r = np.log(close).diff()
    idx = close.index
    out = []
    for k in range(window, len(idx), refresh):
        sub = r.iloc[k - window:k]
        rho = sub.corr(min_periods=min_periods)
        off_mask = ~np.eye(len(rho), dtype=bool)
        off_vals = rho.values[off_mask]
        rbar = float(np.nanmean(off_vals)) if np.isfinite(off_vals).any() else rho_floor
        shrunk = shrink_lambda * rho.values + (1.0 - shrink_lambda) * rbar
        shrunk = np.where(np.isnan(shrunk), rbar, shrunk)
        shrunk = np.clip(shrunk, rho_floor, 1.0)
        np.fill_diagonal(shrunk, 1.0)
        out.append((idx[k], pd.DataFrame(shrunk, index=rho.index, columns=rho.columns)))
    return out


def effective_number_of_bets(rho: "pd.DataFrame") -> float:
    """Meucci-style ENB diagnostic: exp(entropy) of normalized eigenvalues of
    the correlation matrix. Equals N for identity, ->1 as rho->1. DIAGNOSTIC
    CEILING ONLY — never a gate input (spec §3)."""
    vals = np.linalg.eigvalsh(np.asarray(rho, dtype=float))
    vals = np.clip(vals, 1e-12, None)
    p = vals / vals.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def corr_discounted_uniqueness(
    event_time, label_end_time, asset, rho_schedule, grid,
) -> np.ndarray:
    """B0012 v2 FIT-WEIGHTS: correlation-discounted cross-asset uniqueness.

    c_a(t) = n_a(t) + sum_{b != a} rho*_{ab}(t) * n_b(t); u = 1/c; u_bar = mean
    over the event's span days on `grid`. Same-asset concurrency keeps full
    AFML §4.3 weight (shared price path). Cross-asset concurrency is discounted
    by the PIT shrunk correlation (spec §2). Days before the first schedule
    entry — and assets missing from the matrices — use rho*=1 (conservative:
    reduces to the rho=1 rule, never credits unearned independence).

    FIREWALL (spec §1): this feeds sample_weight for fit/search/calibration
    ONLY. Gates, floors and DSR keep consuming `pooled_avg_uniqueness`.
    """
    ev = pd.DatetimeIndex(event_time)
    le = pd.DatetimeIndex(label_end_time)
    asset = np.asarray(asset, dtype=object)
    if not (len(ev) == len(le) == len(asset)):
        raise ValueError("event_time, label_end_time, asset must be aligned")
    grid = pd.DatetimeIndex(grid)
    assets = sorted(set(asset.tolist()))
    a_pos = {a: i for i, a in enumerate(assets)}
    n_days, n_assets = len(grid), len(assets)

    # Open-event counts per (day, asset) via entry/exit difference arrays.
    counts = np.zeros((n_days, n_assets), dtype=float)
    # NOTE (deviation from brief, documented in task-2-report.md): e_idx uses
    # side="left" - 1, NOT side="right" - 1 as in the brief's draft. The brief's
    # version double-counts the label_end_time day as still "open" (inclusive
    # both ends -> a dur=5 event spans 6 grid days), whereas
    # pooled_avg_uniqueness's continuous half-open-segment convention treats
    # the span as [event_time, label_end_time) -> 5 grid days. side="left" - 1
    # reproduces that exclusive-end convention so the two functions' baseline
    # (rho*=1) concurrency counts match exactly, per spec's "mirroring
    # pooled_avg_uniqueness" requirement.
    s_idx = np.clip(grid.searchsorted(ev, side="left"), 0, n_days - 1)
    e_idx = np.clip(grid.searchsorted(le, side="left") - 1, 0, n_days - 1)
    e_idx = np.maximum(e_idx, s_idx)
    for k in range(len(ev)):
        j = a_pos[asset[k]]
        counts[s_idx[k], j] += 1.0
        if e_idx[k] + 1 < n_days:
            counts[e_idx[k] + 1, j] -= 1.0
    counts = np.cumsum(counts, axis=0)

    # rho* per day: schedule step function; rho*=1 before the first entry.
    # Index-array optimization (per implementer NOTE): map each grid day to
    # the schedule slot in effect, avoiding an O(days) python loop per entry.
    ones = np.ones((n_assets, n_assets), dtype=float)
    sched_ts = [ts for ts, _ in rho_schedule]
    sched_mx = []
    for _, m in rho_schedule:
        mx = ones.copy()
        common = [a for a in assets if a in m.index]
        if len(common) < len(assets):
            missing = sorted(set(assets) - set(common))
            warnings.warn(f"corr_discounted_uniqueness: assets {missing} missing "
                          f"from rho matrix — using rho*=1 for them", RuntimeWarning)
        sub = m.loc[common, common].values
        for i, ai in enumerate(common):
            for j, aj in enumerate(common):
                mx[a_pos[ai], a_pos[aj]] = sub[i, j]
        sched_mx.append(mx)

    if sched_ts:
        starts = np.searchsorted(grid.values, pd.DatetimeIndex(sched_ts).values,
                                 side="left")
        # slot[d] = index into sched_mx in effect at grid day d, or -1 if before
        # the first effective_from (conservative rho*=1 warmup).
        slot = np.searchsorted(starts, np.arange(n_days), side="right") - 1
    else:
        slot = np.full(n_days, -1, dtype=int)

    # c per (day, asset): counts @ rho_row, keeping the same-asset term at weight 1.
    inv_c = np.zeros((n_days, n_assets), dtype=float)
    for d in range(n_days):
        mx = sched_mx[slot[d]] if slot[d] >= 0 else ones
        c_row = counts[d] @ mx  # includes own-asset term at rho=1 (diag)
        with np.errstate(divide="ignore"):
            inv_c[d] = np.where(c_row > 0, 1.0 / c_row, 0.0)

    # NOTE (deviation from brief, documented in task-2-report.md): the mean
    # over an event's span days is DURATION-WEIGHTED by the actual wall-clock
    # gap to the next grid day, not a plain arithmetic mean over day count.
    # `pooled_avg_uniqueness` averages 1/concurrency over CONTINUOUS atomic
    # sub-intervals weighted by their real elapsed time — on a business-day
    # grid a Fri->Mon gap carries 3x the weight of a Mon->Tue gap. A plain
    # per-day mean silently disagrees with that (~10-20% off at rho*=1 in the
    # dense-event fixture, entirely a weekend-gap artifact, not a rho effect)
    # and breaks the spec's "mirroring pooled_avg_uniqueness" contract and the
    # single-asset bit-identical test (§5.2). Weighting by the grid's own
    # inter-day duration reproduces pooled_avg_uniqueness's segment durations
    # exactly at rho*=1.
    grid_ns = grid.asi8.astype(np.float64)
    if n_days > 1:
        gaps = np.diff(grid_ns)
        day_dur = np.append(gaps, gaps[-1])
    else:
        day_dur = np.ones(max(n_days, 1), dtype=np.float64)

    weights = np.empty(len(ev), dtype=float)
    for k in range(len(ev)):
        j = a_pos[asset[k]]
        lo, hi = s_idx[k], e_idx[k] + 1
        vals = inv_c[lo:hi, j]
        dur = day_dur[lo:hi]
        mask = vals > 0
        total_dur = dur[mask].sum()
        if total_dur > 0:
            weights[k] = float((vals[mask] * dur[mask]).sum() / total_dur)
        else:
            weights[k] = 1.0
    return weights
