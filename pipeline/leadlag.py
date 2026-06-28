"""BTC->ETH/SOL lead-lag sleeve building blocks (B0167, "LL").

Implements EXACTLY the pre-registration §1.2
(``docs/superpowers/specs/2026-06-12-l2-preregistration.md``):

* :func:`hayashi_yoshida_leadlag` — the lagged Hayashi-Yoshida cross-
  correlation estimator on ASYNCHRONOUS raw ticks. This is the day-one
  (Jun-15) PRECONDITION: BTC must show a lead of >= 30 s (the registered
  timestamp-noise floor, :data:`MIN_LEAD_S`) with positive cross-correlation
  BEFORE any backtest runs, else the sleeve is dead on arrival.

  References: Hayashi & Yoshida (2005), "On covariance estimation of
  non-synchronously observed diffusion processes", Bernoulli 11(2) — the
  overlap-interval covariance estimator; Hoffmann, Rosenbaum & Yoshida
  (2013), "Estimation of the lead-lag parameter from non-synchronous data",
  Bernoulli 19(2) — the lag-shifted HY contrast; Huth & Abergel (2014),
  "High frequency lead/lag relationships — empirical facts", Journal of
  Empirical Finance 26 — the per-lag correlation curve and the lead-lag
  ratio LLR = sum_{l>0} rho(l)^2 / sum_{l<0} rho(l)^2 used as the lead
  direction diagnostic.

* :func:`ll_events` — the registered event detector: BTC 1-minute return
  ``|r| > threshold * sigma_train`` (STRICT, fail-closed on NaN); the
  threshold belongs to the §2 trial registry (3.0 sigma trial 7 / 4.0 sigma
  trial 8) and sigma_train is a TRAINING-period artifact passed in.

* :func:`ll_signal` — the pure follower entry rule: enter in the BTC-move
  direction iff the follower's own concurrent 1-min move is STRICTLY below
  :data:`FOLLOWER_NOT_MOVED_SIGMAS` (= 1, REGISTERED — a module constant,
  not a parameter) sigmas in the SAME direction ("not yet moved"). All
  sigmas are training artifacts passed in by the caller.

* :func:`minute_bars` / :func:`minute_returns` — the 1-minute TIME-bar
  clock both rules run on: right-labeled / right-closed (a bar stamped t
  contains only ticks <= t — fire on bar close, CLAUDE.md no-look-ahead
  invariant), empty minutes carry the last close (zero return).

Everything here is a pure deterministic function of its inputs: no fit
step, no network, no state — the same CLAUDE.md "primaries are pure rule
functions" contract as :mod:`pipeline.osr_intraday`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: REGISTERED (pre-registration §1.2): the follower must NOT have moved >=
#: this many training sigmas in the BTC direction during the event minute.
#: Frozen — deliberately a module constant, not a function parameter.
FOLLOWER_NOT_MOVED_SIGMAS: float = 1.0

#: REGISTERED (pre-registration §1.2): the minimum lead, in seconds, the HY
#: estimate must show for the leader (timestamp-noise floor for
#: broker-arrival timestamps). A correlation peak at a smaller lag — even a
#: strong contemporaneous one — fails the precondition.
MIN_LEAD_S: float = 30.0

#: Default positive-lag grid for the HY scan: 30 s .. 300 s in 30 s steps
#: (the registered floor up to the upper end of the documented 15-60 s
#: BTC->alt lead plus safety margin). The scan always evaluates the
#: symmetric negative lags and lag 0 as well — the LLR and the
#: reversed-direction verdict need them.
DEFAULT_LAGS_S: tuple[float, ...] = tuple(float(s) for s in range(30, 301, 30))


@dataclass(frozen=True)
class LeadLagVerdict:
    """The §1.2 precondition verdict the Jun-15 report records."""

    #: True iff the cross-correlation peak sits at a lag >= MIN_LEAD_S
    #: (leader earlier) AND the peak correlation is positive.
    leader_leads: bool
    #: Lag (seconds) of the correlation peak; positive = leader leads.
    best_lag_s: float
    #: Correlation at the peak.
    best_corr: float
    #: Huth-Abergel lead-lag ratio: sum of rho^2 over positive lags divided
    #: by the same over negative lags (> 1 means the leader side dominates).
    llr: float
    #: The registered lead floor the verdict was evaluated against.
    min_lead_s: float


def _tick_returns_and_intervals(
    ticks: pd.Series, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(log-returns, interval starts, interval ends) in float UTC seconds."""
    if not isinstance(ticks.index, pd.DatetimeIndex) or ticks.index.tz is None:
        raise ValueError(
            f"{name}: tick index must be a tz-aware (UTC) DatetimeIndex."
        )
    if len(ticks) < 2:
        raise ValueError(
            f"{name}: need at least 2 ticks to form one return interval; "
            f"got {len(ticks)}."
        )
    if not ticks.index.is_monotonic_increasing:
        raise ValueError(f"{name}: tick index must be sorted ascending.")
    px = ticks.to_numpy(dtype="float64")
    if not np.all(px > 0):
        raise ValueError(f"{name}: prices must be strictly positive.")
    t = ticks.index.asi8.astype("float64") / 1e9
    r = np.diff(np.log(px))
    return r, t[:-1], t[1:]


def _hy_covariance(
    lr: np.ndarray, lt0: np.ndarray, lt1: np.ndarray,
    fr: np.ndarray, ft0: np.ndarray, ft1: np.ndarray,
    lag_s: float,
) -> float:
    """Lag-shifted Hayashi-Yoshida covariance.

    Leader return intervals are shifted FORWARD by ``lag_s`` (positive lag =
    "the leader's move at time t shows up in the follower at t + lag");
    the HY sum then counts every leader/follower return product whose
    (half-open) intervals overlap in clock time — no synchronization grid,
    no interpolation (Hayashi-Yoshida 2005; lag shift per
    Hoffmann-Rosenbaum-Yoshida 2013).
    """
    a0 = lt0 + lag_s
    a1 = lt1 + lag_s
    # Overlap of (a0, a1] with follower interval (u, v]: v > a0 and u < a1.
    lo = np.searchsorted(ft1, a0, side="right")
    hi = np.searchsorted(ft0, a1, side="left")
    cum = np.concatenate(([0.0], np.cumsum(fr)))
    return float(np.sum(lr * (cum[hi] - cum[lo])))


def hayashi_yoshida_leadlag(
    leader_ticks: pd.Series,
    follower_ticks: pd.Series,
    lags: list[float] | tuple[float, ...] = DEFAULT_LAGS_S,
) -> tuple[pd.DataFrame, LeadLagVerdict]:
    """HY lead-lag cross-correlation scan on asynchronous raw ticks.

    Args:
      leader_ticks: candidate-leader tick prices (BTC), tz-aware UTC index.
      follower_ticks: candidate-follower tick prices (ETH/SOL), same form.
      lags: POSITIVE lag grid in seconds; the scan evaluates the symmetric
        negative lags and lag 0 too (default :data:`DEFAULT_LAGS_S`).

    Returns ``(table, verdict)``:

    * ``table``: one row per scanned lag with columns ``lag_s`` and
      ``corr`` (the lagged HY correlation) — the per-lag diagnostic the
      Jun-15 report records;
    * ``verdict``: :class:`LeadLagVerdict` — the §1.2 precondition outcome
      (peak lag >= :data:`MIN_LEAD_S` with positive correlation).

    Pure and deterministic; never mutates the inputs.
    """
    pos = sorted({float(l) for l in lags})
    if not pos or any(l <= 0 for l in pos):
        raise ValueError(
            f"lags must be a non-empty grid of POSITIVE seconds; got {lags!r}"
            f" — the symmetric negative lags and 0 are added automatically."
        )
    grid = [-l for l in reversed(pos)] + [0.0] + pos

    lr, lt0, lt1 = _tick_returns_and_intervals(leader_ticks, "leader_ticks")
    fr, ft0, ft1 = _tick_returns_and_intervals(follower_ticks,
                                               "follower_ticks")
    norm = float(np.sqrt(np.sum(lr * lr) * np.sum(fr * fr)))
    rows = []
    for lag in grid:
        cov = _hy_covariance(lr, lt0, lt1, fr, ft0, ft1, lag)
        corr = cov / norm if norm > 0 else float("nan")
        rows.append({"lag_s": float(lag), "corr": float(corr)})
    table = pd.DataFrame(rows)

    best = table.loc[table["corr"].idxmax()]
    best_lag = float(best["lag_s"])
    best_corr = float(best["corr"])
    c = table["corr"].to_numpy()
    l = table["lag_s"].to_numpy()
    pos_energy = float(np.sum(c[l > 0] ** 2))
    neg_energy = float(np.sum(c[l < 0] ** 2))
    llr = pos_energy / neg_energy if neg_energy > 0 else float("inf")
    verdict = LeadLagVerdict(
        leader_leads=bool(best_lag >= MIN_LEAD_S and best_corr > 0),
        best_lag_s=best_lag,
        best_corr=best_corr,
        llr=llr,
        min_lead_s=MIN_LEAD_S,
    )
    return table, verdict


def minute_bars(prices: pd.Series) -> pd.DataFrame:
    """1-minute TIME bars from tick prices: right-labeled / right-closed.

    A bar stamped ``t`` contains exactly the ticks in ``(t - 1min, t]`` —
    the boundary tick at ``t`` CLOSES the bar stamped ``t``, so acting on a
    bar at its stamp never reads the future (fire-on-bar-close causality).
    Minutes with no ticks carry the last close forward
    (open = high = low = close = previous close → zero return), keeping the
    grid regular so a 20-minute timeout is a fixed bar horizon.

    Returns an ``open/high/low/close`` frame on the regular 1-min grid.
    """
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.tz is None:
        raise ValueError(
            "minute_bars: price index must be a tz-aware (UTC) DatetimeIndex."
        )
    if len(prices) == 0:
        raise ValueError("minute_bars: empty price series.")
    bars = prices.resample("1min", label="right", closed="right").ohlc()
    close = bars["close"].ffill()
    out = pd.DataFrame({
        "open": bars["open"].fillna(close),
        "high": bars["high"].fillna(close),
        "low": bars["low"].fillna(close),
        "close": close,
    })
    return out


def minute_returns(prices: pd.Series) -> pd.Series:
    """1-minute log close-to-close returns on the :func:`minute_bars` clock.

    Defined LITERALLY as ``log(minute_bars(prices)["close"]).diff()`` so the
    two helpers can never drift apart; the first bar is NaN (no previous
    close), gap minutes are exactly 0.
    """
    return np.log(minute_bars(prices)["close"]).diff()


def ll_events(
    btc_1min_returns: pd.Series,
    sigma_train: float,
    threshold_sigmas: float,
) -> pd.Series:
    """Registered LL event detector: BTC 1-min ``|r| > k * sigma_train``.

    Args:
      btc_1min_returns: BTC 1-minute log returns (:func:`minute_returns`).
      sigma_train: TRAINING-period std of those returns (> 0) — a fitted
        artifact passed in, never recomputed here (leakage boundary).
      threshold_sigmas: the §2 trial parameter (3.0 = trial 7, 4.0 = trial
        8); STRICT inequality — exactly k sigma does not fire.

    NaN returns fail closed (no event). Returns a float Series indexed at
    the event minutes only, value = sign of the BTC move in {-1, +1}.
    """
    if not (sigma_train > 0):
        raise ValueError(f"sigma_train must be > 0, got {sigma_train!r}.")
    if not (threshold_sigmas > 0):
        raise ValueError(
            f"threshold_sigmas must be > 0, got {threshold_sigmas!r}."
        )
    r = btc_1min_returns.astype("float64")
    fire = r.notna() & (r.abs() > float(threshold_sigmas) * float(sigma_train))
    return np.sign(r[fire]).astype("float64")


def ll_signal(
    follower_1min_returns: pd.Series,
    events: pd.Series,
    sigma_train_follower: float,
) -> pd.Series:
    """Pure follower entry rule (§1.2): follow BTC iff "not yet moved".

    Enter the follower IN the BTC-move direction ``d`` iff the follower's
    own concurrent 1-min move satisfies
    ``d * r_follower < FOLLOWER_NOT_MOVED_SIGMAS * sigma_train_follower``
    (STRICT) — i.e. the signed SAME-direction displacement is below one
    training sigma. An opposite-direction move of any size passes the gate
    (the follower has not moved in the BTC direction at all); exactly one
    sigma does not. Fail-closed cases (always 0, never a fabricated entry):
    NaN concurrent return, event minute absent from the follower clock,
    no event.

    Args:
      follower_1min_returns: follower 1-min log returns on its own clock.
      events: output of :func:`ll_events` (signed event minutes).
      sigma_train_follower: TRAINING-period std of the follower's 1-min
        returns (> 0) — fitted artifact passed in.

    Returns a float Series in {-1, 0, +1} indexed exactly like
    ``follower_1min_returns``. Pure, deterministic, causal: minute t uses
    only the BTC and follower returns OF minute t (both close at t).
    """
    if not (sigma_train_follower > 0):
        raise ValueError(
            f"sigma_train_follower must be > 0, got {sigma_train_follower!r}."
        )
    r = follower_1min_returns.astype("float64")
    ev = events.reindex(r.index)  # off-clock event minutes -> NaN -> closed
    bound = FOLLOWER_NOT_MOVED_SIGMAS * float(sigma_train_follower)
    ok = ev.notna() & (ev != 0) & r.notna() & ((ev * r) < bound)
    out = pd.Series(0.0, index=r.index)
    out[ok] = ev[ok]
    return out
