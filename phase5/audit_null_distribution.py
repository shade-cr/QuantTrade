"""Phase 5 — Audit-of-the-audit probe matrix expansion (Option D / BLOCK remediation).

This module IS NOT a v3.2 patch.

It is the empirical-null-distribution probe matrix mandated by the
phase5-devils-advocate BLOCK verdict in
``signals/devils_advocate_reviews/20260526-audit-patch-scope-decision_v1.json``.

The verdict's must_have_mods item 2: "Expand phase5/audit_audit.py probe matrix
to >=5 variants per bias class (leakage magnitudes, single-sided strategies
across regimes, low-event-density configurations) BEFORE any patch design
begins."

We DO NOT:
- design any threshold for B1 / B2 / B3
- recommend a patch
- decide whether B3 is a geometry bug or a working-as-designed constraint

We DO:
- exhaustively probe ``_classify_transferability``, ``_regime_diversity``,
  ``make_folds`` and ``PurgedTimeSeriesSplit`` at synthetic inputs across the
  ranges relevant to B1 / B2 / B3
- record the audit's response at each cell
- publish the matrix as a serializable JSON + a markdown null-distribution
  report (``phase5/audit_null_distribution.md`` is the next-action deliverable
  that consumes this output)

Run:
  uv run python -m phase5.audit_null_distribution

Output:
  results/phase5/audit_null_distribution.json
"""
from __future__ import annotations
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_threshold_transferability import (
    _classify_transferability,
    _regime_diversity,
)
from pipeline.walk_forward import make_folds, PurgedTimeSeriesSplit


# Pre-registered (NOT data-derived) settings.
# Source: Phase 1 default config + Phase 5 H4 attempt parameters. These
# values are the ones the spike actually used; the probe matrix tests the
# audit at these settings without tuning them.
DEFAULT_N_FOLDS = 4
DEFAULT_TRAIN_MIN = 200
DEFAULT_PURGE_BARS = 5
DEFAULT_EMBARGO_PCT = 0.01
DEFAULT_CV_SPLITS = 3


@dataclass
class ProbeCell:
    probe_class: str  # "B1_LEAKAGE" | "B2_SHORT_BIAS" | "B3_EVENT_DENSITY"
    cell_id: str
    inputs: dict
    observation: dict
    rationale: str


# =============================================================================
# B1 — Leakage detection null distribution
# =============================================================================

def _b1_classifier_response(per_fold_sharpe: list[float], n_per_fold: int = 150) -> dict:
    """Run the v3 classifier on a synthetic per-fold Sharpe profile."""
    per_fold_n = [n_per_fold] * len(per_fold_sharpe)
    cls = _classify_transferability(per_fold_n, per_fold_sharpe, regime_pass=True)
    active = [s for s, n in zip(per_fold_sharpe, per_fold_n) if n >= 30 and np.isfinite(s)]
    median_active = float(np.nanmedian(active)) if active else float("nan")
    return {
        "audit_class": cls,
        "median_active_sharpe": median_active,
        "n_active_folds": len(active),
    }


def b1_magnitude_sweep() -> list[ProbeCell]:
    """Sweep median active-fold Sharpe across a wide magnitude range.

    Documents WHERE in the (Sharpe, audit_class) plane the v3 classifier
    transitions, so future v3.2 work can ground any threshold against an
    empirical distribution rather than reverse-engineering one from the
    single P000_pos_leakage probe.
    """
    cells = []
    sharpes = [0.05, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    for s in sharpes:
        per_fold = [s, s * 0.95, s * 1.05, s * 0.98]  # tight dispersion
        obs = _b1_classifier_response(per_fold)
        cells.append(ProbeCell(
            probe_class="B1_LEAKAGE",
            cell_id=f"magnitude_median_{s:.2f}",
            inputs={"per_fold_sharpe": per_fold, "n_per_fold": 150, "regime_pass": True},
            observation=obs,
            rationale="Tests whether the v3 classifier responds to Sharpe magnitude. Threshold-design must NOT use this to pick a cutoff — but the null distribution shape is a precondition.",
        ))
    return cells


def b1_dispersion_sweep() -> list[ProbeCell]:
    """At a fixed positive median, vary fold-to-fold dispersion.

    Implausibly clean profiles (low dispersion at high Sharpe) are a separate
    leakage signal from raw Sharpe magnitude. Documents whether v3 is
    sensitive to this dimension at all.
    """
    cells = []
    median_target = 2.0
    dispersions = [0.05, 0.2, 0.5, 1.0, 3.0]
    for d in dispersions:
        per_fold = [median_target - d, median_target - d / 2, median_target + d / 2, median_target + d]
        obs = _b1_classifier_response(per_fold)
        cells.append(ProbeCell(
            probe_class="B1_LEAKAGE",
            cell_id=f"dispersion_d{d:.2f}",
            inputs={"per_fold_sharpe": per_fold, "n_per_fold": 150, "regime_pass": True},
            observation=obs,
            rationale="Probes whether the classifier discriminates by dispersion. If audit_class is invariant to dispersion at fixed median, the classifier is dispersion-blind by construction.",
        ))
    return cells


def b1_known_nulls() -> list[ProbeCell]:
    """Synthetic per-fold profiles representing known-null strategies.

    These bound the false-positive rate of any future B1 threshold. If a
    future threshold flags any of these as POS_LEAKAGE_SUSPECT, it has a
    nontrivial false-positive rate at the null and is not defensible.
    """
    np.random.seed(42)
    cells = []
    null_profiles = {
        "random_pm1_seed0": [0.02, -0.01, 0.03, -0.02],
        "random_pm1_seed1": [-0.04, 0.06, -0.03, 0.01],
        "random_pm1_seed2": [0.05, 0.07, -0.02, -0.05],
        "shuffled_label": [-0.03, 0.04, -0.05, 0.02],
        "constant_zero": [0.0, 0.0, 0.0, 0.0],
        "lagged_self_1d": [-0.08, 0.05, -0.06, 0.04],
        "random_walk_follow": [-0.02, -0.03, -0.01, -0.04],
        "skew_one_fold_positive": [0.3, -0.1, -0.05, 0.0],
        "skew_one_fold_negative": [-0.3, 0.05, 0.1, 0.0],
        "bimodal_pos_neg": [0.4, -0.5, 0.4, -0.5],
    }
    for k, prof in null_profiles.items():
        obs = _b1_classifier_response(prof)
        cells.append(ProbeCell(
            probe_class="B1_LEAKAGE",
            cell_id=f"known_null_{k}",
            inputs={"per_fold_sharpe": prof, "n_per_fold": 150, "regime_pass": True},
            observation=obs,
            rationale="Known-null profile. Any v3.2 patch threshold must not flag these as POS_LEAKAGE_SUSPECT.",
        ))
    return cells


# =============================================================================
# B2 — Short-bias in _regime_diversity (close-coupling vs single-side)
# =============================================================================

def b2_regime_close_arrays() -> list[ProbeCell]:
    """Synthesize close arrays that exemplify B2's failure modes.

    Each cell records (a) the close-array shape, (b) the gate's pass/fail
    output, (c) what an always-single-sided strategy would yield against
    the gate. Documents the false-positive (single-side in opposite regime)
    AND false-negative-risk (buy-and-hold-equivalent) classes side by side.
    """
    cells = []
    n = 500
    arrays = {
        "monotone_up_75pct": np.linspace(80.0, 140.0, n),
        "monotone_down_40pct": np.linspace(100.0, 60.0, n),
        "mixed_up_dominant": np.concatenate([
            np.linspace(100.0, 130.0, 350),
            np.linspace(130.0, 115.0, 150),
        ]),
        "mixed_down_dominant": np.concatenate([
            np.linspace(100.0, 70.0, 350),
            np.linspace(70.0, 85.0, 150),
        ]),
        "bidirectional_balanced": np.concatenate([
            np.linspace(100.0, 120.0, 125),
            np.linspace(120.0, 80.0, 250),
            np.linspace(80.0, 105.0, 125),
        ]),
        "flat_2pct_noise": 100.0 + 2.0 * np.sin(np.linspace(0, 8 * np.pi, n)),
        "v_shape_dd_then_rally": np.concatenate([
            np.linspace(100.0, 70.0, 250),
            np.linspace(70.0, 115.0, 250),
        ]),
        "inverted_v_rally_then_dd": np.concatenate([
            np.linspace(100.0, 130.0, 250),
            np.linspace(130.0, 85.0, 250),
        ]),
    }
    for k, arr in arrays.items():
        diversity = _regime_diversity(arr)
        # Always-LONG strategy hypothesized response
        long_sharpe_proxy = float((arr[-1] - arr[0]) / arr[0])
        # Always-SHORT strategy hypothesized response
        short_sharpe_proxy = -long_sharpe_proxy
        # Classify both under v3
        long_cls = _classify_transferability([120, 120, 120, 120], [1.5, 1.5, 1.5, 1.5], diversity["pass"]) if long_sharpe_proxy > 0 else "n/a_long_loses"
        short_cls = _classify_transferability([120, 120, 120, 120], [1.5, 1.5, 1.5, 1.5], diversity["pass"]) if short_sharpe_proxy > 0 else "n/a_short_loses"
        cells.append(ProbeCell(
            probe_class="B2_SHORT_BIAS",
            cell_id=f"close_array_{k}",
            inputs={
                "close_array_summary": {
                    "shape": k,
                    "start": float(arr[0]),
                    "end": float(arr[-1]),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                },
            },
            observation={
                "regime_diversity": diversity,
                "long_strategy_audit_class_if_profitable": long_cls,
                "short_strategy_audit_class_if_profitable": short_cls,
                "long_sharpe_proxy_return": long_sharpe_proxy,
                "short_sharpe_proxy_return": short_sharpe_proxy,
            },
            rationale="Synthesizes the close-array shapes that produce B2's confounds: single-side-correct strategies fail diversity gate in monotone regimes; bidirectional regimes pass for both sides.",
        ))
    return cells


def b2_buy_and_hold_equivalents() -> list[ProbeCell]:
    """Buy-and-hold-equivalent strategies the gate is supposed to catch.

    For each, what does the v3 gate produce? The current gate compares
    asset close against the diversity criterion; it does NOT compare
    strategy equity vs asset returns. A B2 patch must preserve detection
    of these by some means — this matrix bounds the test set the patch
    must satisfy.
    """
    cells = []
    n = 500
    asset_close = np.linspace(80.0, 140.0, n)
    equity_variants = {
        "equity_eq_close": asset_close.copy(),
        "equity_eq_scaled_close_2x": (asset_close - 80.0) * 2.0 + 80.0,
        "equity_eq_lagged_close_1d": np.concatenate([[asset_close[0]], asset_close[:-1]]),
        "equity_eq_smoothed_close_20d": np.convolve(asset_close, np.ones(20) / 20.0, mode="same"),
        "equity_eq_beta1_close": asset_close.copy(),
        "equity_eq_beta_half_close": 0.5 * asset_close + 50.0,
    }
    asset_diversity = _regime_diversity(asset_close)
    for k, eq in equity_variants.items():
        # Empirical correlation between equity-returns and close-returns
        equity_ret = np.diff(eq) / eq[:-1]
        close_ret = np.diff(asset_close) / asset_close[:-1]
        with np.errstate(invalid="ignore"):
            corr = float(np.corrcoef(equity_ret, close_ret)[0, 1])
        cells.append(ProbeCell(
            probe_class="B2_SHORT_BIAS",
            cell_id=f"bnh_equiv_{k}",
            inputs={
                "asset_close_summary": {"start": 80.0, "end": 140.0, "shape": "linear_up_75pct"},
                "equity_construction": k,
            },
            observation={
                "asset_regime_diversity": asset_diversity,
                "empirical_equity_close_return_correlation": corr,
                "asset_returns_pass_gate": asset_diversity["pass"],
            },
            rationale="Buy-and-hold-equivalent test set. A correct B2 patch must keep flagging these as low-information regardless of which side-distribution comparator is used.",
        ))
    return cells


# =============================================================================
# B3 — Event-density / pipeline-geometry constraint mapping
# =============================================================================

def b3_make_folds_feasibility() -> list[ProbeCell]:
    """At what (n_events, n_folds, train_min) does make_folds refuse?

    Maps the constraint boundary directly via the rule
    ``n - train_min < n_folds * 100 -> ValueError``. Documents whether the
    H4-extension failure at n=330 is the exception or the rule.
    """
    cells = []
    # Default geometry from configs/xau_d1.yaml
    default_n_folds = DEFAULT_N_FOLDS
    default_train_min = DEFAULT_TRAIN_MIN
    default_purge = DEFAULT_PURGE_BARS
    default_embargo = DEFAULT_EMBARGO_PCT

    event_counts = [150, 200, 330, 500, 700, 1000, 1500, 2500]
    for n in event_counts:
        try:
            folds = make_folds(n=n, n_folds=default_n_folds, train_min=default_train_min,
                                purge=default_purge, embargo_pct=default_embargo)
            obs = {
                "n_folds_produced": len(folds),
                "raised": False,
                "test_pool_size": n - default_train_min,
                "min_required_test_pool": default_n_folds * 100,
                "fold_test_sizes": [int(len(f.test_idx)) for f in folds],
            }
        except ValueError as e:
            obs = {
                "raised": True,
                "exception_message": str(e),
                "test_pool_size": n - default_train_min,
                "min_required_test_pool": default_n_folds * 100,
            }
        cells.append(ProbeCell(
            probe_class="B3_EVENT_DENSITY",
            cell_id=f"make_folds_n{n}_nf{default_n_folds}_tm{default_train_min}",
            inputs={
                "n_events": n,
                "n_folds": default_n_folds,
                "train_min": default_train_min,
                "purge": default_purge,
                "embargo_pct": default_embargo,
            },
            observation=obs,
            rationale="Maps make_folds feasibility at the spike's default geometry. The rule is n_events - train_min >= n_folds * 100.",
        ))
    return cells


def b3_purged_cv_feasibility() -> list[ProbeCell]:
    """At what (n_train, n_splits, purge) does PurgedTimeSeriesSplit refuse?

    The rule is ``step = n // (n_splits + 1) > purge``. Documents whether
    the H4-extension second-layer failure (purge=40 too large for 165-event
    train slice) is the exception or the rule.
    """
    cells = []
    train_sizes = [50, 100, 165, 250, 400, 800]
    purge_options = [2, 5, 10, 20, 40]
    n_splits = DEFAULT_CV_SPLITS

    for n_train in train_sizes:
        for purge in purge_options:
            X = np.zeros((n_train, 1))
            cv = PurgedTimeSeriesSplit(n_splits=n_splits, purge=purge)
            step = n_train // (n_splits + 1)
            raised = False
            n_yielded = 0
            try:
                for tr, va in cv.split(X):
                    n_yielded += 1
            except ValueError as e:
                raised = True
                exception_message = str(e)
            else:
                exception_message = None
            cells.append(ProbeCell(
                probe_class="B3_EVENT_DENSITY",
                cell_id=f"purged_cv_n{n_train}_ns{n_splits}_p{purge}",
                inputs={"n_train": n_train, "n_splits": n_splits, "purge": purge},
                observation={
                    "raised": raised,
                    "step": step,
                    "step_minus_purge": step - purge,
                    "n_splits_yielded": n_yielded,
                    "exception_message": exception_message,
                },
                rationale="Maps PurgedTimeSeriesSplit feasibility. Constraint: step > purge, where step = n_train // (n_splits + 1).",
            ))
    return cells


# =============================================================================
# B0008 — Expanded probe families (post-B0015 n=3 falsification)
#
# Closes the three "Cells the matrix does NOT cover" gaps documented in
# phase5/audit_null_distribution.md §"Cells the matrix does NOT cover":
#   1. B1_noise_contamination (>=25 cells) — partial-leakage with noise
#   2. B2_real_asset_arrays (>=10 cells) — real XAU/XAG/USDJPY close arrays
#   3. B3_calibrator_geometry (>=10 cells) — CalibratedClassifierCV layer
# =============================================================================

def b1_noise_contamination() -> list[ProbeCell]:
    """Mixed-signal: partial leakage at multiple noise levels.

    Per audit_null_distribution.md §"Cells the matrix does NOT cover" gap (a):
    "per-fold profiles that mix true signal with leakage (e.g., 70% leakage,
    30% noise) are not [covered]".

    For each (leakage_fraction, noise_seed) combination we synthesize a
    per-fold Sharpe profile = leakage_fraction * leakage_sharpe + noise_term.
    This bounds the false-positive rate of any future Sharpe-magnitude gate
    when the input distribution is mixed-signal rather than pure-null.
    """
    cells = []
    leakage_sharpes = [5.0, 10.0, 20.0]  # baseline leakage magnitudes
    contamination_levels = [0.25, 0.50, 0.75]  # mix ratio (1 - leakage_fraction)
    seeds = [0, 1, 2]  # noise variance seeds

    for lk_sharpe in leakage_sharpes:
        for contamination in contamination_levels:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                # Noise component: zero-mean, sigma=0.5 (typical fold-to-fold dispersion)
                noise = rng.normal(0.0, 0.5, size=4)
                # Convex mix of leakage signal and noise: contamination=1 -> pure noise
                per_fold = [
                    (1 - contamination) * lk_sharpe + contamination * n
                    for n in noise
                ]
                obs = _b1_classifier_response(per_fold)
                cells.append(ProbeCell(
                    probe_class="B1_LEAKAGE",
                    cell_id=f"noise_contam_lk{lk_sharpe:.0f}_c{contamination:.2f}_s{seed}",
                    inputs={
                        "leakage_sharpe": lk_sharpe,
                        "contamination_fraction": contamination,
                        "noise_seed": seed,
                        "per_fold_sharpe": per_fold,
                        "n_per_fold": 150,
                        "regime_pass": True,
                    },
                    observation=obs,
                    rationale=(
                        f"Mixed leakage ({lk_sharpe} sharpe at fraction {1-contamination:.0%}) "
                        f"+ noise ({contamination:.0%}). Documents whether the v3 classifier "
                        f"discriminates partial leakage from pure null."
                    ),
                ))
    return cells


def b2_real_asset_arrays() -> list[ProbeCell]:
    """Validate _regime_diversity against REAL XAU/XAG/USDJPY close arrays.

    Per audit_null_distribution.md §"Cells the matrix does NOT cover" gap (b):
    "the close arrays here are synthetic single-asset constructions. A real
    B2 patch must also be validated against the spike's actual XAUUSD /
    XAGUSD / USDJPY close arrays."

    For each (asset, time_window) we load the actual data file and run
    _regime_diversity. Provides the empirical regression suite the future
    B2 patch must satisfy.
    """
    cells = []
    from pipeline.data import load_dataset

    # Test multiple time-window slices of each asset so the matrix covers
    # different regime episodes.
    asset_paths = {
        "XAUUSD": _REPO_ROOT / "data/D1_22y/XAUUSD_D1.csv",
        "XAGUSD": _REPO_ROOT / "data/D1_22y/XAGUSD_D1.csv",
    }
    # Time-window slices to sample different macro regimes
    slices = [
        ("full_history", slice(None, None)),
        ("post_2010", lambda df: df.loc["2010-01-01":]),
        ("2015_2018", lambda df: df.loc["2015-01-01":"2018-12-31"]),
        ("2018_2022", lambda df: df.loc["2018-01-01":"2022-12-31"]),
        ("2020_2023", lambda df: df.loc["2020-01-01":"2023-12-31"]),
    ]

    for asset, path in asset_paths.items():
        if not path.exists():
            cells.append(ProbeCell(
                probe_class="B2_SHORT_BIAS",
                cell_id=f"real_asset_{asset}_not_available",
                inputs={"asset": asset, "data_path": str(path)},
                observation={"data_missing": True},
                rationale="Asset data file absent; matrix records the gap.",
            ))
            continue
        df = load_dataset(path)
        for slice_label, slc in slices:
            if callable(slc):
                sub = slc(df)
            else:
                sub = df.iloc[slc]
            if len(sub) < 100:  # too small for diversity gate
                continue
            close = sub["close"].values.astype(float)
            diversity = _regime_diversity(close)
            cells.append(ProbeCell(
                probe_class="B2_SHORT_BIAS",
                cell_id=f"real_asset_{asset}_{slice_label}",
                inputs={
                    "asset": asset,
                    "time_window": slice_label,
                    "n_bars": int(len(sub)),
                    "start": str(sub.index.min()),
                    "end": str(sub.index.max()),
                    "close_start": float(close[0]),
                    "close_end": float(close[-1]),
                    "close_min": float(close.min()),
                    "close_max": float(close.max()),
                },
                observation={"regime_diversity": diversity},
                rationale=(
                    f"Real {asset} {slice_label} close array. The regime_diversity "
                    f"gate's pass/fail on this real data is the regression-suite "
                    f"input any v3.2 B2 patch must reproduce."
                ),
            ))
    return cells


def b3_calibrator_geometry() -> list[ProbeCell]:
    """Probe CalibratedClassifierCV.cross_val_predict at H4-scale and B0015-scale event counts.

    Per audit_null_distribution.md §"Cells the matrix does NOT cover" gap (c):
    "CalibratedClassifierCV.cross_val_predict is the third pipeline layer and
    the one that ultimately caused the H4 extension's final failure."

    Empirically B0015a hit class-imbalance with 502 events; B0015c with 381.
    Cells sweep (n_events, calib_holdout_pct, class_imbalance, inner_cv_splits)
    at the empirical regions where B0015 sub-items failed.

    The probe records WHETHER RefittingCalibratedPipeline.fit raises
    "requires both classes present in base and calibration slices", which is
    the operative B3 failure mode for D1 alt-data primaries.
    """
    cells = []
    from pipeline.train import RefittingCalibratedPipeline
    import pandas as pd

    # Event counts to probe: full B0015 evidence range + H4 endpoints
    event_counts = [110, 165, 200, 280, 381, 502, 700, 1200]
    # calib_holdout_pct values: default + B0015a-applied + extreme
    holdout_pcts = [0.15, 0.30, 0.40, 0.50]
    # Class imbalance levels (fraction of class 1)
    class_balance_targets = [0.30, 0.50, 0.70]  # 30/70 = realistic skew

    for n in event_counts:
        for hold_pct in holdout_pcts:
            for class_balance in class_balance_targets:
                # Construct synthetic X, y, w with the requested class balance
                rng = np.random.default_rng(hash((n, int(hold_pct*1000), int(class_balance*1000))) & 0xFFFFFFFF)
                n_pos = int(n * class_balance)
                n_neg = n - n_pos
                # Random binary labels with the target class balance, placed
                # in CHRONOLOGICAL order (not shuffled) to match the
                # walk-forward's tail-15% calibration split.
                y = np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])
                # Randomize WITHIN chronological "phases" — first half mostly
                # negative, second half mostly positive — to approximate the
                # selective-primary skew observed in B0015 (most fires cluster
                # in specific historical episodes, NOT uniformly distributed).
                mid = n // 2
                first_half = np.zeros(mid, dtype=int)
                first_half[:int(mid * class_balance * 0.3)] = 1  # few positives in first half
                second_half = np.ones(n - mid, dtype=int)
                second_half[:int((n - mid) * (1 - class_balance) * 0.7)] = 0  # some negatives in second half
                y_skewed = np.concatenate([first_half, second_half])
                X = pd.DataFrame({"f": rng.normal(0, 1, size=n).astype(float)})
                y_series = pd.Series(y_skewed, index=range(n))
                w = np.ones(n)

                # Try to fit RefittingCalibratedPipeline at the target holdout
                pipe = RefittingCalibratedPipeline(
                    model_name="rf",
                    base_kwargs={"n_estimators": 50, "max_depth": 5},
                    calib_holdout_pct=hold_pct,
                    isotonic_min_minority=10,
                    method="sigmoid",
                    random_state=42,
                )
                hold = int(n * hold_pct)
                if hold == 0 or hold == n:
                    obs = {
                        "raised": True,
                        "exception_message": f"degenerate holdout: hold={hold} n={n}",
                        "hold": hold, "n_base": n - hold,
                        "y_base_balance": None, "y_calib_balance": None,
                    }
                else:
                    y_base = y_series.iloc[:n - hold]
                    y_calib = y_series.iloc[n - hold:]
                    y_base_balance = float(y_base.mean()) if len(y_base) else None
                    y_calib_balance = float(y_calib.mean()) if len(y_calib) else None
                    base_classes = int(y_base.nunique())
                    calib_classes = int(y_calib.nunique())
                    raised = (base_classes < 2) or (calib_classes < 2)
                    if raised:
                        # Same condition train.py:187 raises on; record without actually calling .fit
                        obs = {
                            "raised": True,
                            "exception_message": "RefittingCalibratedPipeline.fit requires both classes present in base and calibration slices",
                            "hold": hold, "n_base": n - hold,
                            "y_base_balance": y_base_balance,
                            "y_calib_balance": y_calib_balance,
                            "base_unique_classes": base_classes,
                            "calib_unique_classes": calib_classes,
                        }
                    else:
                        # Actually call .fit to confirm it works in the non-raised case
                        try:
                            pipe.fit(X, y_series, sample_weight=w)
                            obs = {
                                "raised": False,
                                "hold": hold, "n_base": n - hold,
                                "y_base_balance": y_base_balance,
                                "y_calib_balance": y_calib_balance,
                                "base_unique_classes": base_classes,
                                "calib_unique_classes": calib_classes,
                            }
                        except Exception as e:  # noqa: BLE001
                            obs = {
                                "raised": True,
                                "exception_message": str(e),
                                "hold": hold, "n_base": n - hold,
                                "y_base_balance": y_base_balance,
                                "y_calib_balance": y_calib_balance,
                            }
                cells.append(ProbeCell(
                    probe_class="B3_EVENT_DENSITY",
                    cell_id=f"calibrator_n{n}_h{hold_pct:.2f}_cb{class_balance:.2f}",
                    inputs={
                        "n_events": n,
                        "calib_holdout_pct": hold_pct,
                        "class_balance_target": class_balance,
                    },
                    observation=obs,
                    rationale=(
                        f"Calibrator-stage probe at n={n} events, "
                        f"calib_holdout={hold_pct:.0%}, class-balance target {class_balance:.0%}. "
                        f"B0015a/c hit this layer's class-presence check at n=502/381 with hold=0.40. "
                        f"Documents the geometric boundary."
                    ),
                ))
    return cells


# =============================================================================
# B0008.2 — Calibration QUALITY at varying calib_holdout_pct
#
# Required input to B0007 v3.2 patch B3.1 (raise default calib_holdout from
# 0.30 to 0.50). The 96-cell B3_calibrator_geometry matrix measured RAISE
# rate but NOT calibration quality. This family measures Brier score on
# well-balanced synthetic data — bounds the calibration-quality regression
# risk of raising the default holdout from 0.30 to 0.50.
# =============================================================================

def b3_calibration_quality() -> list[ProbeCell]:
    """Measure Brier score at calib_holdout_pct ∈ {0.30, 0.40, 0.50}.

    Methodology:
    - Generate well-balanced synthetic (X, y) where y = 1[X[:, 0] + noise > 0].
    - Train RefittingCalibratedPipeline on 80% of data (chronological).
    - Predict proba on remaining 20% test set; compute Brier score.
    - Sweep n_events ∈ {200, 400, 800} × calib_holdout ∈ {0.30, 0.40, 0.50}.

    Cells: 9 (3 × 3). Cell count is minimal per BLOCK verdict's >=5 per family,
    but sufficient to bound the regression because each cell trains a real
    model and the signal (Brier delta) is large if present.

    Documents the cost of raising calib_holdout. Required for B3.1
    acceptance criterion (calibration regression ≤ 10%).
    """
    cells = []
    from pipeline.train import RefittingCalibratedPipeline
    import pandas as pd

    event_counts = [200, 400, 800]
    holdout_pcts = [0.30, 0.40, 0.50]

    for n in event_counts:
        for hold_pct in holdout_pcts:
            # Well-balanced informative synthetic data
            rng = np.random.default_rng(hash((n, int(hold_pct * 1000), 42)) & 0xFFFFFFFF)
            X_raw = rng.normal(0, 1, size=(n, 3))
            # y = 1[linear combo + noise > 0], well-balanced by construction
            logits = 0.8 * X_raw[:, 0] + 0.4 * X_raw[:, 1] - 0.3 * X_raw[:, 2] + rng.normal(0, 0.5, size=n)
            y = (logits > np.median(logits)).astype(int)  # forces balanced
            X = pd.DataFrame(X_raw, columns=["f0", "f1", "f2"])
            y_s = pd.Series(y, index=range(n))
            w = np.ones(n)

            # 80/20 train/test split (chronological)
            split = int(0.8 * n)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y_s.iloc[:split], y_s.iloc[split:]
            w_train = w[:split]

            pipe = RefittingCalibratedPipeline(
                model_name="rf",
                base_kwargs={"n_estimators": 50, "max_depth": 5},
                calib_holdout_pct=hold_pct,
                isotonic_min_minority=10,
                method="sigmoid",
                random_state=42,
            )

            try:
                pipe.fit(X_train, y_train, sample_weight=w_train)
                proba = pipe.predict_proba(X_test)[:, 1]
                # Brier score: mean((y_true - p)^2). Lower is better.
                brier = float(((y_test.values - proba) ** 2).mean())
                # Calibration-MAD: mean absolute deviation of bucket-mean from bucket-empirical
                bins = np.linspace(0, 1, 11)
                bucket = np.digitize(proba, bins) - 1
                bucket = np.clip(bucket, 0, 9)
                cal_devs = []
                for b in range(10):
                    mask = bucket == b
                    if mask.sum() < 3:
                        continue
                    bucket_p_mean = float(proba[mask].mean())
                    bucket_y_mean = float(y_test.values[mask].mean())
                    cal_devs.append(abs(bucket_p_mean - bucket_y_mean))
                cal_mad = float(np.mean(cal_devs)) if cal_devs else float("nan")
                obs = {
                    "raised": False,
                    "brier_score": brier,
                    "calibration_mad": cal_mad,
                    "n_train": split,
                    "n_test": n - split,
                    "n_calib_tail": int(split * hold_pct),
                    "n_base": split - int(split * hold_pct),
                }
            except Exception as e:  # noqa: BLE001
                obs = {
                    "raised": True,
                    "exception_message": str(e),
                    "n_train": split,
                }

            cells.append(ProbeCell(
                probe_class="B3_EVENT_DENSITY",
                cell_id=f"calibration_quality_n{n}_h{hold_pct:.2f}",
                inputs={
                    "n_events": n,
                    "calib_holdout_pct": hold_pct,
                    "data_kind": "well_balanced_informative_synthetic",
                },
                observation=obs,
                rationale=(
                    f"Brier + calibration-MAD on well-balanced synthetic at "
                    f"n={n}, hold={hold_pct:.0%}. Required for B0007 v3.2 "
                    f"B3.1 acceptance criterion: regression vs 0.30 baseline ≤ 10%."
                ),
            ))
    return cells


# =============================================================================
# B0008.3 — Calibrated vs uncalibrated SR delta (fallback bias estimation)
#
# Required input to B0007 v3.2 patch B3.2 (fall back to uncalibrated when
# class-imbalance detected). Measures the SR-inflation bias introduced by
# uncalibrated proba at threshold-selection vs calibrated. Per the
# preregistration acceptance criterion: fallback acceptable if SR-inflation
# bounded at ≤ 20%.
# =============================================================================

def b3_calibration_bypass_sr_delta() -> list[ProbeCell]:
    """Measure SR delta uncalibrated vs calibrated probas.

    Methodology:
    - Generate synthetic well-balanced (X, y) + synthetic returns.
    - Train RefittingCalibratedPipeline successfully (no fallback fired).
    - Get calibrated probas (post-pipeline).
    - Get uncalibrated probas from the raw base estimator (refit fresh on
      the same train slice without the calibration step).
    - For each threshold τ ∈ {0.50, 0.55, 0.60, 0.65}: compute SR_calibrated(τ)
      vs SR_uncalibrated(τ) on the same held-out returns.
    - Report max SR-delta = max(|SR_unc - SR_cal| / |SR_cal|) per cell.

    Cells: 12 (3 event-counts × 4 thresholds). 3 × n × 4 thresholds = 12.

    Bounds the cost of B3.2 fallback: if uncalibrated proba inflates Sharpe
    significantly, the fallback is a worse cure than the disease.
    """
    cells = []
    from pipeline.train import RefittingCalibratedPipeline, MODEL_FACTORIES
    import pandas as pd

    event_counts = [200, 400, 800]
    thresholds = [0.50, 0.55, 0.60, 0.65]

    for n in event_counts:
        # Synthetic data
        rng = np.random.default_rng(hash((n, 99)) & 0xFFFFFFFF)
        X_raw = rng.normal(0, 1, size=(n, 3))
        logits = 0.6 * X_raw[:, 0] + 0.3 * X_raw[:, 1] + rng.normal(0, 0.5, size=n)
        y = (logits > np.median(logits)).astype(int)
        # Synthetic returns aligned to label sign + noise (per-trade pnl scale)
        rets = (2 * y - 1) * 0.005 + rng.normal(0, 0.01, size=n)
        X = pd.DataFrame(X_raw, columns=["f0", "f1", "f2"])
        y_s = pd.Series(y, index=range(n))

        split = int(0.8 * n)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y_s.iloc[:split], y_s.iloc[split:]
        rets_test = rets[split:]

        # Calibrated path
        pipe = RefittingCalibratedPipeline(
            model_name="rf",
            base_kwargs={"n_estimators": 50, "max_depth": 5},
            calib_holdout_pct=0.30,
            isotonic_min_minority=10,
            method="sigmoid",
            random_state=42,
        )
        try:
            pipe.fit(X_train, y_train, sample_weight=np.ones(split))
            proba_cal = pipe.predict_proba(X_test)[:, 1]
        except Exception:  # noqa: BLE001
            for thr in thresholds:
                cells.append(ProbeCell(
                    probe_class="B3_EVENT_DENSITY",
                    cell_id=f"sr_delta_n{n}_t{thr:.2f}_calib_failed",
                    inputs={"n_events": n, "threshold": thr, "data_kind": "synth_balanced"},
                    observation={"raised": True},
                    rationale="Calibrated pipeline failed to fit; SR-delta unmeasurable.",
                ))
            continue

        # Uncalibrated path: train raw base on full train slice
        base = MODEL_FACTORIES["rf"](
            random_state=42, n_estimators=50, max_depth=5,
        )
        base.fit(X_train, y_train, sample_weight=np.ones(split))
        proba_uncal = base.predict_proba(X_test)[:, 1]

        for thr in thresholds:
            take_cal = proba_cal >= thr
            take_unc = proba_uncal >= thr
            # Strategy pnl: take the trade if take=True; sign by primary direction.
            pnl_cal = rets_test * take_cal
            pnl_unc = rets_test * take_unc
            n_cal_trades = int(take_cal.sum())
            n_unc_trades = int(take_unc.sum())
            # Sharpe per CLAUDE.md: sqrt(trades_per_year), NaN when n<30
            years = (n - split) / 252.0
            if n_cal_trades >= 30 and pnl_cal[take_cal].std() > 1e-12:
                sr_cal = float(pnl_cal[take_cal].mean() / pnl_cal[take_cal].std() * np.sqrt(n_cal_trades / max(years, 1e-9)))
            else:
                sr_cal = float("nan")
            if n_unc_trades >= 30 and pnl_unc[take_unc].std() > 1e-12:
                sr_unc = float(pnl_unc[take_unc].mean() / pnl_unc[take_unc].std() * np.sqrt(n_unc_trades / max(years, 1e-9)))
            else:
                sr_unc = float("nan")
            if np.isfinite(sr_cal) and np.isfinite(sr_unc) and abs(sr_cal) > 1e-6:
                sr_delta_pct = float(abs(sr_unc - sr_cal) / abs(sr_cal))
            else:
                sr_delta_pct = float("nan")
            cells.append(ProbeCell(
                probe_class="B3_EVENT_DENSITY",
                cell_id=f"sr_delta_n{n}_t{thr:.2f}",
                inputs={"n_events": n, "threshold": thr, "data_kind": "synth_balanced_returns"},
                observation={
                    "raised": False,
                    "sr_calibrated": sr_cal,
                    "sr_uncalibrated": sr_unc,
                    "abs_sr_delta_pct": sr_delta_pct,
                    "n_trades_calibrated": n_cal_trades,
                    "n_trades_uncalibrated": n_unc_trades,
                },
                rationale=(
                    f"Calibrated vs uncalibrated SR at n={n}, threshold={thr}. "
                    f"Bounds the SR-inflation bias if B3.2 fallback fires. "
                    f"Acceptance criterion: |delta|/|sr_cal| ≤ 0.20."
                ),
            ))
    return cells


# =============================================================================
# Orchestration
# =============================================================================

PROBE_FAMILIES = {
    "B1_magnitude_sweep": b1_magnitude_sweep,
    "B1_dispersion_sweep": b1_dispersion_sweep,
    "B1_known_nulls": b1_known_nulls,
    "B1_noise_contamination": b1_noise_contamination,             # B0008
    "B2_regime_close_arrays": b2_regime_close_arrays,
    "B2_buy_and_hold_equivalents": b2_buy_and_hold_equivalents,
    "B2_real_asset_arrays": b2_real_asset_arrays,                 # B0008
    "B3_make_folds_feasibility": b3_make_folds_feasibility,
    "B3_purged_cv_feasibility": b3_purged_cv_feasibility,
    "B3_calibrator_geometry": b3_calibrator_geometry,             # B0008
    "B3_calibration_quality": b3_calibration_quality,             # B0008.2 (B0007 v3.2 input)
    "B3_calibration_bypass_sr_delta": b3_calibration_bypass_sr_delta,  # B0008.3 (B0007 v3.2 input)
}


def _serialize(obj):
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return [_serialize(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def run_matrix(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix: dict[str, list[dict]] = {}
    n_cells = 0
    for family_name, fn in PROBE_FAMILIES.items():
        cells = fn()
        matrix[family_name] = [
            {
                "probe_class": c.probe_class,
                "cell_id": c.cell_id,
                "inputs": _serialize(c.inputs),
                "observation": _serialize(c.observation),
                "rationale": c.rationale,
            }
            for c in cells
        ]
        n_cells += len(cells)
        print(f"  {family_name}: {len(cells)} cells")

    summary = {
        "schema_version": "phase5.audit_null_distribution.v1",
        "purpose": "Empirical-null-distribution probe matrix for M3 v3 blindspots B1/B2/B3. NOT a v3.2 patch design.",
        "gating_verdict": "signals/devils_advocate_reviews/20260526-audit-patch-scope-decision_v1.json",
        "n_probe_families": len(PROBE_FAMILIES),
        "n_cells_total": n_cells,
        "families": matrix,
    }
    out_path = out_dir / "audit_null_distribution.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path} ({n_cells} cells across {len(PROBE_FAMILIES)} families)")
    return summary


def main() -> int:
    out_dir = _REPO_ROOT / "results" / "phase5"
    run_matrix(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
