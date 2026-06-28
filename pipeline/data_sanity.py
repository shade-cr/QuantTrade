"""Pre-train data validation for multi-asset H4 OHLCV (Phase 2 T1).

Two failure modes the validator distinguishes:
  - HARD errors (raise DataValidationError): the data is unusable.
    Examples: non-positive prices, NaN in OHLC, empty frame.
  - SOFT warnings (collected in a report dict): the data is usable but
    flagged. Examples: fewer bars than expected, OHLC inconsistencies
    (high < max(open, close)), excess zero-volume bars, unexpected gaps.

The cross-asset check is separate from per-asset checks because it only
makes sense when comparing all 8 assets at once (intersection of
timestamps).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class DataValidationError(Exception):
    """Raised on hard data-quality errors (non-positive prices, NaN OHLC)."""


_OHLC_COLS = ("open", "high", "low", "close")


def _check_hard_errors(asset: str, df: pd.DataFrame) -> None:
    """Raise DataValidationError on any hard issue. Returns silently if OK."""
    if len(df) == 0:
        raise DataValidationError(f"{asset}: empty frame")
    for col in _OHLC_COLS:
        if col not in df.columns:
            raise DataValidationError(f"{asset}: missing column {col!r}")
        col_vals = df[col]
        if col_vals.isna().any():
            raise DataValidationError(f"{asset}.{col}: NaN detected")
        if (col_vals <= 0).any():
            raise DataValidationError(f"{asset}.{col}: non-positive prices detected")


def validate_asset_data(
    asset: str,
    df: pd.DataFrame,
    *,
    is_crypto: bool = False,
    min_bars: int = 5000,
    min_years: float = 3.0,
    expected_bar_hours: float = 4.0,
    max_unexpected_gaps: int = 50,
    max_zero_volume_pct: float = 0.05,
) -> dict:
    """Validate a single asset's H4 OHLCV.

    Hard errors raise immediately. Soft warnings are accumulated in
    `report["warnings"]` and returned alongside summary stats.
    """
    _check_hard_errors(asset, df)

    warnings: list[dict] = []
    n_bars = len(df)

    if n_bars < min_bars:
        warnings.append({
            "check": "n_bars",
            "value": n_bars,
            "threshold": min_bars,
            "message": f"only {n_bars} bars (expected ≥ {min_bars})",
        })

    # Coverage in years.
    if isinstance(df.index, pd.DatetimeIndex) and n_bars >= 2:
        span_days = (df.index.max() - df.index.min()).days
        n_years = span_days / 365.25
        if n_years < min_years:
            warnings.append({
                "check": "n_years",
                "value": n_years,
                "threshold": min_years,
                "message": f"only {n_years:.1f} years of history",
            })
    else:
        n_years = None

    # OHLC consistency: high must be >= max(open, close); low must be <= min(open, close).
    bad_hi = int((df["high"] < df[["open", "close"]].max(axis=1)).sum())
    if bad_hi > 0:
        warnings.append({
            "check": "high_lt_max_open_close",
            "value": bad_hi,
            "message": f"{bad_hi} bars with high < max(open, close)",
        })
    bad_lo = int((df["low"] > df[["open", "close"]].min(axis=1)).sum())
    if bad_lo > 0:
        warnings.append({
            "check": "low_gt_min_open_close",
            "value": bad_lo,
            "message": f"{bad_lo} bars with low > min(open, close)",
        })

    # Volume sanity.
    if "volume" in df.columns:
        zero_vol_pct = float((df["volume"] == 0).mean())
        if zero_vol_pct > max_zero_volume_pct:
            warnings.append({
                "check": "zero_volume_pct",
                "value": zero_vol_pct,
                "threshold": max_zero_volume_pct,
                "message": f"{zero_vol_pct:.1%} bars with zero volume",
            })
    else:
        zero_vol_pct = None

    # Gaps: anything larger than 1.5x expected_bar_hours that is NOT a weekend gap.
    if isinstance(df.index, pd.DatetimeIndex) and n_bars >= 2:
        gaps_hours = df.index.to_series().diff().dropna().dt.total_seconds() / 3600
        expected = expected_bar_hours
        if is_crypto:
            # Crypto: any gap > 1.5x expected is unexpected (no weekends).
            unexpected = (gaps_hours > expected * 1.5).sum()
        else:
            # FX/metal: skip the ~48-72h weekend gap; flag anything in between.
            weekend_hours = 48.0
            unexpected = ((gaps_hours > expected * 1.5) & (gaps_hours < weekend_hours * 0.8)).sum()
        n_unexpected = int(unexpected)
        if n_unexpected > max_unexpected_gaps:
            warnings.append({
                "check": "unexpected_gaps",
                "value": n_unexpected,
                "threshold": max_unexpected_gaps,
                "message": f"{n_unexpected} unexpected gaps (> {expected * 1.5:.1f}h)",
            })
    else:
        n_unexpected = None

    return {
        "asset": asset,
        "n_bars": n_bars,
        "n_years": n_years,
        "zero_volume_pct": zero_vol_pct,
        "n_unexpected_gaps": n_unexpected,
        "warnings": warnings,
    }


def validate_multi_asset_data(
    asset_dfs: dict,
    *,
    crypto_assets: set | None = None,
    min_bars: int = 5000,
    min_common_bars: int = 5000,
    **kwargs,
) -> dict:
    """Run per-asset checks on each frame, then verify cross-asset alignment.

    Per-asset hard errors propagate (raise DataValidationError).
    """
    crypto_assets = crypto_assets or set()
    per_asset: dict = {}
    for asset, df in asset_dfs.items():
        per_asset[asset] = validate_asset_data(
            asset, df,
            is_crypto=(asset in crypto_assets),
            min_bars=min_bars,
            **kwargs,
        )

    # Cross-asset: intersection of timestamps across all frames.
    cross_warnings: list[dict] = []
    n_common = 0
    if asset_dfs:
        common = None
        for df in asset_dfs.values():
            idx_set = set(df.index)
            common = idx_set if common is None else common & idx_set
        n_common = len(common) if common is not None else 0
        if n_common < min_common_bars:
            cross_warnings.append({
                "check": "n_common_bars",
                "value": n_common,
                "threshold": min_common_bars,
                "message": (
                    f"only {n_common} timestamps common to all "
                    f"{len(asset_dfs)} assets (expected ≥ {min_common_bars})"
                ),
            })

    return {
        "per_asset": per_asset,
        "cross_asset": {
            "n_common_bars": n_common,
            "warnings": cross_warnings,
        },
    }
