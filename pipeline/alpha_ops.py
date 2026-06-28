"""Operator vocabulary for the WorldQuant-101 alpha screen.

All operators are CAUSAL: a value at bar t uses only data with timestamp <= t.
Cross-sectional operators (rank, scale) act across symbols at a fixed timestamp
in mode="xs"; in mode="per_asset" they degrade to within-asset rolling analogs
so the same alpha expression yields a per-instrument time-series signal.
"""
from __future__ import annotations
import functools
import numpy as np
import pandas as pd

PERIODS_PER_YEAR_D1 = 252


class AlphaContext:
    def __init__(self, fields: dict[str, pd.DataFrame], mode: str, ts_window: int = 252):
        if mode not in ("xs", "per_asset"):
            raise ValueError(f"mode must be 'xs' or 'per_asset', got {mode!r}")
        self.open = fields["open"]
        self.high = fields["high"]
        self.low = fields["low"]
        self.close = fields["close"]
        self.volume = fields["volume"]
        self.mode = mode
        self.ts_window = ts_window

    @functools.cached_property
    def returns(self) -> pd.DataFrame:
        return self.close.pct_change()

    @functools.cached_property
    def vwap(self) -> pd.DataFrame:
        """Proxy for typical price (no true VWAP available)."""
        return (self.high + self.low + self.close) / 3.0

    # --- element-wise helpers ---
    @staticmethod
    def sign(x: pd.DataFrame) -> pd.DataFrame:
        return np.sign(x)

    @staticmethod
    def log(x: pd.DataFrame) -> pd.DataFrame:
        return np.log(x)

    @staticmethod
    def abs_(x: pd.DataFrame) -> pd.DataFrame:
        return x.abs()

    @staticmethod
    def signedpower(x: pd.DataFrame, a: float) -> pd.DataFrame:
        return np.sign(x) * (x.abs() ** a)

    # --- cross-sectional ops (mode-dependent) ---
    def rank(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.mode == "xs":
            return x.rank(axis=1, pct=True)
        # per_asset: within-asset rolling percentile of the latest value
        return x.rolling(self.ts_window, min_periods=self.ts_window).apply(
            lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=True
        )

    def scale(self, x: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
        if self.mode == "xs":
            denom = x.abs().sum(axis=1).replace(0, np.nan)
            return x.mul(a).div(denom, axis=0)
        denom = x.abs().rolling(self.ts_window, min_periods=1).sum().replace(0, np.nan)
        return x.mul(a).div(denom)

    # --- rolling time-series operators ---
    def delay(self, x: pd.DataFrame, d: int) -> pd.DataFrame:
        return x.shift(d)

    def delta(self, x: pd.DataFrame, d: int) -> pd.DataFrame:
        return x - x.shift(d)

    def ts_sum(self, x: pd.DataFrame, d: int) -> pd.DataFrame:
        return x.rolling(d, min_periods=d).sum()

    def ts_mean(self, x: pd.DataFrame, d: int) -> pd.DataFrame:
        return x.rolling(d, min_periods=d).mean()

    def stddev(self, x: pd.DataFrame, d: int) -> pd.DataFrame:
        return x.rolling(d, min_periods=d).std()

    def ts_min(self, x: pd.DataFrame, d: int) -> pd.DataFrame:
        return x.rolling(d, min_periods=d).min()

    def ts_max(self, x: pd.DataFrame, d: int) -> pd.DataFrame:
        return x.rolling(d, min_periods=d).max()

    def ts_argmin(self, x: pd.DataFrame, d: int) -> pd.DataFrame:
        return x.rolling(d, min_periods=d).apply(np.argmin, raw=True)

    def ts_argmax(self, x: pd.DataFrame, d: int) -> pd.DataFrame:
        return x.rolling(d, min_periods=d).apply(np.argmax, raw=True)

    def ts_rank(self, x: pd.DataFrame, d: int) -> pd.DataFrame:
        return x.rolling(d, min_periods=d).apply(
            lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=True
        )

    def product(self, x: pd.DataFrame, d: int) -> pd.DataFrame:
        return x.rolling(d, min_periods=d).apply(np.prod, raw=True)

    def decay_linear(self, x: pd.DataFrame, d: int) -> pd.DataFrame:
        w = np.arange(1, d + 1, dtype=float)
        w /= w.sum()
        return x.rolling(d, min_periods=d).apply(lambda a: float(np.dot(a, w)), raw=True)

    def correlation(self, x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
        return x.rolling(d, min_periods=d).corr(y)

    def covariance(self, x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
        return x.rolling(d, min_periods=d).cov(y)

    def adv(self, d: int) -> pd.DataFrame:
        return self.volume.rolling(d, min_periods=d).mean()
