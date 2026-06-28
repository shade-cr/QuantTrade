"""Tests for pipeline.data.load_dataset."""
from __future__ import annotations
import pandas as pd
import pytest

from pipeline.data import load_dataset, DataValidationError


def test_load_dataset_returns_indexed_dataframe(tmp_csv):
    df = load_dataset(tmp_csv)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].dtype == "float64"


def test_load_dataset_accepts_timestamps_column_alias(tmp_path, synth_ohlcv):
    # Kronos exports use 'timestamps' (plural); MT5 exports use 'time'. Both should work.
    p = tmp_path / "alt.csv"
    synth_ohlcv.rename(columns={"time": "timestamps"}).to_csv(p, index=False)
    df = load_dataset(p)
    assert len(df) == len(synth_ohlcv)


def test_load_dataset_rejects_missing_column(tmp_path, synth_ohlcv):
    p = tmp_path / "broken.csv"
    synth_ohlcv.drop(columns=["volume"]).to_csv(p, index=False)
    with pytest.raises(DataValidationError, match="volume"):
        load_dataset(p)


def test_load_dataset_rejects_non_monotonic_time(tmp_path, synth_ohlcv):
    p = tmp_path / "unsorted.csv"
    synth_ohlcv.sample(frac=1, random_state=1).to_csv(p, index=False)  # shuffled
    with pytest.raises(DataValidationError, match="monoton"):
        load_dataset(p)
