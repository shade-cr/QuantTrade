import pandas as pd
from pathlib import Path
from pipeline.equity_source import normalize_ohlcv, write_contract_csv
from pipeline.data import load_dataset


def _fake_yf_frame() -> pd.DataFrame:
    # yfinance returns a (field, ticker) MultiIndex even for one ticker.
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-03", "2020-01-06"])
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["NVDA"]])
    data = [
        [10, 11, 9, 10.5, 1000],
        [10.5, 12, 10, 11.5, 1200],
        [10.5, 12, 10, 11.5, 1200],   # duplicate timestamp -> keep last
        [11.5, 13, 11, 12.5, 1500],
    ]
    return pd.DataFrame(data, index=idx, columns=cols)


def test_normalize_ohlcv_contract():
    out = normalize_ohlcv(_fake_yf_frame())
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert str(out.index.tz) == "UTC"
    assert out.index.is_monotonic_increasing
    assert not out.index.duplicated().any()
    assert (out[["open", "high", "low", "close"]] > 0).all().all()
    assert out.dtypes.apply(lambda d: d == "float64").all()


def test_round_trips_through_load_dataset(tmp_path: Path):
    out = normalize_ohlcv(_fake_yf_frame())
    csv = tmp_path / "NVDA_D1.csv"
    write_contract_csv(out, csv)
    loaded = load_dataset(csv)
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert len(loaded) == 3  # 4 rows minus 1 duplicate
