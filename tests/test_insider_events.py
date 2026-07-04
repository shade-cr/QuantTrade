"""B0018: PIT opportunistic-insider event machinery."""
import pandas as pd
import pytest

from pipeline.insider_events import (
    INSIDER_FLOW_FEATURES,
    MIN_NOTIONAL_USD,
    classify_opportunistic,
    insider_flow_features,
    opportunistic_knowledge_days,
    qualifying_filings,
)


def _rows(specs):
    """specs: list of (accession, owner, acceptance_utc, trans_date_str, shares, price)."""
    return pd.DataFrame(
        {
            "accession": [s[0] for s in specs],
            "owner_cik": [s[1] for s in specs],
            "acceptance_datetime": pd.to_datetime(
                [s[2] for s in specs], utc=True
            ),
            "trans_date": [s[3] for s in specs],
            "shares": [float(s[4]) for s in specs],
            "price": [float(s[5]) for s in specs],
        }
    )


def test_notional_floor_drops_token_purchases():
    df = _rows(
        [
            ("A1", "O1", "2010-03-02 14:00", "01-MAR-2010", 100, 50.0),   # 5k
            ("A2", "O1", "2010-06-02 14:00", "01-JUN-2010", 400, 50.0),   # 20k
        ]
    )
    f = qualifying_filings(df)
    assert list(f["accession"]) == ["A2"]
    assert (f["notional"] >= MIN_NOTIONAL_USD).all()


def test_multi_row_filing_notional_is_summed():
    df = _rows(
        [
            ("A1", "O1", "2010-03-02 14:00", "01-MAR-2010", 150, 40.0),  # 6k
            ("A1", "O1", "2010-03-02 14:00", "02-MAR-2010", 150, 40.0),  # +6k
        ]
    )
    f = qualifying_filings(df)
    assert len(f) == 1 and f["notional"].iloc[0] == pytest.approx(12000.0)


def _yearly(owner, years, month, accession_prefix):
    """One 20k purchase per year, same calendar month."""
    return [
        (f"{accession_prefix}{y}", owner, f"{y}-{month:02d}-15 14:00",
         f"15-{pd.Timestamp(2000, month, 1).strftime('%b').upper()}-{y}", 400, 50.0)
        for y in years
    ]


def test_routine_insider_excluded_opportunistic_kept():
    # O1 buys every March 2007-2010 -> the 2010 filing is classifiable+routine.
    # O2 buys in different months 2007-2010 -> 2010 filing is opportunistic.
    routine = _yearly("O1", [2007, 2008, 2009, 2010], 3, "R")
    opp = [
        ("P2007", "O2", "2007-02-15 14:00", "15-FEB-2007", 400, 50.0),
        ("P2008", "O2", "2008-07-15 14:00", "15-JUL-2008", 400, 50.0),
        ("P2009", "O2", "2009-11-15 14:00", "15-NOV-2009", 400, 50.0),
        ("P2010", "O2", "2010-05-15 14:00", "15-MAY-2010", 400, 50.0),
    ]
    out = classify_opportunistic(qualifying_filings(_rows(routine + opp)))
    got = dict(zip(out["accession"], out["opportunistic"]))
    assert got["R2010"] is False or got["R2010"] == False  # noqa: E712
    assert bool(got["P2010"]) is True


def test_unclassifiable_excluded_by_default_admitted_under_contingency():
    df = _rows(
        [
            ("A1", "O1", "2009-03-02 14:00", "01-MAR-2009", 400, 50.0),
            ("A2", "O1", "2010-06-02 14:00", "01-JUN-2010", 400, 50.0),
        ]
    )
    strict = classify_opportunistic(qualifying_filings(df))
    assert not strict["opportunistic"].any()
    relaxed = classify_opportunistic(
        qualifying_filings(df), admit_unclassifiable=True
    )
    assert relaxed["opportunistic"].all()


def test_classification_is_pit_prior_filings_only():
    # O1's 2010 filing must not be classified using the LATER 2011-2013 ones.
    specs = _yearly("O1", [2010, 2011, 2012, 2013], 3, "A")
    out = classify_opportunistic(qualifying_filings(_rows(specs)))
    got = dict(zip(out["accession"], out["opportunistic"]))
    assert bool(got["A2010"]) is False  # unclassifiable at its own time
    assert bool(got["A2013"]) is False  # routine by then


def test_after_close_acceptance_rolls_knowledge_day():
    # non-routine history: one purchase per prior year, different months
    base = [
        ("H2007", "O1", "2007-02-15 14:00", "15-FEB-2007", 400, 50.0),
        ("H2008", "O1", "2008-07-15 14:00", "15-JUL-2008", 400, 50.0),
        ("H2009", "O1", "2009-11-15 14:00", "15-NOV-2009", 400, 50.0),
    ]
    late = [("L1", "O1", "2010-04-05 21:30", "05-APR-2010", 400, 50.0)]
    kd = opportunistic_knowledge_days(_rows(base + late))
    assert pd.Timestamp("2010-04-06") in kd  # 21:30 UTC >= close -> next day


def test_flow_features_count_and_never_nan():
    idx = pd.date_range("2010-01-01", periods=100, freq="B", tz="UTC")
    kd = pd.DatetimeIndex([pd.Timestamp("2010-02-01")])
    f = insider_flow_features(idx, kd)
    assert list(f.columns) == list(INSIDER_FLOW_FEATURES)
    assert not f.isna().any().any()
    on_day = f.loc[idx.normalize().tz_localize(None) == pd.Timestamp("2010-02-01")]
    assert (on_day["opp_insider_buys_21d"] == 1.0).all()
    after = f.loc[idx.normalize().tz_localize(None) > pd.Timestamp("2010-03-01")]
    assert (after["opp_insider_buys_21d"] == 0.0).all()
    assert (
        f.loc[idx.normalize().tz_localize(None) == pd.Timestamp("2010-03-20"),
              "opp_insider_buys_63d"] == 1.0
    ).all()
