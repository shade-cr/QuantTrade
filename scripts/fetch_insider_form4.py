"""Fetch Form 4 open-market purchases (code P) for the M3W universe — B0018.

Two stages:

1. **DERA insider-transactions data sets** (quarterly TSV zips, 2006q1+):
   SUBMISSION x REPORTINGOWNER x NONDERIV_TRANS, filtered to
   DOCUMENT_TYPE == "4" (originals only, no 4/A), TRANS_CODE == "P",
   TRANS_ACQUIRED_DISP_CD == "A", officer/director filers, universe issuer
   CIKs. Each quarter's filtered slice is cached as
   data/insider/dera/<quarter>.parquet (zip deleted after processing), so
   reruns only touch missing quarters.

2. **EDGAR acceptance datetimes**: the DERA sets carry only FILING_DATE
   (day granularity). The PIT knowledge moment needs the timestamp (a
   filing accepted after the close must roll to the next session), so we
   fetch the -index page of each surviving accession and parse the
   "Accepted" field. Only qualifying candidates survive to this stage —
   open-market insider buys in mega caps are rare, so this is a small set.

Output: data/insider/form4_p.parquet — one row per (accession, transaction),
columns: accession, issuer_cik, ticker, filing_date, acceptance_datetime
(UTC), owner_cik, owner_name, relationship, trans_date, shares, price.

Run: uv run python scripts/fetch_insider_form4.py
"""
from __future__ import annotations

import io
import re
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UA = {"User-Agent": "QuantTrade research lc@virtualretail.io"}
SLEEP_S = 0.15
DERA_URL = (
    "https://www.sec.gov/files/structureddata/data/"
    "insider-transactions-data-sets/{q}_form345.zip"
)
UNIVERSE_YAML = ROOT / "configs" / "universe_equity_m3w.yaml"
OUT_DIR = ROOT / "data" / "insider"
DERA_DIR = OUT_DIR / "dera"
OUT_PARQUET = OUT_DIR / "form4_p.parquet"
# fail-loud if a quarter this old or older is missing (tail quarters may not
# be published yet and are tolerated)
REQUIRED_THROUGH = "2025q4"


def _universe_tickers() -> list[str]:
    cfg = yaml.safe_load(UNIVERSE_YAML.read_text())
    return [t.upper() for t in cfg["stocks"]]


def _ticker_cik_map(tickers: list[str]) -> dict[int, str]:
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=60
    )
    r.raise_for_status()
    want = set(tickers)
    out: dict[int, str] = {}
    for row in r.json().values():
        t = row["ticker"].upper()
        if t in want:
            out[int(row["cik_str"])] = t
    missing = want - set(out.values())
    if missing:
        raise RuntimeError(f"no CIK found for tickers: {sorted(missing)}")
    return out


def _quarters() -> list[str]:
    today = date.today()
    qs = []
    for y in range(2006, today.year + 1):
        for q in range(1, 5):
            if (y, q) > (today.year, (today.month - 1) // 3 + 1):
                break
            qs.append(f"{y}q{q}")
    return qs


def _read_tsv(zf: zipfile.ZipFile, name: str, usecols: list[str]) -> pd.DataFrame:
    with zf.open(name) as fh:
        return pd.read_csv(
            io.TextIOWrapper(fh, encoding="utf-8", errors="replace"),
            sep="\t",
            usecols=usecols,
            dtype=str,
            low_memory=False,
        )


def _process_quarter(q: str, cik_to_ticker: dict[int, str]) -> pd.DataFrame | None:
    """Download + filter one DERA quarter. Returns None on 404 (unpublished)."""
    out_pq = DERA_DIR / f"{q}.parquet"
    if out_pq.exists():
        return pd.read_parquet(out_pq)
    r = requests.get(DERA_URL.format(q=q), headers=UA, timeout=300)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))

    sub = _read_tsv(
        zf,
        "SUBMISSION.tsv",
        ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK",
         "ISSUERTRADINGSYMBOL"],
    )
    sub["ISSUERCIK"] = pd.to_numeric(sub["ISSUERCIK"], errors="coerce")
    sub = sub[
        (sub["DOCUMENT_TYPE"] == "4") & sub["ISSUERCIK"].isin(cik_to_ticker)
    ].copy()
    if sub.empty:
        empty = pd.DataFrame()
        empty.to_parquet(out_pq)
        return empty
    sub["ticker"] = sub["ISSUERCIK"].astype(int).map(cik_to_ticker)

    own = _read_tsv(
        zf,
        "REPORTINGOWNER.tsv",
        ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
         "RPTOWNER_RELATIONSHIP"],
    )
    rel = own["RPTOWNER_RELATIONSHIP"].fillna("").str.lower()
    own = own[rel.str.contains("officer") | rel.str.contains("director")]

    trans = _read_tsv(
        zf,
        "NONDERIV_TRANS.tsv",
        ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE",
         "TRANS_ACQUIRED_DISP_CD", "TRANS_SHARES", "TRANS_PRICEPERSHARE"],
    )
    trans = trans[
        (trans["TRANS_CODE"] == "P") & (trans["TRANS_ACQUIRED_DISP_CD"] == "A")
    ]

    df = (
        trans.merge(sub, on="ACCESSION_NUMBER")
        .merge(own, on="ACCESSION_NUMBER")
    )
    df = df.rename(
        columns={
            "ACCESSION_NUMBER": "accession",
            "ISSUERCIK": "issuer_cik",
            "FILING_DATE": "filing_date",
            "RPTOWNERCIK": "owner_cik",
            "RPTOWNERNAME": "owner_name",
            "RPTOWNER_RELATIONSHIP": "relationship",
            "TRANS_DATE": "trans_date",
            "TRANS_SHARES": "shares",
            "TRANS_PRICEPERSHARE": "price",
        }
    )[
        ["accession", "issuer_cik", "ticker", "filing_date", "owner_cik",
         "owner_name", "relationship", "trans_date", "shares", "price"]
    ]
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df.to_parquet(out_pq)
    return df


_ACCEPT_RE = re.compile(
    r"Accepted[^0-9]*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})"
)


def _fetch_acceptance(cik: int, accession: str) -> str:
    nodash = accession.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/"
        f"{accession}-index.htm"
    )
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    m = _ACCEPT_RE.search(r.text)
    if not m:
        raise RuntimeError(f"no Accepted timestamp on index page for {accession}")
    return m.group(1)


def main() -> None:
    DERA_DIR.mkdir(parents=True, exist_ok=True)
    tickers = _universe_tickers()
    cik_to_ticker = _ticker_cik_map(tickers)
    print(f"universe: {len(tickers)} tickers, {len(cik_to_ticker)} CIKs")

    frames = []
    for q in _quarters():
        df = _process_quarter(q, cik_to_ticker)
        if df is None:
            if q <= REQUIRED_THROUGH:
                raise RuntimeError(f"DERA quarter {q} missing (404) — required")
            print(f"{q}: not yet published, stopping")
            break
        n = 0 if df.empty else len(df)
        print(f"{q}: {n} P-rows (officer/director, universe)")
        if n:
            frames.append(df)
        time.sleep(SLEEP_S)

    allrows = pd.concat(frames, ignore_index=True)
    print(f"total transaction rows: {len(allrows)}, "
          f"accessions: {allrows['accession'].nunique()}")

    # stage 2: acceptance datetimes (incremental vs existing output)
    accepted: dict[str, str] = {}
    if OUT_PARQUET.exists():
        prev = pd.read_parquet(OUT_PARQUET)
        accepted = dict(
            zip(prev["accession"], prev["acceptance_datetime"].astype(str))
        )
    todo = [
        (int(c), a)
        for a, c in allrows[["accession", "issuer_cik"]]
        .drop_duplicates("accession")
        .itertuples(index=False)
    ]
    todo = [(c, a) for c, a in todo if a not in accepted]
    print(f"fetching acceptance for {len(todo)} accessions from EDGAR ...")
    for i, (cik, acc) in enumerate(todo):
        accepted[acc] = _fetch_acceptance(cik, acc)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(todo)}")
        time.sleep(SLEEP_S)

    allrows["acceptance_datetime"] = pd.to_datetime(
        allrows["accession"].map(accepted)
    ).dt.tz_localize("US/Eastern").dt.tz_convert("UTC")
    if allrows["acceptance_datetime"].isna().any():
        raise RuntimeError("acceptance datetime missing for some accessions")
    allrows = allrows.sort_values(
        ["ticker", "acceptance_datetime"]
    ).reset_index(drop=True)
    allrows.to_parquet(OUT_PARQUET)
    print(f"wrote {OUT_PARQUET} ({len(allrows)} rows)")


if __name__ == "__main__":
    main()
