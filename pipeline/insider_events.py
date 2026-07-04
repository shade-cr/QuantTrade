"""B0018: PIT opportunistic-insider events from Form 4 open-market purchases.

Data source: ``data/insider/form4_p.parquet`` (built by
``scripts/fetch_insider_form4.py`` from the SEC DERA insider-transactions
data sets + per-accession EDGAR acceptance timestamps). One row per
(accession, non-derivative transaction), already filtered to TRANS_CODE "P",
acquired ("A"), officer/director filers, original Form 4s only (no 4/A).

Knowledge discipline (frozen in the B0018 pre-registration): the knowledge
moment is the FILING acceptance timestamp — never the transaction date, which
precedes filing by up to 2 business days. ``effective_knowledge_day`` (shared
with B0017) rolls after-close acceptances to the next session.

Opportunistic classification is the Cohen-Malloy-Pomorski (JF 2012) rule,
applied PIT: at each filing, using only the same insider's PRIOR qualifying
filings for the same issuer —

- classifiable  = >=1 qualifying purchase filing in EACH of the 3 preceding
  calendar years;
- ROUTINE       = classifiable AND some calendar month (of transaction dates)
  contains purchases in all 3 preceding years -> excluded;
- OPPORTUNISTIC = classifiable and not routine -> event;
- unclassifiable (shorter history) -> excluded (conservative).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.earnings_events import effective_knowledge_day

INSIDER_CACHE = Path("data/insider/form4_p.parquet")

# Config-gated meta-features joined by the pooled runner (features.insider_flow)
# — unioned into the tier2 registry test like EARNINGS_CALENDAR_FEATURES.
INSIDER_FLOW_FEATURES = ("opp_insider_buys_21d", "opp_insider_buys_63d")

MIN_NOTIONAL_USD = 10_000.0  # frozen: excludes token purchases
CLASSIFY_YEARS = 3           # frozen: CMP classifiability window


def load_insider_purchases(
    ticker: str, cache: Path = INSIDER_CACHE
) -> pd.DataFrame:
    """Rows for one ticker, ascending by acceptance. Fail loud on absence."""
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} missing — run: uv run python scripts/fetch_insider_form4.py"
        )
    df = pd.read_parquet(cache)
    df = df[df["ticker"] == ticker].copy()
    if df.empty:
        return df
    acc = pd.DatetimeIndex(df["acceptance_datetime"])
    if acc.tz is None:
        raise ValueError(f"{cache}: acceptance_datetime must be tz-aware")
    return df.sort_values("acceptance_datetime").reset_index(drop=True)


def qualifying_filings(purchases: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction rows to FILINGS and apply the notional floor.

    Returns one row per accession: accession, owner_cik, acceptance_datetime,
    notional, trans_ym (frozenset of (year, month) pairs of the filing's
    transaction dates). Rows whose shares or price are unparseable contribute
    0 notional (they cannot qualify a filing on their own — conservative).
    """
    if purchases.empty:
        return pd.DataFrame(
            columns=["accession", "owner_cik", "acceptance_datetime",
                     "notional", "trans_ym"]
        )
    df = purchases.copy()
    df["notional"] = (df["shares"] * df["price"]).fillna(0.0)
    tdate = pd.to_datetime(df["trans_date"], format="%d-%b-%Y", errors="coerce")
    df["_ym"] = list(zip(tdate.dt.year, tdate.dt.month))
    g = df.groupby("accession", sort=False)
    out = pd.DataFrame(
        {
            "owner_cik": g["owner_cik"].first(),
            "acceptance_datetime": g["acceptance_datetime"].first(),
            "notional": g["notional"].sum(),
            "trans_ym": g["_ym"].agg(
                lambda s: frozenset(
                    (int(y), int(m)) for y, m in s if pd.notna(y) and pd.notna(m)
                )
            ),
        }
    ).reset_index()
    out = out[out["notional"] >= MIN_NOTIONAL_USD]
    return out.sort_values("acceptance_datetime").reset_index(drop=True)


def classify_opportunistic(
    filings: pd.DataFrame, admit_unclassifiable: bool = False
) -> pd.DataFrame:
    """CMP classification, strictly PIT per filing.

    ``admit_unclassifiable=True`` is the ONE pre-committed contingency
    (recorded trial): insiders with <3 prior years of history are admitted
    unless provably routine (with <3 years they never can be), instead of
    excluded.
    """
    if filings.empty:
        return filings.assign(opportunistic=pd.Series(dtype=bool))
    flags = []
    for owner, grp in filings.groupby("owner_cik", sort=False):
        grp = grp.sort_values("acceptance_datetime")
        # history = (trans_year, trans_month) pairs from filings STRICTLY
        # before the current one (iteration order guarantees PIT)
        hist_ym: set[tuple[int, int]] = set()
        for _, row in grp.iterrows():
            y = row["acceptance_datetime"].year
            need = {y - k for k in range(1, CLASSIFY_YEARS + 1)}
            hist_years = {yy for yy, _ in hist_ym}
            classifiable = need <= hist_years
            if classifiable:
                routine = any(
                    all((yy, m) in hist_ym for yy in need) for m in range(1, 13)
                )
                opp = not routine
            else:
                opp = bool(admit_unclassifiable)
            flags.append((row["accession"], opp))
            hist_ym.update(row["trans_ym"])
    flag = pd.DataFrame(flags, columns=["accession", "opportunistic"])
    return filings.merge(flag, on="accession", how="left")


def opportunistic_knowledge_days(
    purchases: pd.DataFrame, admit_unclassifiable: bool = False
) -> pd.DatetimeIndex:
    """End-to-end: transaction rows -> sorted unique effective knowledge days
    of qualifying OPPORTUNISTIC purchase filings."""
    filings = classify_opportunistic(
        qualifying_filings(purchases), admit_unclassifiable=admit_unclassifiable
    )
    if filings.empty or not filings["opportunistic"].any():
        return pd.DatetimeIndex([])
    acc = pd.DatetimeIndex(
        filings.loc[filings["opportunistic"], "acceptance_datetime"]
    )
    return pd.DatetimeIndex(sorted(set(effective_knowledge_day(acc))))


def insider_flow_features(
    bar_index: pd.DatetimeIndex, knowledge_days: pd.DatetimeIndex
) -> pd.DataFrame:
    """Trailing counts of opportunistic purchase filings per bar (PIT: an
    event counts from its effective knowledge day). 0 before the first event
    — never NaN (NaN would dropna-kill unrelated bars)."""
    bars = pd.DatetimeIndex(bar_index)
    bar_days = (bars.tz_localize(None) if bars.tz is not None else bars).normalize()
    kd = pd.DatetimeIndex(knowledge_days)
    out = pd.DataFrame(index=bar_index)
    for name, window in zip(INSIDER_FLOW_FEATURES, (21, 63)):
        lo = bar_days - pd.Timedelta(days=window)
        n_hi = np.searchsorted(kd.values, bar_days.values, side="right")
        n_lo = np.searchsorted(kd.values, lo.values, side="right")
        out[name] = (n_hi - n_lo).astype(float)
    return out
