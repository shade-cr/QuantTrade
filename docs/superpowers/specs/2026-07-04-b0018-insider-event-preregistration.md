# B0018 — phase5_insider_event: PRE-REGISTRATION (frozen before any data look)

**Committed 2026-07-04, BEFORE any Form 4 / insider-transaction data was fetched or examined,
and before any event-window return was computed on the panel.** Every parameter below is
frozen; deviations require a logged DSR trial per the amendment discipline established in
B0017. Same reading rules apply.

## Evidence base (literature priors, not measured on our panel)

Cohen, Malloy, Pomorski (JF 2012) "Decoding Inside Information": OPPORTUNISTIC
(non-routine) insider purchases predict ~82 bp/month value-weighted abnormal returns;
ROUTINE trades (same-calendar-month habitual traders) predict nothing. Purchases are
informative; sales are noise (compensation/diversification). Post-publication decay is
unverified — we haircut the prior 50% ex ante and still expect an order of magnitude more
signal per event than the earnings premium (+10.8 bp/event, B0017).

Mechanism: idiosyncratic information timing. Crucially for our effective-N gate, insider
purchases are NOT season-clustered the way earnings are — events spread across the calendar,
so the ρ=1 pooled effective-N should not collapse the way B0017's h10 attempt did.

## Data source (frozen)

- **Primary: SEC structured Insider Transactions data sets** (quarterly TSVs, DERA,
  coverage 2006Q1+ — matches the audit window). Tables used: `SUBMISSION`
  (ACCESSION_NUMBER, ACCEPTANCE_DATETIME — the knowledge moment), `NONDERIV_TRANS`
  (transaction code, date, shares, price), `REPORTINGOWNER` (officer/director flags).
- Knowledge discipline identical to B0017: `effective_knowledge_day` — acceptance
  timestamps ≥ 20:00 UTC roll to the next session. **Signals key off the FILING acceptance
  timestamp, never the transaction date** (transaction date precedes filing by up to 2
  business days = lookahead).
- Amendments (Form 4/A) excluded; only original Form 4 rows.

## Qualifying event (frozen)

A Form 4 filing qualifies iff:
1. Non-derivative transaction, code **"P"** (open-market purchase) — no sells, no option
   exercises, no awards.
2. Filer is an **officer or director** of the issuer.
3. Notional (shares × price, summed over the filing's P rows) **≥ $10,000** — excludes
   token purchases.
4. Filer is classified **OPPORTUNISTIC** (below) at filing time.

## Opportunistic classification (frozen, CMP definition, PIT)

At each filing's acceptance time, using ONLY that insider's PRIOR filings for the same
issuer:
- **Classifiable** requires ≥ 1 qualifying purchase in each of the 3 preceding calendar
  years.
- **ROUTINE** = classifiable AND there exists a calendar month in which the insider
  purchased in all 3 preceding years. Routine insiders' filings are EXCLUDED.
- **OPPORTUNISTIC** = classifiable and not routine → event fires.
- **Unclassifiable** (shorter history) is EXCLUDED (conservative to the mechanism claim).

**Pre-committed contingency (plumbing-only, before any return look):** if the panel-wide
count of qualifying opportunistic events is < 500, ONE amendment is permitted relaxing
rule 4 to "not routine" (i.e., unclassifiable insiders admitted). That amendment is a
RECORDED TRIAL with the standard DSR haircut. No other relaxation is permitted.

## Primary rule (frozen)

- Module: `pipeline/primaries_phase5/phase5_insider_event.py` (INPUT_COLUMNS = ())
- Signal: **+1 (long only) on the first bar at/after the effective knowledge day of each
  qualifying opportunistic purchase filing.** Multiple filings mapping to the same bar =
  one signal. No re-fire for the same ticker within 10 bars of a prior fire (matches the
  holding horizon; prevents cluster double-counting). Never −1.
- Barrier geometry: **h10, tp 1.5 / sl 1.5 ATR** (pinned pooled-audit geometry; the CMP
  effect is a ~1-month drift claim, h10 covers half of it — conservative).
- Effective-N: standard gate (wf_event_floor 799, ρ=1). Horizon amendments follow the
  B0017 discipline: plumbing-only, recorded trial, one shot.

## Meta-features (frozen)

- `opp_insider_buys_21d`, `opp_insider_buys_63d`: trailing counts of qualifying
  opportunistic purchase filings (by effective knowledge day), per ticker. 0 before first
  event; never NaN.

## Falsification criteria (frozen)

- `median_active_fold_sharpe_min`: 0.5
- `n_trades_total_min`: 100
- Breadth: ≥ 5 assets with ≥ 30 trades and positive aggregate best-model Sharpe.
- `survivor_long_bias_discount`: long-only on survivor universe (B0003) — marginal pass
  discounted.
- `stress_concentration_check`: if passed, first robustness read is 2008/2020 episode
  concentration.
- DSR: trials counted per `signals/trial_ledger.json` day-aggregate discipline; this
  family starts at trial 1.

## Known caveats (pre-committed reading rules)

1. Post-2012 publication decay — even a confirmed positive mean must beat costs
   (B0017 lesson: sign ≠ tradeability). The net-of-10bp read is the headline.
2. Mega-cap insiders purchase rarely; sample may concentrate in a few names —
   breadth criterion is load-bearing.
3. The 3-prior-years classifiability rule delays the first possible event to 2009 for
   most insiders (2006 data start) — expected, not a bug.
