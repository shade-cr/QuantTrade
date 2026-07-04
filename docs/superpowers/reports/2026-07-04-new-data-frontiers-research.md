# New-data frontiers: evidence × sources (2026-07-04)

Two parallel investigations after the day's conclusion that the current search space
(parametric primaries + technical/macro/CS features, panel AUC ≈ 0.53) is exhausted with
valid negative verdicts: (A) quant-phd-advisor literature review (LdP corpus + web, post-2015
replication weighted); (B) web research on 2026 source availability/cost/PIT quality.
Full agent outputs recorded in the session; this file is the synthesis.

## The cross-ranked matrix (evidence × horizon fit × data access)

| # | Family | Evidence (replicated) | Data | Cost | Verdict |
|---|---|---|---|---|---|
| 1 | **Overnight/intraday decomposition** | JFE 2019, t≈5; momentum in large caps is overnight | open+close — ALREADY HAVE | $0 | **BUILD FIRST** |
| 2 | Daily microstructure proxies (Amihud, Corwin-Schultz, Roll) | LdP ch.19; Kyle/Amihud top of his MDA ranking | OHLCV — ALREADY HAVE | $0 | Build with #1 (meta-features only) |
| 3 | **Earnings announcement premium** (NOT PEAD-drift — dead in large caps per Martineau 2022) | ~0.30%/event VW, survives specifically in largest ~500 | PIT announcement datetimes 2006→: EDGAR 8-K acceptance timestamps (free, gold-standard PIT); vendor calendars are backfilled — verify against EDGAR | $0 (engineering) | **BUILD SECOND** (phase5_custom event primary + days_to/since_earnings meta-features) |
| 4 | **Form 4 opportunistic insiders** (Cohen-Malloy-Pomorski JF 2012, ~82bp/mo VW; decay unverified → haircut 50% ex ante) | routine/opportunistic filter is the non-naive part | EDGAR XML 2003→, acceptance timestamp = PIT; edgartools parses natively | $0 | **BUILD THIRD** (event primary: opportunistic buy clusters) |
| 5 | Short interest / DTC | strong but concentrated where shorting constrained — attenuated in mega-caps | FINRA API free but ~5yr online depth; QuantQuote sells deep history; borrow fees have NO cheap history (start capturing IBKR SLB snapshots now) | $0-low | Meta-feature, if time |
| 6 | Options IV skew / VRP | real at weekly horizon BUT edge collapses net-of-costs in low-borrow-fee (large-cap) names | ORATS $99-199/mo (2007→); free proxies: VIX term structure, SKEW, P/C (FRED/CBOE) | $99+/mo | DEFER; use free index-level proxies only |
| 7 | News/NLP sentiment | Ke-Kelly-Xiu: 5-day convergence = perfect horizon | nothing cheap reaches 2006 (Tiingo 2017→, AV 2022→; RavenPack $$$) | blocked | DEFER until affordable archive |
| 8 | Analyst revisions | evidence OK | true PIT revisions = IBES-only; cheap substitute = dated ratings actions (Benzinga/FMP) ~2010s→ | ~$29-79/mo | Weak substitute; defer |
| 9 | Index add/drop, TOM, factor momentum, PEAD-drift, VPIN-from-D1 | documented dead / not computable honestly | — | — | SKIP |

## Recommended build order (one month)

1. **Week 1 — $0, zero acquisition risk**: overnight (`log(open_t/close_{t-1})`) + intraday
   (`log(close_t/open_t)`) returns, EWMA components 10/21/60d, tug-of-war spread; Amihud +
   Corwin-Schultz. Add to `build_tier2_features`. PRE-CHECK: audit that our CSVs carry TRUE
   session opens (some free feeds copy prior close — corrupts everything). Fast informative
   null: if panel AUC doesn't move off 0.53, new-info-from-price is dead too.
2. **Weeks 2-3 — earnings-date event primary** (phase5_custom, pre-registered as the
   PREMIUM effect, not drift): +1 entering N days before scheduled announcement; PIT dates
   reconstructed from EDGAR 8-K acceptance timestamps (engineering risk lives here);
   `days_to_earnings` / `days_since_earnings` as meta-features everywhere.
3. **Weeks 3-4 — Form 4 opportunistic insider primary**: EDGAR ingestion for 44 tickers is
   small; routine filter = drop insiders trading same calendar month ≥3 prior years; signal
   ONLY from filing acceptance timestamp (transaction date = 2-day lookahead); buys weighted.
4. **If time — FINRA SI/DTC meta-feature** (publication-lag shifted, FRED-style) + start a
   daily IBKR SLB borrow-rate capture job for future use.

## Purchase decisions (user's)

- **Sharadar Core US Equities (Nasdaq Data Link, ~$599/yr)**: single highest-leverage paid
  buy — PIT fundamentals (datekey dimension) 1990→, survivorship-free prices, S&P membership
  1957→, insiders 2005→, 8-K events. Would ALSO close most of B0004 (PIT vendor). Not
  required for the month-1 plan above (all $0).
- **ORATS ($99-199/mo)**: only affordable per-stock IV history (2007→). Deferred — evidence
  says the options edge nets out in our universe; revisit only if an options-conditional
  hypothesis earns it.

## Guardrails carried from today's skeptic

- All performance numbers above are literature priors, NOT measured on our panel — proposals
  built from them go through the full Loop A immune system with pre-registered criteria.
- The data-snooped bollinger observation (B0014) stays flagged; unrelated to this plan.
- Feed-forward line: plumbing numbers may reach the hypothesizer; measured performance may not.
