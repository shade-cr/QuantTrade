# B0006 — Cross-Sectional Feature Pack v1 (design spec)

**Status:** spec (promoted from `backlog/proposed/B0006.json` on 2026-07-03)
**Trigger:** the pre-registered condition fired — the first M3 pooled run
([2026-07-03-m3-readout.md](../reports/2026-07-03-m3-readout.md)) returned NO EDGE with the meta
signal-starved: own-bar + macro features only, primary OOF AUC ≈ 0.50, effective-N ≈ 690.
**Methodology source:** quant-phd-advisor consult 2026-07-03 (López de Prado corpus + literature;
citations inline). Load-bearing choices below reference it as [ADV].

## 1. Goal

Give the meta-labeler *relative-value* information: whether a name's setup is strong or weak
**relative to its peers**, not just in its own history. The M3 claim under test becomes: does a
cross-sectional (CS) pack lift meta precision/F1 over the own-bar+macro baseline, within the frozen
35-name universe?

## 2. The v1 pack — 6 stock-level features + 2 panel scalars

All stock-level features are **percentile ranks across the 35 names at each date t** (continuous
rank in [0,1], not integer, not z-scores — a 35-wide z has unreliable tails and drifts with panel
vol; ranks are stationary by construction [ADV pitfall 2]). All inputs use data ≤ t only, computed
within the frozen universe.

| # | Feature | Construction | Evidence |
|---|---------|--------------|----------|
| 1 | `cs_mom_12_1_rank` | 252-21d log return, **market-residualized** (see §3), CS rank | Blitz/Huij/Martens 2011 (residual momentum ≈ 2× risk-adjusted vs raw); Moskowitz-Grinblatt 1999 |
| 2 | `cs_52wk_high_rank` | `close / rolling_max_252d(close)`, CS rank (no residualization needed) | George & Hwang 2004 (dominates raw momentum) |
| 3 | `cs_st_reversal_rank` | 21d return residual to the equal-weight basket, CS rank; **kept separate in sign from #1** — do not merge horizons | Blitz et al. 2012 (short-term *residual* reversal) |
| 4 | `cs_idio_vol_rank` | trailing 63d vol of the market-residual return, CS rank | Gu-Kelly-Xiu 2020 (volatility family) |
| 5 | `cs_turnover_rank` | trailing 21d dollar volume ÷ own trailing 252d median dollar volume, CS rank | liquidity family; self-normalized → survivorship-robust |
| 6 | `cs_basket_beta_rank` | 63d beta to the equal-weight universe basket, CS rank | tells the meta whether the name is currently a market proxy (low uniqueness) or idiosyncratic |
| 7 | `cs_dispersion` | panel scalar: CS std of 21d returns across the 35 names (same value all names) | Stivers/Sun (dispersion conditions momentum reliability) |
| 8 | `cs_breadth_200` | panel scalar: fraction of names above their own 200d MA | within-panel breadth; complements VIX/DXY |

Cap respected: with ~690 effective bets and heavy substitution among #1/#2/#3 (one latent trend
factor), effective CS dimensionality supportable is ~3-5; 6+2 is the ceiling, not the floor. If
event-based primaries reduce effective bets well below 690, cut to #1, #2, #5, #7 [ADV].

## 3. Residualization rule — the collider guard

- **v1 residualizes against the equal-weight universe basket (≈ first PC).** Marcenko-Pastur on a
  35-wide panel says only the first 1-3 PCs clear the noise floor; deep-PCA residuals are noise
  [ADV, MLAM §2.2].
- **Deviation from the advisor's first suggestion, with rationale:** [ADV] proposed
  sector-ETF-relative construction for #1. v1 uses market-residual instead because a name→sector-ETF
  map is time-varying over 2006-2026 (XLC did not exist before 2018; pre-2018 telecoms lived in XLK;
  GICS reshuffles) and a wrong static map silently corrupts the feature. The frozen static map in
  Appendix A is provided for a **v1.1 config option** (`residual_vs: sector_etf`), off by default.
- **NEVER sector one-hot dummies as meta inputs/controls.** Conditioning on a collider raises
  in-sample fit while inverting live signs (López de Prado & Zoonekynd 2024, SSRN 4786265).
  Sector information enters only through residualization in feature *construction*.

## 4. Wiring

- New module `pipeline/cross_section.py`: `build_cross_sectional_features(panel: dict[str, DataFrame],
  universe: list[str]) -> dict[str, DataFrame]` — takes the loaded per-ticker OHLCV panel, returns
  per-ticker CS feature frames aligned to each ticker's index. Point-in-time: every value at t uses
  panel data ≤ t. Names not yet listed at t (none in this universe — all have full history) would be
  NaN, never imputed.
- `scripts/run_pooled_equity_d1.py`: after loading all tickers, compute the CS frames once, then
  `features = build_tier2_features(ohlcv, macro).join(cs_frames[ticker])` per ticker. Config-gated:
  `features.cross_sectional: true` in the M3 configs (default false so equity_d1.yaml/NVDA is
  untouched).
- The pack rides the existing pipeline invariants unchanged: same triple-barrier, weights,
  purged folds, calibration. No changes to `run_multi_h4.py`.

## 5. Validation protocol (cheap-first, in order)

1. **Baseline re-run**: M3 stocks config + CS pack, state-based primaries (one full purged WF).
   Read **clustered MDA** (ONC clusters + AFML §8 MDA, F1 scoring per AFML §14.8) on pooled OOF —
   substitution effects make single-feature MDA unreadable for this pack [ADV pitfall 3].
2. **Event-primary run**: same pack with a sparse event-based primary (`cusum_filter` is already
   implemented; the BEAR_QUIET dossiers rank it and `bollinger_meanrev` "higher" on hit-rate). The
   evidence says CS features pay off most with idiosyncratically-timed events (higher uniqueness);
   compare CS-cluster MDA uplift state-based vs event-based [ADV §sub-question 3].
3. Only if a CS cluster shows non-zero MDA proceed to DSR on the fixed 0.55 headline.

## 6. Pre-registered falsification criteria (the pack FAILS if ANY holds)

1. **No cluster signal:** every CS cluster's clustered-MDA bootstrap CI (over folds, F1 scoring)
   includes 0.
2. **No skill uplift:** `np.nanmedian` per-fold meta F1 / precision does not exceed the
   own-bar+macro baseline by more than one fold-to-fold standard error (NaN-safe aggregation,
   min-30-trades gate).
3. **Survivorship artifact:** the uplift disappears when features are recomputed sector-neutralized
   / rank-only (the "edge" was riding absolute winners).
4. **DSR collapse:** headline-threshold Sharpe improvement fails DSR adjustment for the number of
   features/thresholds trialed (increment the trial count for this run).

Passing requires the opposite of all four — and per protocol step 2, ideally a larger uplift under
the event primary than the state primaries.

## 7. Out of scope

- Earnings-surprise / analyst-revision alt-data (spec OQ2 second half — separate item if still
  starved after this pack).
- Point-in-time index membership (B0004) — v1 stays caveated within-universe.
- Per-ticker friction (B0005).
- Lookahead firewall note: these CS features feed the **meta only** (tier-2 features). They are NOT
  added to regime dossiers / the hypothesizer's `available_features` in this phase — dossier
  integration would need its own quasi-circularity vetting pass.

## Appendix A — frozen static name→sector-ETF map (v1.1 option only, off by default)

XLE: XOM, CVX, COP, SLB* · XLK: MSFT, IBM, INTC, CSCO, ORCL, QCOM, T, VZ (pre-2018 telecom
convention) · XLF: C, BAC, JPM, WFC, GS, AIG · XLP: PG, KO, PEP, MO, WMT · XLV: JNJ, PFE, ABT,
MRK, UNH · XLI: GE, MMM, UPS, BA, CAT · XLY: HD, MCD, DIS.
(*SLB only if promoted from alternates; base-35 covers 34 names + GS = 35.)
Known imperfections (GE's 2015-2021 restructuring, T/VZ sector moves) are exactly why v1 defaults
to market-residual.

## References

- Blitz, Huij & Martens (2011), *Residual Momentum*, J. Empirical Finance.
- George & Hwang (2004), *The 52-Week High and Momentum Investing*, J. Finance.
- Moskowitz & Grinblatt (1999), *Do Industries Explain Momentum?*, J. Finance.
- Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*, RFS.
- Blitz, Huij, Lansdorp & Martens (2012), *Short-Term Residual Reversal*.
- López de Prado & Zoonekynd (2024), *Why Has Factor Investing Failed?*, SSRN 4786265.
- López de Prado (2018), AFML §8.3-8.4.2 (substitution effects, clustered MDA), §14.8 (F1 for meta-labeling).
- López de Prado (2020), *Machine Learning for Asset Managers* §2.2 (Marcenko-Pastur).
