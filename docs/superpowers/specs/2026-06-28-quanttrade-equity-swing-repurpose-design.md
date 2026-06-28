# QuantTrade — Equity Swing-Trading Repurpose: Design Spec

**Date:** 2026-06-28
**Status:** Draft for review
**Origin:** Cherry-picked from `D:\PROJECTS\QuantHack` (the "Model to Market" hackathon project, concluded). QuantHack freezes as-is; QuantTrade is a fresh repo with new git history.

---

## 1. Purpose & scope

QuantHack was a López-de-Prado meta-labeling backtest pipeline + an autonomous research loop ("Loop A" / Phase 5, the *alpha generator*), built for a 5-day FX/metals/crypto trading tournament. The hackathon is over.

**QuantTrade repurposes the asset-agnostic research core for a new, friendlier domain: equity swing trading on daily bars.** The 5-day variance-dominated tournament window is gone; equities give us years of clean daily history and a real cross-section — the regime the meta-labeling methodology was actually designed for.

**In scope:**
- A clean equity-research repo seeded only from the reusable core.
- A free equity daily-bar data layer (yfinance, adjusted), behind a swappable seam.
- Re-aiming the alpha generator from metals/FX/crypto onto equities.
- A validation posture that resists the single-name overfit trap.

**Explicitly out of scope (left in QuantHack):** live MT5 execution, the tournament scoring/λ/rung simulator, the barbell lottery sleeve, OSR/L2 intraday order-book machinery, crypto live-ops, the live monitoring webapp, and all hackathon submission docs.

---

## 2. Goals & non-goals

**Goals**
- G1. A standalone repo at `D:\PROJECTS\QuantTrade` containing only asset-agnostic research code, with a fresh `git init` (no hackathon history).
- G2. Daily adjusted equity bars ingestible into the existing strict OHLCV contract (`pipeline/data.py::load_dataset`) with zero changes to that contract.
- G3. The alpha generator runs end-to-end on an equity (NVDA) — the *plumbing smoke test*.
- G4. The alpha generator runs across a liquid equity **cross-section**, where any edge claim must survive broadly (the trustworthy target).

**Non-goals**
- N1. No live or paper execution in this phase (no broker integration). That is a later milestone, deliberately deferred.
- N2. No intraday / L2 / microstructure research (daily bars only). The 6 intraday modules are *not* copied.
- N3. No claim of deployable edge from the NVDA smoke test — it is plumbing validation only.
- N4. No survivorship-bias-free guarantee in phase 1 (yfinance limitation, see §7); acknowledged and bounded, not solved here.

---

## 3. What transfers (cherry-pick manifest)

Derived from a dependency-graph audit of QuantHack (import-level, not filename guesses).

### 3.1 `pipeline/` — CORE (copy)
Asset-agnostic meta-labeling building blocks. Confirmed to import no `execution.*` / `tournament_sim.*` / MT5 code:

- **Data/labels:** `data.py`, `labels.py`, `data_sanity.py`
- **Features:** `features.py`, `frac_features.py`, `microstructure.py`, `regimes.py`
- **Walk-forward/CV:** `walk_forward.py`, `pooled_walk_forward.py`
- **Training/stacking:** `train.py`, `stack.py`, `best_model.py`
- **Weights/metrics:** `sample_weights.py`, `metrics.py`, `thresholds.py`, `threshold_selection.py`, `friction.py`
- **Reporting:** `reporting.py`, `feature_importance.py`
- **Multi-asset:** `cross_asset.py`, `cointegration.py`, `primary_screening.py`, `primary_contracts.py`
- **Macro:** `macro_fetch.py`, `macro_fetch_intraday.py`
- **Primaries:** `primaries_phase5/` (the rule-based primary modules)
- **Utilities (copy as needed):** `cli_console.py`, `registry.py`, `sizing.py`, `session_filter.py`, plus optional feature modules (`cot_features.py`, `sentiment_features.py`, etc.) pulled only if a primary needs them.

> Invariant carried over verbatim: **primaries are deterministic rule-based functions, never ML classifiers** (the QC "squeeze the orange twice" failure). Enforced by `tests/test_primary_input_contract.py`.

### 3.2 `phase5/` — alpha generator (copy)
Imports only `pipeline.*`. Lookahead firewall, regime taxonomy, proposal schema, and DA review contract transfer **unchanged**:

- `regime_stats.py`, `proposal.py`, `run_proposal.py`, `lookahead_lint.py`, `devils_advocate_dispatch.py`, `asset_registry.py`, `__init__.py`
- Plus the orchestrator it subprocesses for the audit step (`scripts/run_xau_d1.py` → renamed `scripts/run_backtest.py`).

### 3.3 Orchestration & agents (copy)
- `scripts/loop_a_tick.py` + its one dependency `execution/state_store.py` (the BoundedIndex util — copied as a standalone util, *not* the rest of `execution/`).
- `.claude/agents/phase5-hypothesizer.md`, `phase5-devils-advocate.md`, `phase5-skeptic.md`
- `.claude/skills/phase5-regime-methodology/`

### 3.4 Project machinery (copy, reset to empty)
- `pyproject.toml` (drop hackathon-only deps: MetaTrader5; keep yfinance, pandas, sklearn, xgboost/lightgbm, etc.)
- `tests/` — only the test files for copied modules.
- `backlog/` + `ideas/` filesystem-DB machinery, **with empty `proposed/done/...` folders** (fresh B-ID space starting B0001).
- A trimmed `CLAUDE.md` (see §6).

### 3.5 NOT copied (stays in QuantHack)
`execution/` (except the one util above), `tournament_sim/`, `webapp/`, `docs/submission/`, the live `signals/` state, and these 12 `pipeline/` hackathon modules: `lottery_sleeve.py`, `stage2_lambda.py`, `osr_intraday.py`, `osr_bundle_writer.py`, `l2_adapter.py`, `tardis_adapter.py`, `intraday_fills.py`, `llm_distillation.py`, `llm_primary.py`, `llm_model_recipes.py`, `llm_prompt.py`, plus the 6 intraday modules (`ticks.py`, `intraday_bars.py`, `intraday_features.py`, `intraday_pool.py`, `intraday_wf.py`, `intraday_gate.py`) deferred until/unless intraday is revisited.

---

## 4. New component — equity data layer

`scripts/fetch_equity_daily.py`, cloned from QuantHack's proven `ingest_gld_volume.py` pattern, with the critical equity change:

- **`auto_adjust=True`** in `yf.download` → split/dividend-adjusted O/H/L/C (unadjusted equity data is unusable; NVDA's splits would corrupt every label).
- Emits the **full 5-column contract** (`open, high, low, close, volume` + UTC `time`), not just close/volume.
- Flattens yfinance's MultiIndex columns; UTC tz-aware; monotonic; dedup — reusing the existing normalize/atomic-write helpers.
- Output: `data/equities/<TICKER>.csv`.

**Seam for later vendor swap:** a thin `EquityDataSource` interface (`fetch_daily(ticker, start, end) -> DataFrame`) with `YFinanceSource` as the first impl, so Norgate/Alpaca/Sharadar drop in behind it without touching the pipeline. (YAGNI: the interface is one method; we don't build adapters we don't yet need.)

---

## 5. Re-aiming the alpha generator for equities

1. **Asset registry:** add an `equity` asset class and an `_EQUITY_PACK` macro feature set (e.g. `vix_level`, `vix_chg_5`, `real_yield_5y_z252d`, `dxy_z252`) to `phase5/asset_registry.py`. FRED stays as the macro source. Register `NVDA` (and later the universe tickers).
2. **Regime dossier:** built by the existing `phase5/regime_stats.py` (unchanged code) — needs the ticker's adjusted OHLCV + a regime parquet from `pipeline/regimes.py`. Same 4-state taxonomy (BULL/BEAR × QUIET/STRESSED), same quantile encoding, same firewall.
3. **Hypothesizer persona:** reframe the *illustrative examples* in `phase5-hypothesizer.md` from precious-metals/dollar mechanisms → equity mechanisms (post-earnings drift, factor rotation, volatility regimes, sector co-movement). **The firewall constraints and proposal schema are unchanged** — only the prose examples change.
4. **Event-density floor:** re-tune the per-cell minimum event count for equity daily bars (QuantHack used 399–599 for FX/metals D1, 300 for custom). Set from each dossier's measured baseline `n_events` with a safety margin.
5. **Config template:** an equity triple-barrier config (`configs/equity_d1.yaml`) — ATR-scaled barriers (already supported), gap-aware exit (already handles earnings/overnight jumps via `triple_barrier_labels`' `exit_price`).

All carried-over pipeline invariants stay intact: `sqrt(trades_per_year)` Sharpe, NaN-not-zero for `n_trades<30`, `should_stack` min-trades gate, FRED `.shift(1)`, PurgedTimeSeriesSplit inner-CV, FrozenEstimator sigmoid calibration, avg_uniqueness sample weights.

---

## 6. Trimmed CLAUDE.md

QuantHack's CLAUDE.md is dominated by hackathon posture (barbell, tournament rules, live-ops). QuantTrade gets a fresh CLAUDE.md keeping only: the pipeline architecture diagram, the "Pipeline invariants — DO NOT BREAK" section, the "Where things live" map (updated paths), the backlog/ideas workflow, the Quant Validation skepticism rule, and Loop A's methodology. All tournament/barbell/live-MT5/submission prose is dropped.

---

## 7. Validation posture (keeps us honest)

The central methodological risk is the **single-name overfit**: an "edge" found on one stock's single price path (NVDA especially — a once-in-a-generation survivor outlier) is almost always the loop fitting the chart, not finding a repeatable effect. The methodology (PSR/DSR, purged walk-forward, audit gates) fights this but cannot rescue n=1.

- **Milestone 0 — NVDA smoke test.** Goal: the full loop (fetch → regime dossier → hypothesizer → devil's advocate → preflight → audit) runs end-to-end on NVDA. Success = it *runs and reports honestly*, **not** that it finds edge. Any positive result is explicitly labeled plumbing-validation-only.
- **Milestone 1 — cross-section (the real target).** Run the generator across a liquid equity universe using the already-built `cross_asset.py` + `pooled_walk_forward.py`. An edge is believed only if it shows up **broadly across names**, not just on the lucky ticker.

**Known limitation (bounded, not solved here):** yfinance has survivorship bias (only currently-listed names, no point-in-time index membership). Acceptable for prototyping; the *trustworthy* cross-section eventually wants point-in-time data (Norgate/Sharadar). Logged as a backlog item, not a phase-1 blocker.

---

## 8. Repo seeding procedure (executed after spec + plan approval)

1. Copy the §3 manifest into `D:\PROJECTS\QuantTrade`, preserving relative paths.
2. Delete/skip the §3.5 non-copied set; rename `run_xau_d1.py` → `run_backtest.py`; extract `state_store.py` as a standalone util.
3. Reset `backlog/` and `ideas/` to empty registries; trim `CLAUDE.md` (§6); prune `pyproject.toml`.
4. `git init` + one initial commit ("seed: equity-swing research core cherry-picked from QuantHack @ <hash>").
5. `uv sync`; run the copied test suite — **green test suite is the seed's acceptance gate** (proves no dangling hackathon imports came across).

---

## 9. Milestones

- **M0 — Seed & green.** Repo created, core copied, `uv run pytest` passes. (Proves clean cherry-pick.)
- **M1 — Equity data.** `fetch_equity_daily.py` pulls adjusted NVDA daily bars into the contract; `load_dataset` accepts them.
- **M2 — NVDA smoke test.** Regime dossier built; one full Loop A tick runs to a recorded outcome on NVDA.
- **M3 — Cross-section.** Universe registered; pooled run across the basket; first honest edge/no-edge readout.

---

## 10. Open questions / risks

- **OQ1.** Universe definition for M3 (S&P 100? a hand-picked liquid 30? sector ETFs as a stepping stone?) — decide before M3.
- **OQ2.** Does the hypothesizer need equity-specific *features* (earnings-surprise, analyst-revision alt-data) to find anything, or is macro+microstructure enough for v1? Start with macro+microstructure; add equity alt-data only if the loop is signal-starved.
- **R1.** yfinance reliability/gaps — mitigated by the data-sanity checks already in `pipeline/data_sanity.py`.
- **R2.** Carrying over a hackathon assumption silently in a copied module — mitigated by the green-test gate (M0) and a grep for `mt5/lottery/tournament/osr` symbols post-copy.
