# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

QuantTrade is an equity swing-trading meta-labeling research project, cherry-picked from the concluded QuantHack hackathon project. The full design spec — including asset universe, feature set, and pipeline repurposing decisions — is [docs/superpowers/specs/2026-06-28-quanttrade-equity-swing-repurpose-design.md](docs/superpowers/specs/2026-06-28-quanttrade-equity-swing-repurpose-design.md). Read it before changing pipeline behavior; the spec encodes decisions that are not obvious from the code alone.

## Commands

Dependencies live in `pyproject.toml` and are installed via [uv](https://docs.astral.sh/uv/). Always run Python through `uv run` so it uses the project's `.venv`.

```powershell
uv sync                                                          # install deps into .venv
copy .env.example .env                                           # then set FRED_API_KEY
uv run python scripts/run_backtest.py --config configs/backtest.yaml --dry-run   # 1 fold, ~timing extrapolation
uv run python scripts/run_backtest.py --config configs/backtest.yaml             # full run (~10-30 min CPU)
uv run pytest -v                                                 # all tests
uv run pytest tests/test_metrics.py::test_sharpe_annualization_uses_trades_per_year -v   # single test
```

**Windows note:** `pytest` is only available inside `.venv\Scripts\python.exe`, not the system Python. Always go through `uv run` (or activate the venv first) — invoking `python` / `pytest` directly will use the wrong interpreter.

## Architecture

The pipeline is a two-stage AFML (López de Prado) meta-labeling flow. End-to-end orchestration is in [scripts/run_backtest.py](scripts/run_backtest.py) (`_run_one_primary`); reusable building blocks are in [pipeline/](pipeline/).

```
ohlcv CSV ──► [data.load_dataset]
              │
              ▼
        [macro_fetch.build_macro_frame]  ──► FRED series, .shift(1) for publication lag
              │
              ▼
       [features.build_tier2_features]   ──► technical + vol-regime + macro
              │
              ▼
   ┌──────────┴──────────┐
   │ Stage 1 primary     │  labels.{ema_crossover,momentum_zscore}_signal → side ∈ {-1, 0, +1}
   └──────────┬──────────┘
              ▼
       [labels.triple_barrier_labels]    ──► TP/SL/timeout label + gap-aware exit_price
              │
              ▼
       [sample_weights.avg_uniqueness]   ──► AFML §4 per-event weights ∈ (0,1]
              │
              ▼
        [walk_forward.make_folds]        ──► expanding-window WF with purge + embargo
              │
              ▼
   for each fold:
     RandomizedSearchCV(inner=PurgedTimeSeriesSplit) → train.fit_calibrated → OOF probs
              │
              ▼
       [stack.should_stack]              ──► ex-ante gate: ≥2 models beat baseline (with min-trades filter) ∧ pair corr < 0.7
              │  if YES
              ▼
       [stack.fit_meta_nested_wf]        ──► LogReg meta-learner, nested WF
              │
              ▼
       [reporting.write_report_md]       ──► report.md + summary.json + OOF parquet + plots
```

Each primary in `cfg["primary"]["candidates"]` runs the full pipeline independently — `results/<primary>/` is the output folder. Promoting one primary to production is a human decision made by reading both reports side by side; the pipeline never auto-selects.

## Pipeline invariants — DO NOT BREAK

These are the load-bearing decisions. Most have a paid-in-blood bug story behind them; check the spec or the inline docstring before relaxing.

- **Sharpe annualization uses `sqrt(trades_per_year)`, NOT `sqrt(252)`.** Meta-labeling pnl is sparse (only on filtered events), so the daily-bars convention is wrong by a factor of ~3. See the docstring on [pipeline/metrics.py](pipeline/metrics.py) `strategy_metrics`. The first Phase 1 run produced Sharpes of 11–62 from this single bug.
- **Sharpe is `NaN` (not `0`) when `n_trades < 30` or `std == 0`.** Downstream aggregations MUST use `np.nanmedian` / `np.nanmean`, and `should_stack` skips NaN folds when counting "beats baseline." Replacing NaN with 0 conflates "no measurement" with "no skill."
- **`should_stack` requires a `min_trades_per_fold` gate.** Without it, a 3-trade all-winners fold can report a Sharpe of 31+ and count as competence. Default is 30; see [pipeline/stack.py](pipeline/stack.py).
- **FRED macro is `.shift(1)` before alignment** to model the 1-day publication lag. `features[t]` may only use macro stamped ≤ `t-1`. Removing this shift produces a silent look-ahead leak that doesn't break tests but inflates the edge.
- **Inner-CV is `PurgedTimeSeriesSplit`, NOT sklearn's `TimeSeriesSplit`.** The latter does not purge between train/val and leaks via overlapping triple-barrier outcome windows.
- **Calibration uses `FrozenEstimator` (sklearn 1.4+ replacement for `cv='prefit'`)**, fitted on a chronological tail-15% holdout. Do NOT use `cv=3` / KFold for `CalibratedClassifierCV` — random folds reintroduce temporal leakage. See [pipeline/train.py](pipeline/train.py) `fit_calibrated`.
- **Default calibration method is `sigmoid` (Platt), not `isotonic`.** Isotonic produces step-function probabilities; on D1 / H4 meta-labeling, every threshold in `[0.55, 0.65]` would select the same trade set and `pct_signals_kept` collapsed to <5%. The override is in the config yaml under `calibration.method`.
- **`triple_barrier_labels` returns `exit_price` per event, gap-aware.** Forward returns use `log(exit_price / entry_close)`, NOT `log(close[t_end] / close[entry])`. The asymmetry between long and short branches is explicit and tested — a single misplaced inequality silently breaks shorts.
- **Threshold selection is NOT done by picking the best from the test-set grid.** The "headline" threshold for the stack-decision is fixed at `0.55` by spec. The full grid is reported as a diagnostic only. Per-fold inner-CV threshold selection is deferred to a future phase — do not implement it as a post-hoc pick.
- **Sample weights via `avg_uniqueness` (AFML §4)** are passed to every fit (`sample_weight=...`) — base estimator, hyperparam search, AND calibration. Removing weights on any one of those reintroduces the concurrent-label overfit.
- **Primaries are deterministic rule-based functions, not fitted classifiers.** Each `*_signal` in [pipeline/labels.py](pipeline/labels.py) and each `phase5_*` module in `pipeline/primaries_phase5/` is a pure function returning `pd.Series` in {-1, 0, +1}. No `.fit()` step, no ML training on the same `tier2_features` the meta sees downstream. A primary trained as an ML classifier on the same data as the meta collapses to the Francesco "squeeze the orange twice" failure case — meta-labeling demonstrably fails to add value in that regime (see [cache/blogs/qc_meta_labeling.txt](cache/blogs/qc_meta_labeling.txt) — the QC critique with controlled experiment). The only validated regime is `rule-based primary + ML meta on richer features`. Enforced by [tests/test_primary_input_contract.py](tests/test_primary_input_contract.py).

## Where things live

- [scripts/run_backtest.py](scripts/run_backtest.py) — orchestrator. `_run_one_primary` is the full per-primary flow; `main()` loops over primaries. Inserts project root onto `sys.path` so `pipeline.*` imports work regardless of how the script is invoked. (Note: the cherry-picked source used `scripts/run_xau_d1.py`; the rename to `run_backtest.py` is a Task 3 deliverable.)
- `configs/` — everything tuneable: date range, triple-barrier params, primary configs, walk-forward folds, hyperparam search, calibration, stacking gates, threshold grid, dry-run knobs. Comments on non-obvious values explain *why* they're set.
- [pipeline/](pipeline/) — pure functions, no side effects outside `reporting.py`. Modules are roughly one-AFML-chapter-each: `labels` (§3), `sample_weights` (§4), `walk_forward` (§7), `metrics` (§14: PSR/DSR), `stack` (stacking gate).
- [tests/](tests/) — one test file per pipeline module. `tests/conftest.py` provides a 500-bar synthetic OHLCV fixture (`synth_ohlcv`) used across tests; prefer that over fetching real data in tests.
- [docs/superpowers/specs/](docs/superpowers/specs/) — design specs (the authoritative source for decisions). [docs/superpowers/plans/](docs/superpowers/plans/) — implementation plans (per-phase task lists).
- [backlog/](backlog/) — filesystem-as-database for B-IDs. One JSON per entry under `proposed/`, `in_progress/`, `blocked/`, `done/`, `discarded/`. Schema + public API in [backlog/SCHEMA.md](backlog/SCHEMA.md). Agents call [backlog/db.py](backlog/db.py); nothing else writes JSON under `backlog/`. Promotion path is `backlog/proposed/ → docs/superpowers/specs/ → docs/superpowers/plans/ → code`. **Active = `proposed + in_progress + blocked`**; recalibration loads only active folders via `db.load_active_entries()`. Lint: `uv run python -m backlog.lint`. Legacy single-file `BACKLOG.md` (from QuantHack) archived at [legacy/BACKLOG_pre_split.md](legacy/BACKLOG_pre_split.md) if present.
- [ideas/](ideas/) — same pattern for raw I-ID captures (no effort/value). Schema in [ideas/SCHEMA.md](ideas/SCHEMA.md). Filled via `/capture` skill (writes `ideas/open/I<NNNN>.json` via `ideas.db.add_entry`). Promotion to a B-ID via `ideas.db.promote_to_backlog(i_id, b_id, reason)` — the idea moves to `ideas/promoted/` and the new B-entry carries `links.spawned_from`. Does NOT feed Loop A's hypothesizer (lookahead firewall forbids text inputs to that agent).
- [scripts/loop_a_tick.py](scripts/loop_a_tick.py) — Loop A staging script (Python half of the autonomous research tick); state at [signals/loop_a_state.json](signals/loop_a_state.json), daily reports at `results/loop_a/<YYYY-MM-DD>.md`.
- `results/<primary>/` — pipeline outputs: `report.md`, `summary.json`, `oof_predictions.parquet`, `metrics_per_fold.json`, `threshold_grid_metrics.json`, `psr_dsr.json`, `mda_per_fold.json`, calibration & equity plots, fitted `models/*.joblib`.
- `cache/fred/` — parquet cache of FRED series, keyed by code. Tests monkeypatch `_make_fred_client` rather than hitting the network.

## Loop A — autonomous research loop (v1)

`/loop-a-tick` runs ONE iteration of the regime-driven autonomous research loop: pick least-recently-explored regime via LRU → dispatch `phase5-hypothesizer` to generate a proposal → dispatch `phase5-devils-advocate` to review → run pre-flight check on PROCEED verdicts → record outcome to `signals/loop_a_state.json` + `results/loop_a/<YYYY-MM-DD>.md`. **Stops at pre-flight**; the full M3 audit (~10-30 min CPU) is a separate manual step (`uv run python -m phase5.run_proposal --proposal signals/proposals/<id>.json`).

- Python staging half lives in [scripts/loop_a_tick.py](scripts/loop_a_tick.py). It's a 4-stage state machine (`stage_propose` → `stage_review` → `run_preflight` → `record_tick`); each invocation advances exactly one stage. Between Python stages, the Claude Code session dispatches the agents via the Agent tool. State enforces ordering — `current_tick.stage` blocks out-of-order calls.
- The hypothesizer is strictly regime-conditioned and never receives idea text from `ideas/` (lookahead firewall, per [.claude/skills/phase5-regime-methodology/SKILL.md](.claude/skills/phase5-regime-methodology/SKILL.md)). Loop A v1 is autonomous via dossier rotation; `ideas/` is parallel human-side capture for the future `/idea-triage` ritual.
- **Custom-primary materialization (Option B)**: a survivor with `primary: phase5_custom` cannot audit directly — the pseudocode is not executable. `phase5.run_proposal` parks it at status `pending_materialization` (the record carries the pseudocode + expected module path). A human writes `pipeline/primaries_phase5/<primary>.py`, has it adversarially reviewed for lookahead (custom code is NOT auto-trusted), then re-runs the same `run_proposal` command — the gate clears once the module exists and the audit proceeds. There is no automated codegen by design.
- Loop B (paper-trade → live → decay-rotation) is the north-star goal but gated on sizing (B-TBD), shut-off, and paper-trade infrastructure. **No live deployment until Loop A produces ≥1 audit-surviving strategy.**

## Workflow notes

- **Committing is allowed without an explicit request.** You may `git commit` on your own when a unit of work is complete and verified (this overrides the default "commit only when the user asks"). Branch first if on `main`. **Pushing still requires the user's explicit OK** — never `git push` autonomously.
- The pipeline runs both `ema_cross` and `momentum_zscore` primaries on every full run by design — do NOT add code that picks one before training meta-labelers (that's cherry-picking).
- Always run `--dry-run` first on any config change that touches `hyperparam_search.n_iter`, `walk_forward.n_folds`, or `models` — it extrapolates per-model wall time and warns if a model is projected over `dry_run.max_minutes_per_model_warn`.
- `pipeline/__init__.py` is empty by design — modules are imported by path, no package-level re-exports.
- The data CSV contract is enforced in [pipeline/data.py](pipeline/data.py) `load_dataset`: columns `{open, high, low, close, volume}` + a time column named `time` or `timestamps`, strictly monotonic, no duplicates, UTC tz-aware.
- **Log ideas to [backlog/](backlog/) instead of relying on conversation memory.** When an improvement, hypothesis, refactor, or feature surfaces in conversation and isn't being implemented immediately, call `backlog.db.add_entry(...)` with `status="proposed"` before moving on (or for raw observations without scope yet, use `/capture` which writes to `ideas/`). Conversation context evaporates across sessions; the registry is the durable store. Promote to a spec when it's time to act. Use `backlog.db.change_status(b_id, "done", reason)` or `("discarded", reason)` — never delete IDs.
- **Record every load-bearing decision in the backlog at the moment it happens — never only in code or commit messages.** A decision that changes the research or deployment posture MUST be appended to the relevant B-ID's history via `backlog.db.append_history(...)` in the same work session, before moving on.

## Workflow / Backlog Process

Follow a spec-first, test-first (TDD) workflow for backlog items: reproduce the issue, create a review-gated B-ID, implement, verify with full test suite, spawn a code review agent to review the code then commit and push. Avoid cheap quick-fixes to underlying data; prefer durable code changes.

## Quant Validation

Be honest and skeptical about results: explicitly check whether 'survivors'/positive signals are false positives (single-class labels, null metrics, calibration) before reporting an edge.
