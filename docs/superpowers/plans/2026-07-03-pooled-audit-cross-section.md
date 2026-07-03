# Pooled Audit Path + Cross-Sectional Features Implementation Plan (B0006 + B0010)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Loop A equity proposals auditable against the pooled 35-name M3 universe (fixing the event-floor starvation that killed ticks 1-2), give the meta the B0006 cross-sectional feature pack, and re-audit the DA-approved-but-starved cusum/BULL_QUIET hypothesis (T002D1) under the new path.

**Architecture:** Three additive layers on the existing M3 machinery. (1) `pipeline/cross_section.py` computes the B0006 pack from the loaded OHLCV panel; `scripts/run_pooled_equity_d1.py` joins it config-gated. (2) The same runner's `build_member_inputs` gains per-member regime gating (each ticker's own `data/regimes/<T>_d1_regimes.parquet`) and feature-override application, mirroring `run_backtest.py`'s single-name logic. (3) `phase5/run_proposal.py` gains a `--pooled-universe` mode: transient POOLED config from the proposal → subprocess the pooled runner → v1 pooled grading (pooled event floor, per-(asset,fold) nanmedian Sharpe, breadth-across-names, long/short split) → `signals/audit_results/<id>.json` with `"mode": "pooled_universe"`. Training itself stays inside the untouched `_run_one_pool`.

**Tech Stack:** Python via `uv run`; pandas/numpy/sklearn; pytest with `synth_ohlcv` fixture; existing M3 artifacts (44 CSVs, 44 regime parquets, universe yaml).

## Global Constraints

(From CLAUDE.md invariants + this project's specs — every task implicitly includes these.)

- All Python through `uv run`. Never bare `python`/`pytest`.
- Sharpe: `sqrt(trades_per_year)` annualization; `NaN` (never 0) under 30 trades or zero std; aggregate with `np.nanmedian`/`np.nanmean`.
- Purged CV everywhere; sigmoid calibration on chronological tail; sample weights to fit + search + calibration — all inside `_run_one_pool`, which this plan MUST NOT modify (`scripts/run_multi_h4.py` is off-limits except where a task says "additive").
- Primaries rule-based; fixed 0.55 headline threshold for stack decisions; the audit's `ev_breakeven_v1` effective threshold is carried as `metrics.audit_effective_threshold` in transient configs.
- Cross-sectional features: percentile ranks (not z-scores), residualize vs the equal-weight basket, NEVER sector one-hot dummies (collider guard), point-in-time within the frozen universe — per `docs/superpowers/specs/2026-07-03-b0006-cross-sectional-features.md` (the binding spec for Task 1).
- Regime gating filters EVENTS BEFORE triple-barrier labeling (weights/labels/folds computed only on in-scope events), mirroring `scripts/run_backtest.py:1089-1099`.
- B0003 caveats bind any positive result: within-universe relative claims only; long/short reported separately; universe changes are recorded trials.
- Frozen interpretation rules are written ex-ante (config headers / audit JSON), never after seeing results.
- Commit per task (no push). Data/results/signals artifacts stay un-versioned (gitignored) — commits carry code, configs, docs, backlog only.

## File Structure

- Create `pipeline/cross_section.py` — B0006 pack builder (pure functions).
- Modify `scripts/run_pooled_equity_d1.py` — CS join, regime gating, feature overrides (all config-gated; default-off keeps M3 re-runs byte-identical).
- Modify `phase5/run_proposal.py` — additive `--pooled-universe` mode (new functions; existing single-name path untouched).
- Modify `phase5/regime_stats.py` + `scripts/build_all_regimes.py` — additive `n_events_audit_window` baseline field.
- Modify `.claude/agents/phase5-hypothesizer.md` — one guidance block (audit-window event counts + schema char limits; partial B0009).
- Tests: `tests/test_cross_section.py`, extend `tests/test_run_pooled_equity_d1.py`, `tests/phase5/test_pooled_audit_path.py`, `tests/phase5/test_baseline_audit_window.py`.

---

### Task 1: B0006 cross-sectional feature pack

**Files:**
- Create: `pipeline/cross_section.py`
- Test: `tests/test_cross_section.py`

**Interfaces:**
- Produces: `build_cross_sectional_features(panel: dict[str, pd.DataFrame], tickers: list[str]) -> dict[str, pd.DataFrame]` — input: per-ticker OHLCV frames (columns open/high/low/close/volume, tz-aware UTC DatetimeIndex); output: per-ticker frames with EXACTLY these 8 columns, aligned to that ticker's index: `cs_mom_12_1_rank`, `cs_52wk_high_rank`, `cs_st_reversal_rank`, `cs_idio_vol_rank`, `cs_turnover_rank`, `cs_basket_beta_rank`, `cs_dispersion`, `cs_breadth_200`. Task 2 consumes this.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cross_section.py
"""B0006 pack: point-in-time construction, rank bounds, panel scalars."""
import numpy as np
import pandas as pd
import pytest

from pipeline.cross_section import build_cross_sectional_features

N, TICKERS = 400, ["AAA", "BBB", "CCC", "DDD", "EEE"]


@pytest.fixture
def panel():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2010-01-04", periods=N, freq="B", tz="UTC")
    out = {}
    for i, t in enumerate(TICKERS):
        r = rng.normal(0.0003 * (i + 1), 0.01 + 0.002 * i, N)
        close = 50.0 * np.exp(np.cumsum(r))
        out[t] = pd.DataFrame({
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": rng.uniform(1e5, 1e6, N),
        }, index=idx)
    return out


EXPECTED = ["cs_mom_12_1_rank", "cs_52wk_high_rank", "cs_st_reversal_rank",
            "cs_idio_vol_rank", "cs_turnover_rank", "cs_basket_beta_rank",
            "cs_dispersion", "cs_breadth_200"]


def test_schema_bounds_and_panel_scalars(panel):
    feats = build_cross_sectional_features(panel, TICKERS)
    assert set(feats) == set(TICKERS)
    for t in TICKERS:
        f = feats[t]
        assert list(f.columns) == EXPECTED
        assert f.index.equals(panel[t].index)
        valid = f.dropna()
        rank_cols = [c for c in EXPECTED if c.endswith("_rank")]
        assert ((valid[rank_cols] >= 0) & (valid[rank_cols] <= 1)).all().all()
    # panel scalars identical across tickers at each t
    for col in ("cs_dispersion", "cs_breadth_200"):
        stacked = pd.concat([feats[t][col] for t in TICKERS], axis=1)
        stacked = stacked.dropna()
        assert (stacked.nunique(axis=1) == 1).all(), f"{col} must be panel-wide"


def test_point_in_time_no_lookahead(panel):
    """Feature values at t must be identical when future rows are truncated."""
    full = build_cross_sectional_features(panel, TICKERS)
    cut = N - 60
    truncated_panel = {t: df.iloc[:cut] for t, df in panel.items()}
    trunc = build_cross_sectional_features(truncated_panel, TICKERS)
    for t in TICKERS:
        a = full[t].iloc[:cut]
        b = trunc[t]
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12)


def test_ranks_are_cross_sectional_not_temporal(panel):
    """At any date, the 5 tickers' momentum ranks must be a permutation of
    the 5 evenly-spaced percentiles — proving the rank is across names."""
    feats = build_cross_sectional_features(panel, TICKERS)
    date = feats[TICKERS[0]]["cs_mom_12_1_rank"].dropna().index[-1]
    vals = sorted(feats[t].loc[date, "cs_mom_12_1_rank"] for t in TICKERS)
    expected = [(i + 1) / len(TICKERS) for i in range(len(TICKERS))]
    assert np.allclose(vals, expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cross_section.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.cross_section'`

- [ ] **Step 3: Implement the pack**

```python
# pipeline/cross_section.py
"""B0006 cross-sectional feature pack v1.

Binding spec: docs/superpowers/specs/2026-07-03-b0006-cross-sectional-features.md.
6 stock-level percentile-rank features + 2 panel scalars, computed point-in-time
within the frozen universe from OHLCV only. Residualization is vs the equal-weight
basket (~first PC); NEVER sector one-hots (collider guard, LdP-Zoonekynd 2024).
All ranks are cross-sectional percentiles in [1/N, 1] via rank(pct=True) at each
date — stationary by construction on a 35-wide panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _panel_frame(panel: dict[str, pd.DataFrame], col: str,
                 tickers: list[str]) -> pd.DataFrame:
    """Union-index frame of one column across tickers (columns = tickers)."""
    return pd.DataFrame({t: panel[t][col] for t in tickers})


def build_cross_sectional_features(
    panel: dict[str, pd.DataFrame], tickers: list[str],
) -> dict[str, pd.DataFrame]:
    close = _panel_frame(panel, "close", tickers).sort_index()
    volume = _panel_frame(panel, "volume", tickers).sort_index()
    logc = np.log(close)
    r1 = logc.diff()

    # Equal-weight basket return (~first PC of a 35-wide large-cap panel).
    basket_r = r1.mean(axis=1)
    # Market-residual daily return: r_i - beta-free demeaning (v1: simple excess
    # vs basket; a rolling-beta residual is the #6 feature's job, not needed here).
    resid_r = r1.sub(basket_r, axis=0)

    # 1. 12-1 momentum, market-residualized: sum of residual log-returns t-252..t-21.
    mom_12_1 = resid_r.rolling(252).sum().shift(21)
    # 2. 52-week-high proximity: close / rolling 252d max (no residualization).
    prox_52wk = close / close.rolling(252).max()
    # 3. Short-term residual reversal: 21d residual return (sign kept raw; the
    #    meta learns the direction — do not pre-negate).
    st_rev = resid_r.rolling(21).sum()
    # 4. Idiosyncratic vol: 63d std of residual returns.
    idio_vol = resid_r.rolling(63).std()
    # 5. Turnover: 21d dollar volume vs own trailing 252d median dollar volume.
    dollar_vol = close * volume
    turnover = dollar_vol.rolling(21).mean() / dollar_vol.rolling(252).median()
    # 6. Rolling 63d beta to the equal-weight basket.
    cov = r1.rolling(63).cov(basket_r)
    beta = cov.div(basket_r.rolling(63).var(), axis=0)

    def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
        return df.rank(axis=1, pct=True)

    ranks = {
        "cs_mom_12_1_rank": cs_rank(mom_12_1),
        "cs_52wk_high_rank": cs_rank(prox_52wk),
        "cs_st_reversal_rank": cs_rank(st_rev),
        "cs_idio_vol_rank": cs_rank(idio_vol),
        "cs_turnover_rank": cs_rank(turnover),
        "cs_basket_beta_rank": cs_rank(beta),
    }
    # Panel scalars (same value for every name at t).
    dispersion = r1.rolling(21).sum().std(axis=1)
    breadth = (close > close.rolling(200).mean()).mean(axis=1)

    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        f = pd.DataFrame(index=panel[t].index)
        for name, frame in ranks.items():
            f[name] = frame[t].reindex(panel[t].index)
        f["cs_dispersion"] = dispersion.reindex(panel[t].index)
        f["cs_breadth_200"] = breadth.reindex(panel[t].index)
        out[t] = f
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cross_section.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```powershell
git add pipeline/cross_section.py tests/test_cross_section.py
git commit -m "feat(b0006): cross-sectional feature pack v1 (6 ranks + 2 panel scalars)"
```

---

### Task 2: Wire CS features into the pooled runner (config-gated)

**Files:**
- Modify: `scripts/run_pooled_equity_d1.py` (in `main()`, around the per-ticker loop)
- Test: extend `tests/test_run_pooled_equity_d1.py`

**Interfaces:**
- Consumes: `build_cross_sectional_features` (Task 1).
- Produces: when the config has `features.cross_sectional: true`, each ticker's tier-2 features are joined with its 8 CS columns BEFORE the dropna/member-build; default (absent/false) leaves behavior byte-identical.

- [ ] **Step 1: Write the failing test** (append to `tests/test_run_pooled_equity_d1.py`; reuse its existing `cfg` fixture, `_features_for` helper, and the mocked-`main()` pattern from `test_count_events_only_writes_artifacts_and_skips_training`)

```python
def test_cross_sectional_join_is_config_gated(synth_ohlcv, cfg, tmp_path, monkeypatch):
    """With features.cross_sectional true, member X gains the 8 cs_* columns;
    without it, X has none. Uses the count-events mocked-main pattern with a
    2-ticker universe so the CS panel is non-degenerate."""
    import scripts.run_pooled_equity_d1 as runner
    # ... (mirror the existing mocked-main test's setup: 2-ticker universe yaml,
    # config yaml written under tmp_path with features: {cross_sectional: true},
    # monkeypatched load_dataset returning synth_ohlcv for both tickers,
    # build_macro_frame -> empty df, build_tier2_features -> _features_for,
    # _run_one_pool -> AssertionError guard, sys.argv with --count-events-only)
    # After main(): read <out>/<ticker>/<primary>/events_side_fwd.parquet exists
    # AND assert the runner-built member X carried cs_ columns by capturing
    # build_member_inputs calls via monkeypatch-wrapping it and recording
    # the features frame's columns.
    seen_cols = {}
    orig = runner.build_member_inputs
    def spy(asset, primary_name, ohlcv, features, cfg_):
        seen_cols[asset] = list(features.columns)
        return orig(asset, primary_name, ohlcv, features, cfg_)
    monkeypatch.setattr(runner, "build_member_inputs", spy)
    # ... run main() ...
    for cols in seen_cols.values():
        assert "cs_mom_12_1_rank" in cols and "cs_breadth_200" in cols
```

(The implementer writes the full test by copying the existing mocked-main test's scaffolding verbatim and adding the spy + a second run with `cross_sectional: false` asserting no `cs_` columns.)

- [ ] **Step 2: Run to verify it fails** — `uv run pytest tests/test_run_pooled_equity_d1.py -v` → new test FAILS (no cs_ columns).

- [ ] **Step 3: Implement in `main()`** — after loading all tickers' OHLCV but before the per-ticker feature build, restructure the loop into two passes:

```python
    # Pass 1: load the panel (needed whole for cross-sectional features).
    panel: dict[str, pd.DataFrame] = {}
    for t in tickers:
        panel[t] = load_dataset(Path(f"data/D1/{t}_D1.csv")).loc[s:e]

    cs_frames: dict[str, pd.DataFrame] = {}
    if (cfg.get("features") or {}).get("cross_sectional"):
        from pipeline.cross_section import build_cross_sectional_features
        cs_frames = build_cross_sectional_features(panel, tickers)

    # Pass 2: per-ticker features + members (unchanged below this point).
    for t in tickers:
        ohlcv = panel[t]
        features = build_tier2_features(ohlcv, macro)
        if cs_frames:
            features = features.join(cs_frames[t])
        features = features.dropna()
        ohlcv = ohlcv.loc[features.index]
        ...
```

- [ ] **Step 4: Run the full runner test file + suite** — `uv run pytest tests/test_run_pooled_equity_d1.py -q && uv run pytest -q` → green.

- [ ] **Step 5: Commit** — `git commit -m "feat(b0006): config-gated cross-sectional join in pooled runner"`

---

### Task 3: Per-member regime gating + feature overrides in the pooled runner

**Files:**
- Modify: `scripts/run_pooled_equity_d1.py` (`build_member_inputs` + config plumbing)
- Test: extend `tests/test_run_pooled_equity_d1.py`

**Interfaces:**
- Consumes: `data/regimes/<TICKER>_d1_regimes.parquet` (column `regime_id`, bar-time index) — exists for all 44 names since the M3 plan's Task 8.
- Produces: config keys `regime_scope: [BULL_QUIET, ...]` (optional; filters each member's events to bars whose OWN regime_id is in scope, BEFORE triple-barrier) and `feature_overrides_add` / `feature_overrides_drop` (optional; drop subtracts from the meta's X; add is validated against available columns via `pipeline.features.feature_add_status` — mirror `run_backtest.py:1149-1168`, writing `feature_overrides_status.json` per (asset, primary)). All four built-in primaries work via the imported `_select_primary` (already true — configs just carry the param block, e.g. `primary: {candidates: [cusum_filter], cusum_filter: {threshold_atr: 2.0}}`).

- [ ] **Step 1: Write failing tests** — two additions:

```python
def test_regime_gate_filters_member_events(synth_ohlcv, cfg, tmp_path, monkeypatch):
    """With regime_scope set and a synthetic regime parquet marking only the
    first half of bars BULL_QUIET, the member's events all fall in that half
    and the count is strictly below the ungated count."""
    # Build regime parquet: regime_id = BULL_QUIET for index[:n//2] else BEAR_QUIET,
    # write to tmp_path; monkeypatch runner._regimes_path(asset) -> that file.
    # Call build_member_inputs with cfg | {"regime_scope": ["BULL_QUIET"]}.
    # Assert: gated member is not None; all gated event_time < midpoint timestamp;
    # len(gated.X) < len(ungated.X).

def test_feature_overrides_drop_removes_column(synth_ohlcv, cfg):
    """cfg feature_overrides_drop=['f_mom'] -> 'f_mom' absent from member X;
    baseline run has it present."""
```

(Implementer fills bodies using the module's existing fixtures/helpers; the regime parquet is `pd.DataFrame({"regime_id": [...]}, index=ohlcv.index)` written with `.to_parquet`.)

- [ ] **Step 2: Verify they fail** (no gating/overrides exist yet).

- [ ] **Step 3: Implement in `build_member_inputs`** — after `events = ...` and before the empty-check:

```python
    scope = cfg.get("regime_scope") or []
    if scope:
        regimes = pd.read_parquet(_regimes_path(asset))
        in_scope_ts = regimes.index[regimes["regime_id"].isin(set(scope))]
        n_before = len(events)
        events = events[events.index.isin(in_scope_ts)]
        print(f"  {asset}/{primary_name}: regime gate kept {len(events)}/{n_before} events")
        if events.empty:
            return None
```

with module-level `def _regimes_path(asset: str) -> Path: return Path(f"data/regimes/{asset}_d1_regimes.parquet")` (monkeypatch seam). For overrides — before the X build:

```python
    fo_drop = cfg.get("feature_overrides_drop", []) or []
    meta_features = features.drop(columns=["_atr_14"]).drop(columns=fo_drop, errors="ignore")
    X = meta_features.loc[valid.index].copy()
```

and after the member is built in `main()`, write `feature_overrides_status.json` per (asset, primary) using `feature_add_status(cfg.get("feature_overrides_add", []), set(meta_features.columns))` imported from `pipeline.features` (exact mirror of run_backtest.py:1158-1168).

- [ ] **Step 4: Full suite green.** `uv run pytest -q`
- [ ] **Step 5: Commit** — `git commit -m "feat(b0010): per-member regime gating + feature overrides in pooled runner"`

---

### Task 4: Pooled-universe audit mode in run_proposal

**Files:**
- Modify: `phase5/run_proposal.py` (additive functions + `--pooled-universe` CLI flag; single-name path untouched)
- Test: `tests/phase5/test_pooled_audit_path.py`

**Interfaces:**
- Consumes: `configs/equity_m3_d1.yaml` as the pooled template; `scripts/run_pooled_equity_d1.py` CLI (`--config`, `--count-events-only`); the proposal dataclass fields (`primary`, `primary_params`, `regime_scope`, `feature_overrides`, `barrier_geometry_attestation` tp/sl, `falsification_criterion`, `threshold_rule`); existing `run_proposal` helpers for the ev-threshold computation; `scripts/report_long_short_split.py` CLI.
- Produces:
  - `build_transient_pooled_config(p: Proposal) -> Path` — loads the M3 stocks config, overlays: `primary.candidates=[p.primary]`, `primary[p.primary]=p.primary_params`, `triple_barrier.tp_atr_mult/sl_atr_mult` from the proposal geometry (horizon stays 40), `regime_scope=list(p.regime_scope)`, `feature_overrides_add/drop`, `features.cross_sectional: true`, `threshold_selection.method: "fixed_ev"`, `metrics.audit_effective_threshold: <ev threshold computed by the existing threshold_rule code>`, `output_dir: results/phase5_pooled/<p.id>`; writes to `phase5/runtime/configs/<p.id>_pooled.yaml`.
  - `run_pooled_audit(p: Proposal) -> dict` — (1) subprocess `uv run python scripts/run_pooled_equity_d1.py --config <transient> --count-events-only`, parse `member_event_counts.json`, POOLED event floor check: `sum(n_events) >= wf_event_floor(cfg.walk_forward.n_folds, cfg.walk_forward.train_min_bars)`; if starved → status `event_floor` (same shape as today's single-name result, plus `"mode": "pooled_universe"` and per-member counts). (2) Otherwise full run (same CLI without the flag), then `report_long_short_split` on the output dir, then v1 grading (below), writing `signals/audit_results/<p.id>.json`.
  - v1 pooled grading (documented as reviewable in the JSON under `grading_version: "pooled_v1"`): `n_trades_total` = trades at the audit threshold summed from per-asset `metrics_per_fold.json` best-model cells; `median_active_fold_sharpe` = `np.nanmedian` over per-(asset, fold) sharpe_net where n_trades≥30; `breadth` = count of assets with ≥30 trades and positive aggregate best-model sharpe; `long_short` = the split JSON. Verdict field `criterion_eval`: each falsification_criterion key evaluated where computable, `null` where the pooled context has no analog (audit_class_in is per-single-name classification → recorded as `not_applicable_pooled_v1`). Final `status`: `event_floor` | `completed_pending_human_read` — the pooled v1 NEVER auto-promotes; a human (or the skeptic agent) reads the artifacts.
  - CLI: `uv run python -m phase5.run_proposal --proposal <path> --pooled-universe`.

- [ ] **Step 1: Write failing tests** (`tests/phase5/test_pooled_audit_path.py`): (a) `build_transient_pooled_config` on a minimal Proposal object (reuse construction patterns from `tests/phase5/test_b0085_param_contract.py`) → yaml exists, primary params/geometry/regime_scope/output_dir all overlaid, `_run_one_pool`-required sections (models, calibration, stacking, best_model, meta_pooling, dry_run) survive from the template; (b) grading unit test on synthetic per-asset artifacts written under tmp_path (two assets × two folds of `metrics_per_fold.json` + a hand-built `long_short_split.json`) → verify n_trades_total, nanmedian (NaN cells skipped, never zero-filled), breadth count; (c) starved path: monkeypatch the subprocess call to emit a `member_event_counts.json` summing below the floor → result JSON has `status: "event_floor"`, `mode: "pooled_universe"`, and no full run attempted.
- [ ] **Step 2: Verify they fail.**
- [ ] **Step 3: Implement** (additive functions; reuse the module's existing subprocess/threshold/json-writing helpers; no edits inside existing functions beyond the `main()` argparse flag + dispatch branch).
- [ ] **Step 4: Full suite green.** `uv run pytest -q`
- [ ] **Step 5: Commit** — `git commit -m "feat(b0010): pooled-universe audit mode in run_proposal (grading pooled_v1)"`

---

### Task 5: Dossier baseline windowing + hypothesizer guidance (B0010 root cause 1, partial B0009)

**Files:**
- Modify: `phase5/regime_stats.py` (additive field), `scripts/build_all_regimes.py` (pass-through arg if needed)
- Modify: `.claude/agents/phase5-hypothesizer.md` (guidance block)
- Test: `tests/phase5/test_baseline_audit_window.py`

**Interfaces:**
- Produces: each `primary_baseline_summary.<primary>` entry gains `n_events_audit_window: int` — the same in-regime event count restricted to bars ≥ `AUDIT_WINDOW_START = "2006-01-01"` (module constant with a docstring pointing at B0010: dossier full-history counts overstate audit-window counts ~2x on long-history names; the hypothesizer must design floors against THIS number). Existing fields unchanged (additive only — old dossiers stay loadable).

- [ ] **Step 1: Failing test**: build a dossier over a synthetic series spanning 2000-2015 with a known event distribution (events before and after 2006) → assert `n_events_audit_window < n_events` and equals the count of events stamped ≥ 2006-01-01. Reuse whatever fixture pattern `tests/phase5/test_build_all_regimes.py` uses for dossier construction.
- [ ] **Step 2: Verify fails.** **Step 3: Implement** (in the baseline-summary builder, compute the same event mask restricted to `ohlcv.index >= AUDIT_WINDOW_START`). **Step 4: Suite green.**
- [ ] **Step 5: Rebuild the 44 equity dossiers**: same command as the M3 plan Task 8 (`build_all_regimes.py --assets <44 tickers> --frequencies D1 --force`). Spot-check one dossier JSON shows the new field.
- [ ] **Step 6: Hypothesizer doc** — add one block to `.claude/agents/phase5-hypothesizer.md`: input arrives INLINE in the dispatch prompt (no tool calls; output = JSON in the final message, the session persists it); `hypothesis`/`causal_story` are 30-800 chars each (target ≤700); event-count design floors must use `n_events_audit_window`, not `n_events`.
- [ ] **Step 7: Commit** — `git commit -m "feat(b0010): audit-window baseline counts in dossiers + hypothesizer guidance (partial B0009)"`

---

### Task 6: EXECUTION — re-audit T002D1 pooled + B0006 validation runs

No new code. Move B0006 and B0010 to `in_progress` first (`backlog.db.change_status`, reason "plan execution started").

- [ ] **Step 1: Dry-run gate.** `uv run python scripts/run_pooled_equity_d1.py --config phase5/runtime/configs/<id>_pooled.yaml --dry-run` is implicit in the audit; instead gate cheaply: run the pooled audit in count-events form first — `uv run python -m phase5.run_proposal --proposal signals/proposals/20260703-ABT-D1-BULL_QUI-T002D1.json --pooled-universe` will do the count-events check itself. Expected: pooled in-regime cusum events across 35 names ≈ 35 × ~360 ≈ 12k ≫ floor → proceeds to the full run (1-3 h CPU; the regime gate + sparse primary shrink training vs the M3 full run).
- [ ] **Step 2: Read the audit JSON** (`signals/audit_results/20260703-ABT-D1-BULL_QUI-T002D1.json`): verify `mode: pooled_universe`, grading fields populated, long/short split present. Apply the DA's binding caveats from `signals/devils_advocate_reviews/20260703-ABT-D1-BULL_QUI-T002D1.json`: long-share of pnl, cs_spread_21 MDA stability, vix_chg_5 domination check.
- [ ] **Step 3: B0006 validation protocol** (spec §5, in order): (a) M3 stocks config + `features.cross_sectional: true`, state primaries — full run into `results/clf_equity_m3_d1_cs/`; (b) same + cusum_filter primary (add `cusum_filter: {threshold_atr: 2.0}` to candidates) into `results/clf_equity_m3_cusum_cs/`. Compare clustered-MDA of cs_* clusters and nanmedian per-fold F1 vs the recorded M3 baseline. Evaluate the spec §6 falsification criteria — any hit = the pack FAILED, record honestly.
- [ ] **Step 4: Readout + records.** Write `docs/superpowers/reports/<date>-pooled-audit-readout.md` (same honesty checklist as the M3 readout: gates, NaN discipline, long/short, one-lucky-fold, verdicts per pre-registered criteria). Append `decision` history entries to B0006 and B0010 with the real numbers; statuses: `done` only if their scope is genuinely complete, else stay `in_progress` with the next step named. Commit docs + backlog.

---

## Self-Review Notes

- B0010 scope: windowing fix → Task 5; pooled audit path → Tasks 3-4; per-episode interim criterion → deliberately NOT built (the pooled path supersedes it; noted here so the omission is a decision, not a gap); T002D1 re-run → Task 6. B0006 scope: pack → Task 1; wiring → Task 2; validation protocol + falsification evaluation → Task 6.
- Types: `build_cross_sectional_features` output joins tier-2 features on the ticker's index (Task 2 join is index-aligned); `regime_scope`/`feature_overrides_*` config keys read by Task 3 are exactly what Task 4's transient config writes; the pooled grading reads the same per-asset artifact filenames `_write_per_asset_oos` produces (verified in the M3 plan).
- Placeholder scan: Tasks 2/3 Step-1 tests are intentionally scaffold-by-reference to the SAME file's existing mocked-main test (the implementer copies working scaffolding rather than risking drift from a stale transcription); all novel logic (CS pack, gating, config overlay, grading) carries complete code or exact field-level contracts.
- Known risk carried from the M3 plan: config keys consumed only inside `_write_per_asset_oos` surface as KeyError only in FULL runs — same mitigation applies (copy the section from configs/multi_h4.yaml; never edit run_multi_h4.py).
