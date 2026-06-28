# QuantTrade Equity-Swing Seed — Implementation Plan (M0–M2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a clean `D:\PROJECTS\QuantTrade` repo seeded only from QuantHack's asset-agnostic research core, add a free yfinance equity-data layer, and prove the meta-labeling engine + alpha-generator scaffolding run end-to-end on NVDA daily bars.

**Architecture:** Copy the 44 core `pipeline/` modules + the `phase5/` alpha generator + orchestrator + tests into a fresh repo (new git history). Add one small data module (`EquityDataSource` seam → `YFinanceSource`) and a fetch CLI. Register an `equity` asset class. Build NVDA regime parquet + dossier. Run the backtest orchestrator on an equity config as the M2 plumbing smoke test.

**Tech Stack:** Python 3.12 (`.venv` via `uv`), pandas, scikit-learn, xgboost/lightgbm, yfinance, pytest. Source repo: `D:\PROJECTS\QuantHack` (read-only origin). Target repo: `D:\PROJECTS\QuantTrade`.

**Spec:** `D:\PROJECTS\QuantTrade\docs\superpowers\specs\2026-06-28-quanttrade-equity-swing-repurpose-design.md`

**Scope note:** This plan covers spec milestones **M0 (seed & green), M1 (equity data), M2 (NVDA smoke)**. Spec **M3 (cross-section)** is a *separate later plan*, gated on the open universe decision (spec §10 OQ1).

## Global Constraints

- **Run Python only through `uv run`** — never bare `python`/`pytest` (wrong interpreter on Windows; pytest lives in `.venv\Scripts`).
- **Use the PowerShell tool** (not Bash) for any process/service/background work on Windows.
- **Fresh git history** — `git init` in QuantTrade; do NOT clone QuantHack's history.
- **No MT5 dependency** — `MetaTrader5` must NOT appear in QuantTrade's `pyproject.toml`.
- **Primaries are deterministic rule-based functions, never ML classifiers** (enforced by `tests/test_primary_input_contract.py`).
- **Pipeline invariants carry over verbatim:** Sharpe uses `sqrt(trades_per_year)`; Sharpe is `NaN` (not 0) when `n_trades<30` or `std==0`; `should_stack` has a `min_trades_per_fold` gate (default 30); FRED macro is `.shift(1)`; inner-CV is `PurgedTimeSeriesSplit`; calibration is `FrozenEstimator` + `sigmoid`; sample weights via `avg_uniqueness` passed to every fit.
- **Equity data must be split/dividend-adjusted** (`auto_adjust=True`); unadjusted equity bars are invalid input.
- **The copied test suite passing is the seed's acceptance gate** — a green suite proves no hackathon import leaked across.

---

### Task 1: Seed the core into QuantTrade and get a green test suite

**Files:**
- Create: `D:\PROJECTS\QuantTrade\` (whole repo tree, copied from QuantHack)
- Create: `D:\PROJECTS\QuantTrade\pyproject.toml` (pruned)
- Create: `D:\PROJECTS\QuantTrade\.gitignore`

**Interfaces:**
- Produces: an importable `pipeline.*` and `phase5.*` package set + a runnable `tests/` suite. Later tasks rely on `pipeline.data.load_dataset`, `pipeline.regimes`, `phase5.asset_registry.ASSET_REGISTRY`, `phase5.regime_stats`.

- [ ] **Step 1: Selective copy from QuantHack** (PowerShell tool). `pipeline/` and `phase5/` copy whole (their modules have import webs; prune in Step 2). `scripts/`, `configs/`, `.claude/` are standalone — copy ONLY the named core files (clean seed, per spec §3).

```powershell
$src = "D:\PROJECTS\QuantHack"; $dst = "D:\PROJECTS\QuantTrade"
# import-web dirs: copy whole, prune in Step 2
Copy-Item -Recurse -Force "$src\pipeline" "$dst\pipeline"
Copy-Item -Recurse -Force "$src\phase5"   "$dst\phase5"
Copy-Item -Recurse -Force "$src\tests"    "$dst\tests"
Copy-Item -Recurse -Force "$src\backlog"  "$dst\backlog"
Copy-Item -Recurse -Force "$src\ideas"    "$dst\ideas"
Copy-Item -Force "$src\uv.lock" "$dst\uv.lock"
Copy-Item -Force "$src\pyproject.toml" "$dst\pyproject.toml"
Copy-Item -Force "$src\.env.example" "$dst\.env.example" -ErrorAction SilentlyContinue
# scripts: selective
New-Item -ItemType Directory -Force "$dst\scripts" | Out-Null
foreach ($f in @("run_xau_d1.py","run_multi_h4.py","loop_a_tick.py","ingest_gld_volume.py")) {
  Copy-Item -Force "$src\scripts\$f" "$dst\scripts\$f" -ErrorAction SilentlyContinue }
# configs: selective
New-Item -ItemType Directory -Force "$dst\configs" | Out-Null
foreach ($f in @("xau_d1.yaml","multi_d1.yaml","multi_h4.yaml","friction.yaml")) {
  Copy-Item -Force "$src\configs\$f" "$dst\configs\$f" -ErrorAction SilentlyContinue }
# .claude: selective agents + the phase5 methodology skill
New-Item -ItemType Directory -Force "$dst\.claude\agents" | Out-Null
foreach ($a in @("phase5-hypothesizer.md","phase5-devils-advocate.md","phase5-skeptic.md","quant-phd-advisor.md")) {
  Copy-Item -Force "$src\.claude\agents\$a" "$dst\.claude\agents\$a" -ErrorAction SilentlyContinue }
Copy-Item -Recurse -Force "$src\.claude\skills\phase5-regime-methodology" "$dst\.claude\skills\phase5-regime-methodology" -ErrorAction SilentlyContinue
# clean compiled artifacts
Get-ChildItem -Path $dst -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
```

- [ ] **Step 2: Prune hackathon modules from `pipeline/` and junk from `phase5/`** (PowerShell tool):

```powershell
$p = "D:\PROJECTS\QuantTrade\pipeline"
$hack = @("lottery_sleeve.py","stage2_lambda.py","osr_intraday.py","osr_bundle_writer.py",
  "l2_adapter.py","tardis_adapter.py","intraday_fills.py","llm_distillation.py",
  "llm_primary.py","llm_model_recipes.py","llm_prompt.py",
  "ticks.py","intraday_bars.py","intraday_features.py","intraday_pool.py","intraday_wf.py","intraday_gate.py")
foreach ($f in $hack) { Remove-Item -Force "$p\$f" -ErrorAction SilentlyContinue }
# phase5: drop hackathon diagnostic junk (reports, logs, fire-rate checks, transient runtime) — keep the .py core
$f5 = "D:\PROJECTS\QuantTrade\phase5"
Get-ChildItem "$f5\*.md","$f5\*.log" -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem "$f5\fire_rate_check_*.py" -ErrorAction SilentlyContinue | Remove-Item -Force
Remove-Item -Recurse -Force "$f5\fire_rate_check_results","$f5\runtime" -ErrorAction SilentlyContinue
# Extract the one util loop_a_tick needs; the rest of execution/ is NOT copied
New-Item -ItemType Directory -Force "$p\util" | Out-Null
Copy-Item -Force "D:\PROJECTS\QuantHack\execution\state_store.py" "$p\util\state_store.py"
```

- [ ] **Step 3: Delete tests for removed modules** (PowerShell tool). Any test file whose subject module was deleted must go, or the suite errors on import.

```powershell
$t = "D:\PROJECTS\QuantTrade\tests"
$hacktests = @("test_lottery_sleeve.py","test_stage2_lambda.py","test_osr_intraday.py","test_osr_bundle*.py",
  "test_l2_adapter.py","test_tardis_adapter.py","test_intraday_fills.py","test_llm_*.py",
  "test_ticks.py","test_intraday_bars.py","test_intraday_features.py","test_intraday_pool.py",
  "test_intraday_wf.py","test_intraday_gate.py")
foreach ($f in $hacktests) { Get-ChildItem "$t\$f" -ErrorAction SilentlyContinue | Remove-Item -Force }
```

- [ ] **Step 4: Write the pruned `pyproject.toml`.** Open QuantHack's `pyproject.toml`, copy it to QuantTrade, then remove the `MetaTrader5` dependency line and rename the project. Verify no `MetaTrader5` remains:

Run: `grep -ri "metatrader5\|MetaTrader5" "D:\PROJECTS\QuantTrade\pyproject.toml"`
Expected: no output.

- [ ] **Step 5: Write `.gitignore`** at `D:\PROJECTS\QuantTrade\.gitignore`:

```
.venv/
__pycache__/
*.pyc
.env
data/
cache/
results/
signals/
catboost_info/
*.log
```

- [ ] **Step 6: Init git + install deps** (PowerShell tool):

```powershell
Set-Location "D:\PROJECTS\QuantTrade"; git init; uv sync
```
Expected: `uv sync` completes, `.venv` created.

- [ ] **Step 7: Grep for leaked hackathon imports.** No surviving core module may import a deleted module or live machinery.

Run: `grep -rn "from execution\|import execution\|lottery_sleeve\|tournament_sim\|osr_\|intraday_fills\|l2_adapter\|MetaTrader5\|import mt5" "D:\PROJECTS\QuantTrade\pipeline" "D:\PROJECTS\QuantTrade\phase5" "D:\PROJECTS\QuantTrade\scripts"`
Expected: no output (the one allowed reference — `state_store` — was relocated to `pipeline/util/` and is imported from there; if `loop_a_tick.py` still imports `from execution.state_store`, fix it to `from pipeline.util.state_store` and re-run).

- [ ] **Step 8: Run the test suite** (the acceptance gate):

Run: `uv run pytest -q`
Expected: PASS (collection succeeds, no `ModuleNotFoundError`). If a test fails ONLY because it references a deleted module, delete that test and re-run. Any *logic* failure is a real defect — stop and investigate, do not delete.

- [ ] **Step 9: Commit:**

```powershell
git add -A; git commit -m "seed: equity-swing research core cherry-picked from QuantHack"
```

---

### Task 2: Reset project machinery (CLAUDE.md, registries, README)

**Files:**
- Modify: `D:\PROJECTS\QuantTrade\CLAUDE.md` (trim)
- Modify: `D:\PROJECTS\QuantTrade\backlog\` (empty the entry folders)
- Modify: `D:\PROJECTS\QuantTrade\ideas\` (empty the entry folders)
- Create: `D:\PROJECTS\QuantTrade\README.md`

**Interfaces:**
- Produces: empty `backlog/{proposed,in_progress,blocked,done,discarded}/` and `ideas/{open,promoted,...}/`; a QuantTrade-specific CLAUDE.md with no hackathon posture.

- [ ] **Step 1: Empty the backlog & ideas entry folders** (PowerShell tool) — keep the `db.py`/`SCHEMA.md`/`lint.py` machinery, drop all hackathon B-IDs and I-IDs:

```powershell
foreach ($s in @("proposed","in_progress","blocked","done","discarded")) {
  Get-ChildItem "D:\PROJECTS\QuantTrade\backlog\$s\*.json" -ErrorAction SilentlyContinue | Remove-Item -Force
}
Get-ChildItem "D:\PROJECTS\QuantTrade\ideas" -Recurse -Filter "I*.json" -ErrorAction SilentlyContinue | Remove-Item -Force
Remove-Item -Force "D:\PROJECTS\QuantTrade\backlog\INDEX.json" -ErrorAction SilentlyContinue
```

- [ ] **Step 2: Rebuild the backlog index** so lint passes on the empty registry:

Run: `uv run python -m backlog.lint`
Expected: passes with 0 active entries (if it errors on a missing INDEX.json, run the project's index-rebuild path first, e.g. `uv run python -m backlog.migrate` or the documented rebuild, then re-lint).

- [ ] **Step 3: Trim `CLAUDE.md`.** Replace QuantHack's CLAUDE.md with a QuantTrade version. KEEP: the pipeline architecture diagram, the "Pipeline invariants — DO NOT BREAK" section, the "Where things live" map (update paths: `run_xau_d1.py`→`run_backtest.py`, drop deleted modules), the backlog/ideas workflow, the Quant Validation skepticism rule, and the Loop A methodology section. DELETE: the entire "Hackathon context & strategy posture" block, the barbell, tournament rules, live-MT5 order-check workflow, and submission deliverables. Add a one-paragraph header stating QuantTrade's purpose (equity swing-trading research, cherry-picked from QuantHack).

- [ ] **Step 4: Write `README.md`** — short: what QuantTrade is, the `uv sync` / `uv run pytest` quickstart, and a pointer to the spec.

- [ ] **Step 5: Commit:**

```powershell
git add -A; git commit -m "chore: reset project machinery for QuantTrade (trim CLAUDE.md, empty registries)"
```

---

### Task 3: Rename the orchestrator to a generic name

**Files:**
- Rename: `scripts/run_xau_d1.py` → `scripts/run_backtest.py`
- Modify: `phase5/run_proposal.py` (the subprocess call that invokes the orchestrator)
- Modify: any `configs/*.yaml` or docs referencing the old name

**Interfaces:**
- Produces: `scripts/run_backtest.py` with identical CLI (`--config <path> [--dry-run]`). `phase5.run_proposal` subprocesses the new name.

- [ ] **Step 1: Find every reference to the old name:**

Run: `grep -rn "run_xau_d1" "D:\PROJECTS\QuantTrade" --include=*.py --include=*.yaml --include=*.md`
Expected: a list including `phase5/run_proposal.py`. Note each hit.

- [ ] **Step 2: Rename the file** (PowerShell tool):

```powershell
Rename-Item "D:\PROJECTS\QuantTrade\scripts\run_xau_d1.py" "run_backtest.py"
```

- [ ] **Step 3: Update each reference** found in Step 1 from `run_xau_d1` to `run_backtest` (the subprocess invocation in `phase5/run_proposal.py` is the load-bearing one).

- [ ] **Step 4: Verify no stale references remain:**

Run: `grep -rn "run_xau_d1" "D:\PROJECTS\QuantTrade"`
Expected: no output.

- [ ] **Step 5: Run the suite to confirm nothing broke:**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit:**

```powershell
git add -A; git commit -m "refactor: rename run_xau_d1 orchestrator to run_backtest"
```

---

### Task 4: Equity data layer (`EquityDataSource` seam + yfinance impl + fetch CLI)

**Files:**
- Create: `pipeline/equity_source.py`
- Create: `scripts/fetch_equity_daily.py`
- Test: `tests/test_equity_source.py`

**Interfaces:**
- Produces:
  - `normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame` — yfinance frame → DatetimeIndex(UTC) frame with float columns `open, high, low, close, volume`.
  - `write_contract_csv(df: pd.DataFrame, path: str | Path) -> None` — writes a CSV with a `time` column readable by `pipeline.data.load_dataset`.
  - `class YFinanceSource` with `fetch_daily(ticker, start=None, end=None) -> pd.DataFrame` (returns a normalized frame, `auto_adjust=True`).
- Consumes: `pipeline.data.load_dataset` (round-trip test).

- [ ] **Step 1: Write the failing test** at `tests/test_equity_source.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails:**

Run: `uv run pytest tests/test_equity_source.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.equity_source'`.

- [ ] **Step 3: Write `pipeline/equity_source.py`:**

```python
"""Equity daily-bar data layer for QuantTrade.

EquityDataSource is the swap seam: YFinanceSource is the free first impl;
a paid vendor (Norgate/Alpaca/Sharadar) drops in behind the same method.
Equity bars MUST be split/dividend-adjusted (auto_adjust=True) — unadjusted
prices corrupt every triple-barrier label.
"""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Protocol

import pandas as pd

_RENAME = {"Open": "open", "High": "high", "Low": "low",
           "Close": "close", "Volume": "volume"}


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    missing = [k for k in _RENAME if k not in df.columns]
    if missing:
        raise ValueError(f"yfinance frame missing columns: {missing}")
    out = (df.rename(columns=_RENAME)[["open", "high", "low", "close", "volume"]]
             .apply(pd.to_numeric, errors="coerce").astype("float64"))
    idx = pd.to_datetime(df.index)
    out.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out = out.dropna().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return out


def write_contract_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "time"
    fd, tmp = tempfile.mkstemp(suffix=".csv", dir=path.parent)
    os.close(fd)
    try:
        out.reset_index().to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class EquityDataSource(Protocol):
    def fetch_daily(self, ticker: str, start: str | None = None,
                    end: str | None = None) -> pd.DataFrame: ...


class YFinanceSource:
    def fetch_daily(self, ticker: str, start: str | None = None,
                    end: str | None = None) -> pd.DataFrame:
        import yfinance as yf
        kwargs = dict(interval="1d", auto_adjust=True, progress=False)
        if start:
            df = yf.download(ticker, start=start, end=end, **kwargs)
        else:
            df = yf.download(ticker, period="max", **kwargs)
        if df is None or df.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker} — retry later.")
        return normalize_ohlcv(df)
```

- [ ] **Step 4: Run the test to verify it passes:**

Run: `uv run pytest tests/test_equity_source.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Write the fetch CLI** at `scripts/fetch_equity_daily.py`:

```python
"""Fetch split/dividend-adjusted daily equity bars into the load_dataset contract.

USAGE:
  uv run python scripts/fetch_equity_daily.py --ticker NVDA
  uv run python scripts/fetch_equity_daily.py --ticker NVDA --out data/D1/NVDA_D1.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.equity_source import YFinanceSource, write_contract_csv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None,
                    help="default data/D1/<TICKER>_D1.csv")
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(f"data/D1/{args.ticker}_D1.csv")
    df = YFinanceSource().fetch_daily(args.ticker, args.start, args.end)
    if len(df) < 1000:
        raise RuntimeError(f"Suspiciously short history for {args.ticker} "
                           f"({len(df)} rows); refusing to write.")
    write_contract_csv(df, out)
    print(f"Wrote {out}: {len(df)} rows, "
          f"{df.index.min().date()} -> {df.index.max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Pull NVDA data (network)** (PowerShell tool):

Run: `uv run python scripts/fetch_equity_daily.py --ticker NVDA`
Expected: `Wrote data/D1/NVDA_D1.csv: <N> rows, 1999-01-22 -> <recent>` with N in the thousands.

- [ ] **Step 7: Confirm the regime resolver finds it:**

Run: `uv run python -c "from pipeline.regimes import _resolve_data_path; print(_resolve_data_path('NVDA','D1',None))"`
Expected: prints a path ending in `NVDA_D1.csv`. (If `FileNotFoundError`, the resolver's primary dir differs — move the CSV to the dir named in the error message, then re-run.)

- [ ] **Step 8: Commit:**

```powershell
git add pipeline/equity_source.py scripts/fetch_equity_daily.py tests/test_equity_source.py
git commit -m "feat: yfinance equity daily-bar data layer with swap seam"
```

---

### Task 5: Register the `equity` asset class and NVDA

**Files:**
- Modify: `phase5/asset_registry.py`
- Test: `tests/test_asset_registry_equity.py`

**Interfaces:**
- Consumes: `phase5.asset_registry.ASSET_REGISTRY`, `AssetSpec`, `dossier_feature_pack()`.
- Produces: `ASSET_REGISTRY["NVDA"]` with `asset_class="equity"`, `frequencies=("D1",)`, base pack `_EQUITY_PACK=("cs_spread_21",)`, and `dossier_feature_pack()` auto-including the 6 macro members.

- [ ] **Step 1: Guard-grep for asset_class equality checks** (so a new class value isn't silently rejected):

Run: `grep -rn "asset_class ==\|asset_class in\|equity_index" "D:\PROJECTS\QuantTrade\pipeline" "D:\PROJECTS\QuantTrade\phase5"`
Expected: the only equality is `== "fx"` (in `dossier_feature_pack`). If any check restricts asset_class to an allow-list, add `"equity"` to it as part of this task.

- [ ] **Step 2: Write the failing test** at `tests/test_asset_registry_equity.py`:

```python
from phase5.asset_registry import ASSET_REGISTRY


def test_nvda_registered_as_equity():
    spec = ASSET_REGISTRY["NVDA"]
    assert spec.asset_class == "equity"
    assert spec.frequencies == ("D1",)


def test_nvda_dossier_pack_inherits_macro():
    pack = ASSET_REGISTRY["NVDA"].dossier_feature_pack()
    # own base feature
    assert "cs_spread_21" in pack
    # macro members auto-appended for non-fx classes
    for m in ("vix_level", "vix_chg_5", "dxy_z252", "real_yield_5y_z252d"):
        assert m in pack
    # metals-only alt-data must NOT leak into equities
    assert "cot_net_noncomm_z52w" not in pack
    assert "gld_dvol_z42" not in pack
```

- [ ] **Step 3: Run the test to verify it fails:**

Run: `uv run pytest tests/test_asset_registry_equity.py -v`
Expected: FAIL with `KeyError: 'NVDA'`.

- [ ] **Step 4: Edit `phase5/asset_registry.py`.** Add the equity pack after `_CRYPTO_PACK` (line ~59) and the NVDA entry inside `ASSET_REGISTRY`:

```python
_EQUITY_PACK = ("cs_spread_21",)  # own-bar liquidity; macro pack auto-appended for non-fx
```
```python
    "NVDA": AssetSpec("NVDA", "equity", ("D1",), _EQUITY_PACK),
```
Also update the `AssetSpec.asset_class` docstring comment (line 30) to include `equity`:
`asset_class: str  # fx | metal | crypto | commodity | equity | equity_index`

- [ ] **Step 5: Run the test to verify it passes:**

Run: `uv run pytest tests/test_asset_registry_equity.py -v`
Expected: PASS.

- [ ] **Step 6: Commit:**

```powershell
git add phase5/asset_registry.py tests/test_asset_registry_equity.py
git commit -m "feat: register equity asset class + NVDA in asset registry"
```

---

### Task 6: Build NVDA regime parquet + dossier

**Files:**
- Create (generated): `data/regimes/NVDA_d1_regimes.parquet`
- Create (generated): `signals/regime_stats/NVDA_d1/{BULL_QUIET,BULL_STRESSED,BEAR_QUIET,BEAR_STRESSED}.json`

**Interfaces:**
- Consumes: `data/D1/NVDA_D1.csv` (Task 4), `ASSET_REGISTRY["NVDA"]` (Task 5).
- Produces: a regime parquet + 4 dossier JSONs the hypothesizer reads.

- [ ] **Step 1: Label NVDA regimes:**

Run: `uv run python -m pipeline.regimes --asset NVDA --frequency D1 --print-sanity`
Expected: writes `data/regimes/NVDA_d1_regimes.parquet`; sanity report prints a 4-state regime distribution over thousands of bars with no all-zero regime.

- [ ] **Step 2: Build the dossier:**

Run: `uv run python -m phase5.regime_stats --asset NVDA --frequency D1 --asset-class equity --out signals/regime_stats/`
Expected: writes 4 JSON files under `signals/regime_stats/NVDA_d1/`. (Macro/COT feature caches are silently skipped if absent — that is expected for the first run; `cs_spread_21` computes from NVDA's own high/low.)

- [ ] **Step 3: Validate the dossier schema** is well-formed and firewall-clean (no dates/years):

Run: `uv run python -c "import json,glob,re; [print(f, json.load(open(f))['regime_id'], json.load(open(f))['sample_sufficient']) for f in glob.glob('signals/regime_stats/NVDA_d1/*.json')]"`
Expected: 4 lines, each a valid `regime_id` and a boolean. At least one regime should be `sample_sufficient=True`.

Run: `grep -rEn "(19|20|21)[0-9]{2}" signals/regime_stats/NVDA_d1/`
Expected: no output (the firewall forbids year tokens in dossiers; a hit means a leak — stop and investigate).

- [ ] **Step 4: Commit the build recipe** (the generated artifacts are gitignored; commit nothing or only a note). If you want the build reproducible, add a one-line `scripts/build_nvda_dossier.ps1` wrapping Steps 1–2 and commit that:

```powershell
git add scripts/build_nvda_dossier.ps1 2>$null; git commit -m "chore: NVDA regime dossier build recipe" --allow-empty
```

---

### Task 7: Equity backtest config + hypothesizer persona reframe

**Files:**
- Create: `configs/equity_d1.yaml`
- Modify: `.claude/agents/phase5-hypothesizer.md` (examples only)
- Modify: `.claude/skills/phase5-regime-methodology/SKILL.md` (event-density floor note)

**Interfaces:**
- Produces: `configs/equity_d1.yaml` consumable by `scripts/run_backtest.py --config configs/equity_d1.yaml`.

- [ ] **Step 1: Create `configs/equity_d1.yaml`** by copying `configs/xau_d1.yaml` and changing only the data/asset-specific keys: point the dataset path at `data/D1/NVDA_D1.csv`, set the output folder (e.g. `results/clf_equity_d1/`), and keep the two rule-based primaries (`ema_cross`, `momentum_zscore`), triple-barrier geometry (ATR-scaled), walk-forward, calibration, and stacking blocks **unchanged** (they are asset-agnostic). Keep the comment explaining any value you do change.

- [ ] **Step 2: Verify the config parses and the dry-run extrapolates timing:**

Run: `uv run python scripts/run_backtest.py --config configs/equity_d1.yaml --dry-run`
Expected: completes 1 fold and prints a per-model wall-time extrapolation with no error. (If it errors on a missing `FRED_API_KEY` for macro features, set it in `.env` — `copy .env.example .env` then add the key — and re-run.)

- [ ] **Step 3: Reframe the hypothesizer persona examples.** In `.claude/agents/phase5-hypothesizer.md`, replace precious-metals/dollar illustrative mechanisms with equity mechanisms (e.g. post-earnings-announcement drift, factor rotation, volatility-regime mean reversion, sector co-movement). **Do NOT touch** the lookahead-firewall constraints, the proposal-schema requirements, the falsification-criterion rules, or the barrier-geometry archetypes — only the domain-flavored example prose changes. Verify the firewall sections are intact:

Run: `grep -in "firewall\|falsif\|barrier geometry\|quantile" ".claude/agents/phase5-hypothesizer.md"`
Expected: those sections still present.

- [ ] **Step 4: Document the equity event-density floor.** In `.claude/skills/phase5-regime-methodology/SKILL.md`, add equity D1 to the event-density floor table, starting at the FX-D1 parity value (399 events) with a note to tighten from each dossier's measured baseline `n_events`.

- [ ] **Step 5: Commit:**

```powershell
git add configs/equity_d1.yaml ".claude/agents/phase5-hypothesizer.md" ".claude/skills/phase5-regime-methodology/SKILL.md"
git commit -m "feat: equity backtest config + reframe hypothesizer for equities"
```

---

### Task 8: NVDA end-to-end smoke test (M2 deliverable)

**Files:**
- Create (generated): `results/clf_equity_d1/<primary>/report.md` + `summary.json`

**Interfaces:**
- Consumes: `configs/equity_d1.yaml`, `data/D1/NVDA_D1.csv`.
- Produces: a completed meta-labeling backtest report on NVDA — the proof the engine runs end-to-end on equities.

- [ ] **Step 1: Run the full backtest on NVDA:**

Run: `uv run python scripts/run_backtest.py --config configs/equity_d1.yaml`
Expected: completes (minutes of CPU); writes `results/clf_equity_d1/ema_cross/` and `.../momentum_zscore/` each containing `report.md`, `summary.json`, `oof_predictions.parquet`, `metrics_per_fold.json`.

- [ ] **Step 2: Read the report honestly.** Open `results/clf_equity_d1/ema_cross/summary.json` and confirm the run produced metrics (per-fold Sharpe may be `NaN` where `n_trades<30` — that is correct behavior, not a bug). **Record the readout as plumbing-validation-only** — per spec §7, a single name is not an edge claim.

Run: `uv run python -c "import json; s=json.load(open('results/clf_equity_d1/ema_cross/summary.json')); print({k:s[k] for k in list(s)[:8]})"`
Expected: prints summary keys/values without error.

- [ ] **Step 3: Log the M2 outcome to the backlog** as the first QuantTrade B-ID (the durable record):

Run: `uv run python -c "from backlog import db; db.add_entry(title='M2 NVDA plumbing smoke test', status='done', body='Meta-labeling engine runs end-to-end on NVDA daily bars via configs/equity_d1.yaml. Plumbing validation only — NOT an edge claim (single-name overfit). Next: M3 cross-section.')"`
Expected: prints a new B-ID (B0001).

- [ ] **Step 4: Commit:**

```powershell
git add backlog/ docs/ 2>$null; git commit -m "test: NVDA end-to-end plumbing smoke passes (M2); log B0001" --allow-empty
```

---

## Self-Review

**Spec coverage:**
- §3 cherry-pick manifest → Task 1 (copy core, delete 12 hackathon modules, extract state_store).
- §3.5 leave-behind → Task 1 Steps 2–3 + leak grep Step 7.
- §4 equity data layer (auto_adjust, seam) → Task 4.
- §5 re-aim (asset class, dossier, persona, event floor, config) → Tasks 5, 6, 7.
- §6 trimmed CLAUDE.md → Task 2 Step 3.
- §7 validation posture (NVDA = plumbing only) → Task 8 Step 2 + backlog note Step 3.
- §8 seeding procedure (copy, rename, reset registries, git init, green-test gate) → Tasks 1, 2, 3.
- §9 milestones M0/M1/M2 → Tasks 1–3 / 4 / 5–8. M3 explicitly deferred to a later plan. ✓
- §10 OQ1/OQ2 → out of scope here (M3); OQ on FRED key surfaced in Task 7 Step 2.

**Placeholder scan:** No TBD/TODO; every code step has complete code; copy/command steps have exact commands + expected output. ✓

**Type consistency:** `normalize_ohlcv`/`write_contract_csv`/`YFinanceSource.fetch_daily` signatures match between Task 4's interface block, implementation, and test. `ASSET_REGISTRY["NVDA"]`/`dossier_feature_pack()` consistent across Task 5. Orchestrator name `run_backtest.py` consistent after Task 3 and used in Tasks 7–8. ✓
