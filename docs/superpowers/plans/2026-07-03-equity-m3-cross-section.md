# Equity M3 Cross-Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the meta-labeling pipeline as a POOLED cross-sectional backtest over a frozen ~35-name t0-selected large-cap universe plus a 9-ETF survivorship-clean control, producing the first honest edge/no-edge readout on equities (B0003, spec Milestone 1).

**Architecture:** A new thin orchestrator `scripts/run_pooled_equity_d1.py` builds per-(ticker, primary) member inputs by mirroring `scripts/run_backtest.py::_run_one_primary` steps exactly (D1 tier-2 features, rule-based primaries, triple-barrier, avg-uniqueness), then delegates Phase B/C/D (pooled concat, time-purged pooled folds, per-model training, per-asset OOS write) to the battle-tested `scripts/run_multi_h4.py::_run_one_pool` — which needs only two additive generalizations (equity bar-duration for the embargo, lgbm/lr hyperparameter spaces). The universe lives in one frozen YAML consumed by the fetch script, the asset registry, and the runner.

**Tech Stack:** Python via `uv run` only. pandas/numpy/sklearn/xgboost/lightgbm, yfinance (via existing `pipeline/equity_source.py`), FRED macro cache, pytest.

## Global Constraints

Every task implicitly includes these (from CLAUDE.md — pipeline invariants):

- All Python runs through `uv run` (pytest lives only in `.venv`).
- Sharpe annualization uses `sqrt(trades_per_year)`, never `sqrt(252)`.
- Sharpe is `NaN` (not `0`) when `n_trades < 30` or `std == 0`; aggregate with `np.nanmedian`/`np.nanmean`.
- FRED macro is `.shift(1)` before alignment (already inside `build_macro_frame` — do not touch).
- Inner CV must be purged (`PurgedTimeSeriesSplit` / `PurgedTimeGroupSplit`), never sklearn `TimeSeriesSplit`.
- Calibration: `FrozenEstimator` on a chronological tail holdout, method `sigmoid`. Never KFold calibration.
- Sample weights (`avg_uniqueness` / `pooled_avg_uniqueness`) flow to base fit + hyperparam search + calibration.
- Primaries are deterministic rule-based functions — no `.fit()` on the meta's features.
- Headline threshold is fixed at 0.55; the grid is diagnostic only.
- The existing H4 pooled path must stay byte-identical: every change to `scripts/run_multi_h4.py` in this plan is ADDITIVE (new dict keys only). `uv run pytest tests/test_run_multi_h4_aggregation.py tests/test_run_multi_h4_planning.py tests/test_pooled_walk_forward.py -v` must pass unchanged after Task 3.
- Committing after each task is allowed without asking; pushing is NOT.
- OQ1 decision (recorded in `backlog/proposed/B0003.json` history): universe = t0-(~2006)-selected large-caps + SPDR sector ETF control. Do not swap tickers ad hoc — any universe change is a recorded trial (B0003 caveat 2).

## File Structure

- `configs/universe_equity_m3.yaml` — THE frozen universe artifact (selection rule, 35 stocks, 9 ETFs, alternates, enumerated delistee exclusions). Single source of truth.
- `pipeline/equity_universe.py` — `load_universe(path)` loader + validation.
- `scripts/fetch_equity_daily.py` — gains `--universe` batch mode (modify).
- `phase5/asset_registry.py` — registers universe tickers from the YAML (modify, additive).
- `scripts/run_multi_h4.py` — `_BAR_DURATION_BY_CLASS` + equity entries; `HP_SPACES` + lgbm/lr (modify, additive).
- `scripts/run_pooled_equity_d1.py` — new orchestrator: Phase-A member builder + pool dispatch.
- `configs/equity_m3_d1.yaml` / `configs/equity_m3_etf_d1.yaml` — stocks run + ETF control run.
- `scripts/report_long_short_split.py` — pooled long/short split reporter (B0003 caveat 1).
- Tests: `tests/test_equity_universe.py`, `tests/test_fetch_equity_universe.py`, `tests/phase5/test_equity_universe_registry.py`, `tests/test_pooled_equity_constants.py`, `tests/test_run_pooled_equity_d1.py`, `tests/test_long_short_split.py`.

---

### Task 1: Frozen universe file + loader

**Files:**
- Create: `configs/universe_equity_m3.yaml`
- Create: `pipeline/equity_universe.py`
- Test: `tests/test_equity_universe.py`

**Interfaces:**
- Produces: `load_universe(path: str | Path) -> dict` with keys `selection_rule: str`, `selected_at: str`, `stocks: list[str]`, `etfs: list[str]`, `alternates: list[str]`, `excluded_delistees: dict[str, str]`. Raises `ValueError` on any validation failure. Tasks 2, 4, 5 consume this.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_equity_universe.py
"""Universe file contract: frozen M3 universe loads and validates (B0003)."""
import pytest

from pipeline.equity_universe import load_universe


def test_frozen_m3_universe_loads_and_is_well_formed():
    u = load_universe("configs/universe_equity_m3.yaml")
    assert len(u["stocks"]) == 35
    assert len(u["etfs"]) == 9
    assert len(set(u["stocks"])) == 35, "duplicate stock tickers"
    assert not set(u["stocks"]) & set(u["etfs"]), "stocks/etfs overlap"
    assert "NVDA" not in u["stocks"], (
        "NVDA must NOT be in the M3 universe — it fails the t0-2006 top-cap rule; "
        "keeping it out is the anti-overfit property of the selection"
    )
    assert u["selection_rule"].strip(), "selection rule must be recorded"
    # Distress delistees must be enumerated (B0003 caveat 1: bounded residual bias).
    for t in ("LEH", "WB", "MER", "FNM"):
        assert t in u["excluded_delistees"]


def test_load_universe_rejects_malformed(tmp_path):
    p = tmp_path / "u.yaml"
    p.write_text("stocks: [AAA, AAA]\netfs: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_universe(p)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_equity_universe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.equity_universe'`

- [ ] **Step 3: Write the universe file**

```yaml
# configs/universe_equity_m3.yaml
# THE frozen M3 cross-section universe (B0003; OQ1 decision 2026-07-03).
# Selection is t0 (~2006) information ONLY — largest/most-liquid US large-caps
# as of early 2006 that still have continuous yfinance history today. This is
# NOT current index membership (that would be the textbook survivorship sin,
# AFML ch.11). Residual bias: 2006 mega-caps that later delisted cannot be
# included; they are ENUMERATED below so the long-side inflation is bounded
# and auditable. Do not edit lists ad hoc — any change is a recorded trial
# (B0003 history, DSR trial count).
selection_rule: >-
  Top 35 by early-2006 market capitalization among US common stocks with
  continuous yfinance daily history from 2006-01-03 or earlier, frozen
  2026-07-03 before any M3 result was observed. Substitution on fetch/sanity
  failure ONLY, in the deterministic order of `alternates`, recorded in
  B0003 history.
selected_at: "2026-07-03"

stocks:
  - XOM
  - GE
  - MSFT
  - C      # 10:1 reverse split 2011 — adjusted history is continuous
  - BAC
  - WMT
  - PG
  - JNJ
  - PFE
  - AIG    # 2008 near-failure survivor kept IN — partially offsets long-side bias
  - MO
  - IBM
  - CVX
  - JPM
  - INTC
  - WFC
  - CSCO
  - KO
  - VZ
  - PEP
  - HD
  - COP
  - T      # SBC->AT&T rename 2005; yfinance history is continuous under T
  - ABT
  - MRK
  - ORCL
  - QCOM
  - MMM
  - UPS
  - DIS
  - MCD
  - GS
  - BA
  - UNH
  - CAT

# Survivorship-clean parallel control group (B0003): the 9 SPDR sector ETFs
# with full history since Dec-1998. XLRE (2015) and XLC (2018) excluded —
# insufficient history for the 2006-2026 window.
etfs:
  - XLB
  - XLE
  - XLF
  - XLI
  - XLK
  - XLP
  - XLU
  - XLV
  - XLY

# Deterministic substitution order if a stock fails the fetch/sanity gate
# (next-largest 2006 caps, still listed).
alternates:
  - AXP
  - SLB
  - TXN
  - LLY
  - AMGN
  - BMY
  - TGT
  - USB

# 2006-era top-cap names that CANNOT be included (delisted / ticker
# discontinuity). Distress exits (the ones whose absence inflates the long
# side): LEH, WB, MER, WM, FNM, FRE, GM. The rest are mostly M&A/neutral.
excluded_delistees:
  LEH: "Lehman Brothers — bankruptcy 2008 (distress)"
  WB: "Wachovia — distressed sale to Wells Fargo 2008 (distress)"
  MER: "Merrill Lynch — distressed sale to Bank of America 2009 (distress)"
  WM: "Washington Mutual — bank failure 2008 (distress)"
  FNM: "Fannie Mae — conservatorship, delisted 2010 (distress)"
  FRE: "Freddie Mac — conservatorship, delisted 2010 (distress)"
  GM: "General Motors — Chapter 11 2009; relisted GM is a different entity (distress)"
  BUD: "Anheuser-Busch — acquired by InBev 2008 (M&A)"
  WYE: "Wyeth — acquired by Pfizer 2009 (M&A)"
  SGP: "Schering-Plough — merged into Merck 2009 (M&A)"
  BLS: "BellSouth — merged into AT&T 2006 (M&A)"
  TWX: "Time Warner — acquired by AT&T 2018 (M&A)"
  DELL: "Dell — LBO 2013; relisted DELL is a different capital structure (going-private)"
  MOT: "Motorola — 2011 split into MSI/MMI (restructuring)"
  UTX: "United Technologies — merged into RTX 2020 (ticker discontinuity)"
  DOW: "Dow Chemical — DowDuPont restructuring 2017-19 (ticker discontinuity)"
  TYC: "Tyco — merged into Johnson Controls 2016 (M&A)"
  S: "Sprint — merged into T-Mobile 2020 (M&A)"
```

- [ ] **Step 4: Write the loader**

```python
# pipeline/equity_universe.py
"""Loader/validator for the frozen M3 equity universe (B0003).

The YAML is the single frozen artifact of the OQ1 universe decision; the
fetch script, asset registry, and pooled runner all consume it through
`load_universe` so no ticker list is ever duplicated in code.
"""
from __future__ import annotations

from pathlib import Path

import yaml

_REQUIRED_KEYS = ("selection_rule", "selected_at", "stocks", "etfs",
                  "alternates", "excluded_delistees")


def load_universe(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"universe file {path}: top level must be a mapping")
    missing = [k for k in _REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"universe file {path}: missing keys {missing}")
    for key in ("stocks", "etfs", "alternates"):
        vals = payload[key]
        if not isinstance(vals, list) or not all(isinstance(t, str) and t for t in vals):
            raise ValueError(f"universe file {path}: {key} must be a list of non-empty strings")
        if len(vals) != len(set(vals)):
            raise ValueError(f"universe file {path}: duplicate tickers in {key}")
    overlap = set(payload["stocks"]) & set(payload["etfs"])
    if overlap:
        raise ValueError(f"universe file {path}: stocks/etfs overlap: {sorted(overlap)}")
    if not isinstance(payload["excluded_delistees"], dict):
        raise ValueError(f"universe file {path}: excluded_delistees must be a mapping")
    if not str(payload["selection_rule"]).strip():
        raise ValueError(f"universe file {path}: selection_rule must be non-empty")
    return payload
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_equity_universe.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```powershell
git add configs/universe_equity_m3.yaml pipeline/equity_universe.py tests/test_equity_universe.py
git commit -m "feat(m3): frozen t0-2006 equity universe + loader (B0003)"
```

---

### Task 2: Batch fetch mode for the universe

**Files:**
- Modify: `scripts/fetch_equity_daily.py`
- Test: `tests/test_fetch_equity_universe.py`

**Interfaces:**
- Consumes: `pipeline.equity_universe.load_universe`, existing `pipeline.equity_source.YFinanceSource.fetch_daily(ticker, start, end) -> pd.DataFrame` and `write_contract_csv(df, path)`.
- Produces: CLI `uv run python scripts/fetch_equity_daily.py --universe configs/universe_equity_m3.yaml` → writes `data/D1/<TICKER>_D1.csv` for every stock+ETF; exit code 1 with a `FAILED:` summary if any ticker fails its gate (≥1000 rows AND first bar ≤ 2006-01-03 in universe mode). No automatic substitution — a failure is reported for a human to apply the `alternates` rule and record it in B0003.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fetch_equity_universe.py
"""--universe batch mode: fetches every stock+ETF, gates short/late histories."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import fetch_equity_daily  # noqa: E402


def _fake_df(start: str, n: int) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    base = np.linspace(100.0, 110.0, n)
    return pd.DataFrame({
        "open": base, "high": base + 1.0, "low": base - 1.0,
        "close": base + 0.5, "volume": np.full(n, 1e6),
    }, index=idx)


def _write_universe(tmp_path: Path) -> Path:
    p = tmp_path / "u.yaml"
    p.write_text(
        "selection_rule: test\nselected_at: '2026-07-03'\n"
        "stocks: [AAAA, BBBB]\netfs: [XLTT]\nalternates: []\n"
        "excluded_delistees: {}\n",
        encoding="utf-8",
    )
    return p


def test_universe_mode_writes_all_tickers(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetch_equity_daily.YFinanceSource, "fetch_daily",
        lambda self, t, s, e: _fake_df("2000-01-03", 1500),
    )
    written = []
    monkeypatch.setattr(
        fetch_equity_daily, "write_contract_csv",
        lambda df, out: written.append(Path(out).name),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["fetch_equity_daily.py", "--universe", str(_write_universe(tmp_path))],
    )
    assert fetch_equity_daily.main() == 0
    assert sorted(written) == ["AAAA_D1.csv", "BBBB_D1.csv", "XLTT_D1.csv"]


def test_universe_mode_gates_late_history_and_exits_nonzero(tmp_path, monkeypatch):
    # BBBB starts 2010 → fails the ≤2006-01-03 gate; others succeed.
    def fake_fetch(self, t, s, e):
        return _fake_df("2010-01-04" if t == "BBBB" else "2000-01-03", 1500)

    monkeypatch.setattr(fetch_equity_daily.YFinanceSource, "fetch_daily", fake_fetch)
    written = []
    monkeypatch.setattr(
        fetch_equity_daily, "write_contract_csv",
        lambda df, out: written.append(Path(out).name),
    )
    monkeypatch.setattr(
        sys, "argv",
        ["fetch_equity_daily.py", "--universe", str(_write_universe(tmp_path))],
    )
    assert fetch_equity_daily.main() == 1
    assert "BBBB_D1.csv" not in written
    assert len(written) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_fetch_equity_universe.py -v`
Expected: FAIL — `main()` errors on unknown `--universe` argument (argparse `SystemExit: 2`).

- [ ] **Step 3: Implement the batch mode**

Replace the whole `main()` in `scripts/fetch_equity_daily.py` (keep the module docstring and imports; add `from datetime import date` and the loader import):

```python
"""Fetch split/dividend-adjusted daily equity bars into the load_dataset contract.

USAGE:
  uv run python scripts/fetch_equity_daily.py --ticker NVDA
  uv run python scripts/fetch_equity_daily.py --ticker NVDA --out data/D1/NVDA_D1.csv
  uv run python scripts/fetch_equity_daily.py --universe configs/universe_equity_m3.yaml
"""
from __future__ import annotations
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.equity_source import YFinanceSource, write_contract_csv
from pipeline.equity_universe import load_universe

# Universe mode gate: every member must cover the M3 backtest window from its
# start, or the pooled panel is unbalanced across folds (B0003).
_UNIVERSE_HISTORY_FLOOR = date(2006, 1, 3)


def _fetch_one(src: YFinanceSource, ticker: str, start, end, out: Path,
               enforce_floor: bool) -> None:
    df = src.fetch_daily(ticker, start, end)
    if len(df) < 1000:
        raise RuntimeError(f"suspiciously short history ({len(df)} rows)")
    first = df.index.min().date()
    if enforce_floor and first > _UNIVERSE_HISTORY_FLOOR:
        raise RuntimeError(f"history starts {first} > {_UNIVERSE_HISTORY_FLOOR}")
    write_contract_csv(df, out)
    print(f"Wrote {out}: {len(df)} rows, "
          f"{df.index.min().date()} -> {df.index.max().date()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--universe", default=None,
                    help="universe yaml; fetches every stock+etf into data/D1/")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None,
                    help="single-ticker mode only; default data/D1/<TICKER>_D1.csv")
    args = ap.parse_args()
    if bool(args.ticker) == bool(args.universe):
        ap.error("exactly one of --ticker / --universe is required")

    src = YFinanceSource()
    if args.ticker:
        out = Path(args.out) if args.out else Path(f"data/D1/{args.ticker}_D1.csv")
        _fetch_one(src, args.ticker, args.start, args.end, out, enforce_floor=False)
        return 0

    u = load_universe(args.universe)
    failed: list[tuple[str, str]] = []
    for ticker in list(u["stocks"]) + list(u["etfs"]):
        try:
            _fetch_one(src, ticker, args.start, args.end,
                       Path(f"data/D1/{ticker}_D1.csv"), enforce_floor=True)
        except Exception as exc:  # noqa: BLE001 — keep batch going, report at end
            failed.append((ticker, f"{type(exc).__name__}: {exc}"))
    if failed:
        print("FAILED: " + "; ".join(f"{t} ({r})" for t, r in failed))
        print("Apply the alternates rule from the universe file manually and "
              "record the substitution in B0003 history.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fetch_equity_universe.py tests/test_equity_universe.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```powershell
git add scripts/fetch_equity_daily.py tests/test_fetch_equity_universe.py
git commit -m "feat(m3): --universe batch fetch with history-floor gate (B0003)"
```

---

### Task 3: Register the universe in the asset registry + equity pooled constants

**Files:**
- Modify: `phase5/asset_registry.py` (append after the `ASSET_REGISTRY` dict, ~line 72)
- Modify: `scripts/run_multi_h4.py` (two additive dict edits: `_BAR_DURATION_BY_CLASS` ~line 1365, `HP_SPACES` ~line 125)
- Test: `tests/phase5/test_equity_universe_registry.py`, `tests/test_pooled_equity_constants.py`

**Interfaces:**
- Consumes: `pipeline.equity_universe.load_universe`, existing `AssetSpec`, `_EQUITY_PACK`.
- Produces: every universe ticker present in `ASSET_REGISTRY` with `asset_class` `"equity"` (stocks) / `"equity_index"` (ETFs), `frequencies=("D1",)` — this is what lets `scripts/build_all_regimes.py --assets ...` build Loop-A dossiers later. Also: `run_multi_h4._BAR_DURATION_BY_CLASS["equity"]` and `["equity_index"]` = `pd.Timedelta(days=365.25 / 252)`; `run_multi_h4.HP_SPACES["lgbm"]` and `["lr"]` identical to `scripts/run_backtest.py` lines 84-91.

- [ ] **Step 1: Write the failing tests**

```python
# tests/phase5/test_equity_universe_registry.py
"""M3 universe tickers register as D1 equity/equity_index specs (B0003)."""
from phase5.asset_registry import ASSET_REGISTRY
from pipeline.equity_universe import load_universe


def test_universe_tickers_registered_with_correct_class():
    u = load_universe("configs/universe_equity_m3.yaml")
    for t in u["stocks"]:
        spec = ASSET_REGISTRY[t]
        assert spec.asset_class == "equity"
        assert spec.frequencies == ("D1",)
        assert "cs_spread_21" in spec.feature_pack
    for t in u["etfs"]:
        spec = ASSET_REGISTRY[t]
        assert spec.asset_class == "equity_index"
        assert spec.frequencies == ("D1",)


def test_legacy_entries_untouched():
    assert ASSET_REGISTRY["NVDA"].asset_class == "equity"
    assert ASSET_REGISTRY["XAUUSD"].asset_class == "metal"
```

```python
# tests/test_pooled_equity_constants.py
"""Equity D1 additions to the pooled machinery (B0003): embargo + HP spaces."""
import pandas as pd

from scripts.run_multi_h4 import (
    HP_SPACES,
    _BAR_DURATION_BY_CLASS,
    _pooled_embargo_td,
)
from pipeline.train import MODEL_FACTORIES


def test_equity_bar_duration_registered():
    # D1 equity: 252 bars/year → ~1.45 calendar days per bar.
    assert _BAR_DURATION_BY_CLASS["equity"] == pd.Timedelta(days=365.25 / 252)
    assert _BAR_DURATION_BY_CLASS["equity_index"] == pd.Timedelta(days=365.25 / 252)


def test_pooled_embargo_covers_equity_horizon():
    members = [{"asset_class": "equity"}, {"asset_class": "equity_index"}]
    cfg = {"triple_barrier": {"horizon": 40}}
    td = _pooled_embargo_td(members, cfg)
    # 40 bars × 365.25/252 days ≈ 58 calendar days — NOT the fx fallback (~9.4d).
    assert pd.Timedelta(days=57) < td < pd.Timedelta(days=59)


def test_lgbm_and_lr_hp_spaces_present_and_buildable():
    for name in ("lgbm", "lr"):
        assert name in HP_SPACES
        assert name in MODEL_FACTORIES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/phase5/test_equity_universe_registry.py tests/test_pooled_equity_constants.py -v`
Expected: registry test FAILS with `KeyError: 'XOM'`; constants tests FAIL with `KeyError: 'equity'` / `assert 'lgbm' in HP_SPACES`.

- [ ] **Step 3: Implement — registry**

Append to `phase5/asset_registry.py` immediately after the `ASSET_REGISTRY` literal (keep NVDA's explicit entry — it is deliberately NOT in the M3 universe):

```python
_EQUITY_INDEX_PACK = ("cs_spread_21",)  # sector ETFs: own-bar liquidity + auto macro pack

_M3_UNIVERSE_PATH = Path("configs/universe_equity_m3.yaml")


def _load_m3_universe_specs() -> dict[str, AssetSpec]:
    """M3 cross-section universe (B0003): registry entries generated from the
    frozen universe yaml so the ticker list is never duplicated in code.
    Missing file → empty (keeps the registry importable in stripped checkouts).
    """
    if not _M3_UNIVERSE_PATH.exists():
        return {}
    import yaml
    payload = yaml.safe_load(_M3_UNIVERSE_PATH.read_text(encoding="utf-8"))
    specs: dict[str, AssetSpec] = {}
    for t in payload.get("stocks", []):
        specs[t] = AssetSpec(t, "equity", ("D1",), _EQUITY_PACK)
    for t in payload.get("etfs", []):
        specs[t] = AssetSpec(t, "equity_index", ("D1",), _EQUITY_INDEX_PACK)
    return specs


ASSET_REGISTRY.update(_load_m3_universe_specs())
```

- [ ] **Step 4: Implement — pooled constants**

In `scripts/run_multi_h4.py`, extend `_BAR_DURATION_BY_CLASS` (~line 1365):

```python
_BAR_DURATION_BY_CLASS = {
    "fx": pd.Timedelta(days=365.25 / 1560),
    "metal": pd.Timedelta(days=365.25 / 1560),
    "crypto": pd.Timedelta(days=365.25 / 2190),
    # M3 (B0003): D1 equity bars — 252 trading days/year. Sizing the pooled
    # embargo off the fx H4 fallback would under-embargo D1 by ~6x.
    "equity": pd.Timedelta(days=365.25 / 252),
    "equity_index": pd.Timedelta(days=365.25 / 252),
}
```

And extend `HP_SPACES` (~line 125) with the two entries copied VERBATIM from `scripts/run_backtest.py` lines 84-91:

```python
    # M3 (B0003): equity D1 pooled runs use the B0053 model set (lgbm/lr) —
    # spaces copied verbatim from scripts/run_backtest.py so pooled and
    # per-asset paths search identical grids.
    "lgbm": {"num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.05, 0.1],
             "n_estimators": [200, 400], "min_child_samples": [10, 20, 40],
             "reg_lambda": [0.5, 1.0, 5.0]},
    "lr": {"C": [0.01, 0.1, 1.0], "max_iter": [500, 1000]},
```

- [ ] **Step 5: Run tests — new AND the H4-parity guard**

Run: `uv run pytest tests/phase5/test_equity_universe_registry.py tests/test_pooled_equity_constants.py tests/test_run_multi_h4_aggregation.py tests/test_run_multi_h4_planning.py tests/test_pooled_walk_forward.py -v`
Expected: all pass (H4 tests unchanged — the edits are additive dict keys only).

- [ ] **Step 6: Commit**

```powershell
git add phase5/asset_registry.py scripts/run_multi_h4.py tests/phase5/test_equity_universe_registry.py tests/test_pooled_equity_constants.py
git commit -m "feat(m3): register universe in asset registry; equity embargo + lgbm/lr pooled HP spaces (B0003)"
```

---

### Task 4: Pooled equity D1 runner + configs

**Files:**
- Create: `scripts/run_pooled_equity_d1.py`
- Create: `configs/equity_m3_d1.yaml`, `configs/equity_m3_etf_d1.yaml`
- Test: `tests/test_run_pooled_equity_d1.py`

**Interfaces:**
- Consumes: `scripts.run_backtest._select_primary(name, ohlcv, features, cfg) -> pd.Series`; `scripts.run_multi_h4._run_one_pool(primary_name, pool_key, members, cfg, schema, weight_balance, pooled_uniqueness, train_min_frac, out_root, dry_run)`; `pipeline.labels.triple_barrier_labels`, `pipeline.labels.compute_primary_state`; `pipeline.sample_weights.avg_uniqueness`; `pipeline.data.load_dataset`; `pipeline.macro_fetch.build_macro_frame`; `pipeline.features.build_tier2_features`; `pipeline.equity_universe.load_universe`.
- Produces: `build_member_inputs(asset: str, primary_name: str, ohlcv: pd.DataFrame, features: pd.DataFrame, cfg: dict) -> dict | None` returning the member dict `_run_one_pool` consumes (keys: `asset, primary_name, asset_class, bars_per_year, cost_bps, X, y, w, side, fwd_ret, event_time, label_end_time, pool_key`). CLI: `uv run python scripts/run_pooled_equity_d1.py --config configs/equity_m3_d1.yaml [--dry-run] [--count-events-only] [--assets XOM,GE]`. Per-(asset, primary) it also writes `<output_dir>/<asset>/<primary>/events_side_fwd.parquet` (columns `side`, `fwd_ret`, index = event timestamps) — Task 5's reporter joins these against the pooled OOF parquets.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_run_pooled_equity_d1.py
"""Phase-A member builder mirrors run_backtest._run_one_primary alignment
invariants; runner CLI plumbing (count-events mode)."""
import numpy as np
import pandas as pd
import pytest

from scripts.run_pooled_equity_d1 import build_member_inputs


@pytest.fixture
def cfg():
    return {
        "asset_class": "equity",
        "triple_barrier": {"horizon": 40, "tp_atr_mult": 3.0,
                           "sl_atr_mult": 1.0, "atr_period": 14},
        "primary": {
            "candidates": ["ema_cross", "momentum_zscore"],
            "ema_cross": {"fast": 20, "slow": 50, "dead_zone_atr": 0.25},
            "momentum_zscore": {"lookback": 20, "threshold": 0.3},
        },
        "metrics": {"cost_per_trade_bps": 10, "bars_per_year": 252},
    }


def _features_for(ohlcv: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=ohlcv.index)
    tr = (ohlcv["high"] - ohlcv["low"]).rolling(14).mean()
    feats["_atr_14"] = tr
    r = np.log(ohlcv["close"]).diff()
    feats["z_r20"] = (r - r.rolling(20).mean()) / r.rolling(20).std()
    feats["f_mom"] = r.rolling(5).sum()
    return feats.dropna()


def test_member_alignment_invariants(synth_ohlcv, cfg):
    features = _features_for(synth_ohlcv)
    ohlcv = synth_ohlcv.loc[features.index]
    m = build_member_inputs("TEST", "ema_cross", ohlcv, features, cfg)
    assert m is not None, "synthetic series should produce ema_cross events"
    n = len(m["X"])
    assert n == len(m["y"]) == len(m["w"]) == len(m["fwd_ret"]) \
        == len(m["event_time"]) == len(m["label_end_time"])
    assert not m["X"].isnull().any().any()
    for col in ("primary_side", "primary_strength", "bars_since_signal"):
        assert col in m["X"].columns
    assert "_atr_14" not in m["X"].columns
    assert not m["fwd_ret"].isnull().any()
    assert (m["label_end_time"] >= m["event_time"]).all()
    assert m["asset_class"] == "equity"
    assert m["bars_per_year"] == 252
    assert m["cost_bps"] == 10
    assert set(np.unique(m["y"])) <= {0, 1}
    assert (m["w"] > 0).all() and (m["w"] <= 1).all()


def test_member_returns_none_when_no_signals(synth_ohlcv, cfg):
    features = _features_for(synth_ohlcv)
    ohlcv = synth_ohlcv.loc[features.index]
    cfg["primary"]["momentum_zscore"]["threshold"] = 99.0  # unreachable
    assert build_member_inputs("TEST", "momentum_zscore", ohlcv, features, cfg) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_run_pooled_equity_d1.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.run_pooled_equity_d1'`

- [ ] **Step 3: Write the runner**

```python
# scripts/run_pooled_equity_d1.py
"""M3 pooled cross-sectional equity D1 orchestrator (B0003).

Phase A (here): per-(ticker, primary) member inputs, mirroring
scripts/run_backtest.py::_run_one_primary steps 1:1 so single-name and pooled
paths build identical X/y/w/fwd_ret. Phases B/C/D (pooled concat, time-purged
folds, training, per-asset OOS reports) are delegated to the battle-tested
scripts/run_multi_h4.py::_run_one_pool.

USAGE:
  uv run python scripts/run_pooled_equity_d1.py --config configs/equity_m3_d1.yaml --count-events-only
  uv run python scripts/run_pooled_equity_d1.py --config configs/equity_m3_d1.yaml --dry-run
  uv run python scripts/run_pooled_equity_d1.py --config configs/equity_m3_d1.yaml
"""
from __future__ import annotations
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.data import load_dataset
from pipeline.equity_universe import load_universe
from pipeline.features import build_tier2_features
from pipeline.labels import compute_primary_state, triple_barrier_labels
from pipeline.macro_fetch import build_macro_frame
from pipeline.sample_weights import avg_uniqueness
from scripts.run_backtest import _select_primary
from scripts.run_multi_h4 import _run_one_pool


def build_member_inputs(
    asset: str,
    primary_name: str,
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    cfg: dict,
) -> dict | None:
    """Phase A for one (ticker, primary) — mirrors _run_one_primary exactly:
    signal -> triple barrier -> avg_uniqueness -> X(+primary state) -> dropna
    -> aligned y/w/fwd_ret/event & label-end timestamps."""
    cost_bps = float(cfg["metrics"]["cost_per_trade_bps"])

    sig = _select_primary(primary_name, ohlcv, features, cfg)
    events = pd.DataFrame({"side": sig[sig != 0].astype(int)})
    if events.empty:
        print(f"  {asset}/{primary_name}: 0 primary signals — skipping.")
        return None

    atr = features["_atr_14"]
    labels = triple_barrier_labels(
        ohlcv, events, atr,
        horizon=cfg["triple_barrier"]["horizon"],
        tp_mult=cfg["triple_barrier"]["tp_atr_mult"],
        sl_mult=cfg["triple_barrier"]["sl_atr_mult"],
    )
    valid = labels[labels["t_end_idx"] < len(ohlcv)]
    if valid.empty:
        print(f"  {asset}/{primary_name}: 0 events survive triple-barrier — skipping.")
        return None

    idx_pos = {ts: i for i, ts in enumerate(ohlcv.index)}
    t_starts_all = np.array([idx_pos[ts] for ts in valid.index])
    t_ends_all = valid["t_end_idx"].values
    w_all = avg_uniqueness(t_starts_all, t_ends_all, n_bars=len(ohlcv))

    state = compute_primary_state(valid["side"], cap=60)  # D1 cap, as run_backtest
    X = features.drop(columns=["_atr_14"]).loc[valid.index].copy()
    X["primary_side"] = state["primary_side"].values
    if primary_name == "ema_cross":
        spread = (ohlcv["close"].ewm(span=cfg["primary"]["ema_cross"]["fast"]).mean()
                  - ohlcv["close"].ewm(span=cfg["primary"]["ema_cross"]["slow"]).mean())
        X["primary_strength"] = (spread / atr).loc[valid.index].values
    else:
        X["primary_strength"] = features["z_r20"].loc[valid.index].values
    X["bars_since_signal"] = state["bars_since_signal"].values

    pre_drop_index = X.index
    X = X.dropna()
    if X.empty:
        print(f"  {asset}/{primary_name}: all events NaN-dropped — skipping.")
        return None
    y = valid["label"].loc[X.index]
    keep_mask = pre_drop_index.isin(X.index)
    w = w_all[keep_mask]
    assert len(w) == len(X) == len(y)

    side = X["primary_side"]
    close = ohlcv["close"].values
    valid_kept = valid.loc[X.index]
    entry_close = close[[idx_pos[ts] for ts in X.index]]
    exit_price = valid_kept["exit_price"].values
    fwd_ret = pd.Series(np.log(exit_price / entry_close), index=X.index)
    assert not fwd_ret.isnull().any(), "fwd_ret contains NaN — check exit_price alignment"

    event_time = pd.DatetimeIndex(X.index)
    label_end_time = pd.DatetimeIndex(
        ohlcv.index[valid_kept["t_end_idx"].astype(int).values]
    )

    return {
        "asset": asset,
        "primary_name": primary_name,
        "asset_class": cfg["asset_class"],
        "bars_per_year": int(cfg["metrics"]["bars_per_year"]),
        "cost_bps": cost_bps,
        "X": X, "y": y, "w": w, "side": side, "fwd_ret": fwd_ret,
        "event_time": event_time, "label_end_time": label_end_time,
        "pool_key": cfg["asset_class"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--count-events-only", action="store_true",
                    help="build members, print per-member + pooled event counts, exit")
    ap.add_argument("--assets", default=None,
                    help="comma-separated subset of universe tickers")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    u = load_universe(cfg["universe_path"])
    tickers = list(u[cfg["universe_segment"]])
    if args.assets:
        subset = set(args.assets.split(","))
        tickers = [t for t in tickers if t in subset]

    s, e = cfg["date_range"]["start"], cfg["date_range"]["end"]
    macro = build_macro_frame(s, e, Path("cache/fred"))
    out_root = Path(cfg["output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    members_by_primary: dict[str, list[dict]] = defaultdict(list)
    counts: list[dict] = []
    for t in tickers:
        ohlcv = load_dataset(Path(f"data/D1/{t}_D1.csv")).loc[s:e]
        features = build_tier2_features(ohlcv, macro).dropna()
        ohlcv = ohlcv.loc[features.index]
        for primary in cfg["primary"]["candidates"]:
            m = build_member_inputs(t, primary, ohlcv, features, cfg)
            if m is None:
                continue
            counts.append({"asset": t, "primary": primary, "n_events": len(m["X"])})
            members_by_primary[primary].append(m)
            d = out_root / t / primary
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"side": m["side"].values, "fwd_ret": m["fwd_ret"].values},
                index=m["event_time"],
            ).to_parquet(d / "events_side_fwd.parquet")

    (out_root / "member_event_counts.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8")
    for primary in cfg["primary"]["candidates"]:
        total = sum(c["n_events"] for c in counts if c["primary"] == primary)
        n_members = sum(1 for c in counts if c["primary"] == primary)
        print(f"POOL {primary}: {n_members} members, {total} pooled events")
    if args.count_events_only:
        return 0

    mp = cfg["meta_pooling"]
    for primary, members in members_by_primary.items():
        if not members:
            continue
        _run_one_pool(
            primary_name=primary,
            pool_key=members[0]["pool_key"],
            members=members,
            cfg=cfg,
            schema=mp.get("schema", "core"),
            weight_balance=mp.get("weight_balance", "per_class"),
            pooled_uniqueness=bool(mp.get("pooled_uniqueness", True)),
            train_min_frac=float(mp.get("train_min_frac", 0.5)),
            out_root=out_root,
            dry_run=args.dry_run,
        )
    print("M3 pooled run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Write the two configs**

`configs/equity_m3_d1.yaml`:

```yaml
# M3 pooled cross-section — 35 t0-2006 large-caps (B0003).
# Consumed by scripts/run_pooled_equity_d1.py. Sections mirror equity_d1.yaml
# where shared; pooled-specific keys mirror multi_h4.yaml's meta_pooling.
asset_class: equity
universe_path: configs/universe_equity_m3.yaml
universe_segment: stocks

date_range:
  start: "2006-01-01"
  # End pinned inside the FRED cache span (cache/fred updated 2026-05-25);
  # same end as equity_d1.yaml so NVDA-smoke and M3 windows match.
  end: "2026-04-30"

triple_barrier:
  # Same asymmetric-3R geometry as equity_d1.yaml (see rationale there).
  horizon: 40
  tp_atr_mult: 3.0
  sl_atr_mult: 1.0
  atr_period: 14

primary:
  candidates: [ema_cross, momentum_zscore]
  ema_cross:
    fast: 20
    slow: 50
    dead_zone_atr: 0.25
  momentum_zscore:
    lookback: 20
    threshold: 0.3

features:
  tier: 2
  gld_volume: false

walk_forward:
  # Pooled folds come from make_pooled_time_folds (train_min_frac below);
  # n_folds=4 is affordable because the pool has ~35x NVDA's event count.
  # train_min_bars is used only for the wf_event_floor pre-flight print.
  n_folds: 4
  train_min_bars: 1500
  purge_bars: 20
  embargo_pct: 0.01

models:
  - xgb
  - lgbm
  - rf
  - lr

hyperparam_search:
  method: random
  n_iter: 20
  cv_splits: 3
  cv_type: purged_time_series
  cv_purge_bars: 20

calibration:
  method: sigmoid
  cv: prefit
  calib_holdout_pct: 0.15
  min_minority_for_isotonic: 50

stacking:
  baseline: trade_all_primary
  min_models_beating_baseline: 2
  min_folds_beating_baseline: 3
  max_oof_corr: 0.7
  min_trades_per_fold: 30
  meta_learner: logistic_regression
  meta_C: 1.0
  meta_penalty: l2
  meta_n_folds: 4

best_model:
  min_trades_per_fold: 30
  min_folds_with_trades: 2

metrics:
  # Flat 10 bps is deliberately conservative for mega-caps (B0005 refines
  # to per-ticker spreads later; do not lower this before B0005).
  cost_per_trade_bps: 10
  bars_per_year: 252
  threshold_grid: [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62]
  threshold_selection: inner_cv

threshold_selection:
  method: inner_cv
  inner_splits: 3
  min_trades_per_inner_fold: 20

meta_pooling:
  scope: within_class
  schema: core
  weight_balance: per_class
  pooled_uniqueness: true
  train_min_frac: 0.5

output_dir: results/clf_equity_m3_d1
random_seed: 42

dry_run:
  enabled_via_cli_flag: true
  n_iter: 5
  max_minutes_per_model_warn: 8
  max_minutes_per_model_abort: 15
```

`configs/equity_m3_etf_d1.yaml` — identical except these four keys (create by copying the file and changing ONLY these):

```yaml
# M3 survivorship-clean CONTROL — 9 SPDR sector ETFs (B0003 caveat 3).
# Ex-ante interpretation rule (FROZEN, do not reinterpret after results):
#   uplift on stocks AND ETFs        -> credibility rises materially
#   uplift on stocks, NOT on ETFs    -> flag: possibly universe-contaminated
#                                       OR genuine idiosyncratic dispersion — ambiguous
#   ETF run fails min-trades gates   -> report "control underpowered"; count neither way
asset_class: equity_index
universe_segment: etfs
metrics:
  # ETFs trade at ~1-2 bps spreads; 3 bps keeps slippage margin without the
  # 10 bps stock cost punishing the lower-vol barrier geometry (B0005).
  cost_per_trade_bps: 3
  bars_per_year: 252
  threshold_grid: [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62]
  threshold_selection: inner_cv
output_dir: results/clf_equity_m3_etf_d1
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_run_pooled_equity_d1.py -v`
Expected: 2 passed. (The `synth_ohlcv` fixture comes from `tests/conftest.py`.)

- [ ] **Step 6: Run the full suite (imports from run_backtest/run_multi_h4 must not break anything)**

Run: `uv run pytest -q`
Expected: all pass (same skip set as before this plan).

- [ ] **Step 7: Commit**

```powershell
git add scripts/run_pooled_equity_d1.py configs/equity_m3_d1.yaml configs/equity_m3_etf_d1.yaml tests/test_run_pooled_equity_d1.py
git commit -m "feat(m3): pooled equity D1 runner + stock/ETF-control configs (B0003)"
```

---

### Task 5: Long/short split reporter

**Files:**
- Create: `scripts/report_long_short_split.py`
- Test: `tests/test_long_short_split.py`

**Interfaces:**
- Consumes: per-(asset, primary) `oof_predictions.parquet` (columns = model names, float probs, index = event timestamps — written by `_write_per_asset_oos`) and `events_side_fwd.parquet` (columns `side`, `fwd_ret`, same index — written by Task 4's runner).
- Produces: `split_metrics(pnl: pd.Series, years: float) -> dict` and `long_short_split(oof: pd.DataFrame, events: pd.DataFrame, model: str, threshold: float, cost_bps: float) -> dict`; CLI `uv run python scripts/report_long_short_split.py --results results/clf_equity_m3_d1 --cost-bps 10` walks `<results>/<asset>/<primary>/`, pools all assets per (primary, model), applies the FIXED 0.55 headline threshold, writes `<results>/long_short_split.json`.

**Why:** B0003 caveat 1 — long-side absolute performance is inflated by residual survivorship, short-side deflated. The M3 report must show sides separately; a long-only-driven "edge" is a red flag, not a result.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_long_short_split.py
"""Per-side pooled metrics: NaN-not-zero under 30 trades; sides split correctly."""
import numpy as np
import pandas as pd

from scripts.report_long_short_split import long_short_split


def _fixtures(n_long=60, n_short=60):
    idx = pd.date_range("2015-01-01", periods=n_long + n_short, freq="B", tz="UTC")
    side = np.array([1] * n_long + [-1] * n_short)
    rng = np.random.default_rng(7)
    # Longs win on average, shorts lose (side * fwd_ret is the trade pnl).
    fwd = np.where(side == 1,
                   rng.normal(0.01, 0.02, n_long + n_short),
                   rng.normal(0.01, 0.02, n_long + n_short))
    events = pd.DataFrame({"side": side, "fwd_ret": fwd}, index=idx)
    oof = pd.DataFrame({"lr": np.full(len(idx), 0.60)}, index=idx)
    return oof, events


def test_sides_are_split_and_signed_correctly():
    oof, events = _fixtures()
    out = long_short_split(oof, events, model="lr", threshold=0.55, cost_bps=0.0)
    assert out["long"]["n_trades"] == 60
    assert out["short"]["n_trades"] == 60
    # Same positive fwd_ret both sides -> longs profit, shorts lose.
    assert out["long"]["mean_pnl_per_trade"] > 0
    assert out["short"]["mean_pnl_per_trade"] < 0


def test_nan_sharpe_below_30_trades():
    oof, events = _fixtures(n_long=60, n_short=10)
    out = long_short_split(oof, events, model="lr", threshold=0.55, cost_bps=0.0)
    assert out["short"]["n_trades"] == 10
    assert np.isnan(out["short"]["sharpe_net"]), "NaN, never 0, under 30 trades"
    assert not np.isnan(out["long"]["sharpe_net"])


def test_threshold_filters_trades():
    oof, events = _fixtures()
    oof.iloc[:30, 0] = 0.40  # below threshold -> dropped
    out = long_short_split(oof, events, model="lr", threshold=0.55, cost_bps=0.0)
    assert out["long"]["n_trades"] == 30
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_long_short_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.report_long_short_split'`

- [ ] **Step 3: Write the reporter**

```python
# scripts/report_long_short_split.py
"""Pooled long/short split report for M3 runs (B0003 caveat 1).

Survivor-universe long-side results are inflated and short-side deflated;
the M3 claim is only ever 'meta adds value over primary WITHIN this
universe', reported per side. Pools events across all assets per
(primary, model) at the FIXED 0.55 headline threshold.

USAGE:
  uv run python scripts/report_long_short_split.py --results results/clf_equity_m3_d1 --cost-bps 10
"""
from __future__ import annotations
import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HEADLINE_THRESHOLD = 0.55  # fixed by spec — never selected from the test grid
MIN_TRADES_FOR_SHARPE = 30  # pipeline invariant: NaN (not 0) below this


def split_metrics(pnl: pd.Series, years: float) -> dict:
    n = int(len(pnl))
    out = {"n_trades": n,
           "mean_pnl_per_trade": float(pnl.mean()) if n else None,
           "hit_ratio": float((pnl > 0).mean()) if n else None}
    sd = float(pnl.std(ddof=1)) if n > 1 else 0.0
    if n < MIN_TRADES_FOR_SHARPE or sd == 0.0 or years <= 0:
        out["sharpe_net"] = float("nan")
    else:
        trades_per_year = n / years
        out["sharpe_net"] = float(pnl.mean() / sd * np.sqrt(trades_per_year))
    return out


def long_short_split(oof: pd.DataFrame, events: pd.DataFrame, model: str,
                     threshold: float, cost_bps: float) -> dict:
    # POSITIONAL alignment, not index joins: pooling concatenates many assets
    # whose event timestamps overlap, so the index is NOT unique. Caller must
    # pass row-aligned frames (same order, same length).
    if len(oof) != len(events) or not (oof.index == events.index).all():
        raise ValueError("oof and events must be row-aligned (same index, same order)")
    p = oof[model].to_numpy(dtype=float)
    side = events["side"].to_numpy()
    fwd = events["fwd_ret"].to_numpy(dtype=float)
    take = ~np.isnan(p) & (p >= threshold)
    span_years = max(
        (events.index.max() - events.index.min()).days / 365.25, 1e-9,
    ) if len(events) else 0.0
    result = {}
    for name, sval in (("long", 1), ("short", -1)):
        mask = take & (side == sval)
        pnl = pd.Series(side[mask] * fwd[mask] - cost_bps / 1e4)
        result[name] = split_metrics(pnl, span_years)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--cost-bps", type=float, required=True)
    args = ap.parse_args()
    root = Path(args.results)

    pooled: dict[tuple[str, str], list[tuple[pd.DataFrame, pd.DataFrame]]] = {}
    for ev_path in sorted(root.glob("*/*/events_side_fwd.parquet")):
        oof_path = ev_path.parent / "oof_predictions.parquet"
        if not oof_path.exists():
            continue
        primary = ev_path.parent.name
        oof = pd.read_parquet(oof_path)
        ev = pd.read_parquet(ev_path)
        if len(oof) != len(ev) or not (oof.index == ev.index).all():
            print(f"  SKIP {ev_path.parent}: oof/events misaligned")
            continue
        for model in oof.columns:
            pooled.setdefault((primary, model), []).append((oof[[model]], ev))

    report: dict[str, dict] = {}
    for (primary, model), parts in sorted(pooled.items()):
        # Both lists concat in the same per-asset order -> positional alignment
        # is preserved even though timestamps repeat across assets.
        oof_all = pd.concat([o for o, _ in parts], axis=0)
        ev_all = pd.concat([e for _, e in parts], axis=0)
        report[f"{primary}/{model}"] = long_short_split(
            oof_all, ev_all, model=model,
            threshold=HEADLINE_THRESHOLD, cost_bps=args.cost_bps,
        )

    out = root / "long_short_split.json"
    out.write_text(json.dumps(
        {"threshold": HEADLINE_THRESHOLD, "cost_bps": args.cost_bps,
         "note": ("Survivor universe: long side inflated, short side deflated. "
                  "Within-universe relative claims only (B0003 caveat 1)."),
         "pools": report}, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(report)} primary/model pools)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_long_short_split.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```powershell
git add scripts/report_long_short_split.py tests/test_long_short_split.py
git commit -m "feat(m3): pooled long/short split reporter (B0003 caveat 1)"
```

---

### Task 6: Ingest universe data + event-density gate (EXECUTION — network + verification)

No new code. This task moves B0003 to `in_progress`, pulls the data, and runs the two cheap gates the OQ1 decision demands BEFORE any full run.

- [ ] **Step 1: Mark B0003 in progress**

```powershell
uv run python -c "import sys; sys.path.insert(0, '.'); from backlog import db; db.change_status('B0003', 'in_progress', 'M3 plan approved; starting data ingestion')"
```

- [ ] **Step 2: Fetch the universe (network; ~2-5 min)**

Run: `uv run python scripts/fetch_equity_daily.py --universe configs/universe_equity_m3.yaml`
Expected: 44 lines `Wrote data/D1/<T>_D1.csv: NNNN rows, YYYY-MM-DD -> ...`, exit 0.
If any ticker FAILS the gate: do NOT improvise — substitute from `alternates` in order, edit `configs/universe_equity_m3.yaml`, and record the substitution:

```powershell
uv run python -c "import sys; sys.path.insert(0, '.'); from backlog import db; db.append_history('B0003', 'field_update', 'universe substitution per frozen alternates rule: <OUT> -> <IN>, reason: <fetch failure detail>')"
```

- [ ] **Step 3: Verify contract-load of every CSV**

```powershell
uv run python -c "import sys; sys.path.insert(0, '.'); from pipeline.data import load_dataset; from pipeline.equity_universe import load_universe; u = load_universe('configs/universe_equity_m3.yaml'); [print(t, len(load_dataset(f'data/D1/{t}_D1.csv'))) for t in u['stocks'] + u['etfs']]"
```

Expected: 44 lines, each ≥ 5000 rows, no exception.

- [ ] **Step 4: Event-density gate on a 5-ticker probe (the OQ1 go/no-go)**

Run: `uv run python scripts/run_pooled_equity_d1.py --config configs/equity_m3_d1.yaml --count-events-only --assets XOM,MSFT,JPM,KO,CAT`
Expected output shape: `POOL ema_cross: 5 members, NNN pooled events` (and same for momentum_zscore).

**Decision rule (frozen in B0003):** per-name events/yr = pooled events ÷ 5 ÷ ~20.3 yrs. If < 3 events/yr/name for BOTH primaries, STOP — record in B0003 history and consult the user (the fallback is broadening the universe toward option A, a scope change). Otherwise proceed.

- [ ] **Step 5: Full-universe event counts + record the gate outcome**

Run: `uv run python scripts/run_pooled_equity_d1.py --config configs/equity_m3_d1.yaml --count-events-only`
Then: `uv run python scripts/run_pooled_equity_d1.py --config configs/equity_m3_etf_d1.yaml --count-events-only`

```powershell
uv run python -c "import sys; sys.path.insert(0, '.'); from backlog import db; db.append_history('B0003', 'decision', 'event-density gate PASSED: <X> pooled events (stocks), <Y> (ETF control); proceeding to dry-run')"
```

- [ ] **Step 6: Commit the data + gate evidence**

```powershell
git add data/D1/ results/clf_equity_m3_d1/member_event_counts.json results/clf_equity_m3_etf_d1/member_event_counts.json backlog/
git commit -m "data(m3): universe OHLCV ingested; event-density gate passed (B0003)"
```

---

### Task 7: Dry-run timing gate, full runs, honest readout (EXECUTION)

- [ ] **Step 1: Dry-run both configs (timing extrapolation)**

Run: `uv run python scripts/run_pooled_equity_d1.py --config configs/equity_m3_d1.yaml --dry-run`
Expected: `PRE-FLIGHT` line with `pooled_effective_N`, then per-model timing lines `[pool ema_cross/equity] xgb: N.N min over 1 fold(s)`. Extrapolate: `minutes × 4 folds × 2 primaries`. If any single model extrapolates past ~60 min, stop and consult the user before burning CPU-days.

Run: `uv run python scripts/run_pooled_equity_d1.py --config configs/equity_m3_etf_d1.yaml --dry-run`
Expected: same shape, much faster.

- [ ] **Step 2: Full stocks run (long: expect 1-4 h CPU)**

Run: `uv run python scripts/run_pooled_equity_d1.py --config configs/equity_m3_d1.yaml`
Expected: per-asset outputs under `results/clf_equity_m3_d1/<TICKER>/<primary>/` (`summary.json`, `psr_dsr.json`, `metrics_per_fold.json`, `oof_predictions.parquet`, ...), no Phase-D ERROR lines.

Known risk: `_write_per_asset_oos` (Phase D) runs only in the FULL run (dry-run returns early), so a config key it needs but our configs lack surfaces here as `KeyError`. Fix by copying the corresponding section from `configs/multi_h4.yaml` into both M3 configs (schema parity) — do NOT edit `run_multi_h4.py` to paper over it.

- [ ] **Step 3: Full ETF control run**

Run: `uv run python scripts/run_pooled_equity_d1.py --config configs/equity_m3_etf_d1.yaml`
Expected: same tree under `results/clf_equity_m3_etf_d1/`.

- [ ] **Step 4: Long/short split reports**

Run: `uv run python scripts/report_long_short_split.py --results results/clf_equity_m3_d1 --cost-bps 10`
Run: `uv run python scripts/report_long_short_split.py --results results/clf_equity_m3_etf_d1 --cost-bps 3`
Expected: `long_short_split.json` written in each results root.

- [ ] **Step 5: The honest readout (quant-validation discipline)**

Write `results/clf_equity_m3_d1/M3_READOUT.md` answering, with numbers from the artifacts, in this order:

1. Coverage: members per primary that survived Phase A; pooled event counts; `pooled_effective_N` vs floor.
2. Per the FIXED 0.55 threshold and `should_stack` gates: how many (primary, model) pools beat baseline in ≥3 folds with ≥30 trades? (NaN folds skipped, never zero-filled.)
3. PSR/DSR of the best candidates — report the DSR *after* familywise deflation, and say explicitly how many trials the DSR assumed.
4. Long/short split: does any apparent edge survive on the SHORT side? Long-only edge in a survivor universe = flagged, not celebrated.
5. ETF control per the frozen interpretation rule (in `configs/equity_m3_etf_d1.yaml` header): both / stocks-only / underpowered.
6. False-positive checklist: single-class folds? zero-trade models? calibration collapse (`pct_signals_kept` < 5%)? one lucky fold driving the aggregate (recompute with `np.nanmedian` across folds)?
7. Verdict: `EDGE-CANDIDATE (within-universe, pending B0004 PIT re-run)` or `NO EDGE` — nothing stronger is claimable from a survivor universe (B0003 caveat 1).

- [ ] **Step 6: Record the readout in B0003 and commit**

```powershell
uv run python -c "import sys; sys.path.insert(0, '.'); from backlog import db; db.append_history('B0003', 'decision', 'M3 first readout: <VERDICT>. Stocks: <n pools passing gates>/<total>; ETF control: <both|stocks-only|underpowered>; long/short: <summary>. Full analysis in results/clf_equity_m3_d1/M3_READOUT.md')"
git add results/clf_equity_m3_d1/ results/clf_equity_m3_etf_d1/ backlog/
git commit -m "results(m3): first pooled cross-section readout (B0003)"
```

Note: whether B0003 then moves to `done` (edge or honest no-edge both complete the item) or spawns follow-ups (B0006 cross-sectional features if signal-starved) is a HUMAN decision made reading `M3_READOUT.md` — do not auto-promote.

---

### Task 8: Regime dossiers for the universe (Loop A enablement)

The pooled backtest doesn't need dossiers, but B0003 scope includes them: they are what lets Loop A hypothesize on the universe next.

- [ ] **Step 1: Build dossiers for the universe tickers only**

```powershell
$u = uv run python -c "import sys; sys.path.insert(0, '.'); from pipeline.equity_universe import load_universe; u = load_universe('configs/universe_equity_m3.yaml'); print(','.join(u['stocks'] + u['etfs']))"
uv run python scripts/build_all_regimes.py --assets $u --frequencies D1
```

Expected: per-ticker regime parquets under `data/regimes/` and dossier folders under `signals/regime_stats/<TICKER>_d1/` (4 regime JSONs each, like `signals/regime_stats/NVDA_d1/`). Tickers whose history is under the burn-in floor are refused loudly — that's the guard working, not a bug; list any refusals in the commit message.

- [ ] **Step 2: Run the full test suite one last time**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 3: Commit**

```powershell
git add data/regimes/ signals/regime_stats/
git commit -m "feat(m3): regime dossiers for the M3 universe (Loop A enablement, B0003)"
```

---

## Self-Review Notes (author-checked against B0003 + spec)

- B0003 scope line 1 (frozen selection rule, t0, enumerated delistees) → Task 1. Line 2 (batch fetch + registry + dossiers) → Tasks 2, 3, 8. Line 3 (pooled config + dry-run first) → Tasks 4, 7. Line 4 (ETF control + ex-ante interpretation rule) → Task 4 config header + Task 7 Step 5. Line 5 (long/short separate, within-universe claims only) → Tasks 5, 7. Line 6 (this plan doc) → this file.
- The five-ticker density probe and the "<3 events/yr/name → broaden toward A" fallback from the OQ1 decision are Task 6 Step 4, with an explicit stop-and-consult rather than silent universe swap (caveat 2).
- Types are consistent: `build_member_inputs` produces exactly the member keys `_run_one_pool` consumes (verified against `scripts/run_multi_h4.py:1506-1599`); `events_side_fwd.parquet` (Task 4) is what `report_long_short_split.py` (Task 5) joins against `oof_predictions.parquet` (written by `_write_per_asset_oos` at `scripts/run_multi_h4.py:1162`).
- Deliberately out of scope: per-ticker friction (B0005 — configs carry conservative flat costs with comments), cross-sectional features (B0006 — triggered only if the readout is signal-starved), PIT vendor (B0004 — the acceptance gate for anything Loop-B-bound), sizing.
