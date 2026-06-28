"""Batch driver: build regime parquets + dossiers for every registry asset.

For each (ticker, frequency) cell:
  - skip (no write) if the CSV has fewer than min_bars_for_attempt bars;
  - isolate per-cell failures (bad CSV / missing file) into the roll-up;
  - reuse pipeline.regimes.label_regimes + phase5.regime_stats helpers;
  - write artifacts atomically with a manifest for idempotency (Task 6);
  - emit a roll-up report (Task 7).

Run: uv run python scripts/build_all_regimes.py [--assets XAUUSD,BTCUSD] [--frequencies D1,H4] [--force]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from pipeline.data import load_dataset, DataValidationError
from pipeline.regimes import label_regimes
from phase5.regime_stats import build_dossier_features, build_regime_dossiers, build_primary_baselines
from phase5.asset_registry import dossier_dirname
from pipeline.features import _atr


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def build_one_cell(
    *,
    ticker: str,
    frequency: str,
    asset_class: str,
    feature_pack: tuple[str, ...],
    data_path: Path,
    regimes_dir: Path,
    dossiers_dir: Path,
    min_bars: int,
    force: bool = False,
) -> dict:
    """Build one (ticker, frequency) cell. Returns a roll-up row dict; never raises
    on per-cell data problems (records status='failed' instead)."""
    row = {"ticker": ticker, "frequency": frequency, "status": "", "detail": "",
           "n_total_bars": 0, "n_labeled_bars": 0, "regimes_sufficient": []}
    try:
        df = load_dataset(data_path)
    except (DataValidationError, FileNotFoundError, ValueError) as e:
        row["status"] = "failed"
        row["detail"] = str(e)
        return row

    row["n_total_bars"] = int(len(df))
    if len(df) < min_bars:
        row["status"] = "skipped_insufficient_history"
        row["detail"] = f"{len(df)} bars < min_bars {min_bars}"
        return row

    regimes_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = regimes_dir / f"{ticker}_{frequency.lower()}_manifest.json"
    csv_hash = _sha256_file(data_path)
    if not force and manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text())
            if prior.get("csv_sha256") == csv_hash:
                row["status"] = "skipped_up_to_date"
                row["n_labeled_bars"] = int(prior.get("n_labeled_bars", 0))
                row["regimes_sufficient"] = prior.get("regimes_sufficient", [])
                return row
        except (json.JSONDecodeError, OSError):
            pass  # corrupt manifest => rebuild

    regimes = label_regimes(df["close"], frequency=frequency)
    n_labeled = int(regimes["regime_id"].notna().sum())
    if n_labeled == 0:
        # B0066: cleared min_bars but burn-in + min_dwell exceeds history => no labeled
        # regime. Honest skip — write NOTHING (no parquet/dossiers/manifest) so the cell
        # is re-attempted once more history arrives.
        row["status"] = "skipped_no_labeled_bars"
        row["detail"] = (f"{len(df)} bars cleared min_bars {min_bars} but label_regimes "
                         f"produced 0 labeled bars (burn-in + min_dwell exceeds history)")
        return row
    parquet_path = regimes_dir / f"{ticker}_{frequency.lower()}_regimes.parquet"
    regimes.to_parquet(parquet_path)

    features_df = build_dossier_features(df, regimes, asset=ticker, feature_pack=feature_pack)
    atr = _atr(df["high"], df["low"], df["close"])
    primary_baselines = build_primary_baselines(df, atr, regimes, frequency=frequency)
    dossiers = build_regime_dossiers(
        regimes, features_df, asset_class=asset_class,
        primary_baselines=primary_baselines, frequency=frequency,
    )

    import tempfile
    out_dir = dossiers_dir / dossier_dirname(ticker, frequency)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{ticker}_{frequency}_", dir=out_dir.parent))
    for regime_id, dossier in dossiers.items():
        (tmp_dir / f"{regime_id}.json").write_text(
            json.dumps(dossier, indent=2, default=str), encoding="utf-8"
        )
    for f in tmp_dir.glob("*.json"):
        os.replace(f, out_dir / f.name)  # atomic per-file rename
    tmp_dir.rmdir()

    row["n_labeled_bars"] = n_labeled
    row["regimes_sufficient"] = [r for r, d in dossiers.items() if d["sample_sufficient"]]
    manifest_path.write_text(json.dumps({
        "ticker": ticker, "frequency": frequency, "csv_sha256": csv_hash,
        "git_sha": _git_sha(), "n_labeled_bars": row["n_labeled_bars"],
        "regimes_sufficient": row["regimes_sufficient"],
        "dossiers": [str((out_dir / f"{r}.json")) for r in dossiers],
    }, indent=2), encoding="utf-8")
    row["status"] = "built"
    return row


def write_rollup(rows: list[dict], *, out_dir: Path, date_str: str) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"build_{date_str}.md"
    json_path = out_dir / f"build_{date_str}.json"
    lines = [f"# Regime build roll-up {date_str}", "",
             "| ticker | freq | status | labeled/total | sufficient regimes | detail |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['ticker']} | {r['frequency']} | {r['status']} | "
            f"{r['n_labeled_bars']}/{r['n_total_bars']} | "
            f"{', '.join(r['regimes_sufficient']) or '—'} | {r.get('detail', '')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return md_path, json_path


def main() -> int:
    from phase5.asset_registry import ASSET_REGISTRY, resolve_csv
    ap = argparse.ArgumentParser(description="Build regime parquets + dossiers for all registry assets")
    ap.add_argument("--assets", default=None, help="comma-separated tickers; default = all in registry")
    ap.add_argument("--frequencies", default="D1,H4", help="comma-separated; default D1,H4")
    ap.add_argument("--regimes-dir", default="data/regimes")
    ap.add_argument("--dossiers-dir", default="signals/regime_stats")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tickers = args.assets.split(",") if args.assets else list(ASSET_REGISTRY)
    frequencies = args.frequencies.split(",")
    rows = []
    for ticker in tickers:
        spec = ASSET_REGISTRY[ticker]
        for freq in frequencies:
            if freq not in spec.frequencies:
                continue
            try:
                data_path = resolve_csv(ticker, freq)
            except FileNotFoundError as e:
                rows.append({"ticker": ticker, "frequency": freq, "status": "failed",
                             "detail": str(e), "n_total_bars": 0, "n_labeled_bars": 0,
                             "regimes_sufficient": []})
                continue
            row = build_one_cell(
                ticker=ticker, frequency=freq, asset_class=spec.asset_class,
                feature_pack=spec.dossier_feature_pack(), data_path=data_path,
                regimes_dir=Path(args.regimes_dir), dossiers_dir=Path(args.dossiers_dir),
                min_bars=spec.min_bars_for_attempt(freq), force=args.force,
            )
            rows.append(row)
            print(f"  {ticker} {freq}: {row['status']} "
                  f"(labeled {row['n_labeled_bars']}/{row['n_total_bars']}, "
                  f"sufficient={row['regimes_sufficient']})", flush=True)
    from datetime import date
    md_path, json_path = write_rollup(rows, out_dir=Path("results/regimes"),
                                      date_str=date.today().strftime("%Y%m%d"))
    print(f"Roll-up: {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
