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

    # Pass 1: load the whole OHLCV panel (needed in full for cross-sectional
    # features, which reduce across tickers at each date).
    panel: dict[str, pd.DataFrame] = {}
    for t in tickers:
        panel[t] = load_dataset(Path(f"data/D1/{t}_D1.csv")).loc[s:e]

    cs_frames: dict[str, pd.DataFrame] = {}
    if (cfg.get("features") or {}).get("cross_sectional"):
        from pipeline.cross_section import build_cross_sectional_features
        cs_frames = build_cross_sectional_features(panel, tickers)

    members_by_primary: dict[str, list[dict]] = defaultdict(list)
    counts: list[dict] = []
    # Pass 2: per-ticker tier-2 features (+ cs_ join before dropna) + members.
    for t in tickers:
        ohlcv = panel[t]
        features = build_tier2_features(ohlcv, macro)
        if cs_frames:
            features = features.join(cs_frames[t])
        features = features.dropna()
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
