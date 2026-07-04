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
from pipeline.features import build_tier2_features, feature_add_status
from pipeline.labels import compute_primary_state, triple_barrier_labels
from pipeline.macro_fetch import build_macro_frame
from pipeline.sample_weights import (
    avg_uniqueness,
    corr_discounted_uniqueness,
    effective_number_of_bets,
    pooled_avg_uniqueness,
    rolling_panel_rho,
)
from scripts.run_backtest import _select_primary
from scripts.run_multi_h4 import _run_one_pool


def _regimes_path(asset: str) -> Path:
    """Monkeypatch seam for tests; production default matches the M3 Task 8
    regime-parquet layout (data/regimes/<TICKER>_d1_regimes.parquet)."""
    return Path(f"data/regimes/{asset}_d1_regimes.parquet")


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

    # Regime gating (additive, mirrors run_backtest.py's regime_mask_path
    # block): filter events to bars whose OWN regime_id is in scope BEFORE
    # triple-barrier labeling, so weights/labels/folds are computed only on
    # in-scope events.
    scope = cfg.get("regime_scope") or []
    if scope:
        regimes = pd.read_parquet(_regimes_path(asset))
        in_scope_ts = regimes.index[regimes["regime_id"].isin(set(scope))]
        n_before = len(events)
        events = events[events.index.isin(in_scope_ts)]
        print(f"  {asset}/{primary_name}: regime gate kept {len(events)}/{n_before} events")
        if events.empty:
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
    # B0010: feature_overrides_drop subtracts from the meta's X (mirrors
    # run_backtest.py's feature_overrides_drop / primary_feature_blacklist
    # handling). The primary already received the full `features` above via
    # _select_primary; this drop only affects the meta's view from here on.
    fo_drop = cfg.get("feature_overrides_drop", []) or []
    meta_features = features.drop(columns=["_atr_14"]).drop(columns=fo_drop, errors="ignore")
    X = meta_features.loc[valid.index].copy()
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
        "meta_feature_columns": list(meta_features.columns),
        "pool_key": cfg["asset_class"],
    }


def _apply_fit_weight_mode(
    members_by_primary: dict[str, list[dict]],
    panel: dict[str, pd.DataFrame],
    cfg: dict,
) -> tuple[dict[str, list[dict]], bool, dict[str, dict]]:
    """B0012 v2 wiring (spec §3-4). Decides the fit-weight mode from
    ``cfg["meta_pooling"]["fit_weight"]`` and returns:

    - ``members_by_primary`` with ``m["w"]`` overwritten in-place for
      ``corr_discounted_v2`` (untouched otherwise).
    - ``pooled_uniqueness_flag``: the ONE thing that may vary in the
      downstream ``_run_one_pool`` call across modes (the firewall — spec
      §1/§4's "Firewall assertion in code").
    - ``diagnostics_by_primary``: unconditional per-pool effective-N record
      (closes B0011), written by the caller to
      ``<output_dir>/effective_n_<primary>.json`` in EVERY mode.

    The conservative rho=1 ``effective_n_rho1`` (via `pooled_avg_uniqueness`)
    is computed IDENTICALLY regardless of mode — this is the gate-monotonicity
    invariant (spec §5.4): no gate input may ever depend on `fit_weight`.
    """
    mp = cfg.get("meta_pooling") or {}
    mode = mp.get("fit_weight", "rho1_pooled") or "rho1_pooled"
    valid_modes = {"rho1_pooled", "corr_discounted_v2", "per_asset"}
    if mode not in valid_modes:
        raise ValueError(
            f"meta_pooling.fit_weight={mode!r} is not one of {sorted(valid_modes)}; "
            "refusing to fall back silently to rho1_pooled"
        )
    pooled_uniqueness_flag = bool(mp.get("pooled_uniqueness", True))

    rho_schedule = None
    union_grid = None
    if mode == "corr_discounted_v2":
        pooled_uniqueness_flag = False
        for df in panel.values():
            union_grid = df.index if union_grid is None else union_grid.union(df.index)
        union_grid = union_grid.sort_values()
        close_frame = pd.DataFrame({t: panel[t]["close"].reindex(union_grid) for t in panel})
        rho_schedule = rolling_panel_rho(close_frame)
    elif mode == "per_asset":
        pooled_uniqueness_flag = False

    diagnostics_by_primary: dict[str, dict] = {}
    for primary, members in members_by_primary.items():
        if not members:
            continue
        event_time_all = pd.DatetimeIndex(
            np.concatenate([np.asarray(m["event_time"]) for m in members])
        )
        label_end_all = pd.DatetimeIndex(
            np.concatenate([np.asarray(m["label_end_time"]) for m in members])
        )
        asset_all = np.concatenate(
            [np.full(len(m["event_time"]), m["asset"], dtype=object) for m in members]
        )
        raw_n = int(len(event_time_all))

        # Gate-side quantity: UNCHANGED regardless of fit_weight mode.
        w_rho1_all = pooled_avg_uniqueness(event_time_all, label_end_all)
        effective_n_rho1 = float(w_rho1_all.sum())

        enb_ceiling = None
        enb_ceiling_reason = None
        rho_panel_mean_last = None

        if mode == "corr_discounted_v2":
            v2_weights = corr_discounted_uniqueness(
                event_time_all, label_end_all, asset_all, rho_schedule, union_grid,
            )
            offset = 0
            for m in members:
                n = len(m["event_time"])
                m["w"] = np.asarray(v2_weights[offset: offset + n])
                offset += n
            fit_weight_sum = float(v2_weights.sum())
            if rho_schedule:
                last_rho = rho_schedule[-1][1]
                enb_ceiling = effective_number_of_bets(last_rho)
                off_mask = ~np.eye(len(last_rho), dtype=bool)
                off_vals = last_rho.values[off_mask]
                rho_panel_mean_last = float(np.nanmean(off_vals)) if off_vals.size else None
            else:
                enb_ceiling_reason = "no rho schedule produced (insufficient history)"
        else:
            # rho1_pooled / per_asset: weights untouched; the diagnostic
            # correlation schedule is NOT built here (would be spec-legal
            # only "if cheap" and adds no value when nothing consumes it).
            # fit_weight_sum must describe the weights the fit ACTUALLY uses:
            # with pooled_uniqueness on, _run_one_pool discards m["w"] and
            # fits on the rho=1 pooled weights (sum == effective_n_rho1).
            if mode == "rho1_pooled" and pooled_uniqueness_flag:
                fit_weight_sum = effective_n_rho1
            else:
                fit_weight_sum = float(sum(np.asarray(m["w"]).sum() for m in members))
            enb_ceiling_reason = "fit_weight mode does not use a rho schedule"

        diagnostics_by_primary[primary] = {
            "primary": primary,
            "pool_key": members[0]["pool_key"],
            "fit_weight_mode": mode,
            "raw_n": raw_n,
            "effective_n_rho1": effective_n_rho1,
            "fit_weight_sum": fit_weight_sum,
            "enb_ceiling": enb_ceiling,
            "enb_ceiling_reason": enb_ceiling_reason,
            "rho_panel_mean_last": rho_panel_mean_last,
            "computed_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }

    return members_by_primary, pooled_uniqueness_flag, diagnostics_by_primary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--count-events-only", action="store_true",
                    help="build members, print per-member + pooled event counts, exit")
    ap.add_argument("--effective-n-only", action="store_true",
                    help="stop after computing/persisting effective_n_<primary>.json "
                         "(no model fitting) — cheap floor check before a full run")
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

            # B0010: feature_overrides_add is validated (not applied — all
            # tier2 columns already flow to the meta minus feature_overrides_drop)
            # and the evidence is recorded per (asset, primary) so the audit
            # artifact can prove which features the meta actually saw. Exact
            # mirror of run_backtest.py:1158-1168.
            fo_add = cfg.get("feature_overrides_add", []) or []
            fo_drop = cfg.get("feature_overrides_drop", []) or []
            fo_add_status = feature_add_status(fo_add, set(m["meta_feature_columns"]))
            (d / "feature_overrides_status.json").write_text(
                json.dumps({
                    "add_requested": list(fo_add),
                    "add_status": fo_add_status,
                    "drop_applied": list(fo_drop),
                    "meta_feature_count": len(m["meta_feature_columns"]),
                }, indent=2),
                encoding="utf-8",
            )

    (out_root / "member_event_counts.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8")
    for primary in cfg["primary"]["candidates"]:
        total = sum(c["n_events"] for c in counts if c["primary"] == primary)
        n_members = sum(1 for c in counts if c["primary"] == primary)
        print(f"POOL {primary}: {n_members} members, {total} pooled events")
    if args.count_events_only:
        return 0

    mp = cfg["meta_pooling"]
    members_by_primary, pooled_uniqueness_flag, diagnostics_by_primary = _apply_fit_weight_mode(
        members_by_primary, panel, cfg
    )
    for primary, diag in diagnostics_by_primary.items():
        (out_root / f"effective_n_{primary}.json").write_text(
            json.dumps(diag, indent=2), encoding="utf-8"
        )
        print(f"  EFFECTIVE-N [{primary}]: raw_n={diag['raw_n']}  "
              f"effective_n_rho1={diag['effective_n_rho1']:.1f}  "
              f"fit_weight_mode={diag['fit_weight_mode']}")
    if args.effective_n_only:
        return 0

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
            pooled_uniqueness=pooled_uniqueness_flag,
            train_min_frac=float(mp.get("train_min_frac", 0.5)),
            out_root=out_root,
            dry_run=args.dry_run,
        )
    print("M3 pooled run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
