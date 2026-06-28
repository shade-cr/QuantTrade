"""Phase 5 orchestrator — deterministic Python driver for the hypothesis loop.

The orchestrator is the single source of truth for what happens in what
order during a Phase 5 spike day. It does NO LLM reasoning itself; at each
decision point it writes a `next_action_<id>.json` payload, pauses, and
expects a sibling Claude Code session to invoke the relevant agent via the
Agent tool and write the response back to `agent_response_<id>.json`.

This Day-1 skeleton supports the Day 2+ subcommands:
  orchestrate dossiers   — build per-regime dossiers (delegates to phase5.regime_stats)
  orchestrate propose    — emit a next_action payload for the hypothesizer
  orchestrate review     — emit a next_action payload for the devil's advocate
  orchestrate run        — run a proposal through the pipeline + auditor (delegates to phase5.run_proposal)
  orchestrate skeptic    — emit a next_action payload for the daily skeptic
  orchestrate status     — print signals/index.csv summary

Day 1 ships the skeleton; Day 2+ fills in `propose`, `review`, `run`, and
`skeptic` as the orchestrator drives the actual loop.
"""
from __future__ import annotations
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from phase5 import regime_stats, devils_advocate_dispatch as dad
from pipeline.primary_contracts import primary_param_schema_for_payload


SIGNALS_DIR = Path("signals")
INDEX_CSV = SIGNALS_DIR / "index.csv"
INDEX_HEADER = (
    "id", "asset", "regime", "primary", "status",
    "devils_advocate_verdict", "audit_class", "sharpe", "skeptic_verdict", "created_at",
)


def _ensure_index() -> None:
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_CSV.exists():
        with INDEX_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(INDEX_HEADER)


def cmd_status() -> int:
    """Print the day's index.csv summary."""
    _ensure_index()
    print(f"Reading {INDEX_CSV}")
    with INDEX_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print("No proposals yet. Run `orchestrate propose --asset XAUUSD --regime BEAR_QUIET`.")
        return 0
    print(f"{'id':30s} {'asset':8s} {'regime':16s} {'primary':18s} {'status':12s} {'audit':22s}")
    for r in rows:
        print(
            f"{r['id']:30s} {r['asset']:8s} {r['regime']:16s} {r['primary']:18s} "
            f"{r['status']:12s} {r.get('audit_class','-'):22s}"
        )
    return 0


def cmd_dossiers(asset: str, frequency: str, asset_class: str) -> int:
    """Build per-regime dossiers (Day 2 prerequisite)."""
    import subprocess, sys
    cmd = [
        sys.executable, "-m", "phase5.regime_stats",
        "--asset", asset, "--frequency", frequency, "--asset-class", asset_class,
    ]
    print(f"+ {' '.join(cmd)}")
    return subprocess.call(cmd)


def cmd_propose(asset: str, regime: str, asset_class: str, action_id: str | None = None, frequency: str = "D1") -> int:
    """Emit a next_action payload for the hypothesizer (Day 2+).

    The orchestrator does NOT invoke the agent — it writes the payload and
    pauses. The parent Claude Code session invokes the phase5-hypothesizer
    agent via the Agent tool and writes the response back to
    phase5/runtime/agent_response_<action_id>.json.
    """
    if action_id is None:
        action_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_propose_{asset}_{regime}"

    from phase5.asset_registry import dossier_dirname
    dossier_path = Path("signals/regime_stats") / dossier_dirname(asset, frequency) / f"{regime}.json"
    if not dossier_path.exists():
        print(
            f"ERROR: dossier not found at {dossier_path}. Run `orchestrate dossiers "
            f"--asset {asset} --frequency D1 --asset-class {asset_class}` first.",
            flush=True,
        )
        return 1

    dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
    payload = {
        "agent": "phase5-hypothesizer",
        "action_id": action_id,
        "input": {
            "asset_class": dossier["asset_class"],
            "regime_id": regime,
            "regime_stats_dossier": dossier,
            "available_features": list(dossier["features_quantile_summary"].keys()),
            # B0085: built-in primaries carry their exact canonical param schema so
            # the hypothesizer emits matching keys (phase5_custom stays free-form).
            "available_primaries": primary_param_schema_for_payload(),
            "id_hint": datetime.now(timezone.utc).strftime("%Y%m%d") + f"-{asset}-{frequency}-{regime[:8]}-PRO",
        },
        "output_path": f"phase5/runtime/agent_response_{action_id}.json",
    }
    runtime_dir = Path("phase5/runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    out_path = runtime_dir / f"next_action_{action_id}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"PAUSED: invoke phase5-hypothesizer agent with this payload, then write response to {payload['output_path']}.")
    return 0


def cmd_review(decision_type: str, decision_payload_path: str, action_id: str | None = None) -> int:
    """Emit a next_action payload for the devil's advocate."""
    if action_id is None:
        action_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_review_{decision_type}"

    decision_payload = json.loads(Path(decision_payload_path).read_text(encoding="utf-8"))
    payload = dad.make_decision_payload(
        decision_type=decision_type,
        decision_payload=decision_payload,
    )
    out_path = dad.submit_for_review({
        "agent": "phase5-devils-advocate",
        "action_id": action_id,
        "input": payload,
        "output_path": f"phase5/runtime/agent_response_{action_id}.json",
    }, action_id)
    print(f"Wrote {out_path}")
    print("PAUSED: invoke phase5-devils-advocate agent with this payload.")
    return 0


def cmd_run(proposal_path: str) -> int:
    """Run a proposal through the pipeline + auditor (Day 2+)."""
    import subprocess, sys
    cmd = [sys.executable, "-m", "phase5.run_proposal", "--proposal", proposal_path]
    print(f"+ {' '.join(cmd)}")
    return subprocess.call(cmd)


def cmd_skeptic(date_str: str) -> int:
    """Emit a next_action payload for the daily skeptic."""
    payload = {
        "agent": "phase5-skeptic",
        "action_id": f"skeptic_{date_str}",
        "input": {
            "review_date": date_str,
            "instructions": (
                "Read the day's artifacts under signals/ and produce the markdown "
                "report at signals/skeptic_reviews/<YYYYMMDD>.md per your agent definition."
            ),
        },
        "output_path": f"signals/skeptic_reviews/{date_str}.md",
    }
    runtime_dir = Path("phase5/runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    out_path = runtime_dir / f"next_action_skeptic_{date_str}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"PAUSED: invoke phase5-skeptic agent; write report to {payload['output_path']}.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="orchestrate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status")
    p = sub.add_parser("dossiers")
    p.add_argument("--asset", required=True)
    p.add_argument("--frequency", default="D1")
    p.add_argument("--asset-class", required=True)

    p = sub.add_parser("propose")
    p.add_argument("--asset", required=True)
    p.add_argument("--regime", required=True, choices=("BULL_QUIET", "BULL_STRESSED", "BEAR_QUIET", "BEAR_STRESSED"))
    p.add_argument("--asset-class", required=True)
    p.add_argument("--action-id", default=None)
    p.add_argument("--frequency", default="D1")

    p = sub.add_parser("review")
    p.add_argument("--decision-type", required=True)
    p.add_argument("--decision-payload", required=True, help="path to JSON payload under review")
    p.add_argument("--action-id", default=None)

    p = sub.add_parser("run")
    p.add_argument("--proposal", required=True, help="path to signals/proposals/<id>.json")

    p = sub.add_parser("skeptic")
    p.add_argument("--date", required=True, help="YYYYMMDD")

    args = ap.parse_args()

    if args.cmd == "status":
        return cmd_status()
    if args.cmd == "dossiers":
        return cmd_dossiers(args.asset, args.frequency, args.asset_class)
    if args.cmd == "propose":
        return cmd_propose(args.asset, args.regime, args.asset_class, args.action_id, args.frequency)
    if args.cmd == "review":
        return cmd_review(args.decision_type, args.decision_payload, args.action_id)
    if args.cmd == "run":
        return cmd_run(args.proposal)
    if args.cmd == "skeptic":
        return cmd_skeptic(args.date)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
