"""B0104 — LRU parity: the new `lru_by_cell`-based `_regime_lru_pick` must return
the IDENTICAL (asset, frequency, regime) pick as the old full-history scan.

This is the correctness anchor for Component 3 of the Bounded State Store spec
(docs/superpowers/specs/2026-05-30-bounded-state-store-design.md). The old scan
is kept inline below as the reference oracle. We fuzz a corpus of synthetic tick
histories and assert the new implementation agrees on every one.
"""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_lat():
    spec = importlib.util.spec_from_file_location(
        "loop_a_tick", REPO_ROOT / "scripts" / "loop_a_tick.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ASSETS = [
    ("XAUUSD", "D1"), ("XAGUSD", "D1"), ("BTCUSD", "D1"),
    ("ETHUSD", "D1"), ("BTCUSD", "H4"), ("ETHUSD", "H4"),
]
REGIMES = ["BULL_QUIET", "BULL_STRESSED", "BEAR_QUIET", "BEAR_STRESSED"]


def _oracle_pick(state: dict) -> tuple[str, str, str]:
    """REFERENCE ORACLE — the pre-B0104 full-history scan, verbatim logic.

    Collapses regime_history to max(last_ticked) per (asset, frequency, regime)
    cell, then picks the least-recently-explored eligible cell. Eligibility here
    is "every cell in asset_scope x regime_scope" (the test makes all dossiers
    sample_sufficient, so the dossier filter is a no-op and parity is isolated to
    the history->cell collapse + sort, which is what B0104 changes).
    """
    history_by_key: dict[tuple[str, str, str], str] = {}
    for entry in state.get("regime_history", []):
        key = (entry["asset"], entry.get("frequency", "D1"), entry["regime"])
        if key not in history_by_key or entry["last_ticked"] > history_by_key[key]:
            history_by_key[key] = entry["last_ticked"]

    eligible: list[tuple[str, str, str, str]] = []
    for cell in state["asset_scope"]:
        asset, frequency = cell["asset"], cell["frequency"]
        for regime in state["regime_scope"]:
            last = history_by_key.get((asset, frequency, regime), "")
            eligible.append((last, asset, frequency, regime))

    eligible.sort()
    _, asset, frequency, regime = eligible[0]
    return asset, frequency, regime


def _build_lru_by_cell(history: list[dict]) -> dict[str, str]:
    """Replicate record_tick's projection: max(last_ticked) per cell key."""
    lru: dict[str, str] = {}
    for e in history:
        key = f"{e['asset']}|{e.get('frequency', 'D1')}|{e['regime']}"
        if key not in lru or e["last_ticked"] > lru[key]:
            lru[key] = e["last_ticked"]
    return lru


def _make_state(history: list[dict], with_projection: bool) -> dict:
    state = {
        "asset_scope": [{"asset": a, "frequency": f} for a, f in ASSETS],
        "regime_scope": list(REGIMES),
        "regime_history": history,
    }
    if with_projection:
        state["lru_by_cell"] = _build_lru_by_cell(history)
    return state


def _random_history(rng: random.Random, n: int) -> list[dict]:
    history = []
    for i in range(n):
        a, f = rng.choice(ASSETS)
        regime = rng.choice(REGIMES)
        # ISO-comparable timestamps; lexicographic order == chronological order.
        ts = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:00:00+00:00"
        history.append({"asset": a, "frequency": f, "regime": regime, "last_ticked": ts})
    return history


def test_lru_parity_new_matches_old_scan_on_corpus(monkeypatch):
    """For a fuzzed corpus of histories, the new lru_by_cell pick == old scan pick."""
    lat = _load_lat()
    # Make every cell eligible: dossier present + sample_sufficient. Isolates the
    # parity check to the history->cell collapse, which is what B0104 rewrites.
    monkeypatch.setattr(lat, "_load_dossier", lambda a, f, r: {"sample_sufficient": True})

    rng = random.Random(0xB0104)
    mismatches = []
    for trial in range(300):
        n = rng.randint(0, 60)
        history = _random_history(rng, n)
        old = _oracle_pick(_make_state(history, with_projection=False))
        new = lat._regime_lru_pick(_make_state(history, with_projection=True))
        if old != new:
            mismatches.append((trial, n, old, new))

    assert not mismatches, f"LRU parity broken on {len(mismatches)} trials: {mismatches[:5]}"


def test_lru_parity_v1_fallback_without_projection(monkeypatch):
    """Backward-compat: a v1 state with regime_history but no lru_by_cell must
    still produce the same pick (the function derives the projection on the fly)."""
    lat = _load_lat()
    monkeypatch.setattr(lat, "_load_dossier", lambda a, f, r: {"sample_sufficient": True})

    rng = random.Random(0xFA11)
    for _ in range(100):
        history = _random_history(rng, rng.randint(0, 40))
        old = _oracle_pick(_make_state(history, with_projection=False))
        new = lat._regime_lru_pick(_make_state(history, with_projection=False))
        assert old == new


def test_lru_skips_sample_insufficient_via_projection(monkeypatch):
    """A cell whose dossier is sample_insufficient is excluded even if it's the
    least-recently-ticked in lru_by_cell."""
    lat = _load_lat()

    def fake_dossier(a, f, r):
        if r == "BEAR_STRESSED":
            return {"sample_sufficient": False}
        return {"sample_sufficient": True}

    monkeypatch.setattr(lat, "_load_dossier", fake_dossier)
    # BEAR_STRESSED never ticked (would be the LRU min "") but is ineligible.
    history = [
        {"asset": a, "frequency": f, "regime": rr, "last_ticked": "2026-03-01T00:00:00+00:00"}
        for a, f in ASSETS for rr in ("BULL_QUIET", "BULL_STRESSED", "BEAR_QUIET")
    ]
    state = _make_state(history, with_projection=True)
    asset, frequency, regime = lat._regime_lru_pick(state)
    assert regime != "BEAR_STRESSED"
