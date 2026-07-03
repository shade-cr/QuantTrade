"""Universe file contract: frozen M3 universe loads and validates (B0003)."""
import copy

import pytest
import yaml

from pipeline.equity_universe import load_universe


def _valid_payload() -> dict:
    """A complete, well-formed universe payload that passes every guard.

    Each parametrized malformed-input case below starts from a deep copy of
    this payload and mutates exactly one field, so it's guaranteed to hit the
    intended validation branch rather than short-circuiting on an earlier one
    (e.g. missing keys).
    """
    return {
        "selection_rule": "top-N by market cap at t0, frozen",
        "selected_at": "2020-01-01",
        "stocks": ["AAA", "BBB", "CCC"],
        "etfs": ["DDD", "EEE"],
        "alternates": ["FFF"],
        "excluded_delistees": {"LEH": "bankruptcy 2008"},
    }


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


def _mutate(mutator):
    payload = _valid_payload()
    mutator(payload)
    return payload


@pytest.mark.parametrize(
    "case_id, payload",
    [
        (
            "duplicate_tickers_in_stocks",
            _mutate(lambda p: p.__setitem__("stocks", ["AAA", "AAA", "BBB"])),
        ),
        (
            "ticker_overlap_stocks_etfs",
            _mutate(lambda p: p.__setitem__("etfs", ["AAA", "EEE"])),
        ),
        (
            "stocks_not_a_list",
            _mutate(lambda p: p.__setitem__("stocks", "AAA")),
        ),
        (
            "excluded_delistees_not_a_dict",
            _mutate(lambda p: p.__setitem__("excluded_delistees", ["LEH"])),
        ),
        (
            "selection_rule_blank",
            _mutate(lambda p: p.__setitem__("selection_rule", "   ")),
        ),
        (
            "missing_required_key",
            _mutate(lambda p: p.pop("selected_at")),
        ),
    ],
)
def test_load_universe_rejects_malformed(tmp_path, case_id, payload):
    p = tmp_path / f"u_{case_id}.yaml"
    p.write_text(yaml.safe_dump(copy.deepcopy(payload)), encoding="utf-8")
    with pytest.raises(ValueError):
        load_universe(p)
