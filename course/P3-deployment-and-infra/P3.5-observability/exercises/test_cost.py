import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "platform"))
from cost.model import Node, Replica, breakeven_utilization, gpu_usd_per_hour, replica_usd_per_hour, tenant_costs, usd_per_million, waste  # noqa: E402

ONE = Node("fixture-1gpu", usd_per_hour=1.0, gpus=1)
EIGHT = Node("fixture-8gpu", usd_per_hour=32.0, gpus=8)


def test_token_share_allocation():
    r = Replica("r1", ONE, 1, {"a": 3_000_000, "b": 1_000_000})
    c = tenant_costs([r])
    assert c["a"]["usd_per_hour"] == pytest.approx(0.75) and c["b"]["usd_per_hour"] == pytest.approx(0.25)
    assert c["a"]["usd_per_million"] == pytest.approx(0.25) == c["b"]["usd_per_million"]


def test_tp_and_mig_shares():
    assert replica_usd_per_hour(Replica("tp2", EIGHT, 2, {})) == pytest.approx(8.0)
    assert replica_usd_per_hour(Replica("mig", EIGHT, 1 / 7, {})) == pytest.approx(4.0 / 7)


def test_spot_overhead():
    spot = Node("spot", usd_per_hour=0.4, gpus=1, spot=True, interruption_overhead=0.05)
    assert gpu_usd_per_hour(spot) == pytest.approx(0.42)


def test_waste_and_breakeven():
    r = Replica("r1", ONE, 1, {"a": 1_000_000})
    assert waste([r], {"r1": 4_000_000})["r1"] == pytest.approx(0.75)
    # full utilization: $1/h / 4M tok/h = $0.25/1M; vs an API at $1/1M -> break even at 25% utilization
    assert usd_per_million(1.0, 4_000_000) == pytest.approx(0.25)
    assert breakeven_utilization(1.0, 4_000_000, 1.0) == pytest.approx(0.25)
