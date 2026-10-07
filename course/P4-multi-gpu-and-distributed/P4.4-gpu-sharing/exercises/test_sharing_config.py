"""Exercise 2: device-plugin sharing configs pass schema checks (T0)."""
from pathlib import Path

import yaml

from partitioning import fairness, sharing_config as sc

ROOT = Path(__file__).resolve().parents[4]


def test_generated_configs_valid():
    assert sc.validate(sc.time_slicing(4)) == []
    assert sc.validate(sc.mps(4)) == []


def test_validator_catches_mistakes():
    bad = sc.time_slicing(1)
    assert any("replicas" in e for e in sc.validate(bad))
    bad = sc.time_slicing(4)
    bad["sharing"]["timeSlicing"]["failRequestsGreaterThanOne"] = False
    assert any("failRequestsGreaterThanOne" in e for e in sc.validate(bad))
    both = {"version": "v1", "sharing": {**sc.time_slicing(2)["sharing"], **sc.mps(2)["sharing"]}}
    assert sc.validate(both)


def test_repo_configmap_entries_valid():
    cm = yaml.safe_load((ROOT / "platform/partitioning/k8s/time-slicing.yaml").read_text())
    for key, body in cm["data"].items():
        assert sc.validate(yaml.safe_load(body)) == [], key


def test_jain():
    assert fairness.jain([10, 10, 10, 10]) == 1.0
    assert abs(fairness.jain([40, 0, 0, 0]) - 0.25) < 1e-12
    assert fairness.jain_weighted([30, 10], [3, 1]) == 1.0      # 3:1 shares, 3:1 throughput → fair
