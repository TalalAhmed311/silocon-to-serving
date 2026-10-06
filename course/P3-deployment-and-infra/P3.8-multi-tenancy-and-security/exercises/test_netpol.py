"""P3.8 exercise 1 (static half): the NetworkPolicies say what the platform needs, and nothing more."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
POLS = {d["metadata"]["name"]: d for d in yaml.safe_load_all((ROOT / "platform/tenancy/networkpolicies.yaml").read_text()) if d}


def test_default_deny_both_directions():
    d = POLS["default-deny"]["spec"]
    assert d["podSelector"] == {} and set(d["policyTypes"]) == {"Ingress", "Egress"}
    assert "ingress" not in d and "egress" not in d


def test_backends_only_from_gateway_and_monitoring():
    ing = POLS["backends"]["spec"]["ingress"]
    sources = [f for rule in ing for f in rule["from"]]
    assert {"podSelector": {"matchLabels": {"app": "gateway"}}} in sources
    assert all("podSelector" in s or s["namespaceSelector"]["matchLabels"]["kubernetes.io/metadata.name"] == "monitoring" for s in sources)


def test_backend_egress_blocks_imds_and_private_ranges():
    eg = POLS["backends"]["spec"]["egress"]
    blocks = [t["ipBlock"] for rule in eg for t in rule["to"] if "ipBlock" in t]
    for b in blocks:
        assert "169.254.169.254/32" in b.get("except", []), "IMDS must be blocked"
        assert "10.0.0.0/8" in b.get("except", [])
    assert all(p["port"] == 443 for rule in eg for p in rule["ports"])


def test_backend_pods_carry_the_role_label():
    for f in ("vllm.yaml", "mockllm.yaml"):
        dep = next(d for d in yaml.safe_load_all((ROOT / "platform/deploy/base" / f).read_text()) if d and d["kind"] == "Deployment")
        assert dep["spec"]["template"]["metadata"]["labels"].get("s2s/role") == "backend", f
