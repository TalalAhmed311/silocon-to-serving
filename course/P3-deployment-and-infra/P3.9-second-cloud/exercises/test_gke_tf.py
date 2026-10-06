"""Static policy checks on infra/gcp/gke (T0, no terraform binary needed)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
D = ROOT / "infra/gcp/gke"
MAIN = (D / "main.tf").read_text()
ALL = "\n".join(p.read_text() for p in D.glob("*.tf"))


def blocks(kind):
    """Bodies of `resource "<kind>" "<name>" { ... }` (brace-matched)."""
    out = []
    for m in re.finditer(rf'resource "{kind}" "(\w+)" \{{', MAIN):
        i, depth = m.end(), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(MAIN[i], 0)
            i += 1
        out.append((m[1], MAIN[m.end():i - 1]))
    return out


def test_private_nodes_and_authorized_networks():
    (_, c), = blocks("google_container_cluster")
    assert re.search(r"enable_private_nodes\s*=\s*true", c)
    assert "master_authorized_networks_config" in c
    assert "0.0.0.0/0" not in ALL


def test_gpu_pool_scales_to_zero():
    pools = dict(blocks("google_container_node_pool"))
    gpu = [b for b in pools.values() if "guest_accelerator" in b]
    assert gpu, "no GPU pool"
    for b in gpu:
        assert re.search(r"min_node_count\s*=\s*0\b", b)


def test_every_pool_hides_node_metadata():
    for name, b in blocks("google_container_node_pool"):
        assert 'mode = "GKE_METADATA"' in b, name


def test_budget_and_teardown():
    assert blocks("google_billing_budget")
    assert re.search(r"deletion_protection\s*=\s*false", MAIN)


def test_no_secrets_in_tfvars_example():
    ex = (D / "terraform.tfvars.example").read_text()
    assert not re.search(r"(?i)(secret|token|password|private_key)\s*=", ex)
