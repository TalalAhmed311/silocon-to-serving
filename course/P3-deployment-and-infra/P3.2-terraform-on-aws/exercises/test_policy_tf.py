"""Exercise 3 — policy tests over the Terraform source (T0, no terraform binary needed).

These are deliberately simple text checks; a real setup uses OPA/conftest or checkov custom policies. Extend them:
the TODO test is skipped until you write it.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SINGLE = ROOT / "infra/aws/single-node"
EKS = ROOT / "infra/aws/eks"


def tf(dirpath: Path) -> str:
    return "\n".join(p.read_text() for p in sorted(dirpath.glob("*.tf")))


def test_single_node_has_no_ingress_blocks():
    assert not re.search(r"^\s*ingress\s*\{", tf(SINGLE), re.M)


def test_single_node_requires_imdsv2():
    assert re.search(r'http_tokens\s*=\s*"required"', tf(SINGLE))


def test_root_volume_encrypted():
    assert re.search(r"encrypted\s*=\s*true", tf(SINGLE))


def test_eks_api_not_open_to_world():
    src = tf(EKS) + (EKS / "terraform.tfvars.example").read_text()
    assert "0.0.0.0/0" not in src
    assert "endpoint_public_access_cidrs" in tf(EKS)


def test_gpu_nodes_are_tainted():
    assert re.search(r'key\s*=\s*"nvidia.com/gpu"', tf(EKS))


def test_no_hardcoded_secrets():
    pat = re.compile(r'(secret|token|password)\s*=\s*"[^"$]{8,}"', re.I)
    for d in (SINGLE, EKS):
        assert not pat.search(tf(d)), f"literal secret-looking value in {d}"


def test_your_policy_iam_has_no_wildcard_actions():
    pytest.skip("exercise 3: write a check that no IAM statement uses actions = [\"*\"] or \"s3:*\"; delete this line")
