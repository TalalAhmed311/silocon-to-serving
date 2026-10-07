import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("policy", ROOT / "platform/deploy/policy.py")
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def test_all_deployments_pass_policy():
    res = policy.check_tree(ROOT / "platform/deploy")
    assert res, "no Deployments found"
    bad = {k: v for k, v in res.items() if v}
    assert not bad, bad


def test_policy_catches_a_bad_gpu_pod():
    d = {"metadata": {"name": "x"}, "spec": {"template": {"spec": {"containers": [
        {"name": "c", "image": "img:latest", "resources": {"limits": {"nvidia.com/gpu": 1}},
         "env": [{"name": "HF_TOKEN", "value": "hf_abc"}]}]}}}}
    msgs = " ".join(policy.check_deployment(d))
    for needle in ("memory", "runAsNonRoot", "readinessProbe", "pin the image", "Secret", "tolerate", "startupProbe"):
        assert needle in msgs, needle
