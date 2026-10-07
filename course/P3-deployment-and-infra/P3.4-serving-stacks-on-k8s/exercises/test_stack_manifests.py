"""T0: the P3.4 stack manifests parse and keep the GPU scheduling contract (taint toleration, GPU pool, GPU limits)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4] / "platform/deploy"


def test_kserve_isvc():
    d = yaml.safe_load((ROOT / "kserve/inferenceservice.yaml").read_text())
    p = d["spec"]["predictor"]
    assert any(t["key"] == "nvidia.com/gpu" for t in p["tolerations"])
    assert p["model"]["resources"]["limits"]["nvidia.com/gpu"] in ("1", 1)


def test_rayservice_gpu_group():
    d = yaml.safe_load((ROOT / "rayserve/rayservice.yaml").read_text())
    wg = d["spec"]["rayClusterConfig"]["workerGroupSpecs"][0]
    c = wg["template"]["spec"]["containers"][0]
    assert c["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert wg["minReplicas"] == 0
