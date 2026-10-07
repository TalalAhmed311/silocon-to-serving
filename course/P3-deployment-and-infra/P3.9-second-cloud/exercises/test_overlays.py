from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
OV = ROOT / "platform/deploy/overlays"


def load(cloud):
    return yaml.safe_load((OV / cloud / "kustomization.yaml").read_text())


def test_same_resources_only_images_differ():
    eks, gke = load("eks"), load("gke")
    assert eks["resources"] == gke["resources"]
    assert eks["configMapGenerator"] == gke["configMapGenerator"]
    allowed = {"apiVersion", "kind", "resources", "images", "configMapGenerator"}
    for k in (eks, gke):
        assert set(k) <= allowed, set(k) - allowed
    assert [i["name"] for i in eks["images"]] == [i["name"] for i in gke["images"]]
    assert eks["images"][0]["newTag"] == gke["images"][0]["newTag"], "same engine version on both clouds"


def test_same_gateway_config():
    strip = lambda p: [ln for ln in p.read_text().splitlines() if not ln.startswith("#")]  # noqa: E731
    assert strip(OV / "eks/gateway.yaml") == strip(OV / "gke/gateway.yaml")


def test_gpu_pool_label_is_set_by_both_terraform_roots():
    assert '"s2s/pool" = "gpu"' in (ROOT / "infra/gcp/gke/main.tf").read_text()
    eks = (ROOT / "infra/aws/eks/main.tf").read_text()
    assert "s2s/pool" in eks
