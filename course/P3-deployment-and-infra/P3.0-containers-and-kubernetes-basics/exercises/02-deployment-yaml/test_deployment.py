import os
from pathlib import Path

import pytest
import yaml


def _path() -> Path:
    here = Path(__file__).resolve().parent
    if os.environ.get("S2S_SOLUTIONS") == "1":
        return here.parent / "solutions" / here.name / "deployment.yaml"
    return here / "deployment.yaml"


@pytest.fixture(scope="module")
def docs():
    return [d for d in yaml.safe_load_all(_path().read_text()) if d]


@pytest.fixture(scope="module")
def deployment(docs):
    found = [d for d in docs if d.get("kind") == "Deployment"]
    assert len(found) == 1, "write exactly one Deployment"
    return found[0]


@pytest.fixture(scope="module")
def service(docs):
    found = [d for d in docs if d.get("kind") == "Service"]
    assert len(found) == 1, "write exactly one Service"
    return found[0]


def container(deployment):
    return deployment["spec"]["template"]["spec"]["containers"][0]


def test_two_replicas(deployment):
    assert deployment["spec"]["replicas"] == 2


def test_selector_matches_pod_labels(deployment):
    sel = deployment["spec"]["selector"]["matchLabels"]
    labels = deployment["spec"]["template"]["metadata"]["labels"]
    assert sel, "selector.matchLabels must not be empty"
    assert all(labels.get(k) == v for k, v in sel.items()), "the pod template labels must contain every selector label"


def test_image_is_pinned(deployment):
    image = container(deployment)["image"]
    assert image and (":" in image.rsplit("/", 1)[-1] or "@sha256:" in image), "pin the image with a tag"
    assert not image.endswith(":latest")


def test_container_port_declared(deployment):
    ports = container(deployment).get("ports") or []
    assert [p["containerPort"] for p in ports] == [5678]


def test_readiness_probe_hits_the_container_port(deployment):
    probe = container(deployment).get("readinessProbe")
    assert probe and "httpGet" in probe, "add a readinessProbe with httpGet"
    assert probe["httpGet"]["port"] in (5678, "5678")
    assert probe["httpGet"]["path"].startswith("/")


def test_requests_and_limits_for_cpu_and_memory(deployment):
    res = container(deployment).get("resources") or {}
    for side in ("requests", "limits"):
        assert {"cpu", "memory"} <= set(res.get(side) or {}), f"resources.{side} needs cpu and memory"


def test_requests_do_not_exceed_limits(deployment):
    res = container(deployment)["resources"]

    def millicpu(v):
        v = str(v)
        return float(v[:-1]) if v.endswith("m") else float(v) * 1000

    units = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30}

    def mem(v):
        v = str(v)
        for u, k in units.items():
            if v.endswith(u):
                return float(v[: -len(u)]) * k
        return float(v)

    assert millicpu(res["requests"]["cpu"]) <= millicpu(res["limits"]["cpu"])
    assert mem(res["requests"]["memory"]) <= mem(res["limits"]["memory"])


def test_service_selects_the_same_pods(deployment, service):
    labels = deployment["spec"]["template"]["metadata"]["labels"]
    sel = service["spec"]["selector"]
    assert sel and all(labels.get(k) == v for k, v in sel.items())


def test_service_ports_route_to_the_container(deployment, service):
    port = service["spec"]["ports"][0]
    assert port["port"] == 80
    assert port["targetPort"] in (5678, "5678")
    assert service["spec"].get("type", "ClusterIP") == "ClusterIP", "a public Service type is not needed here"
