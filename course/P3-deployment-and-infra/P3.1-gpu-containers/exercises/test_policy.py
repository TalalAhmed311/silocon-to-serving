import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("pol", ROOT / "course/P3-deployment-and-infra/P3.1-gpu-containers/examples/02_dockerfile_policy.py")
pol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pol)


def test_linter_catches_each_rule():
    bad = "FROM python\nADD https://x/y.tar /\nENV HF_TOKEN=abc\nCOPY model.safetensors /m\n"
    msgs = " ".join(pol.check(bad))
    for rule in ("P1", "P2", "P3", "P4", "P5"):
        assert rule in msgs


def test_good_file_passes_strict():
    good = "FROM python:3.12-slim@sha256:" + "a" * 64 + "\nRUN true\nUSER 10001\n"
    assert pol.check(good, strict=True) == []


@pytest.mark.parametrize("f", sorted((ROOT / "env").glob("Dockerfile.*")), ids=lambda p: p.name)
def test_repo_dockerfiles_basic_policy(f):
    assert pol.check(f.read_text()) == [], "env/ Dockerfiles must pass the basic policy"


@pytest.mark.parametrize("f", sorted((ROOT / "env").glob("Dockerfile.*")), ids=lambda p: p.name)
def test_repo_dockerfiles_strict_policy(f):
    problems = pol.check(f.read_text(), strict=True)
    if problems:
        pytest.xfail("exercise 1: pin base images by digest — " + "; ".join(problems))
