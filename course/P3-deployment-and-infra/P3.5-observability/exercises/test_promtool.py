import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4] / "platform/observability/rules"


@pytest.mark.skipif(shutil.which("promtool") is None, reason="install promtool (Prometheus release) to run rule tests")
@pytest.mark.parametrize("test_file", sorted(p.name for p in ROOT.glob("*_test.yaml")))
def test_rules(test_file):
    r = subprocess.run(["promtool", "test", "rules", test_file], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(shutil.which("promtool") is None, reason="install promtool")
@pytest.mark.parametrize("rule_file", ["slo.yaml", "cost.yaml"])
def test_rules_are_valid(rule_file):
    r = subprocess.run(["promtool", "check", "rules", rule_file], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
