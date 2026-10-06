"""make_prometheusrule.py — wrap rules/*.yaml (plain Prometheus format, unit-tested with promtool) into one PrometheusRule CR."""
from pathlib import Path

import yaml

here = Path(__file__).resolve().parent / "rules"
groups = []
for f in ("slo.yaml", "cost.yaml"):
    groups += yaml.safe_load((here / f).read_text())["groups"]
print(yaml.safe_dump({"apiVersion": "monitoring.coreos.com/v1", "kind": "PrometheusRule",
                      "metadata": {"name": "s2s", "namespace": "monitoring"}, "spec": {"groups": groups}}, sort_keys=False))
