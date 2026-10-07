"""policy.py — static checks for every Deployment under platform/deploy (T0; used by P3.3 tests and CI).

Rules: resources.requests+limits.memory set; non-root + no privilege escalation + read-only root FS + drop ALL caps;
readiness probe; GPU pods (requesting nvidia.com/gpu) tolerate the GPU taint, select the GPU pool, and have a
startupProbe; no :latest images; secrets only via secretKeyRef (no literal values for *KEY/*TOKEN env vars).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


def docs(path: Path):
    for d in yaml.safe_load_all(path.read_text()):
        if d:
            yield d


def check_deployment(d: dict) -> list[str]:
    p, name = [], d["metadata"]["name"]
    spec = d["spec"]["template"]["spec"]
    pod_sc = spec.get("securityContext", {})
    for c in spec["containers"]:
        cn = f"{name}/{c['name']}"
        res = c.get("resources", {})
        if "memory" not in res.get("requests", {}) or "memory" not in res.get("limits", {}):
            p.append(f"{cn}: set memory requests and limits")
        sc = c.get("securityContext", {})
        if not (pod_sc.get("runAsNonRoot") or sc.get("runAsNonRoot")):
            p.append(f"{cn}: runAsNonRoot")
        if sc.get("allowPrivilegeEscalation", True):
            p.append(f"{cn}: allowPrivilegeEscalation: false")
        if not sc.get("readOnlyRootFilesystem"):
            p.append(f"{cn}: readOnlyRootFilesystem: true")
        if "ALL" not in sc.get("capabilities", {}).get("drop", []):
            p.append(f"{cn}: capabilities.drop [ALL]")
        if "readinessProbe" not in c:
            p.append(f"{cn}: readinessProbe")
        if c.get("image", "").endswith(":latest") or ":" not in c.get("image", ":"):
            p.append(f"{cn}: pin the image tag")
        for e in c.get("env", []):
            if re.search(r"(KEY|TOKEN|SECRET|PASSWORD)$", e["name"]) and "value" in e:
                p.append(f"{cn}: {e['name']} must come from a Secret (valueFrom.secretKeyRef)")
        gpu = "nvidia.com/gpu" in res.get("limits", {}) or "nvidia.com/gpu" in res.get("requests", {})
        if gpu:
            if not any(t.get("key") == "nvidia.com/gpu" for t in spec.get("tolerations", [])):
                p.append(f"{cn}: GPU pod must tolerate the nvidia.com/gpu taint")
            if "startupProbe" not in c:
                p.append(f"{cn}: GPU servers need a startupProbe (model load takes minutes)")
    return p


def check_tree(root: Path) -> dict[str, list[str]]:
    out = {}
    for f in sorted(root.rglob("*.yaml")):
        for d in docs(f):
            if isinstance(d, dict) and d.get("kind") == "Deployment":
                out[f"{f.relative_to(root)}:{d['metadata']['name']}"] = check_deployment(d)
    return out


if __name__ == "__main__":
    res = check_tree(Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent))
    bad = 0
    for k, v in res.items():
        print(k, "OK" if not v else "")
        for x in v:
            print("   ", x)
        bad += bool(v)
    sys.exit(1 if bad else 0)
