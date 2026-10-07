"""sharing_config.py — build and validate NVIDIA k8s-device-plugin sharing configs (time-slicing, MPS) (T0).

Schema per the k8s-device-plugin README at the version the GPU Operator v26.7.1 ships (UNVERIFIED field names):
    version: v1
    sharing:
      timeSlicing:            # or `mps:`
        renameByDefault: false
        failRequestsGreaterThanOne: true
        resources:
          - name: nvidia.com/gpu
            replicas: 4
"""
from __future__ import annotations


def time_slicing(replicas: int, rename: bool = False) -> dict:
    return {"version": "v1", "sharing": {"timeSlicing": {"renameByDefault": rename, "failRequestsGreaterThanOne": True,
                                                          "resources": [{"name": "nvidia.com/gpu", "replicas": replicas}]}}}


def mps(replicas: int) -> dict:
    return {"version": "v1", "sharing": {"mps": {"resources": [{"name": "nvidia.com/gpu", "replicas": replicas}]}}}


def validate(cfg: dict) -> list[str]:
    errs = []
    if cfg.get("version") != "v1":
        errs.append("version must be v1")
    sharing = cfg.get("sharing") or {}
    modes = [k for k in ("timeSlicing", "mps") if k in sharing]
    if len(modes) != 1:
        errs.append("exactly one of sharing.timeSlicing / sharing.mps")
        return errs
    block = sharing[modes[0]]
    res = block.get("resources") or []
    if not res:
        errs.append("resources must be a non-empty list")
    for r in res:
        if r.get("name") != "nvidia.com/gpu":
            errs.append(f"unexpected resource {r.get('name')!r}")
        rep = r.get("replicas")
        if not isinstance(rep, int) or rep < 2:
            errs.append("replicas must be an integer >= 2 (1 means no sharing)")
        elif rep > 48:
            errs.append("replicas > 48 is almost certainly a mistake (UNVERIFIED MPS client limit)")
    if modes[0] == "timeSlicing" and not block.get("failRequestsGreaterThanOne", False):
        errs.append("set failRequestsGreaterThanOne: a pod asking for 2 shared 'GPUs' would get 2 slices of ONE GPU")
    return errs
