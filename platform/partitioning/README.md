# partitioning: #11, the GPU partitioning lab (stretch)

| File | What |
|---|---|
| `mig.py` | MIG profile tables (A100-40/80GB, H100-80GB; UNVERIFIED against the MIG guide), a first-fit-decreasing layout planner, and mig-parted config output |
| `sharing_config.py` | build and validate k8s-device-plugin **time-slicing** and **MPS** configs |
| `fairness.py` | Jain's fairness index (plain and share-weighted) and the contention table row |
| `contention.sh` | T3: N vLLM servers on one GPU under time-slicing, MPS or MIG, loaded at once by #4 |
| `cli.py` | `mig` (plan a layout for a tenant mix) and `table` (contention results → the bench row) |
| `k8s/` | GPU Operator ConfigMaps for time-slicing/MPS and for the MIG manager |

Built in [P4.4](../../course/P4-multi-gpu-and-distributed/P4.4-gpu-sharing/README.md).
