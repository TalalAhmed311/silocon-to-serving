# P4.3 examples

| File | Tier | What |
|---|---|---|
| [`01_ddp_cpu.py`](01_ddp_cpu.py) | T0 | DDP with gloo: replicas stay identical because gradients are all-reduced |
| [`02_dcp_save_load.py`](02_dcp_save_load.py) | T0 | a DCP round trip of model + optimizer state, bitwise equal, and what's on disk |
| [`03_ray/`](03_ray/README.md) | T3 | #9 as a RayJob on EKS (primary path) |
| [`03_slurm/`](03_slurm/README.md) | T3 | #9 on Slurm/ParallelCluster (documented alternative) |
| [`04_fault_inject.sh`](04_fault_inject.sh) | T0 / T3 | kill a rank or a pod mid-run, resume, and print the RPO/RTO table |
| [`platform/training/`](../../../../platform/training/README.md) | T0 → T3 | **#9** itself |
