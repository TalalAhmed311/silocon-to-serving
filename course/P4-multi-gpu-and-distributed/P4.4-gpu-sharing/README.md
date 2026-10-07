# P4.4: GPU sharing (MIG, time-slicing, MPS) (+ #11, stretch)

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for MIG layout planning, sharing-config validation and fairness math. ![T3](https://img.shields.io/badge/tier-T3%20A100%2FH100-red) for the contention matrix (MIG needs an A100/H100-class GPU) |
| **Time** | ≈25 min reading + ≈9 h hands-on |
| **Prerequisites** | P3.3 (GPU Operator), P3.8 (tenancy models), P2.4 (#4 load testing) |
| **You will build** | **#11**: a MIG layout planner, validated sharing configs, and a measured contention matrix with fairness |

## Learning objectives

1. Compare **MIG** (hardware partitions), **time-slicing** (no isolation) and **MPS** (concurrent kernels, partial isolation) on isolation, utilization and failure blast radius.
2. Configure each one through the GPU Operator: MIG manager, device-plugin sharing configs.
3. Pick a MIG layout for a tenant mix, and know where the simple model breaks (placement rules).
4. Measure throughput, tail latency and **Jain's fairness index** under contention.

---

## 1. Three ways to share one GPU

| | MIG | time-slicing | MPS |
|---|---|---|---|
| what's shared | nothing: separate SMs, L2 slices, memory and copy engines per instance | the whole GPU, taking turns (context switches) | the whole GPU, with kernels from several processes at the same time |
| memory isolation | **yes** (hardware) | no: one pod can OOM the others | partial (per-client limits possible) |
| fault isolation | yes, per instance | no | no: a fault can take down all clients |
| performance isolation | yes: fixed SMs and bandwidth share | no: tail latency suffers under contention | partial (thread percentage limits) |
| utilization | can strand capacity if tenants don't fit the profiles | high | highest for small kernels |
| GPUs | A100, H100 and later (data-center) | any | any (Volta+) |
| K8s resource | `nvidia.com/mig-3g.40gb` (mixed strategy) | `nvidia.com/gpu` × replicas | `nvidia.com/gpu` × replicas |

A rule of thumb for serving: **MIG** for multi-tenant isolation with known, steady models. **MPS** for many small models from one trust domain. **Time-slicing** only for dev and bursty, latency-tolerant work.

## 2. MIG in slices

An A100/H100 has **7 compute slices** and **8 memory slices**. A profile `Ng.Mgb` takes N compute slices and a fixed number of memory slices (`3g.40gb` takes 4 of 8 on an 80 GB card). Profiles must also start at allowed positions, so two `3g` and one `1g` fit, but some other combinations don't. `platform/partitioning/mig.py` plans with the slice totals only and says so. Tables come from the NVIDIA MIG User Guide (UNVERIFIED until you check them against your driver's guide).

## 3. Measuring contention

For each mode, run N tenants (one vLLM server each) at the same time with #4 and record per-tenant tok/s and p99 ITL. Fairness:

```
Jain(x) = (Σ xᵢ)² / (n · Σ xᵢ²)        1 = perfectly even, 1/n = one tenant gets everything
```

Weight by entitlement (`xᵢ / shareᵢ`) when tenants pay for different shares.

> **Predict first.** 4 tenants, each serving a 1B-class model, on one A100-80GB. Rank the modes (exclusive × 4 GPUs as the baseline, 4 × `1g.10gb` MIG, time-slicing × 4, MPS × 4) by (a) aggregate tok/s, (b) p99 ITL, (c) Jain when one tenant floods. Write the ranking down before running.

---

## Walkthrough

```bash
uv run pytest course/P4-multi-gpu-and-distributed/P4.4-gpu-sharing/exercises
PYTHONPATH=platform uv run python -m partitioning.cli mig --gpu A100-80GB --tenant chat:35:3 --tenant rerank:9 --tenant embed:8 --tenant small:18
```

The T3 path is in [aws.md](aws.md).

## What you should see

- The planner prints the GPU count, each GPU's profiles with slice usage, and a mig-parted YAML.
- On a p4d (`TODO(run-on: p4d.24xlarge)`): MIG has the flattest p99 ITL and Jain near 1 under a flood. Time-slicing has the worst tails. MPS has the highest aggregate throughput with small models and the lowest fairness without limits.

## Bench

| mode | tenants | per-tenant tok/s | p99 ITL | Jain fairness index |
|---|---|---|---|---|
| (see exercise 3) | | | | `TODO(run-on: p4d.24xlarge)` |

## Exercises

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [Choose a MIG layout for a tenant mix](exercises/01-mig-layout.md) | T0 | `test_layout.py` |
| 2 | [Time-slicing and MPS configs pass schema checks](exercises/02-sharing-config.md) | T0 | `test_sharing_config.py` |
| 3 | [#11 contention matrix](exercises/03-contention.md) | T3 | your table |
| 4 | *(hard)* [A fair scheduler across MPS clients](exercises/04-fair-mps.md) | T3 | share-weighted Jain before and after |

## Common mistakes

- Treating time-slicing as isolation. It doesn't isolate memory, faults or performance.
- Asking for `nvidia.com/gpu: 2` on a time-sliced node and getting two slices of one GPU.
- Choosing MIG profiles by memory only. A tenant's SLO may need more compute slices than its memory implies.
- Forgetting MIG reconfiguration drains the GPU: every pod on it is evicted.
- Reporting average throughput only. Contention shows up in p99 and in fairness.

## Go deeper

- NVIDIA MIG User Guide (profiles and placement) and MPS docs. GPU Operator docs: MIG manager, time-slicing.
- `NVIDIA/k8s-device-plugin` README (sharing configs). mig-parted README.
- Silicon to Scale ch. 16 (cluster management). Jain, Chiu, Hawe (1984), the fairness index.

**Next:** [P5 CUDA deep dive](../../P5-cuda-deep-dive/syllabus.md).
