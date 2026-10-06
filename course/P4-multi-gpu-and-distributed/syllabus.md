# P4 — Multi-GPU, GPU sharing, distributed jobs

**Weeks 21–24 · Lane A ≈ 44 h · Tier T3 (multi-GPU: a 4×L4 `g6.12xlarge`-class instance for TP practice (GPU count to verify against the EC2 types page), `p4d.24xlarge` or a Capacity Block for NVLink/EFA/MIG) with T0 simulators · Lane B: L3 (until week 24)**

**Thread through the phase:** communication is just another roofline. You predict collective time from message size and link bandwidth, measure it with `nccl-tests`, then choose TP/PP/DP/EP for serving and training with those numbers. Projects #9 and #11.

Pinned: NCCL `v2.32.3-1`, nccl-tests `v2.21.1`, torchtitan `v0.3.0`, awsome-distributed-training `d4325212` (SHA; latest tag is pre-reorg), Ray `2.59.0`.

**Cost rule:** P-instances are expensive, so every T3 lab is scripted to finish in one short session: launch → run the scripted benchmark → copy results to S3 → destroy. Labs default to a Capacity Block or spot where safe. Each `aws.md` states the expected wall-clock and $ (TODO(verify-prices) at build time).

| Module | Time | Project | Animations |
|---|---|---|---|
| P4.1 NCCL collectives and bandwidth math | 0.5 h + 10 h | — | **ring all-reduce step by step** |
| P4.2 Parallelism strategies (TP/PP/DP/EP, FSDP) | 0.6 h + 11 h | — | **TP split of an MLP across 2/4 GPUs** |
| P4.3 Distributed jobs, checkpointing, fault tolerance | 0.5 h + 12 h | **#9 checkpointed distributed training** | — |
| P4.4 GPU sharing: MIG, time-slicing, MPS | 0.4 h + 9 h | **#11 GPU partitioning lab** (stretch) | — |

---

## P4.1 — NCCL collectives and bandwidth math

**Objectives.** (1) Semantics of broadcast, reduce, all-reduce, reduce-scatter, all-gather and all-to-all. (2) Ring and tree algorithms and their α–β cost models. (3) Algorithm bandwidth vs **bus bandwidth**, following nccl-tests `doc/PERFORMANCE.md` at the pinned SHA. (4) Link math for PCIe, NVLink and EFA: predict all-reduce time for a TP layer's activation size. (5) Read `NCCL_DEBUG=INFO` topology output.

**Examples.** `01_collectives_numpy.py` (T0: simulate ring all-reduce on N fake ranks, verify against `sum`) · `02_torch_dist_gloo.py` (T0: the same collectives with the gloo backend on CPU processes) · `03_nccl_tests.sh` (T3: build and run `all_reduce_perf -b 8 -e 4G -f 2 -g <n>`).

**Exercises.** (1) Ring all-reduce simulator: correctness + step count = 2(N−1) (pytest) · (2) α–β model fitted to nccl-tests output · (3) busbw from algbw for each collective (formula tests matching PERFORMANCE.md) · (4) *(hard)* reduce-scatter + all-gather = all-reduce, shown in the simulator and in the timing.

**Bench.** nccl-tests output parsed into `size | time | algbw | busbw | % of link peak`. Expected busbw ranges are cited from AWS/NVIDIA docs or marked `TODO(run-on: p4d.24xlarge)`.

**Animation (required).** `animations/p4-ring-allreduce.html`: chunks travel around the ring through the reduce-scatter phase, then the all-gather phase.

**Sources.** NCCL, nccl-tests (pinned) · HF Ultra-Scale Playbook · How to Scale Your Model · Silicon to Scale ch 14 (networking) · AWS EFA docs.

## P4.2 — Parallelism strategies

**Objectives.** (1) Tensor parallelism: column- then row-split MLP (Megatron style) and its single all-reduce per block. (2) Pipeline parallelism: bubbles and microbatches. (3) Data parallelism and FSDP (ZeRO-3-style sharding), expert parallelism and all-to-all. (4) For serving: choose TP vs PP vs replicas from P1.5's capacity calculator plus P4.1's comm cost. (5) Run vLLM with `--tensor-parallel-size 2/4` and measure ITL against the prediction.

**Examples.** `01_tp_mlp_numpy.py` (T0: split an MLP across simulated ranks, verify vs the unsplit output) · `02_tp_torch_gloo.py` (T0) · `03_vllm_tp.sh` (T3) · `04_fsdp2_toy.py` (T3, torchtitan-style).

**Exercises.** (1) TP MLP numerics match unsplit at fp32 `atol=1e-5` (T0) · (2) PP bubble fraction formula vs simulation · (3) add TP to the D3 capacity calculator (comm time per token) · (4) *(hard)* measured vLLM TP=1/2/4 ITL vs the model's prediction, with the gaps explained.

**Animation (required).** `animations/p4-tp-mlp.html`: weight matrices split across 2/4 GPUs, activations flowing, the all-reduce point highlighted.

**Sources.** Ultra-Scale Playbook · How to Scale Your Model · torchtitan · handbook `data-tensor-pipeline-expert-hybrid-parallelism.md` · Silicon to Scale ch 10 (training), ch 13 (MoE).

## P4.3 — Distributed jobs, checkpointing, fault tolerance (+ #9)

**Objectives.** (1) Launch multi-node jobs with Slurm (ParallelCluster) **or** Ray on EKS; the course picks one primary path and documents the other. (2) PyTorch Distributed Checkpoint (DCP): sharded save/load and resharding on resume. (3) Fault injection: kill a rank, lose a node, then resume from the last checkpoint. Measure lost work and RTO. (4) **#9:** FSDP (+TP optional) training of a small model with periodic DCP checkpoints, fault injection and automatic resume.

**Examples.** `01_ddp_cpu.py` (T0, gloo) · `02_dcp_save_load.py` (T0: DCP works on CPU with gloo) · `03_slurm/` and `03_ray/` job specs (T3) · `04_fault_inject.sh`.

**Exercises.** (1) DCP round trip, bitwise-equal state (T0) · (2) resume with a different world size (resharding, T0 with 2→4 processes) · (3) checkpoint interval optimum (Young/Daly formula) vs simulation · (4) **#9** on T3, with an RTO/RPO table · (5) *(hard)* async checkpointing and its overhead.

**Sources.** torchtitan (FSDP2, checkpointing) · PyTorch DCP docs · awsome-distributed-training (Slurm/EKS + EFA) · Ray docs · AWS ParallelCluster docs.

## P4.4 — GPU sharing: MIG, time-slicing, MPS (+ #11, stretch)

**Objectives.** (1) MIG profiles on A100/H100 (memory and SM slices, hard isolation), time-slicing (no isolation) and MPS (concurrent kernels, partial isolation). (2) Configure each through the GPU Operator (MIG manager, device-plugin sharing configs). (3) **#11:** run small-model serving on each mode and measure throughput, tail latency under contention and fairness.

**Exercises.** (1) Choose a MIG layout for a tenant mix (T0 bin-packing function with tests) · (2) device-plugin time-slicing config passes schema checks (T0) · (3) **#11** contention matrix on T3 · (4) *(hard)* fair scheduler across MPS clients.

**Bench.** `mode | tenants | per-tenant tok/s | p99 ITL | Jain fairness index`. `TODO(run-on: p4d.24xlarge)`.

**Sources.** NVIDIA MIG user guide · GPU Operator MIG docs · k8s-device-plugin sharing docs · Silicon to Scale ch 16 (cluster management).
