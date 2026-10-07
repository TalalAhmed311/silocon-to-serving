# P4.3: Distributed jobs, checkpointing, fault tolerance (+ #9)

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for DDP/FSDP on CPU (gloo), DCP round trips, resharding 2→4, fault injection with resume, and checkpoint-interval math. ![T3](https://img.shields.io/badge/tier-T3%20EKS%20%2B%20GPUs-red) for #9 as a RayJob on EKS with real failures |
| **Time** | ≈30 min reading + ≈12 h hands-on |
| **Prerequisites** | P4.1–P4.2 (collectives, FSDP), P3.2–P3.3 (EKS, K8s) |
| **You will build** | **#9**: FSDP2 training with DCP checkpoints, automatic resume, fault injection, and a measured RTO/RPO table |

**Primary path: Ray on EKS** (KubeRay `RayJob`), chosen because it reuses the P3 cluster, identity and observability. **Slurm on ParallelCluster** is documented as the alternative. Both run the same `platform/training/train.py`.

## Learning objectives

1. Launch multi-process and multi-node training with `torchrun`, Ray Train, or Slurm `srun`, and know what each one sets up (rendezvous, ranks, restarts).
2. Save and load **sharded** state with PyTorch Distributed Checkpoint (DCP), including resharding to a different world size.
3. Make checkpoints **crash-consistent**: commit markers, barriers, pruning that never deletes the only good copy.
4. Inject failures and measure **RPO** (work lost) and **RTO** (time to recover). Choose the checkpoint interval with Young/Daly.

---

## 1. Launchers

| | torchrun | Ray Train on KubeRay | Slurm + torchrun |
|---|---|---|---|
| who starts processes | torchrun's elastic agent, per node | Ray workers (actors) in pods | `srun`, per node |
| rendezvous | `--rdzv-backend c10d` | Ray sets up the process group | first host of the allocation |
| on failure | `--max-restarts` restarts the local group | `FailureConfig(max_failures)` restarts the worker group. KubeRay replaces lost pods | `--requeue` resubmits the job |
| fits with | anything | the P3 EKS platform | HPC clusters, FSx for Lustre |

The training code shouldn't care. `train.py` reads `RANK`/`WORLD_SIZE` through `init_process_group` and **always resumes from the latest committed checkpoint**, so "restart it" is the whole recovery procedure under every launcher.

## 2. DCP: sharded, reshardable checkpoints

FSDP2 parameters and optimizer states are DTensors: each rank holds a shard. `dcp.save(state, checkpoint_id=dir)` writes one file per rank plus a `.metadata` file that records every tensor's global shape and where each shard lives. `dcp.load` reads whichever byte ranges the *current* layout needs. Save on 2 ranks, load on 4: exercise 2 does exactly that. `get_state_dict` and `set_state_dict` (from `torch.distributed.checkpoint.state_dict`) produce and consume those sharded dicts for model plus optimizer.

**Crash consistency** is your job, not DCP's. `platform/training/checkpoint.py` writes `COMMITTED` from rank 0 only **after a barrier**, so a checkpoint is visible to resume only once every rank's shard is on disk. Pruning keeps the newest `keep` committed checkpoints and never deletes a newer in-progress one.

## 3. Failures, RPO, RTO

- **RPO** (recovery point): steps since the last committed checkpoint. On average it's τ/2 for interval τ.
- **RTO** (recovery time): detect + reschedule + restart + load + warm-up. Process crashes recover in seconds. Node loss waits for a new instance (minutes).
- **Interval**: `τ* ≈ √(2·C·M)` (Young). For a fixed per-GPU failure rate, M shrinks linearly with the number of GPUs, so big jobs checkpoint more often. Async checkpointing cuts the blocking part of C.

> **Predict first.** #9 on 4 GPUs: a checkpoint takes C = 10 s, and you checkpoint every 100 steps at 0.5 s/step. A process crash at step 1,337: what is the RPO in steps and in seconds? If the pod is rescheduled in 20 s and loading takes 15 s, what's the RTO? Run the exercise and compare.

---

## Walkthrough

```bash
uv run --extra torch torchrun --nproc-per-node 2 course/P4-multi-gpu-and-distributed/P4.3-distributed-jobs-and-checkpointing/examples/01_ddp_cpu.py
uv run --extra torch python course/P4-multi-gpu-and-distributed/P4.3-distributed-jobs-and-checkpointing/examples/02_dcp_save_load.py
uv run --extra torch bash course/P4-multi-gpu-and-distributed/P4.3-distributed-jobs-and-checkpointing/examples/04_fault_inject.sh local
uv run --extra torch pytest course/P4-multi-gpu-and-distributed/P4.3-distributed-jobs-and-checkpointing/exercises       # add -m slow for resharding
```

The T3 path is in [aws.md](aws.md).

## What you should see

- `01`: both ranks print the same parameter checksum at every logged step.
- `02`: `DCP round trip: model and optimizer state bitwise equal (…)` and the checkpoint files: `__0_0.distcp` + `.metadata`.
- `04 local`: run 1 dies at step 37. Run 2 logs `"resumed_from": 19`, and the table shows an RPO of 17 steps.
- On EKS (`TODO(run-on: EKS g6.12xlarge)`): the RayJob restarts after a pod deletion and continues from the last commit. Node loss takes minutes longer, mostly waiting for the new node.

## Exercises

| # | Exercise | Tier | Test |
|---|---|---|---|
| 1 | [DCP round trip, bitwise-equal state; commit markers](exercises/01-dcp.md) | T0 | `test_dcp.py`, `test_checkpoint_logic.py` |
| 2 | [Resume with a different world size (2 → 4)](exercises/02-reshard.md) | T0 (slow) | `test_reshard.py` |
| 3 | [Optimal checkpoint interval: Young/Daly vs simulation](exercises/03-interval.md) | T0 | `test_interval.py` |
| 4 | [#9 on T3, with an RTO/RPO table](exercises/04-project-9.md) | T3 | the table + `training.rto` |
| 5 | *(hard)* [Async checkpointing and its overhead](exercises/05-async.md) | T0 + T3 | your measurements |

## Common mistakes

- Treating a checkpoint directory as valid because it exists. Use a commit marker after a barrier.
- Seeding the data loader by rank. Resuming at a different world size then changes the data order.
- Saving only the model, not the optimizer, LR schedule, step and RNG state. The resumed run then diverges.
- Checkpoints on the node's local disk: node loss takes them with it. Use shared storage or S3.
- One global checkpoint interval for every job size. τ* shrinks as the job grows.

## Go deeper

- PyTorch DCP docs (`torch.distributed.checkpoint`, `state_dict` helpers, `async_save`). torchrun/elastic docs.
- torchtitan `v0.3.0`: FSDP2 + checkpointing at production scale.
- Ray Train docs (`TorchTrainer`, `FailureConfig`), KubeRay `RayJob`. awsome-distributed-training `d4325212` (EKS and ParallelCluster with EFA). AWS ParallelCluster docs.
- Young (1974) and Daly (2006) on checkpoint intervals. Silicon to Scale ch. 10, 16.

**Next:** [P4.4 GPU sharing: MIG, time-slicing, MPS](../P4.4-gpu-sharing/README.md).
