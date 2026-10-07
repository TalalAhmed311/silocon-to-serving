# training: #9, checkpointed distributed training

| File | What |
|---|---|
| `train.py` | a tiny transformer LM trained with FSDP2. DCP checkpoints every N steps, automatic resume from the latest **committed** checkpoint, fault injection (`--die-at-step`), deterministic data so a resumed run matches an uninterrupted one |
| `checkpoint.py` | DCP save/load, the `COMMITTED` marker (written by rank 0 after a barrier), latest-checkpoint discovery, pruning |
| `faults.py` | Young/Daly optimal checkpoint interval and a Monte-Carlo failure simulator |
| `rto.py` | RPO (work lost) and RTO (time to recover) from the logs of a failed run and its resumed run |
| `ray/` | **primary multi-node path**: a KubeRay `RayJob` plus a Ray Train launcher with `FailureConfig` restarts |
| `slurm/` | the documented alternative: a ParallelCluster `sbatch` with `--requeue` |

Built in [P4.3](../../course/P4-multi-gpu-and-distributed/P4.3-distributed-jobs-and-checkpointing/README.md). Infra: [`infra/aws/distributed`](../../infra/aws/distributed/README.md).
