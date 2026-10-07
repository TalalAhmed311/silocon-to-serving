# 03_slurm: #9 on Slurm (documented alternative, T3)

[`platform/training/slurm/train.sbatch`](../../../../../platform/training/slurm/train.sbatch) runs the same `train.py` with `torchrun` under `srun`, with `--requeue` so Slurm resubmits after a node failure. To build the cluster, use AWS ParallelCluster with FSx for Lustre, following awsome-distributed-training `d4325212` (its ParallelCluster templates include EFA, NCCL tests and health checks).

Why it isn't the primary path here: the course already runs EKS (P3), so Ray on the same cluster reuses the observability, autoscaling and identity work. Slurm is still the default on most large training clusters, so read the sbatch file line by line. `--requeue`, `--time` and the rendezvous over the first host are the parts that matter.
