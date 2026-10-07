#!/bin/bash
# #9 on Slurm (AWS ParallelCluster) — the documented alternative to Ray on EKS. Not the primary path; not CI-tested.
# Cluster: ParallelCluster with a p4d/g6.12xlarge queue + FSx for Lustre at /fsx (awsome-distributed-training d4325212
# has full ParallelCluster configs incl. EFA). Submit: sbatch platform/training/slurm/train.sbatch
#SBATCH --job-name=s2s-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00              # hard stop
#SBATCH --requeue                    # node failure → Slurm requeues; train.py resumes from the latest checkpoint
#SBATCH --output=/fsx/logs/%x-%j.out
set -euo pipefail
export NCCL_DEBUG=WARN
# EFA (p4d/p5): FI_PROVIDER=efa and the aws-ofi-nccl plugin from the DLAMI / the awsome-distributed-training image.
head=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
srun torchrun --nnodes "$SLURM_NNODES" --nproc-per-node 4 --rdzv-backend c10d --rdzv-endpoint "$head:29500" \
  --rdzv-id "$SLURM_JOB_ID" /fsx/silocon-to-serving/platform/training/train.py \
  --device cuda --dim 1024 --layers 12 --heads 16 --seq 512 --bs 8 --steps 2000 --ckpt /fsx/ckpt/s2s --ckpt-every 100
