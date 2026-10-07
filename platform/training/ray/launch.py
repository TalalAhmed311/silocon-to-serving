"""launch.py — run #9's training loop under Ray Train (TorchTrainer) with automatic restart on worker failure.

Ray Train owns process launch and torch.distributed setup; our train.main() still does its own DCP checkpointing and
resume-from-latest, so recovery works the same under torchrun, Slurm and Ray. Ray's FailureConfig(max_failures=N)
restarts the worker group after a failure; train.main() then finds the latest COMMITTED checkpoint and continues.
API names per Ray 2.59.0 docs (ray.train.torch.TorchTrainer, ScalingConfig, RunConfig, FailureConfig) — UNVERIFIED
against your installed version.

Note: checkpoints are written to local disk path --ckpt-local on shared storage (FSx/EFS) or synced to --storage (S3)
by rank 0 after each commit; the simple path for the course is a shared filesystem mount.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def train_fn(config):
    import torch.distributed as dist
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from training import train
    if dist.is_initialized():                       # Ray already initialised the process group; train.main does too
        dist.destroy_process_group()
    train.main(config["argv"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--storage", required=True, help="s3://bucket/prefix for Ray run state")
    ap.add_argument("--ckpt-local", default="/mnt/shared/ckpt", help="shared FS path for DCP checkpoints")
    ap.add_argument("--max-failures", type=int, default=3)
    a = ap.parse_args()
    from ray.train import FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer
    argv = ["--device", "cuda", "--dim", "1024", "--layers", "12", "--heads", "16", "--seq", "512", "--bs", "8",
            "--steps", str(a.steps), "--ckpt", a.ckpt_local, "--ckpt-every", "100"]
    trainer = TorchTrainer(
        train_fn, train_loop_config={"argv": argv},
        scaling_config=ScalingConfig(num_workers=a.workers, use_gpu=True),
        run_config=RunConfig(storage_path=a.storage, name="s2s-train", failure_config=FailureConfig(max_failures=a.max_failures)),
    )
    print(trainer.fit())


if __name__ == "__main__":
    main()
