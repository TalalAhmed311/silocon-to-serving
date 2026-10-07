"""04_fsdp2_toy.py — FSDP2 (`fully_shard`) on a toy transformer: memory per rank vs DDP (T3; runs on CPU/gloo for a
smoke test if your torch build supports it — results there show sharding, not speed).

Run (GPU box): torchrun --nproc-per-node 4 course/P4-multi-gpu-and-distributed/P4.2-parallelism-strategies/examples/04_fsdp2_toy.py
Expected: per-rank parameter bytes ≈ total / world (sharded), loss decreasing; compare peak CUDA memory with --ddp.
The FSDP2 API (torch.distributed.fsdp.fully_shard) is per torch 2.6+ docs (UNVERIFIED for your version); torchtitan
v0.3.0 is the production-grade reference for FSDP2 + TP + checkpointing.
"""
import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddp", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(dev)
    torch.manual_seed(0)
    layers = [nn.TransformerEncoderLayer(512, 8, 2048, batch_first=True) for _ in range(8)]
    model = nn.Sequential(*layers).to(dev)
    if a.ddp:
        model = nn.parallel.DistributedDataParallel(model)
    else:
        from torch.distributed.fsdp import fully_shard
        for layer in layers:
            fully_shard(layer)              # each layer's params become DTensor shards; all-gathered just in time
        fully_shard(model)
    local = sum(p.to_local().numel() if hasattr(p, "to_local") else p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(a.steps):
        x = torch.randn(8, 128, 512, device=dev)
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        if rank == 0 and step % 5 == 0:
            print(f"step {step} loss {loss.item():.4f}")
    if rank == 0:
        mem = f", peak CUDA {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB" if cuda else ""
        print(f"{'DDP' if a.ddp else 'FSDP2'}: local params per rank = {local / 1e6:.2f} M (world {world}){mem}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
