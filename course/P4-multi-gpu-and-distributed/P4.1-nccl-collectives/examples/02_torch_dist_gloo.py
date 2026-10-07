"""02_torch_dist_gloo.py — the real collectives with torch.distributed on CPU processes (gloo backend, T0).

Run: uv run --extra torch python course/P4-multi-gpu-and-distributed/P4.1-nccl-collectives/examples/02_torch_dist_gloo.py --world 4
Expected: one line per collective with the result check and a small timing table (CPU loopback: the numbers only show
the shape — time grows ~linearly with size; latency dominates small messages).
Swap backend="nccl" and device="cuda" on a multi-GPU box (T3) and the same code exercises NVLink/PCIe/EFA.
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world: int, port: int):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    x = torch.full((8,), float(rank + 1))
    y = x.clone(); dist.all_reduce(y)                                         # noqa: E702
    assert torch.all(y == sum(range(1, world + 1)))
    g = [torch.empty(8) for _ in range(world)]; dist.all_gather(g, x)         # noqa: E702
    assert all(torch.all(g[r] == r + 1) for r in range(world))
    b = x.clone(); dist.broadcast(b, src=0)                                   # noqa: E702
    assert torch.all(b == 1)   # (reduce_scatter: gloo support varies by version — exercised with NCCL in 03 instead)
    if rank == 0:
        print("all_reduce / all_gather / broadcast OK")
        print("| bytes | all_reduce ms |\n|---|---|")
    for lg in range(10, 25, 2):
        t = torch.ones(1 << (lg - 2))
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(5):
            dist.all_reduce(t)
        dist.barrier()
        if rank == 0:
            print(f"| {1 << lg} | {(time.perf_counter() - t0) / 5 * 1e3:.3f} |")
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--port", type=int, default=29512)
    a = ap.parse_args()
    mp.spawn(worker, args=(a.world, a.port), nprocs=a.world)
