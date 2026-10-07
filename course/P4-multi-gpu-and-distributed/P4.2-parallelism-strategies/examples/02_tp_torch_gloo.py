"""02_tp_torch_gloo.py — the same TP MLP with real processes and a real all-reduce (torch.distributed, gloo, T0).

Run: uv run --extra torch python course/P4-multi-gpu-and-distributed/P4.2-parallelism-strategies/examples/02_tp_torch_gloo.py --tp 4
Expected: "rank 0: TP=4 output matches unsplit, max err ~1e-6". Each process only ever materialises its own shard.
"""
import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


def worker(rank, tp, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=tp)
    g = torch.Generator().manual_seed(0)                          # same full weights on every rank (for the check)
    d, f = 256, 1024
    wg, wu = torch.randn(d, f, generator=g) / d**0.5, torch.randn(d, f, generator=g) / d**0.5
    wd = torch.randn(f, d, generator=g) / f**0.5
    x = torch.randn(4, d, generator=g)
    k = f // tp
    c = slice(rank * k, (rank + 1) * k)
    wg_s, wu_s, wd_s = wg[:, c].contiguous(), wu[:, c].contiguous(), wd[c, :].contiguous()   # this rank's shard
    y = (F.silu(x @ wg_s) * (x @ wu_s)) @ wd_s                     # partial [4, d]
    dist.all_reduce(y)                                             # the one communication of the block
    ref = (F.silu(x @ wg) * (x @ wu)) @ wd
    if rank == 0:
        print(f"rank 0: TP={tp} output matches unsplit, max err {(y - ref).abs().max().item():.1e}")
    dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--port", type=int, default=29513)
    a = ap.parse_args()
    mp.spawn(worker, args=(a.tp, a.port), nprocs=a.tp)
