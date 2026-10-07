"""train.py — #9: a small transformer LM trained with FSDP2, checkpointed with DCP, resumable after any failure.

T0 (CPU, gloo):  torchrun --nproc-per-node 2 platform/training/train.py --steps 60 --ckpt-every 20 --ckpt /tmp/ckpt
T3 (GPUs):       torchrun --nproc-per-node 4 platform/training/train.py --device cuda --dim 1024 --layers 12 ...
                 or as a RayJob on EKS (ray/rayjob.yaml) — the primary multi-node path; Slurm in slurm/train.sbatch.
Resume is automatic: on start, every rank loads the latest COMMITTED checkpoint under --ckpt, so "restart the job"
is the whole recovery procedure. Fault injection: --die-at-step N --die-rank R (os._exit, no cleanup — like a crash).

Data: deterministic synthetic tokens keyed by (step, rank), so a resumed run sees exactly the batches it would have
seen: loss after resume must match an uninterrupted run (the T0 test checks this).
Prints one JSON line per logged step: {"step", "loss", "world", "resumed_from", "t"} — parsed by the RTO/RPO checker.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training import checkpoint as ck  # noqa: E402


class TinyLM(nn.Module):
    def __init__(self, vocab: int, dim: int, layers: int, heads: int, seq: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(seq, dim))
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(dim, heads, 4 * dim, batch_first=True, norm_first=True)
                                    for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.register_buffer("mask", torch.triu(torch.full((seq, seq), float("-inf")), 1), persistent=False)

    def forward(self, x):
        h = self.emb(x) + self.pos[: x.shape[1]]
        for b in self.blocks:
            h = b(h, src_mask=self.mask[: x.shape[1], : x.shape[1]])
        return self.head(self.norm(h))


def batch(step: int, rank: int, world: int, bs: int, seq: int, vocab: int, device):
    """Global batch for `step` is bs*world sequences; rank r takes its slice. Independent of world size per sequence:
    sequence i of step s is always generated from seed (s, i), so resharding 2→4 keeps the data stream identical."""
    xs = []
    assert (2 * bs) % world == 0, "global batch (2*bs sequences) must divide by the world size"
    per = 2 * bs // world                                  # GLOBAL batch fixed at 2*bs sequences, any world size
    for i in range(rank * per, (rank + 1) * per):
        g = torch.Generator().manual_seed(step * 1_000_003 + i)
        start = torch.randint(0, vocab, (1,), generator=g)
        xs.append((start + torch.arange(seq + 1) * 7) % vocab)   # a learnable pattern: +7 mod vocab
    t = torch.stack(xs).to(device)
    return t[:, :-1], t[:, 1:]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--ckpt", default="checkpoints/run")
    ap.add_argument("--ckpt-every", type=int, default=20)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--bs", type=int, default=4, help="sequences per rank at world size 2 (global batch = 2*bs)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--no-fsdp", action="store_true", help="plain DDP-free single model (debugging)")
    ap.add_argument("--die-at-step", type=int, default=-1)
    ap.add_argument("--die-rank", type=int, default=0)
    a = ap.parse_args(argv)

    cuda = a.device == "cuda"
    dist.init_process_group("nccl" if cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    torch.manual_seed(0)                                   # identical init on every rank before sharding
    model = TinyLM(a.vocab, a.dim, a.layers, a.heads, a.seq).to(device)
    if not a.no_fsdp:
        from torch.distributed.fsdp import fully_shard     # FSDP2 (torch ≥ 2.6 API; UNVERIFIED for older versions)
        for b in model.blocks:
            fully_shard(b)
        fully_shard(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    start, resumed_from = 0, None
    last = ck.latest_committed(a.ckpt)
    if last is not None:
        msd, osd = get_state_dict(model, opt)
        state = {"model": msd, "optim": osd, "meta": {"step": 0}}
        ck.load(state, a.ckpt, last)
        set_state_dict(model, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
        start, resumed_from = last + 1, last
    t0 = time.time()
    for step in range(start, a.steps):
        if step == a.die_at_step and rank == a.die_rank:
            print(json.dumps({"event": "injected_failure", "step": step, "rank": rank}), flush=True)
            os._exit(17)
        x, y = batch(step, rank, world, a.bs, a.seq, a.vocab, device)
        loss = nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        lv = loss.detach().clone()
        dist.all_reduce(lv)
        if rank == 0 and (step % 10 == 0 or step == a.steps - 1):
            print(json.dumps({"step": step, "loss": round(lv.item() / world, 5), "world": world,
                              "resumed_from": resumed_from, "t": round(time.time() - t0, 3)}), flush=True)
        if (step + 1) % a.ckpt_every == 0 or step == a.steps - 1:
            msd, osd = get_state_dict(model, opt)
            ck.save({"model": msd, "optim": osd, "meta": {"step": step}}, a.ckpt, step, rank)
            if rank == 0:
                ck.prune(a.ckpt, a.keep)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
