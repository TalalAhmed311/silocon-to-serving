"""01_ddp_cpu.py — DistributedDataParallel on CPU processes (gloo): the smallest real distributed training job (T0).

Run: uv run --extra torch torchrun --nproc-per-node 2 course/P4-multi-gpu-and-distributed/P4.3-distributed-jobs-and-checkpointing/examples/01_ddp_cpu.py
Expected: both ranks print identical parameter checksums after every step (DDP all-reduces gradients, so replicas
never diverge) and the loss decreases.
"""
import torch
import torch.distributed as dist
import torch.nn as nn

dist.init_process_group("gloo")
rank = dist.get_rank()
torch.manual_seed(0)
model = nn.parallel.DistributedDataParallel(nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 1)))
opt = torch.optim.SGD(model.parameters(), lr=0.05)
w_true = torch.arange(16.0) / 16
for step in range(30):
    g = torch.Generator().manual_seed(step * 100 + rank)          # each rank sees different data
    x = torch.randn(32, 16, generator=g)
    loss = (model(x).squeeze(-1) - x @ w_true).pow(2).mean()
    loss.backward()                                               # gradient all-reduce happens here, bucketed
    opt.step()
    opt.zero_grad()
    if step % 10 == 0 or step == 29:
        checksum = sum(p.sum().item() for p in model.parameters())
        print(f"rank {rank} step {step} loss {loss.item():.4f} param checksum {checksum:.6f}")
dist.destroy_process_group()
