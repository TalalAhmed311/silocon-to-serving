"""02_dcp_save_load.py — PyTorch Distributed Checkpoint round trip on CPU (T0), single process or under torchrun.

Run: uv run --extra torch python course/P4-multi-gpu-and-distributed/P4.3-distributed-jobs-and-checkpointing/examples/02_dcp_save_load.py
Expected: "DCP round trip: model and optimizer state bitwise equal (N tensors)" and a listing of the checkpoint dir:
one .distcp file per rank plus a .metadata file that maps every tensor's shards — which is what makes resharding work.
"""
import os
import tempfile

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 8))
opt = torch.optim.AdamW(model.parameters())
model(torch.randn(4, 32)).sum().backward()
opt.step()                                                         # populate Adam state (exp_avg, exp_avg_sq, step)

state = {"model": model.state_dict(), "optim": opt.state_dict()}
d = tempfile.mkdtemp()
dcp.save(state, checkpoint_id=d)

model2 = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 8))
opt2 = torch.optim.AdamW(model2.parameters())
model2(torch.randn(4, 32)).sum().backward()
opt2.step()                                                        # same structure, different values
state2 = {"model": model2.state_dict(), "optim": opt2.state_dict()}
dcp.load(state2, checkpoint_id=d)                                  # in place
model2.load_state_dict(state2["model"])
opt2.load_state_dict(state2["optim"])

n = 0
for a, b in zip(model.state_dict().values(), model2.state_dict().values()):
    assert torch.equal(a, b); n += 1                               # noqa: E702
for k, s in opt.state_dict()["state"].items():
    for name, v in s.items():
        assert torch.equal(torch.as_tensor(v), torch.as_tensor(opt2.state_dict()["state"][k][name])); n += 1  # noqa: E702
print(f"DCP round trip: model and optimizer state bitwise equal ({n} tensors)")
print("checkpoint files:", sorted(os.listdir(d)))
