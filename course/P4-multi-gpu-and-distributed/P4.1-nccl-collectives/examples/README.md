# P4.1 examples

| File | Tier | What |
|---|---|---|
| [`01_collectives_numpy.py`](01_collectives_numpy.py) | T0 | ring all-reduce on N simulated ranks, checked against `sum`; prints the 2(N−1) step count |
| [`02_torch_dist_gloo.py`](02_torch_dist_gloo.py) | T0 | real `torch.distributed` collectives across CPU processes (gloo), plus a size sweep |
| [`03_nccl_tests.sh`](03_nccl_tests.sh) | T3 | build nccl-tests v2.21.1 and sweep five collectives with `NCCL_DEBUG=INFO` |
| [`../commmodel.py`](../commmodel.py) | T0 | the α–β cost model and the busbw factors, imported by P4.2 |
