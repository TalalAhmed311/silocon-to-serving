# Exercise 2: buckets and padding in the engine (T2)

Read `s2s_engine/cuda_graph.py` and `TorchRunner.decode`. Then:

1. Run `tests/test_torch_runner.py -m gpu` (graphs on and off must match the NumPy reference).
2. Change the bucket list to `(1, 3, 8, 24, 64)`; check correctness; measure padding waste on a trace where the running batch size is uniform in 1..64: mean padded rows per step for powers of two vs your list.
3. Measure capture time and the graph pool's memory (`torch.cuda.memory_reserved()` before/after) for 4, 8 and 16 buckets.

Deliverable: a table (bucket list, mean padding, capture s, pool MB) and your choice with one sentence of justification. `TODO(run-on: g6.xlarge)`.
