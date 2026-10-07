"""Capture a launch-bound toy step (many tiny kernels) in a CUDA graph and compare replay with eager. T2.

Run: uv run --extra torch python course/P6-engine-internals/P6.6-cuda-graph-capture/examples/01_capture_replay.py
TODO(run-on: g6.xlarge)
"""
import time

import torch

assert torch.cuda.is_available(), "needs a GPU (T2)"
dev = torch.device("cuda")
W = [torch.randn(256, 256, device=dev) / 16 for _ in range(50)]


def step(x):                                    # 50 × (matmul + 3 elementwise) = 200 small kernels
    for w in W:
        x = torch.nn.functional.silu(x @ w) * 0.5 + x
    return x


static_in = torch.randn(1, 256, device=dev)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):                      # warm-up outside capture
    for _ in range(3):
        step(static_in)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_out = step(static_in)

x = torch.randn(1, 256, device=dev)
static_in.copy_(x)                              # new input → copy INTO the captured buffer
g.replay()
print("replay == eager:", torch.allclose(static_out, step(x), atol=1e-5))


def bench(fn, n=200):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e6


print(f"eager : {bench(lambda: step(static_in)):8.1f} µs/step")
print(f"replay: {bench(g.replay):8.1f} µs/step")
