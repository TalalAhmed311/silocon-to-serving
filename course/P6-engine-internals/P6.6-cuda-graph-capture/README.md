# P6.6: CUDA graph capture

| | |
|---|---|
| **Tier** | ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) — graphs need a GPU. The reading and the rules are T0 |
| **Time** | ≈25 min reading + ≈7 h hands-on |
| **Prerequisites** | P5.1 (launches, streams), P5.9 (profiling with nsys), P6.3 (block tables) |
| **You will build** | decode-step graph capture per batch-size bucket with padding (`platform/engine/v1/s2s_engine/cuda_graph.py`) and its measurement |

## Learning objectives

1. Say what a CUDA graph records and why replay is cheaper than eager launches.
2. List what capture forbids — dynamic shapes, new allocations at replay, host syncs, data-dependent Python control flow — and rewrite a decode step to satisfy it.
3. Capture one graph per padded batch size, share one memory pool, and pick buckets.
4. Measure launch-overhead savings with nsys at small and large batch, and explain the crossover.

---

## 1. Why graphs

A Llama decode step at batch 1 is dozens of kernels per layer (norms, projections, RoPE, attention, MLP, residuals) × 32 layers — a few thousand launches, each doing microseconds of memory-bound work. Each eager launch costs CPU time in Python, the PyTorch dispatcher and the driver. If the CPU can't enqueue kernels faster than the GPU finishes them, the GPU idles between kernels: **launch-bound**. In nsys that's visible as gaps between kernels on the GPU row.

A CUDA graph records the whole sequence of kernels (and memcpys) with their arguments once; `replay()` submits all of it with one call. The cost of thousands of launches becomes one.

## 2. The rules (and how `TorchRunner.decode` follows them)

| capture requires | because | in #0 v1 |
|---|---|---|
| fixed shapes | kernel launch configs are frozen | pad the batch up to a bucket (1, 2, 4, 8, … 64); pad block tables to `max_blocks_per_seq` |
| fixed addresses | pointers are baked into the graph | inputs copied **into** persistent buffers; output read from a persistent tensor |
| no allocation from a fresh pool at replay | memory addresses must stay valid | capture inside `torch.cuda.graph(g, pool=…)`; all buckets share one pool (captured largest first) |
| no host sync (`.item()`, `.cpu()`, printing a tensor) | the CPU isn't there at replay | masks (`masked_fill`) instead of Python `if`s on tensor values |
| no data-dependent control flow | only the recorded path replays | attention over all `max_blocks · bs` slots, masked by `positions` |

Padded rows decode token 0 at position 0 through block 0 — a harmless write to slot 0, which is why the engine **reserves block 0** when graphs are on. Prefill stays eager (its shapes vary with every chunk); vLLM additionally captures *piecewise* graphs around attention to cover mixed batches (`docs/design/cuda_graphs.md` at the pin).

> **Predict first.** Measure one eager decode step at batch 1 on the L4 first; call it X ms. If launch overhead is ≈ 5 µs × N kernels, how many kernels make up half the step? Count them with nsys, then measure the graph replay. At batch 64, is the saving larger or smaller in **absolute** ms? In **relative** terms?

## 3. Buckets

More buckets → less padding waste, more capture time and memory (every bucket's activations live in the shared pool, sized by the largest). vLLM's default capture sizes are a list you can read in its config at the pin; ours are powers of two up to 64. Above the largest bucket, `DecodeGraphs.run` falls back to eager — at large batch the step is long enough that launch overhead no longer matters.

## Walkthrough

```bash
# T2 (see aws.md)
uv run --extra torch python course/P6-engine-internals/P6.6-cuda-graph-capture/examples/01_capture_replay.py
uv run --extra torch pytest platform/engine/v1/tests/test_torch_runner.py -m gpu -q
nsys profile -o decode_eager  uv run --extra torch python course/P6-engine-internals/P6.6-cuda-graph-capture/examples/02_decode_itl.py --eager
nsys profile -o decode_graphs uv run --extra torch python course/P6-engine-internals/P6.6-cuda-graph-capture/examples/02_decode_itl.py
```

## What you should see

- `01_capture_replay.py`: `replay == eager: True` and a µs-per-call comparison for a 200-op toy step; replay is far cheaper at this size.
- `test_torch_runner.py`: greedy outputs with and without graphs equal the NumPy reference.
- `02_decode_itl.py` prints ms per decode step for B ∈ {1, 8, 32}, eager vs graphs. In nsys, the eager trace has gaps between short kernels; the graph trace doesn't.

## Bench

| B | eager ms/step | graph ms/step | saving | GPU |
|---|---|---|---|---|
| 1 | `TODO(run-on: g6.xlarge)` | | | L4 |
| 8 | | | | |
| 32 | | | | |

## Exercises

| # | Exercise | Tier | Check |
|---|---|---|---|
| 1 | [Capture and replay a decode step in PyTorch](exercises/01-capture.md) | T2 | `examples/01_capture_replay.py` asserts |
| 2 | [Batch-size buckets + padding in the engine](exercises/02-buckets.md) | T2 | `tests/test_torch_runner.py` (graphs) |
| 3 | *(hard)* [Graph-safe sampling](exercises/03-graph-sampling.md) | T2 | your test |

## Common mistakes

- Feeding new tensors to `replay()`: the graph reads the *captured* addresses. Copy into the static inputs.
- Capturing without warm-up: lazy initialisation (cuBLAS handles, autotuning) happens inside the capture and fails or bakes in junk.
- Keeping a reference to a graph's output across replays and being surprised it changed: it's the same memory.
- `.item()` hidden in the model (e.g. a `max(positions)` to size a buffer) — capture errors or, worse, freezes a value.

## Go deeper

- CUDA C++ Programming Guide, "CUDA Graphs". PyTorch docs, "CUDA Graphs" (`torch.cuda.graph`, `make_graphed_callables`).
- vLLM `docs/design/cuda_graphs.md` and `vllm/compilation/cuda_graph.py` at `v0.31.0` (verify paths). SGLang `cuda_graph_runner.py`. nano-vllm `model_runner.py` `capture_cudagraph`.

**Next:** [P6.7 Project: Tiny Inference Engine (GPU)](../P6.7-project-tiny-engine-gpu/README.md).
