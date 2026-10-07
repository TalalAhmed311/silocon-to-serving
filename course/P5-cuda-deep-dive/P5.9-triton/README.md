# P5.9: Triton

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) for correctness (Triton interpreter on CPU). ![T2](https://img.shields.io/badge/tier-T2%20one%20GPU-orange) for performance (sm_80+ recommended: `g6.xlarge`) |
| **Time** | ≈25 min reading + ≈9 h hands-on |
| **Prerequisites** | P5.5, P5.6–P5.8 (you'll rewrite those kernels) |
| **Reading** | Triton docs + tutorials 01, 02, 03, 05, 06 at `v3.8.0` · Triton Puzzles · GPU MODE Triton lecture |
| **D4** | `platform/kernels/triton_kernels/` |

## Learning objectives

1. Think in Triton's **block-level** model: a program owns a tile, and you write tensor ops on it, while the compiler handles threads, shared memory and (with `tl.dot`) tensor-core layouts.
2. Use masks (`tl.load(..., mask, other)`), `tl.dot`, `tl.max`/`tl.sum` reductions, `constexpr` specialization and autotune configs.
3. Rewrite softmax, RMSNorm, matmul and FA-2 forward, and compare each with your CUDA version.
4. Test GPU kernels on a laptop with the interpreter (`TRITON_INTERPRET=1`).

---

## 1. The model in one table

| CUDA (P5.1–P5.8) | Triton |
|---|---|
| threads, warps, `threadIdx` | **programs** (`tl.program_id`), each operating on whole blocks of data |
| explicit shared memory + `__syncthreads` | implicit: the compiler stages tiles and inserts barriers |
| fragment layouts, `ldmatrix`, `mma.sync` | `tl.dot(a, b)` on fp16/bf16 blocks → tensor cores |
| `cp.async` stages, by hand | `num_stages` in a `triton.Config` |
| bounds checks per thread | `mask=` on loads and stores |
| templates for tile sizes | `tl.constexpr` parameters + `@triton.autotune` |

What you give up: control over register allocation, warp specialization, exact smem layouts, and anything that doesn't map to block-level ops. For FA-3-style Hopper kernels, people go back to CUDA, CUTLASS, or Triton's lower-level extensions.

## 2. The four rewrites

- **softmax**: the row fits in one block (`BLOCK = next_power_of_2(cols)`), so it's 1 read and 1 write. Rows longer than the block use the online max/sum loop from P5.5.
- **RMSNorm**: two loops over the row (Σx², then scale), fp32 accumulation, optional fused residual. The same shape as vLLM's `fused_add_rms_norm`.
- **matmul**: tiles + `tl.dot`, masks for ragged K, **grouped ordering** (consecutive programs share A row panels, so they hit in L2), autotuning over tile shapes and stages. fp32 inputs use `input_precision="ieee"`, so the comparison with SGEMM is fair (no TF32).
- **FA-2 forward**: P5.8's algorithm with both matmuls as `tl.dot`. On an L4 this often beats the CUDA-core FA-2 from P5.8 by a wide margin, and that margin is the tensor cores.

> **Predict first.** For each of the four kernels, will Triton be faster or slower than your CUDA version on an L4? (Hint: it depends on whether your CUDA version used tensor cores and pipelining.) Write the predictions before running `bench_triton_vs_cuda.py`.

---

## Walkthrough

```bash
uv run --extra torch pytest course/P5-cuda-deep-dive/P5.9-triton/exercises -m torch       # T0, interpreter
S2S_SOLUTIONS=1 uv run --extra torch pytest course/P5-cuda-deep-dive/P5.9-triton/exercises -m torch
```

The T2 path is in [aws.md](aws.md).

## What you should see

- Interpreter: all tests pass, slowly (seconds per test). It's Python executing the block ops.
- GPU: softmax and RMSNorm near your CUDA GB/s. fp16 matmul near cuBLAS (`tl.dot` uses tensor cores). FA-2 well ahead of the CUDA-core FA-2.

`TODO(run-on: g6.xlarge)`

## Bench

| kernel | unit | CUDA (D4) | Triton (ours) | PyTorch |
|---|---|---|---|---|
| softmax 65536×1024 | GB/s | | | `TODO(run-on: g6.xlarge)` |
| rmsnorm 16384×4096 | GB/s | | | |
| matmul 4096³ fp32 / fp16 | TFLOP/s | | | |
| FA-2 fwd N = 4096 | TFLOP/s | | | |

## Exercises

See [exercises/README.md](exercises/README.md): four rewrites with pytest on T0 (interpreter) and T2 (real), plus Triton Puzzles (hard).

## Common mistakes

- Forgetting `other=` on masked loads, so padding lanes read garbage that ends up in a `max` or `sum`.
- Non-power-of-two `tl.arange` sizes (not allowed). Pad with masks.
- Comparing fp32 `tl.dot` (TF32 by default on Ampere+) with IEEE SGEMM.
- Autotuning inside a benchmark loop: the first call compiles and tunes. Warm up first (`triton.testing.do_bench` does).

## Go deeper

- Triton tutorials 01–06 at `v3.8.0` (run and annotate). Triton Puzzles.
- Triton's `06-fused-attention.py`: the production-style FA-2 with backward, pipelining and causal stages.
- GPU MODE Triton lectures.

**Next:** [P5.10 Wiring a custom op into PyTorch/vLLM (#12)](../P5.10-custom-op-into-serving/README.md).
