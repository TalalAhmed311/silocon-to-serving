# kernels: D4 kernels repo (+ #12 custom-kernel path)

The kernels you write across P5, with tests and benches, as one header-only library. Lane B solutions and the #0 v1 engine (P6) reuse them.

| Path | What | Module |
|---|---|---|
| `include/d4/common.cuh` | warp/block reductions, ceil_div, device queries | — |
| `include/d4/elementwise.cuh`, `transpose.cuh` | grid-stride and `float4` vector add; the transpose ladder | P5.1, P5.2 |
| `include/d4/reduce.cuh`, `scan.cuh` | the 6-rung reduction ladder; 3-phase and decoupled look-back scans | P5.4 |
| `include/d4/softmax.cuh`, `norms.cuh` | safe / online / in-register softmax; RMSNorm + fused residual (fp32/fp16/bf16) | P5.5 |
| `include/d4/gemm.cuh` | SGEMM rungs 1–7 + cuBLAS baseline | P5.6 |
| `include/d4/hgemm.cuh` | WMMA, `mma.sync`+`ldmatrix`, `cp.async` pipelines; int8 `dp4a` | P5.7 |
| `include/d4/attention.cuh` | naive, FA-1 structure, FA-2 forward (+causal); paged decode attention | P5.8 |
| `triton_kernels/` | softmax, RMSNorm, matmul, FA-2 in Triton (interpreter-testable) | P5.9 |
| `torch_ext/`, `vllm_plugin/` | **#12**: `torch.ops.s2s.fused_add_rms_norm` (CUDA + Triton) and the vLLM plugin that swaps it in | P5.10 |
| `tests/*.cu` | CTest: every kernel against CPU, cuBLAS or CUB references | — |
| `tests_py/` | pytest: the custom op (opcheck, torch.compile, GPU) | P5.10 |
| `bench/d4_bench.cu` | `d4_bench all|reduce|scan|softmax|rmsnorm|gemm|hgemm|attention`, writing `results/<family>.jsonl` | — |
| `bench/ncu_to_table.py` | `ncu --csv` → bench table with "% of peak" | P5.3 |

```bash
cmake -S platform/kernels -B build/d4 -DCMAKE_CUDA_ARCHITECTURES=native && cmake --build build/d4 -j
ctest --test-dir build/d4 --output-on-failure && ./build/d4/d4_bench all
```

Requirements: CUDA 12.4+, CMake 3.24+, an sm_75+ GPU (tensor-core `mma.sync`/`cp.async` variants need sm_80+, and they skip themselves at runtime below that). Status: written for the course and **not yet compiled or run** (`TODO(run)` in GAPS.md). Expect a first build to surface typos.
