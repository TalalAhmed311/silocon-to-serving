# L5 — GEMM ladder, tensor cores, async copies, double buffering

**Weeks 28–31 (with P5.6–P5.7) · daily · Tier T2 required (cuBLAS comparison). Tensor-core rungs need sm_80+ (A10G/L4).**

**Objectives.** Implement all 8 GEMM-ladder rungs on the LeetGPU GEMM problem and in the local harness:
1. naive
2. coalesced
3. shared-memory tiling
4. 1D register blocking
5. 2D register blocking
6. vectorized loads + transposed A tile
7. warp tiling + autotuning + double buffering
8. tensor cores

Then INT8 and batched GEMM.

**Problems.** See [leetgpu-map.md](leetgpu-map.md). Core:
- General Matrix Multiplication (GEMM)
- Batched Matrix Multiplication
- FP16 Batched Matrix Multiplication
- INT8 Quantized MatMul

**Exit check.**
- SGEMM ≥70% of cuBLAS SGEMM at N=4096.
- Tensor-core HGEMM ≥50% of cuBLAS HGEMM at N=4096.

Both run on the same GPU and are reported as median TFLOP/s. `TODO(run-on: g6.xlarge)`.

**Sources.**
- siboehm SGEMM_CUDA and article
- CUTLASS/CuTe docs
- Lane A P5.6–P5.7
