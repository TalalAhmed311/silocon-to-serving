# Exercise 1: WMMA HGEMM (sm_70+, T2)

Implement `hgemm` in `kernel.cuh` with `nvcuda::wmma`: fp16 inputs, fp32 accumulation, fp32 output.

1. A 128×128 block tile, 8 warps. Each warp owns 64×32 = 4×2 accumulator fragments of 16×16.
2. Stage A (128×32) and B (32×128) tiles in shared memory with 16-byte vector loads. Pad rows by 8 halves. The pointer passed to `load_matrix_sync` must be **32-byte aligned** and `ldm` a multiple of 8 halves. Check that your padding keeps both true.
3. For each `kk ∈ {0, 16}`: load 4 A and 2 B fragments, then 8 `mma_sync`.
4. `store_matrix_sync` straight to global memory (row-major, `ldm = N`).

**Tolerance (derive it):** fp16 inputs are exactly representable, products of two fp16 values are exact in fp32, and only the fp32 accumulation order differs from cuBLAS. So the expected error is ~K·2⁻²⁴·|C| ≈ 1e-5 relative. The test's `2e-3` is generous on purpose, because the internal accumulation order of tensor cores isn't specified. If yours needs more than that, something's wrong.

`--bench` prints % of cuBLAS at N = 4096 (L5 exit check: ≥ 50%, also reachable with exercise 2).
