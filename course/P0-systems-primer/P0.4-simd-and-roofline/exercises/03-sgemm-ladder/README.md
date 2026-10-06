# Exercise 3 — D1: the SGEMM ladder (medium)

Implement four rungs in `sgemm.hpp`. Each computes `C += A·B` for row-major `A[M×K]`, `B[K×N]`, `C[M×N]`:

| function | what to do |
|---|---|
| `sgemm_naive` | **given**: the `i, j, k` triple loop |
| `sgemm_reorder` | the `i, k, j` order: hoist `A[i][k]`, stream rows of `B` and `C` |
| `sgemm_tiled` | `i, k, j` over `TI × TK × TJ` blocks (start with 64 × 256 × 256), with edge clamping |
| `sgemm_simd` | a register-tiled micro-kernel: a 4 × (2·W) block of `C` held in 8 `simd::vf` accumulators across the K loop, broadcasting `A[i+r][k]` with `simd::set1` and loading two vectors of row `k` of `B`. Scalar code handles the leftover rows/columns |

**Hints**

1. Write `sgemm_reorder` first. Most of the speed comes from the loop order.
2. For the micro-kernel, draw the 4 × 16 block of C (AVX2) and ask: per k, which 4 scalars of A and which 16 floats of B do I need?
3. Get the edges right with the simplest possible code: the micro-kernel for full tiles, a scalar `i, k, j` loop for the rest.
4. Each rung must give the same answer within fp32 rounding. The test uses `rtol=1e-4`, which allows for different summation orders over K ≤ 1024.

**Test:** all rungs vs a float64 reference for shapes (1,1,1), (3,5,7), (64,64,64), (65,33,129), (128,256,96).
**Bench:** `uv run python bench/sgemm_bench.py` times every rung at N = 256…2048 and prints `% of measured peak`.
