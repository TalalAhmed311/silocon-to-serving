// Exercise 1 starter: HGEMM with WMMA (sm_70+). fp16 A[M×K], B[K×N] row-major; fp32 C[M×N].
// Shapes are multiples of 128 (M, N) and 32 (K). Suggested structure: 128×128 block tile, 8 warps, each warp owning a
// 64×32 sub-tile = 4×2 wmma 16×16 accumulator fragments; A/B tiles staged in shared memory with +8 halves padding.
#pragma once
#include <cuda_fp16.h>
#include <mma.h>

inline void hgemm(int M, int N, int K, const __half* A, const __half* B, float* C) {
  (void)M; (void)N; (void)K; (void)A; (void)B; (void)C;   // TODO
}
