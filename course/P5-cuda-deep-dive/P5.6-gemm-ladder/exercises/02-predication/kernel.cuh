// Exercise 2 starter (hard): make the 2D register-blocked SGEMM correct for ANY M, N, K (not multiples of the tile).
// Approach: guard the global loads (zero-fill out-of-range tile elements) and the final stores. Keep the fast path
// branch-free when the tile is fully inside the matrix (check once per block).
#pragma once
#include <cuda_runtime.h>

inline void sgemm_any(int M, int N, int K, const float* A, const float* B, float* C) {
  (void)M; (void)N; (void)K; (void)A; (void)B; (void)C;   // TODO
}
