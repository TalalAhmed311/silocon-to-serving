// Exercise 3 starter (D1). Fill in reorder, tiled and simd. naive is given.
#pragma once
#include <algorithm>
#include "simd.hpp"

namespace d1 {

// C[M×N] += A[M×K] · B[K×N], row-major.
inline void sgemm_naive(int M, int N, int K, const float* A, const float* B, float* C) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0;
      for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];  // B walked down a column: stride N
      C[i * N + j] += acc;
    }
}

inline void sgemm_reorder(int M, int N, int K, const float* A, const float* B, float* C) {
  (void)M; (void)N; (void)K; (void)A; (void)B; (void)C;  // TODO: i, k, j order (C must end up += A·B)
}

inline void sgemm_tiled(int M, int N, int K, const float* A, const float* B, float* C) {
  (void)M; (void)N; (void)K; (void)A; (void)B; (void)C;  // TODO: blocked i, k, j (C must end up += A·B)
}

inline void sgemm_simd(int M, int N, int K, const float* A, const float* B, float* C) {
  (void)M; (void)N; (void)K; (void)A; (void)B; (void)C;  // TODO: 4 x 2W register-tiled micro-kernel (C must end up += A·B)
}

}  // namespace d1
