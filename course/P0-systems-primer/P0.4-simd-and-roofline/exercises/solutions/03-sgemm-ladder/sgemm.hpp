// Exercise 3 solution (D1).
#pragma once
#include <algorithm>
#include "simd.hpp"

namespace d1 {

inline void sgemm_naive(int M, int N, int K, const float* A, const float* B, float* C) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float acc = 0;
      for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
      C[i * N + j] += acc;
    }
}

// Rung 1: i, k, j. The inner loop now reads B and writes C contiguously — every cache line is fully used,
// and the compiler auto-vectorizes it (y += a * x, a SAXPY per (i, k)).
inline void sgemm_reorder(int M, int N, int K, const float* A, const float* B, float* C) {
  for (int i = 0; i < M; ++i)
    for (int k = 0; k < K; ++k) {
      const float a = A[i * K + k];
      const float* b = B + k * N;
      float* c = C + i * N;
      for (int j = 0; j < N; ++j) c[j] += a * b[j];
    }
}

// Rung 2: block i, k, j so the B block (TK × TJ floats = 256 KB here) stays in L2 while TI rows of A sweep over it.
inline void sgemm_tiled(int M, int N, int K, const float* A, const float* B, float* C) {
  constexpr int TI = 64, TK = 256, TJ = 256;
  for (int i0 = 0; i0 < M; i0 += TI)
    for (int k0 = 0; k0 < K; k0 += TK)
      for (int j0 = 0; j0 < N; j0 += TJ) {
        const int i1 = std::min(M, i0 + TI), k1 = std::min(K, k0 + TK), j1 = std::min(N, j0 + TJ);
        for (int i = i0; i < i1; ++i)
          for (int k = k0; k < k1; ++k) {
            const float a = A[i * K + k];
            const float* b = B + k * N;
            float* c = C + i * N;
            for (int j = j0; j < j1; ++j) c[j] += a * b[j];
          }
      }
}

// Rung 3: register tiling. A 4 × (2W) tile of C lives in 8 vector registers for the whole K loop, so C is read
// and written once per tile instead of once per k. Per k: 4 broadcasts of A, 2 loads of B, 8 FMAs.
// 8 independent accumulators also hide FMA latency (§1 of the lesson).
inline void sgemm_simd(int M, int N, int K, const float* A, const float* B, float* C) {
  constexpr int MR = 4, NR = 2 * simd::W;
  constexpr int TK = 256;  // keep the 4 A rows' K-slice and B's K-slice of NR columns in L1/L2
  const int Mf = M / MR * MR, Nf = N / NR * NR;
  for (int k0 = 0; k0 < K; k0 += TK) {
    const int k1 = std::min(K, k0 + TK);
    for (int i = 0; i < Mf; i += MR)
      for (int j = 0; j < Nf; j += NR) {
        simd::vf c[MR][2];
        for (int r = 0; r < MR; ++r) {
          c[r][0] = simd::load(C + (i + r) * N + j);
          c[r][1] = simd::load(C + (i + r) * N + j + simd::W);
        }
        for (int k = k0; k < k1; ++k) {
          const simd::vf b0 = simd::load(B + k * N + j), b1 = simd::load(B + k * N + j + simd::W);
          for (int r = 0; r < MR; ++r) {
            const simd::vf a = simd::set1(A[(i + r) * K + k]);
            c[r][0] = simd::fma(a, b0, c[r][0]);
            c[r][1] = simd::fma(a, b1, c[r][1]);
          }
        }
        for (int r = 0; r < MR; ++r) {
          simd::store(C + (i + r) * N + j, c[r][0]);
          simd::store(C + (i + r) * N + j + simd::W, c[r][1]);
        }
      }
    // Edges: leftover columns for the full row-tiles, then leftover rows for all columns. Scalar i, k, j.
    for (int i = 0; i < Mf; ++i)
      for (int k = k0; k < k1; ++k)
        for (int j = Nf; j < N; ++j) C[i * N + j] += A[i * K + k] * B[k * N + j];
    for (int i = Mf; i < M; ++i)
      for (int k = k0; k < k1; ++k)
        for (int j = 0; j < N; ++j) C[i * N + j] += A[i * K + k] * B[k * N + j];
  }
}

}  // namespace d1
