// Exercise 1 starter: rungs 1–2 given; implement rungs 3–7 (README) behind sgemm(rung, …). Row-major fp32,
// C[M×N] = A[M×K]·B[K×N]. Size contract per rung is the same as d4/gemm.cuh (sgemm_supported below).
#pragma once
#include <cuda_runtime.h>

__global__ void ex_r1(int M, int N, int K, const float* A, const float* B, float* C) {
  const int m = blockIdx.x * 32 + threadIdx.x, n = blockIdx.y * 32 + threadIdx.y;
  if (m >= M || n >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
  C[(size_t)m * N + n] = acc;
}

__global__ void ex_r2(int M, int N, int K, const float* A, const float* B, float* C) {
  const int n = blockIdx.x * 32 + threadIdx.x, m = blockIdx.y * 32 + threadIdx.y;
  if (m >= M || n >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
  C[(size_t)m * N + n] = acc;
}

// TODO rung 3: 32×32 shared-memory tiles
// TODO rung 4: 64×64 block tile, BK = 8, 512 threads × 8 outputs (1D register blocking)
// TODO rung 5: 128×128 block tile, BK = 8, 256 threads × 8×8 outputs (2D register blocking)
// TODO rung 6: rung 5 + float4 loads, A tile transposed in smem
// TODO rung 7: rung 6 + warp tiling + double buffering

inline bool sgemm_supported(int rung, int M, int N, int K) {
  if (rung <= 3) return true;
  if (rung == 4) return M % 64 == 0 && N % 64 == 0 && K % 8 == 0;
  return M % 128 == 0 && N % 128 == 0 && K % 8 == 0;
}

inline void sgemm(int rung, int M, int N, int K, const float* A, const float* B, float* C) {
  const dim3 b(32, 32);
  if (rung == 1) ex_r1<<<dim3((M + 31) / 32, (N + 31) / 32), b>>>(M, N, K, A, B, C);
  else if (rung == 2) ex_r2<<<dim3((N + 31) / 32, (M + 31) / 32), b>>>(M, N, K, A, B, C);
  // else: TODO
}
