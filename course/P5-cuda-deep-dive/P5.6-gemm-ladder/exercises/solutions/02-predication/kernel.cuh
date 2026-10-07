// Reference: rung-5 structure (128×128×8 tiles, 8×8 per thread) with predicated loads and stores.
#pragma once
#include <cuda_runtime.h>

template <int BM = 128, int BN = 128, int BK = 8, int TM = 8, int TN = 8>
__global__ void __launch_bounds__((BM * BN) / (TM * TN)) sgemm_pred(int M, int N, int K, const float* __restrict__ A,
                                                                     const float* __restrict__ B, float* __restrict__ C) {
  constexpr int NT = (BM * BN) / (TM * TN), AS = NT / BK, BS = NT / BN;
  __shared__ float As[BM * BK], Bs[BK * BN];
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  const int tr = threadIdx.x / (BN / TN), tc = threadIdx.x % (BN / TN);
  const int ar = threadIdx.x / BK, ac = threadIdx.x % BK, br = threadIdx.x / BN, bc = threadIdx.x % BN;
  float acc[TM * TN] = {0.f}, rm[TM], rn[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
    for (int o = 0; o < BM; o += AS) {
      const int m = m0 + ar + o, k = k0 + ac;
      As[(ar + o) * BK + ac] = (m < M && k < K) ? A[(size_t)m * K + k] : 0.f;     // zero-fill: contributes nothing
    }
#pragma unroll
    for (int o = 0; o < BK; o += BS) {
      const int k = k0 + br + o, n = n0 + bc;
      Bs[(br + o) * BN + bc] = (k < K && n < N) ? B[(size_t)k * N + n] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) rm[i] = As[(tr * TM + i) * BK + k];
#pragma unroll
      for (int j = 0; j < TN; ++j) rn[j] = Bs[k * BN + tc * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i * TN + j] += rm[i] * rn[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int m = m0 + tr * TM + i, n = n0 + tc * TN + j;
      if (m < M && n < N) C[(size_t)m * N + n] = acc[i * TN + j];
    }
}

inline void sgemm_any(int M, int N, int K, const float* A, const float* B, float* C) {
  sgemm_pred<><<<dim3((N + 127) / 128, (M + 127) / 128), 256>>>(M, N, K, A, B, C);
}
