// d4/gemm.cuh — the P5.6 SGEMM ladder, rungs 1–7: C[M×N] = A[M×K] · B[K×N], fp32, row-major.
// Written for this course after reading siboehm's SGEMM_CUDA article (rungs follow its order; code is original).
// Size requirements: rungs 1–3 any M, N, K; rung 4 needs M%64 = N%64 = K%8 = 0; rungs 5–7 need M%128 = N%128 = K%8 = 0
// (the "hard" exercise adds predication). Baseline: cuBLAS SGEMM (gemm_cublas below).
#pragma once
#include <cublas_v2.h>

#include "common.cuh"

namespace d4 {

// Rung 1: one thread per C element; threadIdx.x walks ROWS → a warp reads 32 different rows of A (uncoalesced).
__global__ void sgemm_r1(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
  const int m = blockIdx.x * 32 + threadIdx.x, n = blockIdx.y * 32 + threadIdx.y;
  if (m >= M || n >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
  C[(size_t)m * N + n] = acc;
}

// Rung 2: same math, threadIdx.x walks COLUMNS → B and C accesses coalesced, A broadcast within the warp.
__global__ void sgemm_r2(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
  const int n = blockIdx.x * 32 + threadIdx.x, m = blockIdx.y * 32 + threadIdx.y;
  if (m >= M || n >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
  C[(size_t)m * N + n] = acc;
}

// Rung 3: 32×32 shared-memory tiles: each element of A and B is loaded from global once per tile, reused 32 times.
__global__ void sgemm_r3(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
  __shared__ float As[32][32], Bs[32][32];
  const int tx = threadIdx.x, ty = threadIdx.y, n = blockIdx.x * 32 + tx, m = blockIdx.y * 32 + ty;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += 32) {
    As[ty][tx] = (m < M && k0 + tx < K) ? A[(size_t)m * K + k0 + tx] : 0.f;
    Bs[ty][tx] = (k0 + ty < K && n < N) ? B[(size_t)(k0 + ty) * N + n] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 32; ++k) acc += As[ty][k] * Bs[k][tx];   // As: broadcast across the warp; Bs: conflict-free
    __syncthreads();
  }
  if (m < M && n < N) C[(size_t)m * N + n] = acc;
}

// Rung 4: 1D register blocking. Block tile 64×64, BK = 8, each of 512 threads computes TM = 8 outputs in a column:
// one Bs value is reused for 8 FMAs from registers.
__global__ void __launch_bounds__(512) sgemm_r4(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
  constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
  __shared__ float As[BM * BK], Bs[BK * BN];
  const int tc = threadIdx.x % BN, tr = threadIdx.x / BN;          // tr in 0..7
  A += (size_t)blockIdx.y * BM * K;
  B += (size_t)blockIdx.x * BN;
  C += (size_t)blockIdx.y * BM * N + (size_t)blockIdx.x * BN;
  const int ar = threadIdx.x / BK, ac = threadIdx.x % BK, br = threadIdx.x / BN, bc = threadIdx.x % BN;
  float acc[TM] = {0.f};
  for (int k0 = 0; k0 < K; k0 += BK) {
    As[ar * BK + ac] = A[(size_t)ar * K + ac];
    Bs[br * BN + bc] = B[(size_t)br * N + bc];
    __syncthreads();
    A += BK;
    B += (size_t)BK * N;
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      const float b = Bs[k * BN + tc];
#pragma unroll
      for (int i = 0; i < TM; ++i) acc[i] += As[(tr * TM + i) * BK + k] * b;
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) C[(size_t)(tr * TM + i) * N + tc] = acc[i];
}

// Rung 5: 2D register blocking. Block tile 128×128, BK = 8, 256 threads × (TM×TN = 8×8) outputs: per k, 8 + 8 smem
// loads feed 64 FMAs (arithmetic intensity per smem load: 4 FMAs).
template <int BM = 128, int BN = 128, int BK = 8, int TM = 8, int TN = 8>
__global__ void __launch_bounds__((BM * BN) / (TM * TN)) sgemm_r5(int M, int N, int K, const float* __restrict__ A,
                                                                   const float* __restrict__ B, float* __restrict__ C) {
  constexpr int NT = (BM * BN) / (TM * TN);
  __shared__ float As[BM * BK], Bs[BK * BN];
  const int tr = threadIdx.x / (BN / TN), tc = threadIdx.x % (BN / TN);
  A += (size_t)blockIdx.y * BM * K;
  B += (size_t)blockIdx.x * BN;
  C += (size_t)blockIdx.y * BM * N + (size_t)blockIdx.x * BN;
  const int ar = threadIdx.x / BK, ac = threadIdx.x % BK, br = threadIdx.x / BN, bc = threadIdx.x % BN;
  constexpr int AS = NT / BK, BS = NT / BN;                      // rows covered per loading pass
  float acc[TM * TN] = {0.f}, rm[TM], rn[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
    for (int o = 0; o < BM; o += AS) As[(ar + o) * BK + ac] = A[(size_t)(ar + o) * K + ac];
#pragma unroll
    for (int o = 0; o < BK; o += BS) Bs[(br + o) * BN + bc] = B[(size_t)(br + o) * N + bc];
    __syncthreads();
    A += BK;
    B += (size_t)BK * N;
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
    for (int j = 0; j < TN; ++j) C[(size_t)(tr * TM + i) * N + tc * TN + j] = acc[i * TN + j];
}

// Rung 6: rung 5 + float4 global loads, A tile stored TRANSPOSED (As[k][m]) so the per-k column of A is contiguous
// (vectorisable smem reads), float4 stores of C. Requires K % 4 == 0, N % 4 == 0 (plus the rung-5 multiples).
template <int BM = 128, int BN = 128, int BK = 8, int TM = 8, int TN = 8>
__global__ void __launch_bounds__((BM * BN) / (TM * TN)) sgemm_r6(int M, int N, int K, const float* __restrict__ A,
                                                                   const float* __restrict__ B, float* __restrict__ C) {
  constexpr int NT = (BM * BN) / (TM * TN);
  static_assert(BM * BK == 4 * NT && BK * BN == 4 * NT, "each thread loads exactly one float4 of A and of B per tile");
  __shared__ __align__(16) float As[BK * BM];
  __shared__ __align__(16) float Bs[BK * BN];
  const int tr = threadIdx.x / (BN / TN), tc = threadIdx.x % (BN / TN);
  A += (size_t)blockIdx.y * BM * K;
  B += (size_t)blockIdx.x * BN;
  C += (size_t)blockIdx.y * BM * N + (size_t)blockIdx.x * BN;
  const int ar = threadIdx.x / (BK / 4), ac4 = threadIdx.x % (BK / 4);
  const int br = threadIdx.x / (BN / 4), bc4 = threadIdx.x % (BN / 4);
  float acc[TM * TN] = {0.f}, rm[TM], rn[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    const float4 a = reinterpret_cast<const float4*>(A + (size_t)ar * K)[ac4];
    As[(ac4 * 4 + 0) * BM + ar] = a.x;
    As[(ac4 * 4 + 1) * BM + ar] = a.y;
    As[(ac4 * 4 + 2) * BM + ar] = a.z;
    As[(ac4 * 4 + 3) * BM + ar] = a.w;
    reinterpret_cast<float4*>(Bs + br * BN)[bc4] = reinterpret_cast<const float4*>(B + (size_t)br * N)[bc4];
    __syncthreads();
    A += BK;
    B += (size_t)BK * N;
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; i += 4) { const float4 v = *reinterpret_cast<const float4*>(As + k * BM + tr * TM + i); rm[i] = v.x; rm[i + 1] = v.y; rm[i + 2] = v.z; rm[i + 3] = v.w; }
#pragma unroll
      for (int j = 0; j < TN; j += 4) { const float4 v = *reinterpret_cast<const float4*>(Bs + k * BN + tc * TN + j); rn[j] = v.x; rn[j + 1] = v.y; rn[j + 2] = v.z; rn[j + 3] = v.w; }
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
    for (int j = 0; j < TN; j += 4)
      *reinterpret_cast<float4*>(C + (size_t)(tr * TM + i) * N + tc * TN + j) =
          make_float4(acc[i * TN + j], acc[i * TN + j + 1], acc[i * TN + j + 2], acc[i * TN + j + 3]);
}

// Rung 7: rung 6 + WARP TILING (each warp owns a WM×WN sub-tile, so a warp's smem reads cover a compact region and
// its lanes share A/B fragments) + DOUBLE BUFFERING (the next K-tile is fetched from global into registers while the
// current one is computed from shared memory; one __syncthreads per tile). Template parameters are what autotune.py
// sweeps. Constraint checks are static_asserts.
template <int BM = 128, int BN = 128, int BK = 8, int WM = 64, int WN = 32, int TM = 8, int TN = 8>
__global__ void __launch_bounds__((BM * BN) / (TM * TN)) sgemm_r7(int M, int N, int K, const float* __restrict__ A,
                                                                   const float* __restrict__ B, float* __restrict__ C) {
  constexpr int NT = (BM * BN) / (TM * TN);
  constexpr int WARPS_N = BN / WN, LANES_N = WN / TN;              // warps per block row, lanes per warp row
  static_assert(NT % 32 == 0 && (BM / WM) * (BN / WN) == NT / 32, "warp tiles must exactly cover the block tile");
  static_assert((WM / TM) * (WN / TN) == 32, "a warp tile must be 32 thread tiles");
  static_assert(BM * BK == 4 * NT && BK * BN == 4 * NT, "one float4 of A and of B per thread per tile");
  __shared__ __align__(16) float As[2][BK * BM];
  __shared__ __align__(16) float Bs[2][BK * BN];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int row0 = (warp / WARPS_N) * WM + (lane / LANES_N) * TM;  // this thread's first C row inside the block tile
  const int col0 = (warp % WARPS_N) * WN + (lane % LANES_N) * TN;
  A += (size_t)blockIdx.y * BM * K;
  B += (size_t)blockIdx.x * BN;
  C += (size_t)blockIdx.y * BM * N + (size_t)blockIdx.x * BN;
  const int ar = threadIdx.x / (BK / 4), ac4 = threadIdx.x % (BK / 4);
  const int br = threadIdx.x / (BN / 4), bc4 = threadIdx.x % (BN / 4);
#define D4_STASH(buf, a, b)                                        \
  do {                                                             \
    As[buf][(ac4 * 4 + 0) * BM + ar] = (a).x;                      \
    As[buf][(ac4 * 4 + 1) * BM + ar] = (a).y;                      \
    As[buf][(ac4 * 4 + 2) * BM + ar] = (a).z;                      \
    As[buf][(ac4 * 4 + 3) * BM + ar] = (a).w;                      \
    reinterpret_cast<float4*>(Bs[buf] + br * BN)[bc4] = (b);       \
  } while (0)
  float4 ga = reinterpret_cast<const float4*>(A + (size_t)ar * K)[ac4];
  float4 gb = reinterpret_cast<const float4*>(B + (size_t)br * N)[bc4];
  D4_STASH(0, ga, gb);
  __syncthreads();
  float acc[TM * TN] = {0.f}, rm[TM], rn[TN];
  const int tiles = K / BK;
  for (int t = 0; t < tiles; ++t) {
    const int cur = t & 1;
    if (t + 1 < tiles) {                                           // issue next tile's global loads first
      ga = reinterpret_cast<const float4*>(A + (size_t)ar * K + (size_t)(t + 1) * BK)[ac4];
      gb = reinterpret_cast<const float4*>(B + (size_t)(br + (t + 1) * BK) * N)[bc4];
    }
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; i += 4) { const float4 v = *reinterpret_cast<const float4*>(As[cur] + k * BM + row0 + i); rm[i] = v.x; rm[i + 1] = v.y; rm[i + 2] = v.z; rm[i + 3] = v.w; }
#pragma unroll
      for (int j = 0; j < TN; j += 4) { const float4 v = *reinterpret_cast<const float4*>(Bs[cur] + k * BN + col0 + j); rn[j] = v.x; rn[j + 1] = v.y; rn[j + 2] = v.z; rn[j + 3] = v.w; }
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i * TN + j] += rm[i] * rn[j];
    }
    if (t + 1 < tiles) D4_STASH(cur ^ 1, ga, gb);                  // the other buffer: nobody reads it this tile
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; j += 4)
      *reinterpret_cast<float4*>(C + (size_t)(row0 + i) * N + col0 + j) =
          make_float4(acc[i * TN + j], acc[i * TN + j + 1], acc[i * TN + j + 2], acc[i * TN + j + 3]);
}
#undef D4_STASH

inline bool sgemm_supported(int rung, int M, int N, int K) {
  if (rung <= 3) return true;
  if (rung == 4) return M % 64 == 0 && N % 64 == 0 && K % 8 == 0;
  return M % 128 == 0 && N % 128 == 0 && K % 8 == 0;
}

inline void sgemm(int rung, int M, int N, int K, const float* A, const float* B, float* C, cudaStream_t s = 0) {
  switch (rung) {
    case 1: sgemm_r1<<<dim3(ceil_div(M, 32), ceil_div(N, 32)), dim3(32, 32), 0, s>>>(M, N, K, A, B, C); break;
    case 2: sgemm_r2<<<dim3(ceil_div(N, 32), ceil_div(M, 32)), dim3(32, 32), 0, s>>>(M, N, K, A, B, C); break;
    case 3: sgemm_r3<<<dim3(ceil_div(N, 32), ceil_div(M, 32)), dim3(32, 32), 0, s>>>(M, N, K, A, B, C); break;
    case 4: sgemm_r4<<<dim3(N / 64, M / 64), 512, 0, s>>>(M, N, K, A, B, C); break;
    case 5: sgemm_r5<><<<dim3(N / 128, M / 128), 256, 0, s>>>(M, N, K, A, B, C); break;
    case 6: sgemm_r6<><<<dim3(N / 128, M / 128), 256, 0, s>>>(M, N, K, A, B, C); break;
    default: sgemm_r7<><<<dim3(N / 128, M / 128), 256, 0, s>>>(M, N, K, A, B, C); break;
  }
}

// cuBLAS is column-major: computing row-major C = A·B is the column-major product Cᵀ = Bᵀ·Aᵀ, i.e. swap A and B.
inline void gemm_cublas(cublasHandle_t h, int M, int N, int K, const float* A, const float* B, float* C) {
  const float one = 1.f, zero = 0.f;
  cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N);
}

}  // namespace d4
