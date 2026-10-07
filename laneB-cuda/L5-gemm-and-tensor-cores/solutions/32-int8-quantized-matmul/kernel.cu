// LeetGPU #32 INT8 Quantized MatMul — Lane B L5 solution. A: M×K int8, B: K×N int8 (row-major), symmetric per-tensor
// scales sA, sB, sC: C_q = clamp(round(Σ_k A·B · sA·sB / sC), −128, 127) as int8. Accumulate in int32 with __dp4a.
// Our quantization convention (symmetric, per-tensor, round-half-even); check the statement (zero points, per-channel
// scales, output dtype) — the requantization line is the only part that changes.
#include <cuda_runtime.h>
#include <cstdint>

// 32×32 output tile per block, k in steps of 32: A rows and B COLUMNS packed 4 int8 per int in smem, so the inner loop
// is 8 __dp4a per output (4 MACs each). B is transposed while loading (it is row-major K×N; dp4a needs k-contiguous).
__global__ void qgemm_dp4a(const int8_t* __restrict__ A, const int8_t* __restrict__ B, int8_t* __restrict__ C,
                           int M, int N, int K, float scale) {
  __shared__ int As[32][9], Bt[32][9];                      // 32 rows/cols × 8 packed ints (+1 pad)
  const int tx = threadIdx.x, ty = threadIdx.y;              // block (32, 8): thread computes rows ty·4..+3 at column tx
  const int m0 = blockIdx.y * 32, n0 = blockIdx.x * 32;
  int acc[4] = {0, 0, 0, 0};
  for (int k0 = 0; k0 < K; k0 += 32) {
    for (int r = ty; r < 32; r += 8) {                       // A: thread tx < 8 packs 4 consecutive k of row r
      if (tx < 8) {
        int packed = 0;
        for (int q = 0; q < 4; ++q) {
          const int m = m0 + r, k = k0 + tx * 4 + q;
          const int v = (m < M && k < K) ? A[(size_t)m * K + k] : 0;
          packed |= (v & 0xFF) << (8 * q);
        }
        As[r][tx] = packed;
      }
    }
    for (int kq = ty; kq < 8; kq += 8) {                     // B: column tx, 4 consecutive k packed (transpose on load)
      int packed = 0;
      for (int q = 0; q < 4; ++q) {
        const int k = k0 + kq * 4 + q, n = n0 + tx;
        const int v = (k < K && n < N) ? B[(size_t)k * N + n] : 0;   // coalesced across tx for each (kq, q)
        packed |= (v & 0xFF) << (8 * q);
      }
      Bt[tx][kq] = packed;
    }
    __syncthreads();
#pragma unroll
    for (int kq = 0; kq < 8; ++kq)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[i] = __dp4a(As[ty * 4 + i][kq], Bt[tx][kq], acc[i]);
    __syncthreads();
  }
  const int n = n0 + tx;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int m = m0 + ty * 4 + i;
    if (m < M && n < N) {
      const float v = rintf(acc[i] * scale);                 // round half to even
      C[(size_t)m * N + n] = static_cast<int8_t>(fminf(fmaxf(v, -128.f), 127.f));
    }
  }
}

void solve(const int8_t* A, const int8_t* B, int8_t* C, int M, int N, int K, float scale_A, float scale_B, float scale_C) {
  qgemm_dp4a<<<dim3((N + 31) / 32, (M + 31) / 32), dim3(32, 8)>>>(A, B, C, M, N, K, scale_A * scale_B / scale_C);
}
