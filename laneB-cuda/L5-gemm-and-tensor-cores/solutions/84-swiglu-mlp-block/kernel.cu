// LeetGPU #84 SwiGLU MLP Block — Lane B L5 solution. y = (silu(x·Wg) ⊙ (x·Wu))·Wd; x: M×d, Wg, Wu: d×f, Wd: f×d.
// Kernel 1 computes the gate and up tiles for the same output block and applies silu(g)·u in the EPILOGUE, so the
// two M×f intermediates are never written — only h. Kernel 2 is a plain GEMM h·Wd.
#include <cuda_runtime.h>

// ---- tiled fp32 GEMM core (same in every L5 fp32 solution; any M, N, K via zero-filled loads and guarded stores) ----
// 64×64 block tile, BK = 16, 256 threads, 4×4 outputs per thread (2D register blocking, P5.6 rung 5 + predication).
constexpr int GBM = 64, GBN = 64, GBK = 16, GTM = 4, GTN = 4;

// acc[i][j] += Σ_k A[m0+…][k]·B[k][n0+…] for this thread's 4×4 tile. lda/ldb are row strides.
__device__ __forceinline__ void gemm_tile(const float* __restrict__ A, const float* __restrict__ B, int M, int N, int K,
                                          int lda, int ldb, int m0, int n0, float acc[GTM][GTN]) {
  __shared__ float As[GBK][GBM + 4], Bs[GBK][GBN + 4];      // A stored transposed (k-major) so rows of a k-slice are contiguous
  const int tr = threadIdx.x / (GBN / GTN), tc = threadIdx.x % (GBN / GTN);
  for (int k0 = 0; k0 < K; k0 += GBK) {
    for (int e = threadIdx.x; e < GBM * GBK; e += blockDim.x) {           // A tile: 64 rows × 16 k
      const int r = e / GBK, k = e % GBK, m = m0 + r, kk = k0 + k;
      As[k][r] = (m < M && kk < K) ? A[(size_t)m * lda + kk] : 0.f;
    }
    for (int e = threadIdx.x; e < GBK * GBN; e += blockDim.x) {           // B tile: 16 k × 64 cols (coalesced)
      const int k = e / GBN, c = e % GBN, kk = k0 + k, n = n0 + c;
      Bs[k][c] = (kk < K && n < N) ? B[(size_t)kk * ldb + n] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < GBK; ++k) {
      float a[GTM], b[GTN];
#pragma unroll
      for (int i = 0; i < GTM; ++i) a[i] = As[k][tr * GTM + i];
#pragma unroll
      for (int j = 0; j < GTN; ++j) b[j] = Bs[k][tc * GTN + j];
#pragma unroll
      for (int i = 0; i < GTM; ++i)
#pragma unroll
        for (int j = 0; j < GTN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
}
// ----------------------------------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(256) gemm_k(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
  const int m0 = blockIdx.y * GBM, n0 = blockIdx.x * GBN;
  float acc[GTM][GTN] = {};
  gemm_tile(A, B, M, N, K, K, N, m0, n0, acc);
  const int tr = threadIdx.x / (GBN / GTN), tc = threadIdx.x % (GBN / GTN);
#pragma unroll
  for (int i = 0; i < GTM; ++i)
#pragma unroll
    for (int j = 0; j < GTN; ++j) {
      const int m = m0 + tr * GTM + i, n = n0 + tc * GTN + j;
      if (m < M && n < N) {
        float* c = C + (size_t)m * N + n;
        *c = alpha * acc[i][j] + (beta != 0.f ? beta * *c : 0.f);   // beta = 0 must not read C (it may be garbage/NaN)
      }
    }
}

void solve(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta) {
  gemm_k<<<dim3((N + GBN - 1) / GBN, (M + GBM - 1) / GBM), 256>>>(A, B, C, M, N, K, alpha, beta);
}

__global__ void __launch_bounds__(256) gate_up_swiglu(const float* x, const float* Wg, const float* Wu, float* h, int M, int F, int D) {
  const int m0 = blockIdx.y * GBM, n0 = blockIdx.x * GBN;
  float g[GTM][GTN] = {}, u[GTM][GTN] = {};
  gemm_tile(x, Wg, M, F, D, D, F, m0, n0, g);           // x tiles are re-read for the second product (L2 hits);
  gemm_tile(x, Wu, M, F, D, D, F, m0, n0, u);           // a fused version shares one x tile across both (README)
  const int tr = threadIdx.x / (GBN / GTN), tc = threadIdx.x % (GBN / GTN);
  for (int i = 0; i < GTM; ++i)
    for (int j = 0; j < GTN; ++j) {
      const int m = m0 + tr * GTM + i, n = n0 + tc * GTN + j;
      if (m < M && n < F) h[(size_t)m * F + n] = g[i][j] / (1.f + __expf(-g[i][j])) * u[i][j];
    }
}

__global__ void __launch_bounds__(256) gemm_plain(const float* A, const float* B, float* C, int M, int N, int K) {
  const int m0 = blockIdx.y * GBM, n0 = blockIdx.x * GBN;
  float acc[GTM][GTN] = {};
  gemm_tile(A, B, M, N, K, K, N, m0, n0, acc);
  const int tr = threadIdx.x / (GBN / GTN), tc = threadIdx.x % (GBN / GTN);
  for (int i = 0; i < GTM; ++i)
    for (int j = 0; j < GTN; ++j) {
      const int m = m0 + tr * GTM + i, n = n0 + tc * GTN + j;
      if (m < M && n < N) C[(size_t)m * N + n] = acc[i][j];
    }
}

void solve(const float* x, const float* Wg, const float* Wu, const float* Wd, float* y, int M, int D, int F) {
  float* h = nullptr;
  cudaMalloc(&h, sizeof(float) * M * F);
  gate_up_swiglu<<<dim3((F + GBN - 1) / GBN, (M + GBM - 1) / GBM), 256>>>(x, Wg, Wu, h, M, F, D);
  gemm_plain<<<dim3((D + GBN - 1) / GBN, (M + GBM - 1) / GBM), 256>>>(h, Wd, y, M, D, F);
  cudaFree(h);
}
