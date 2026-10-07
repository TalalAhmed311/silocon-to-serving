// LeetGPU #57 FP16 Batched Matrix Multiplication — Lane B L5 solution (exit check is on P5.7's HGEMM ≥ 50% cuBLAS).
// C[b] = A[b]·B[b], fp16 inputs, fp32 accumulation, fp16 output, any M, N, K. WMMA (sm_70+): block tile 64×64,
// 4 warps (2×2), each warp 32×32 = 2×2 fragments of 16×16; K step 16. Loads zero-fill out-of-range elements; the
// result goes through shared memory so the store can be guarded for ragged edges.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

constexpr int WB = 64, WK = 16, WPAD = 8;

__global__ void __launch_bounds__(128) hbmm_wmma(const __half* A, const __half* B, __half* C, int M, int N, int K) {
#if __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  __shared__ __align__(32) __half As[WB][WK + WPAD];
  __shared__ __align__(32) __half Bs[WK][WB + WPAD];
  __shared__ __align__(32) float Cs[WB][WB + 4];
  const size_t b = blockIdx.z;
  A += b * M * K;
  B += b * K * N;
  C += b * M * N;
  const int m0 = blockIdx.y * WB, n0 = blockIdx.x * WB, warp = threadIdx.x / 32, wm = warp / 2, wn = warp % 2;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
  for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);
  for (int k0 = 0; k0 < K; k0 += WK) {
    for (int e = threadIdx.x; e < WB * WK; e += blockDim.x) {
      const int r = e / WK, k = e % WK;
      As[r][k] = (m0 + r < M && k0 + k < K) ? A[(size_t)(m0 + r) * K + k0 + k] : __float2half(0.f);
    }
    for (int e = threadIdx.x; e < WK * WB; e += blockDim.x) {
      const int k = e / WB, c = e % WB;
      Bs[k][c] = (k0 + k < K && n0 + c < N) ? B[(size_t)(k0 + k) * N + n0 + c] : __float2half(0.f);
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb[2];
    for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(fa[i], &As[wm * 32 + i * 16][0], WK + WPAD);
    for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(fb[j], &Bs[0][wn * 32 + j * 16], WB + WPAD);
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    __syncthreads();
  }
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) wmma::store_matrix_sync(&Cs[wm * 32 + i * 16][wn * 32 + j * 16], acc[i][j], WB + 4, wmma::mem_row_major);
  __syncthreads();
  for (int e = threadIdx.x; e < WB * WB; e += blockDim.x) {
    const int r = e / WB, c = e % WB;
    if (m0 + r < M && n0 + c < N) C[(size_t)(m0 + r) * N + n0 + c] = __float2half(Cs[r][c]);
  }
#endif
}

void solve(const __half* A, const __half* B, __half* C, int batch, int M, int N, int K) {
  hbmm_wmma<<<dim3((N + WB - 1) / WB, (M + WB - 1) / WB, batch), 128>>>(A, B, C, M, N, K);
}
