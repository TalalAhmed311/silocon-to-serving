// S2S_MIN_SM 80
// Exercise 2 starter: HGEMM with ldmatrix + mma.sync.m16n8k16 (sm_80+). The PTX wrappers (d4::ldmatrix_x4,
// d4::ldmatrix_x2_trans, d4::mma_16816) and the tile loader are given — write the warp's fragment loop and the
// accumulator store yourself (the two places where the PTX ISA's fragment layouts matter).
#pragma once
#include <d4/hgemm.cuh>

__global__ void ex_hgemm_mma(int M, int N, int K, const __half* __restrict__ A, const __half* __restrict__ B, float* __restrict__ C) {
#if __CUDA_ARCH__ >= 800
  __shared__ __align__(16) __half As[d4::A_TILE];
  __shared__ __align__(16) __half Bs[d4::B_TILE];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, wm = warp / 4, wn = warp % 4;
  const int m0 = blockIdx.y * d4::HBM, n0 = blockIdx.x * d4::HBN;
  float acc[4][4][4] = {};
  for (int k0 = 0; k0 < K; k0 += d4::HBK) {
    d4::load_tiles_sync(As, Bs, A, B, N, K, m0, n0, k0);
    __syncthreads();
    // TODO: for kk in {0, 16}: ldmatrix_x4 four 16×16 A fragments (warp rows wm·64 + mi·16), ldmatrix_x2_trans four
    // 16×8 B fragments (warp cols wn·32 + ni·8), then 16 mma_16816 calls into acc[mi][ni].
    __syncthreads();
  }
  // TODO: store acc with the m16n8 accumulator layout: lane → rows g, g+8 and cols 2t, 2t+1 (g = lane/4, t = lane%4).
  (void)acc; (void)wm; (void)wn; (void)lane; (void)C; (void)M;
#endif
}

inline void hgemm_mma(int M, int N, int K, const __half* A, const __half* B, float* C) {
  ex_hgemm_mma<<<dim3(N / d4::HBN, M / d4::HBM), d4::H_THREADS>>>(M, N, K, A, B, C);
}
