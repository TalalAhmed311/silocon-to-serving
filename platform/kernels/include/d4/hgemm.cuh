// d4/hgemm.cuh — tensor-core HGEMM (P5.7): fp16 A[M×K], B[K×N] row-major, fp32 accumulate, fp32 C[M×N].
//   hgemm_wmma      : nvcuda::wmma 16×16×16 fragments (sm_70+)
//   hgemm_mma       : ldmatrix + mma.sync.m16n8k16 PTX, explicit fragment layouts (sm_80+)
//   hgemm_mma_async : the same + cp.async multi-stage shared-memory pipeline (sm_80+), STAGES = 2 or 3
// All three: block tile 128×128, BK = 32, 8 warps (2×4), warp tile 64×32. Shapes: M%128 = N%128 = K%32 = 0.
// Baseline: cuBLAS cublasGemmEx (fp16 in, fp32 compute/out). L5 exit check: ≥ 50% of cuBLAS.
#pragma once
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>

#include "common.cuh"

namespace d4 {

constexpr int HBM = 128, HBN = 128, HBK = 32, HPAD = 8;   // +8 halves (16 B) per row: no ldmatrix bank conflicts
constexpr int H_THREADS = 256;

// ---- WMMA (sm_70+) -------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(H_THREADS) hgemm_wmma(int M, int N, int K, const __half* __restrict__ A,
                                                         const __half* __restrict__ B, float* __restrict__ C) {
#if __CUDA_ARCH__ >= 700
  using namespace nvcuda;
  __shared__ __align__(32) __half As[HBM][HBK + HPAD];
  __shared__ __align__(32) __half Bs[HBK][HBN + HPAD];
  const int warp = threadIdx.x / 32, wm = warp / 4, wn = warp % 4;          // warp tile at (wm·64, wn·32)
  const int m0 = blockIdx.y * HBM, n0 = blockIdx.x * HBN;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);
  for (int k0 = 0; k0 < K; k0 += HBK) {
#pragma unroll
    for (int c = threadIdx.x; c < HBM * HBK / 8; c += H_THREADS) {          // 16-byte chunks (8 halves)
      const int r = c / (HBK / 8), col = (c % (HBK / 8)) * 8;
      *reinterpret_cast<uint4*>(&As[r][col]) = *reinterpret_cast<const uint4*>(A + (size_t)(m0 + r) * K + k0 + col);
    }
#pragma unroll
    for (int c = threadIdx.x; c < HBK * HBN / 8; c += H_THREADS) {
      const int r = c / (HBN / 8), col = (c % (HBN / 8)) * 8;
      *reinterpret_cast<uint4*>(&Bs[r][col]) = *reinterpret_cast<const uint4*>(B + (size_t)(k0 + r) * N + n0 + col);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < HBK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[4];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb[2];
#pragma unroll
      for (int i = 0; i < 4; ++i) wmma::load_matrix_sync(fa[i], &As[wm * 64 + i * 16][kk], HBK + HPAD);
#pragma unroll
      for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(fb[j], &Bs[kk][wn * 32 + j * 16], HBN + HPAD);
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma::store_matrix_sync(C + (size_t)(m0 + wm * 64 + i * 16) * N + n0 + wn * 32 + j * 16, acc[i][j], N, wmma::mem_row_major);
#endif
}

// ---- PTX helpers (sm_80+) -----------------------------------------------------------------------------------------
__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], const void* smem) {
#if __CUDA_ARCH__ >= 800
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
#endif
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t r[2], const void* smem) {
#if __CUDA_ARCH__ >= 800
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
#endif
}

// D = A·B + D, m16n8k16, A row-major fragment (4 regs), B "col" fragment (2 regs), fp32 accumulators (4).
__device__ __forceinline__ void mma_16816(float d[4], const uint32_t a[4], const uint32_t b[2]) {
#if __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
#if __CUDA_ARCH__ >= 800
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(gmem));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

// Shared-memory tile views for the mma kernels (row strides in halves).
constexpr int A_LD = HBK + HPAD, B_LD = HBN + HPAD;
constexpr int A_TILE = HBM * A_LD, B_TILE = HBK * B_LD;   // halves per stage

// One BK = 32 slice of MMAs for a warp: 2 × k16 steps × (4 m16 tiles × 4 n8 tiles).
// Fragment mapping (PTX ISA, mma.m16n8k16): A via ldmatrix.x4 — lane l supplies the address of row (l % 16),
// column 8·(l / 16) of the 16×16 A tile; B via ldmatrix.x2.trans of the row-major K×N tile — lane l (< 16) supplies
// row k = l, so the transposed 8×8 pieces land as the "col" B fragment.
__device__ __forceinline__ void warp_mma_tile(const __half* As, const __half* Bs, int wm, int wn, int lane, float acc[4][4][4]) {
#pragma unroll
  for (int kk = 0; kk < HBK; kk += 16) {
    uint32_t a[4][4], b[4][2];
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) ldmatrix_x4(a[mi], As + (wm * 64 + mi * 16 + (lane % 16)) * A_LD + kk + (lane / 16) * 8);
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) ldmatrix_x2_trans(b[ni], Bs + (kk + (lane % 16)) * B_LD + wn * 32 + ni * 8);
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) mma_16816(acc[mi][ni], a[mi], b[ni]);
  }
}

// Accumulator layout (m16n8): c0,c1 at (row g, cols 2t, 2t+1), c2,c3 at (row g+8, same cols); g = lane/4, t = lane%4.
__device__ __forceinline__ void store_acc(float* C, int N, int m0, int n0, int wm, int wn, int lane, const float acc[4][4][4]) {
  const int g = lane / 4, t = lane % 4;
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int r = m0 + wm * 64 + mi * 16 + g, c = n0 + wn * 32 + ni * 8 + 2 * t;
      *reinterpret_cast<float2*>(C + (size_t)r * N + c) = make_float2(acc[mi][ni][0], acc[mi][ni][1]);
      *reinterpret_cast<float2*>(C + (size_t)(r + 8) * N + c) = make_float2(acc[mi][ni][2], acc[mi][ni][3]);
    }
}

__device__ __forceinline__ void load_tiles_sync(__half* As, __half* Bs, const __half* A, const __half* B, int N, int K, int m0, int n0, int k0) {
#pragma unroll
  for (int c = threadIdx.x; c < HBM * HBK / 8; c += H_THREADS) {
    const int r = c / (HBK / 8), col = (c % (HBK / 8)) * 8;
    *reinterpret_cast<uint4*>(As + r * A_LD + col) = *reinterpret_cast<const uint4*>(A + (size_t)(m0 + r) * K + k0 + col);
  }
#pragma unroll
  for (int c = threadIdx.x; c < HBK * HBN / 8; c += H_THREADS) {
    const int r = c / (HBN / 8), col = (c % (HBN / 8)) * 8;
    *reinterpret_cast<uint4*>(Bs + r * B_LD + col) = *reinterpret_cast<const uint4*>(B + (size_t)(k0 + r) * N + n0 + col);
  }
}

__device__ __forceinline__ void load_tiles_async(__half* As, __half* Bs, const __half* A, const __half* B, int N, int K, int m0, int n0, int k0) {
#pragma unroll
  for (int c = threadIdx.x; c < HBM * HBK / 8; c += H_THREADS) {
    const int r = c / (HBK / 8), col = (c % (HBK / 8)) * 8;
    cp_async_16(As + r * A_LD + col, A + (size_t)(m0 + r) * K + k0 + col);
  }
#pragma unroll
  for (int c = threadIdx.x; c < HBK * HBN / 8; c += H_THREADS) {
    const int r = c / (HBN / 8), col = (c % (HBN / 8)) * 8;
    cp_async_16(Bs + r * B_LD + col, B + (size_t)(k0 + r) * N + n0 + col);
  }
}

// ---- mma.sync, synchronous loads (sm_80+) ------------------------------------------------------------------------
__global__ void __launch_bounds__(H_THREADS) hgemm_mma(int M, int N, int K, const __half* __restrict__ A,
                                                        const __half* __restrict__ B, float* __restrict__ C) {
#if __CUDA_ARCH__ >= 800
  __shared__ __align__(16) __half As[A_TILE];
  __shared__ __align__(16) __half Bs[B_TILE];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, wm = warp / 4, wn = warp % 4;
  const int m0 = blockIdx.y * HBM, n0 = blockIdx.x * HBN;
  float acc[4][4][4] = {};
  for (int k0 = 0; k0 < K; k0 += HBK) {
    load_tiles_sync(As, Bs, A, B, N, K, m0, n0, k0);
    __syncthreads();
    warp_mma_tile(As, Bs, wm, wn, lane, acc);
    __syncthreads();
  }
  store_acc(C, N, m0, n0, wm, wn, lane, acc);
#endif
}

// ---- mma.sync + cp.async pipeline (sm_80+) -----------------------------------------------------------------------
// STAGES shared-memory buffers. Prologue issues STAGES-1 tiles; each iteration waits until the oldest outstanding
// group (tile t) has landed, issues tile t+STAGES-1 into the buffer freed last iteration, then computes tile t.
template <int STAGES>
__global__ void __launch_bounds__(H_THREADS) hgemm_mma_async(int M, int N, int K, const __half* __restrict__ A,
                                                              const __half* __restrict__ B, float* __restrict__ C) {
#if __CUDA_ARCH__ >= 800
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __half* As = reinterpret_cast<__half*>(smem_raw);
  __half* Bs = As + STAGES * A_TILE;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, wm = warp / 4, wn = warp % 4;
  const int m0 = blockIdx.y * HBM, n0 = blockIdx.x * HBN, tiles = K / HBK;
  float acc[4][4][4] = {};
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < tiles) load_tiles_async(As + s * A_TILE, Bs + s * B_TILE, A, B, N, K, m0, n0, s * HBK);
    cp_async_commit();                                     // commit even when empty: keeps group counting uniform
  }
  for (int t = 0; t < tiles; ++t) {
    cp_async_wait<STAGES - 2>();                           // tile t has arrived (for this thread's copies)
    __syncthreads();                                       // …and for everyone's; buffer (t-1)%S is now free
    const int nt = t + STAGES - 1;
    if (nt < tiles) load_tiles_async(As + (nt % STAGES) * A_TILE, Bs + (nt % STAGES) * B_TILE, A, B, N, K, m0, n0, nt * HBK);
    cp_async_commit();
    warp_mma_tile(As + (t % STAGES) * A_TILE, Bs + (t % STAGES) * B_TILE, wm, wn, lane, acc);
  }
  store_acc(C, N, m0, n0, wm, wn, lane, acc);
#endif
}

inline bool hgemm_supported(int variant, int M, int N, int K) {
  const int cc = compute_capability();
  if (variant == 0 && cc < 70) return false;
  if (variant >= 1 && cc < 80) return false;
  return M % HBM == 0 && N % HBN == 0 && K % HBK == 0;
}

// variant: 0 = WMMA, 1 = mma.sync, 2 = mma.sync + cp.async (2 stages), 3 = mma.sync + cp.async (3 stages)
inline void hgemm(int variant, int M, int N, int K, const __half* A, const __half* B, float* C, cudaStream_t s = 0) {
  const dim3 grid(N / HBN, M / HBM);
  if (variant == 0) { hgemm_wmma<<<grid, H_THREADS, 0, s>>>(M, N, K, A, B, C); return; }
  if (variant == 1) { hgemm_mma<<<grid, H_THREADS, 0, s>>>(M, N, K, A, B, C); return; }
  const int stages = variant == 2 ? 2 : 3;
  const size_t smem = size_t(stages) * (A_TILE + B_TILE) * sizeof(__half);
  if (stages == 2) {
    cudaFuncSetAttribute(hgemm_mma_async<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    hgemm_mma_async<2><<<grid, H_THREADS, smem, s>>>(M, N, K, A, B, C);
  } else {
    cudaFuncSetAttribute(hgemm_mma_async<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    hgemm_mma_async<3><<<grid, H_THREADS, smem, s>>>(M, N, K, A, B, C);
  }
}

// Row-major fp16 GEMM with fp32 output via cuBLAS (column-major): Cᵀ = Bᵀ·Aᵀ.
inline void hgemm_cublas(cublasHandle_t h, int M, int N, int K, const __half* A, const __half* B, float* C) {
  const float one = 1.f, zero = 0.f;
  cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &zero, C, CUDA_R_32F, N,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// ---- int8 GEMM with __dp4a (sm_61+) ------------------------------------------------------------------------------
// C[m][n] = Σ_k A[m][k]·Bt[n][k], A: M×K int8 row-major, Bt: N×K int8 row-major (B transposed, so both operands are
// contiguous in k — the layout quantized weights usually have), C int32. 32×32 tiles of packed char4 in smem.
__global__ void igemm_dp4a(int M, int N, int K, const int8_t* __restrict__ A, const int8_t* __restrict__ Bt, int32_t* __restrict__ C) {
  __shared__ int As[32][9], Bs[32][9];                      // 32 rows × (32 k / 4) packed, +1 pad
  const int tx = threadIdx.x, ty = threadIdx.y;              // block (32, 8): each thread 4 rows
  const int n = blockIdx.x * 32 + tx;
  int acc[4] = {0, 0, 0, 0};
  for (int k0 = 0; k0 < K; k0 += 32) {
    for (int r = ty; r < 32; r += 8) {
      if (tx < 8) {
        const int m = blockIdx.y * 32 + r, nn = blockIdx.x * 32 + r;
        As[r][tx] = (m < M) ? reinterpret_cast<const int*>(A + (size_t)m * K + k0)[tx] : 0;
        Bs[r][tx] = (nn < N) ? reinterpret_cast<const int*>(Bt + (size_t)nn * K + k0)[tx] : 0;
      }
    }
    __syncthreads();
#pragma unroll
    for (int k4 = 0; k4 < 8; ++k4)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[i] = __dp4a(As[ty * 4 + i][k4], Bs[tx][k4], acc[i]);   // 4 int8 MACs per instr
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int m = blockIdx.y * 32 + ty * 4 + i;
    if (m < M && n < N) C[(size_t)m * N + n] = acc[i];
  }
}

// K must be a multiple of 32 (pad weights at quantization time).
inline void igemm(int M, int N, int K, const int8_t* A, const int8_t* Bt, int32_t* C, cudaStream_t s = 0) {
  igemm_dp4a<<<dim3(ceil_div(N, 32), ceil_div(M, 32)), dim3(32, 8), 0, s>>>(M, N, K, A, Bt, C);
}

}  // namespace d4
