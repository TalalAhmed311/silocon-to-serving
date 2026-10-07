// d4/attention.cuh — attention forward (P5.8), fp16 Q/K/V/O with fp32 math, head_dim D = 64, layout [B, H, N, D].
//   attn_naive : S = QKᵀ·scale (materialised, fp32, B·H·N·N), softmax rows, O = P·V — three kernels, O(N²) HBM traffic
//   attn_fa1   : FlashAttention-1 structure: OUTER loop over K/V tiles, inner over Q tiles; running (m, l, O) kept in
//                HBM between outer iterations (never materialises S, but rereads/rewrites O)
//   attn_fa2   : FlashAttention-2 structure: one block per Q tile, inner loop over K/V tiles, (m, l, O) in REGISTERS,
//                one rescale per K/V tile, causal tile skipping. CUDA-core math (the tensor-core version is the
//                exercise extension). L6 exit check: fa2 ≥ 5× naive at N = 4096.
// Plus paged_decode_attention: q_len = 1 over a block-table KV cache (P5.8 exercise 4, feeds P6.3).
#pragma once
#include <cuda_fp16.h>

#include "common.cuh"
#include "softmax.cuh"

namespace d4 {

constexpr int AD = 64;   // head dimension

// ---- naive ----------------------------------------------------------------------------------------------------------
__global__ void attn_scores(const __half* __restrict__ Q, const __half* __restrict__ K, float* __restrict__ S, int N, float scale, bool causal) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y, bh = blockIdx.z;
  if (j >= N) return;
  const __half* q = Q + ((size_t)bh * N + i) * AD;
  const __half* k = K + ((size_t)bh * N + j) * AD;
  float s = 0.f;
#pragma unroll 8
  for (int d = 0; d < AD; ++d) s += __half2float(q[d]) * __half2float(k[d]);
  S[((size_t)bh * N + i) * N + j] = (causal && j > i) ? -INFINITY : s * scale;
}

__global__ void attn_pv(const float* __restrict__ P, const __half* __restrict__ V, __half* __restrict__ O, int N) {
  const int d = threadIdx.x, i = blockIdx.x, bh = blockIdx.y;              // block = one output row, D threads
  const float* p = P + ((size_t)bh * N + i) * N;
  const __half* v = V + (size_t)bh * N * AD;
  float acc = 0.f;
  for (int j = 0; j < N; ++j) acc += p[j] * __half2float(v[(size_t)j * AD + d]);
  O[((size_t)bh * N + i) * AD + d] = __float2half(acc);
}

// S must hold B·H·N·N floats (that's the point: at N = 4096 and B·H = 16, 1 GiB).
inline void attn_naive(const __half* Q, const __half* K, const __half* V, __half* O, float* S, int BH, int N, bool causal, cudaStream_t s = 0) {
  const float scale = 1.f / sqrtf(float(AD));
  attn_scores<<<dim3(ceil_div(N, 256), N, BH), 256, 0, s>>>(Q, K, S, N, scale, causal);
  softmax(2, S, S, BH * N, N, s);                                          // in place, row-wise
  attn_pv<<<dim3(N, BH), AD, 0, s>>>(S, V, O, N);
}

// ---- FA-2 forward (and the building block for FA-1) -----------------------------------------------------------------
constexpr int FBR = 64;   // query rows per block (one thread per row)
constexpr int FBC = 32;   // keys per K/V tile

// One thread owns one query row: q[64], o[64] and the scores of the current tile s[32] live in registers. The block
// stages each K/V tile in shared memory (fp32) once; every thread then reads K[j][·] and V[j][·] — the same address
// across the warp → broadcast. Per tile: scores → tile max → ONE rescale of (o, l) → accumulate p·V.
__global__ void __launch_bounds__(FBR) attn_fa2(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
                                               __half* __restrict__ O, int N, float scale, bool causal) {
  __shared__ float Ks[FBC][AD + 1], Vs[FBC][AD + 1];       // +1: the cooperative load writes rows without conflicts
  const int bh = blockIdx.y, i0 = blockIdx.x * FBR, i = i0 + threadIdx.x;
  const size_t base = (size_t)bh * N * AD;
  float q[AD], o[AD];
#pragma unroll
  for (int d = 0; d < AD; ++d) { q[d] = i < N ? __half2float(Q[base + (size_t)i * AD + d]) * scale : 0.f; o[d] = 0.f; }
  float m = -INFINITY, l = 0.f;
  const int kv_end = causal ? min(N, i0 + FBR) : N;        // causal: tiles entirely to the right of this Q tile skipped
  for (int j0 = 0; j0 < kv_end; j0 += FBC) {
    for (int e = threadIdx.x; e < FBC * AD; e += FBR) {     // cooperative, coalesced K/V tile load
      const int r = e / AD, c = e % AD, j = j0 + r;
      Ks[r][c] = j < N ? __half2float(K[base + (size_t)j * AD + c]) : 0.f;
      Vs[r][c] = j < N ? __half2float(V[base + (size_t)j * AD + c]) : 0.f;
    }
    __syncthreads();
    float s[FBC], tmax = -INFINITY;
#pragma unroll
    for (int jj = 0; jj < FBC; ++jj) {
      const int j = j0 + jj;
      float acc = 0.f;
#pragma unroll
      for (int d = 0; d < AD; ++d) acc = fmaf(q[d], Ks[jj][d], acc);
      s[jj] = (j >= N || (causal && j > i)) ? -INFINITY : acc;
      tmax = fmaxf(tmax, s[jj]);
    }
    const float m_new = fmaxf(m, tmax);
    if (m_new != -INFINITY) {                               // rows with everything masked so far: nothing to do
      const float corr = __expf(m - m_new);                 // m = -inf on the first tile → corr = 0, o and l are 0
      l *= corr;
#pragma unroll
      for (int d = 0; d < AD; ++d) o[d] *= corr;
#pragma unroll
      for (int jj = 0; jj < FBC; ++jj) {
        const float p = __expf(s[jj] - m_new);              // masked: exp(-inf) = 0
        l += p;
#pragma unroll
        for (int d = 0; d < AD; ++d) o[d] = fmaf(p, Vs[jj][d], o[d]);
      }
      m = m_new;
    }
    __syncthreads();
  }
  if (i < N) {
    const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
    for (int d = 0; d < AD; ++d) O[base + (size_t)i * AD + d] = __float2half(o[d] * inv);
  }
}

inline void attn_flash2(const __half* Q, const __half* K, const __half* V, __half* O, int BH, int N, bool causal, cudaStream_t s = 0) {
  attn_fa2<<<dim3(ceil_div(N, FBR), BH), FBR, 0, s>>>(Q, K, V, O, N, 1.f / sqrtf(float(AD)), causal);
}

// ---- FA-1 structure ---------------------------------------------------------------------------------------------------
// Outer loop over K/V tiles (one kernel launch per tile here, to make the HBM round-trips explicit): every launch
// re-reads Q and the running (m, l, O_unnormalised) from global memory, folds one K/V tile in, and writes them back.
// Same math as FA-2, more HBM traffic — which is exactly FA-2's first improvement.
__global__ void __launch_bounds__(FBR) attn_fa1_step(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
                                                    float* __restrict__ Oacc, float* __restrict__ M, float* __restrict__ L,
                                                    int N, int j0, float scale, bool causal) {
  __shared__ float Ks[FBC][AD + 1], Vs[FBC][AD + 1];
  const int bh = blockIdx.y, i = blockIdx.x * FBR + threadIdx.x;
  const size_t base = (size_t)bh * N * AD;
  for (int e = threadIdx.x; e < FBC * AD; e += FBR) {
    const int r = e / AD, c = e % AD, j = j0 + r;
    Ks[r][c] = j < N ? __half2float(K[base + (size_t)j * AD + c]) : 0.f;
    Vs[r][c] = j < N ? __half2float(V[base + (size_t)j * AD + c]) : 0.f;
  }
  __syncthreads();
  if (i >= N) return;
  const size_t row = (size_t)bh * N + i;
  float s[FBC], tmax = -INFINITY;
#pragma unroll
  for (int jj = 0; jj < FBC; ++jj) {
    const int j = j0 + jj;
    float acc = 0.f;
    for (int d = 0; d < AD; ++d) acc = fmaf(__half2float(Q[base + (size_t)i * AD + d]) * scale, Ks[jj][d], acc);
    s[jj] = (j >= N || (causal && j > i)) ? -INFINITY : acc;
    tmax = fmaxf(tmax, s[jj]);
  }
  const float m = M[row], m_new = fmaxf(m, tmax);
  if (m_new == -INFINITY) return;
  const float corr = __expf(m - m_new);
  float l = L[row] * corr;
  float* o = Oacc + row * AD;
  for (int d = 0; d < AD; ++d) o[d] *= corr;                // read-modify-write of O in HBM every K/V tile
  for (int jj = 0; jj < FBC; ++jj) {
    const float p = __expf(s[jj] - m_new);
    l += p;
    for (int d = 0; d < AD; ++d) o[d] = fmaf(p, Vs[jj][d], o[d]);
  }
  M[row] = m_new;
  L[row] = l;
}

__global__ void attn_fa1_finish(const float* __restrict__ Oacc, const float* __restrict__ L, __half* __restrict__ O, long long rows) {
  for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < rows * AD; e += (long long)gridDim.x * blockDim.x) {
    const float l = L[e / AD];
    O[e] = __float2half(l > 0.f ? Oacc[e] / l : 0.f);
  }
}

__global__ void attn_fa1_init(float* Oacc, float* M, float* L, long long rows) {
  for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < rows * AD; e += (long long)gridDim.x * blockDim.x) {
    Oacc[e] = 0.f;
    if (e % AD == 0) { M[e / AD] = -INFINITY; L[e / AD] = 0.f; }
  }
}

// Workspace: Oacc B·H·N·D floats, M and L B·H·N floats each.
inline void attn_flash1(const __half* Q, const __half* K, const __half* V, __half* O, float* Oacc, float* M, float* L,
                        int BH, int N, bool causal, cudaStream_t s = 0) {
  const long long rows = (long long)BH * N;
  attn_fa1_init<<<sm_count() * 8, 256, 0, s>>>(Oacc, M, L, rows);
  for (int j0 = 0; j0 < N; j0 += FBC)
    attn_fa1_step<<<dim3(ceil_div(N, FBR), BH), FBR, 0, s>>>(Q, K, V, Oacc, M, L, N, j0, 1.f / sqrtf(float(AD)), causal);
  attn_fa1_finish<<<sm_count() * 8, 256, 0, s>>>(Oacc, L, O, rows);
}

// ---- decode attention over a paged KV cache (q_len = 1) --------------------------------------------------------------
// q: [B, H, D]; k_cache, v_cache: [num_blocks, block_size, Hkv, D]; block_table: [B, max_blocks_per_seq];
// seq_lens: [B]; out: [B, H, D]. GQA: query head h reads KV head h / (H / Hkv). One block per (b, h), 4 warps; each
// warp walks a strided subset of the sequence's tokens with its own online-softmax state; warps merge at the end.
constexpr int PD_WARPS = 4;

__global__ void __launch_bounds__(PD_WARPS * 32) paged_decode_k(const __half* __restrict__ q, const __half* __restrict__ k_cache,
                                                               const __half* __restrict__ v_cache, const int* __restrict__ block_table,
                                                               const int* __restrict__ seq_lens, __half* __restrict__ out, int H,
                                                               int Hkv, int block_size, int max_blocks, float scale) {
  const int b = blockIdx.y, h = blockIdx.x, kvh = h / (H / Hkv), warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int len = seq_lens[b];
  // each lane owns 2 of the 64 dims
  const __half2 qh = reinterpret_cast<const __half2*>(q + ((size_t)b * H + h) * AD)[lane];
  const float q0 = __low2float(qh) * scale, q1 = __high2float(qh) * scale;
  float m = -INFINITY, l = 0.f, o0 = 0.f, o1 = 0.f;
  for (int t = warp; t < len; t += PD_WARPS) {
    const int phys = block_table[(size_t)b * max_blocks + t / block_size], off = t % block_size;
    const size_t kv = (((size_t)phys * block_size + off) * Hkv + kvh) * AD;
    const __half2 kk = reinterpret_cast<const __half2*>(k_cache + kv)[lane];
    const float s = warp_sum(q0 * __low2float(kk) + q1 * __high2float(kk));   // every lane gets the full dot product
    const float m_new = fmaxf(m, s), corr = __expf(m - m_new), p = __expf(s - m_new);
    const __half2 vv = reinterpret_cast<const __half2*>(v_cache + kv)[lane];
    l = l * corr + p;
    o0 = o0 * corr + p * __low2float(vv);
    o1 = o1 * corr + p * __high2float(vv);
    m = m_new;
  }
  __shared__ float sm[PD_WARPS], sl[PD_WARPS], so[PD_WARPS][AD];
  if (lane == 0) { sm[warp] = m; sl[warp] = l; }
  so[warp][2 * lane] = o0;
  so[warp][2 * lane + 1] = o1;
  __syncthreads();
  if (warp == 0) {
    float M = -INFINITY;
    for (int w = 0; w < PD_WARPS; ++w) M = fmaxf(M, sm[w]);
    float L = 0.f, a0 = 0.f, a1 = 0.f;
    for (int w = 0; w < PD_WARPS; ++w) {
      if (sm[w] == -INFINITY) continue;                       // a warp that saw no tokens
      const float c = __expf(sm[w] - M);
      L += sl[w] * c;
      a0 += so[w][2 * lane] * c;
      a1 += so[w][2 * lane + 1] * c;
    }
    const float inv = L > 0.f ? 1.f / L : 0.f;
    reinterpret_cast<__half2*>(out + ((size_t)b * H + h) * AD)[lane] = __floats2half2_rn(a0 * inv, a1 * inv);
  }
}

inline void paged_decode_attention(const __half* q, const __half* k_cache, const __half* v_cache, const int* block_table,
                                   const int* seq_lens, __half* out, int B, int H, int Hkv, int block_size, int max_blocks,
                                   cudaStream_t s = 0) {
  paged_decode_k<<<dim3(H, B), PD_WARPS * 32, 0, s>>>(q, k_cache, v_cache, block_table, seq_lens, out, H, Hkv, block_size,
                                                      max_blocks, 1.f / sqrtf(float(AD)));
}

}  // namespace d4
