// LeetGPU #96 INT8 KV-Cache Attention — Lane B L6 solution. Decode attention (one query per sequence) over an int8
// KV cache with per-(sequence, kv-head, token) scales: k = k_q · k_scale, v = v_q · v_scale (symmetric).
// q: [B, H, D] fp32; k_q, v_q: [B, Hkv, L, D] int8; k_scale, v_scale: [B, Hkv, L]; lens: [B]; out: [B, H, D].
// Our layout/scale granularity (per token); check the statement (per-channel or per-tensor scales change one line).
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

constexpr int WARPS = 4, DMAX = 128;

// One block per (b, h); warps take tokens round-robin; each lane owns D/32 dims (≤ 4). Dequantise in registers:
// the cache is read as int8 — 4× fewer bytes than fp32, 2× fewer than fp16 — which is the whole point for decode.
__global__ void __launch_bounds__(WARPS * 32) int8_decode(const float* __restrict__ q, const int8_t* __restrict__ kq,
                                                          const int8_t* __restrict__ vq, const float* __restrict__ ks,
                                                          const float* __restrict__ vs, const int* __restrict__ lens,
                                                          float* __restrict__ out, int H, int Hkv, int L, int D) {
  const int b = blockIdx.y, h = blockIdx.x, kvh = h / (H / Hkv), warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int per = (D + 31) / 32;                          // dims per lane
  const float scale = rsqrtf(float(D));
  float qr[DMAX / 32], o[DMAX / 32];
  for (int t = 0; t < per; ++t) { const int dd = lane + 32 * t; qr[t] = dd < D ? q[((size_t)b * H + h) * D + dd] * scale : 0.f; o[t] = 0.f; }
  float m = -INFINITY, l = 0.f;
  const size_t base = ((size_t)b * Hkv + kvh) * L;
  const int len = lens[b];
  for (int j = warp; j < len; j += WARPS) {
    const int8_t* kr = kq + (base + j) * D;
    float s = 0.f;
    for (int t = 0; t < per; ++t) { const int dd = lane + 32 * t; if (dd < D) s += qr[t] * float(kr[dd]); }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
    s *= ks[base + j];                                    // scale the dot product once instead of each element
    const float m_new = fmaxf(m, s), c = __expf(m - m_new), pj = __expf(s - m_new);
    const float vsc = vs[base + j] * pj;
    const int8_t* vr = vq + (base + j) * D;
    l = l * c + pj;
    for (int t = 0; t < per; ++t) { const int dd = lane + 32 * t; o[t] = o[t] * c + (dd < D ? vsc * float(vr[dd]) : 0.f); }
    m = m_new;
  }
  __shared__ float sm[WARPS], sl[WARPS], so[WARPS][DMAX];
  if (lane == 0) { sm[warp] = m; sl[warp] = l; }
  for (int t = 0; t < per; ++t) { const int dd = lane + 32 * t; if (dd < D) so[warp][dd] = o[t]; }
  __syncthreads();
  if (warp == 0) {
    float M = -INFINITY;
    for (int w = 0; w < WARPS; ++w) M = fmaxf(M, sm[w]);
    float Lsum = 0.f;
    for (int w = 0; w < WARPS; ++w) if (sm[w] != -INFINITY) Lsum += sl[w] * __expf(sm[w] - M);
    for (int t = 0; t < per; ++t) {
      const int dd = lane + 32 * t;
      if (dd >= D) continue;
      float a = 0.f;
      for (int w = 0; w < WARPS; ++w) if (sm[w] != -INFINITY) a += so[w][dd] * __expf(sm[w] - M);
      out[((size_t)b * H + h) * D + dd] = Lsum > 0.f ? a / Lsum : 0.f;
    }
  }
}

void solve(const float* q, const int8_t* k_q, const int8_t* v_q, const float* k_scale, const float* v_scale, const int* lens,
           float* out, int B, int H, int Hkv, int L, int D) {
  int8_decode<<<dim3(H, B), WARPS * 32>>>(q, k_q, v_q, k_scale, v_scale, lens, out, H, Hkv, L, D);
}
