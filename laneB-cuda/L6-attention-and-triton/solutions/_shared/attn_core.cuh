// attn_core.cuh — the one flexible FA-2-style attention forward used by the L6 attention solutions (fp32 in/out,
// fp32 math, CUDA cores). Copy it into your LeetGPU submission together with the problem's solve(); locally the
// solutions #include it. Features, all by parameter: causal (bottom-right aligned when Lq ≠ Lk), sliding window,
// ALiBi slopes, attention sinks, GQA/MQA head mapping, cross-attention (Lq ≠ Lk), and variable-length batches
// (cu_seqlens, packed [total_tokens, heads, d] layout). Head dim d ≤ DMAX (template), any d via masked loops.
//
// Algorithm (P5.8): one block = FBR query rows (one thread each) of one (sequence, head); loop over K/V tiles of FBC
// keys staged in shared memory; per tile: scores → tile max → ONE rescale of (o, l) → accumulate p·V; causal/window
// tiles that are fully masked are skipped. Output o / l.
#pragma once
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

namespace attn {

struct Params {
  const float *q, *k, *v;
  float* o;
  int B = 1, H = 1, Hkv = 1;            // query heads, key/value heads (H % Hkv == 0)
  int Lq = 0, Lk = 0, d = 64;           // lengths (max lengths when varlen), head dim
  // element strides: [b][h][i][d] addressing for q/o (q*) and k/v (k*); varlen ignores the *_b strides
  long long qb = 0, qh = 0, qi = 0, kb = 0, kh = 0, ki = 0;
  float scale = 0.f;                    // 0 → 1/sqrt(d)
  bool causal = false;
  int window = 0;                       // > 0: key j visible to query position p only if p − j < window
  const float* alibi = nullptr;         // per-head slope: score += −slope · (p − j)
  const float* sinks = nullptr;         // per-head sink logit: an extra softmax term with no value vector
  const int* cu_q = nullptr;            // varlen: cumulative sequence starts [B + 1] (queries)
  const int* cu_k = nullptr;            // varlen: same for keys (nullptr → same as cu_q)
};

constexpr int FBR = 64, FBC = 32;

template <int DMAX>
__global__ void __launch_bounds__(FBR) fwd(Params p) {
  __shared__ float Ks[FBC][DMAX + 1], Vs[FBC][DMAX + 1];
  const int bh = blockIdx.y, b = bh / p.H, h = bh % p.H, kvh = h / (p.H / p.Hkv);
  int Lq = p.Lq, Lk = p.Lk;
  long long qoff, koff;
  if (p.cu_q) {                                         // varlen: this sequence's slice of the packed token axis
    const int* cuk = p.cu_k ? p.cu_k : p.cu_q;
    Lq = p.cu_q[b + 1] - p.cu_q[b];
    Lk = cuk[b + 1] - cuk[b];
    qoff = (long long)p.cu_q[b] * p.qi + (long long)h * p.qh;
    koff = (long long)cuk[b] * p.ki + (long long)kvh * p.kh;
  } else {
    qoff = (long long)b * p.qb + (long long)h * p.qh;
    koff = (long long)b * p.kb + (long long)kvh * p.kh;
  }
  const int i0 = blockIdx.x * FBR, i = i0 + threadIdx.x;
  if (i0 >= Lq) return;                                 // whole block beyond this (varlen) sequence: uniform exit
  const int shift = Lk - Lq;                            // query i sits at key position i + shift (bottom-right)
  const float scale = p.scale > 0.f ? p.scale : rsqrtf(float(p.d));
  const float slope = p.alibi ? p.alibi[h] : 0.f;
  float q[DMAX], o[DMAX];
#pragma unroll
  for (int t = 0; t < DMAX; ++t) {
    q[t] = (t < p.d && i < Lq) ? p.q[qoff + (long long)i * p.qi + t] * scale : 0.f;
    o[t] = 0.f;
  }
  float m = -INFINITY, l = 0.f;
  if (p.sinks) { m = p.sinks[h]; l = 1.f; }             // sink: e^{sink − m} = 1 in the denominator, nothing in o
  // key range this block can see
  int j_lo = 0, j_hi = Lk;
  if (p.causal) j_hi = min(Lk, i0 + FBR + shift);
  if (p.window > 0) j_lo = max(0, i0 + shift - p.window + 1);
  j_lo = (j_lo / FBC) * FBC;
  for (int j0 = j_lo; j0 < j_hi; j0 += FBC) {
    for (int e = threadIdx.x; e < FBC * DMAX; e += FBR) {
      const int r = e / DMAX, c = e % DMAX, j = j0 + r;
      const bool ok = j < Lk && c < p.d;
      Ks[r][c] = ok ? p.k[koff + (long long)j * p.ki + c] : 0.f;
      Vs[r][c] = ok ? p.v[koff + (long long)j * p.ki + c] : 0.f;
    }
    __syncthreads();
    float s[FBC], tmax = -INFINITY;
    const int pos = i + shift;
#pragma unroll
    for (int jj = 0; jj < FBC; ++jj) {
      const int j = j0 + jj;
      float acc = 0.f;
#pragma unroll
      for (int t = 0; t < DMAX; ++t) acc = fmaf(q[t], Ks[jj][t], acc);
      bool visible = j < Lk && i < Lq;
      if (p.causal && j > pos) visible = false;
      if (p.window > 0 && pos - j >= p.window) visible = false;
      s[jj] = visible ? acc - slope * float(pos - j) : -INFINITY;
      tmax = fmaxf(tmax, s[jj]);
    }
    const float m_new = fmaxf(m, tmax);
    if (m_new != -INFINITY) {
      const float corr = __expf(m - m_new);
      l *= corr;
#pragma unroll
      for (int t = 0; t < DMAX; ++t) o[t] *= corr;
#pragma unroll
      for (int jj = 0; jj < FBC; ++jj) {
        const float pj = __expf(s[jj] - m_new);
        l += pj;
#pragma unroll
        for (int t = 0; t < DMAX; ++t) o[t] = fmaf(pj, Vs[jj][t], o[t]);
      }
      m = m_new;
    }
    __syncthreads();
  }
  if (i < Lq) {
    const float inv = l > 0.f ? 1.f / l : 0.f;
    for (int t = 0; t < p.d; ++t) p.o[qoff + (long long)i * p.qi + t] = o[t] * inv;
  }
}

// grid: x = query tiles (of the longest sequence), y = B·H
inline void launch(const Params& p) {
  const dim3 grid((p.Lq + FBR - 1) / FBR, p.B * p.H);
  if (p.d <= 32) fwd<32><<<grid, FBR>>>(p);
  else if (p.d <= 64) fwd<64><<<grid, FBR>>>(p);
  else fwd<128><<<grid, FBR>>>(p);                      // register-heavy: expect spills; fine for correctness
}

// Contiguous [B, H, L, d] strides.
inline void set_bhld(Params& p) {
  p.qi = p.d; p.qh = (long long)p.Lq * p.d; p.qb = (long long)p.H * p.Lq * p.d;
  p.ki = p.d; p.kh = (long long)p.Lk * p.d; p.kb = (long long)p.Hkv * p.Lk * p.d;
}

}  // namespace attn
