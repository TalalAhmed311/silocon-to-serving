// LeetGPU #87 Speculative Decoding Verification — Lane B L6 solution. Per sequence b, with k draft tokens:
//   for i < k: x = draft[b][i]; accept if u[b][i] < min(1, p[b][i][x] / q[b][i][x]) (exact rejection sampling);
//   at the first rejection i: emit a token sampled from normalise(max(0, p_i − q_i)) and stop;
//   if all k are accepted: emit a bonus token sampled from p_k.
// Inputs: draft [B, k] int, q_probs [B, k, V], p_probs [B, k+1, V], u_accept [B, k], u_sample [B] uniforms.
// Outputs: tokens [B, k+1] (first n_out valid), n_out [B]. Same semantics as platform/specdec/core.py (P2.6) and
// P6.5. Our signature; check the statement (it may pass logits instead of probabilities → softmax first).
#include <cuda_runtime.h>

constexpr int VT = 256;

__device__ __forceinline__ float warp_incl(float v) {
  const int lane = threadIdx.x & 31;
  for (int o = 1; o < 32; o <<= 1) { const float u = __shfl_up_sync(0xffffffffu, v, o); if (lane >= o) v += u; }
  return v;
}

// Inverse-CDF sample of index j with weight w(j) over [0, V): smallest j with Σ_{≤ j} w > u · Σ w (u in [0, 1)).
template <class W>
__device__ int sample_block(W w, int V, float u) {
  __shared__ float wt[VT / 32];
  __shared__ float total_s, run_s;
  __shared__ int found, found_min;
  float local = 0.f;
  for (int j = threadIdx.x; j < V; j += VT) local += w(j);
  for (int o = 16; o > 0; o >>= 1) local += __shfl_xor_sync(0xffffffffu, local, o);
  if ((threadIdx.x & 31) == 0) wt[threadIdx.x / 32] = local;
  __syncthreads();
  if (threadIdx.x == 0) { float t = 0.f; for (int i = 0; i < VT / 32; ++i) t += wt[i]; total_s = t; run_s = 0.f; found = -1; found_min = 0x7fffffff; }
  __syncthreads();
  const float target = u * total_s;
  for (int c0 = 0; c0 < V; c0 += VT) {                    // scan chunk by chunk until the crossing is found
    const int j = c0 + threadIdx.x;
    const float x = j < V ? w(j) : 0.f;
    float inc = warp_incl(x);
    __syncthreads();
    if ((threadIdx.x & 31) == 31) wt[threadIdx.x / 32] = inc;
    __syncthreads();
    float before = run_s;
    for (int i = 0; i < threadIdx.x / 32; ++i) before += wt[i];
    const float cum = before + inc;
    if (j < V && x > 0.f && cum > target && cum - x <= target) atomicMin(&found_min, j);   // smallest j with CDF(j) > u·total
    __syncthreads();
    if (threadIdx.x == VT - 1) run_s = cum;
    if (threadIdx.x == 0 && found_min != 0x7fffffff) found = found_min;
    __syncthreads();
    if (found >= 0) break;
  }
  if (found < 0) {                                        // rounding left target just above the total: last positive weight
    if (threadIdx.x == 0) { for (int j = V - 1; j >= 0; --j) if (w(j) > 0.f) { found = j; break; } }
    __syncthreads();
  }
  return found;
}

__global__ void __launch_bounds__(VT) verify_k(const int* __restrict__ draft, const float* __restrict__ qp, const float* __restrict__ pp,
                                              const float* __restrict__ ua, const float* __restrict__ us,
                                              int* __restrict__ tokens, int* __restrict__ n_out, int k, int V) {
  const int b = blockIdx.x;
  __shared__ int n_acc;
  if (threadIdx.x == 0) {                                 // k is small (≤ 8): the accept chain is sequential by nature
    int n = 0;
    for (; n < k; ++n) {
      const int x = draft[b * k + n];
      const float p = pp[((size_t)b * (k + 1) + n) * V + x], q = qp[((size_t)b * k + n) * V + x];
      if (!(ua[b * k + n] < fminf(1.f, q > 0.f ? p / q : 1.f))) break;
      tokens[b * (k + 1) + n] = x;
    }
    n_acc = n;
  }
  __syncthreads();
  const int n = n_acc;
  const float* p = pp + ((size_t)b * (k + 1) + n) * V;
  int tok;
  if (n < k) {
    const float* q = qp + ((size_t)b * k + n) * V;
    tok = sample_block([&](int j) { return fmaxf(p[j] - q[j], 0.f); }, V, us[b]);   // residual distribution
  } else {
    tok = sample_block([&](int j) { return p[j]; }, V, us[b]);                       // bonus token from the target
  }
  if (threadIdx.x == 0) { tokens[b * (k + 1) + n] = tok; n_out[b] = n + 1; }
}

void solve(const int* draft, const float* q_probs, const float* p_probs, const float* u_accept, const float* u_sample,
           int* tokens, int* n_out, int B, int k, int V) {
  verify_k<<<B, VT>>>(draft, q_probs, p_probs, u_accept, u_sample, tokens, n_out, k, V);
}
