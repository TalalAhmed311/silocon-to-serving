// LeetGPU #102 Variable-Length Causal Attention — Lane B L6 solution. A batch of sequences of different lengths packed
// along one token axis: Q, K, V: [total_tokens, H, d]; cu_seqlens: [B + 1] prefix offsets (cu[0] = 0, cu[B] = total).
// Causal within each sequence; no padding, no cross-sequence attention. max_len sizes the grid; blocks past a short
// sequence's end exit immediately.
#include "../_shared/attn_core.cuh"

void solve(const float* Q, const float* K, const float* V, float* output, const int* cu_seqlens, int B, int max_len, int H, int d) {
  attn::Params p;
  p.q = Q; p.k = K; p.v = V; p.o = output;
  p.B = B; p.H = p.Hkv = H; p.Lq = p.Lk = max_len; p.d = d; p.causal = true;
  p.cu_q = cu_seqlens;
  p.qi = p.ki = (long long)H * d; p.qh = p.kh = d;
  attn::launch(p);
}
