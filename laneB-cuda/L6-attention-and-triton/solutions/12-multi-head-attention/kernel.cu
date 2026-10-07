// LeetGPU #12 Multi-Head Attention — Lane B L6 solution. Q, K, V: N × d_model with h heads packed in the last dim
// ([N, h, d_model/h]); each head attends independently (non-causal); output in the same packed layout. Heads go in the
// grid (blockIdx.y) and are addressed purely through strides — no transpose kernels.
#include "../_shared/attn_core.cuh"

void solve(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h) {
  attn::Params p;
  p.q = Q; p.k = K; p.v = V; p.o = output;
  p.H = p.Hkv = h; p.Lq = p.Lk = N; p.d = d_model / h;
  p.qi = p.ki = d_model;              // next token: skip all heads
  p.qh = p.kh = p.d;                  // next head: skip one head's dims
  p.qb = p.kb = 0;
  attn::launch(p);
}
