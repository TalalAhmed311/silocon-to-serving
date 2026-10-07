// LeetGPU #26 Multi-Head Cross-Attention — Lane B L6 solution. Q: Lq × d_model (decoder states), K, V: Lk × d_model
// (encoder states), h heads packed in the last dim; non-causal; Lq ≠ Lk. Same addressing as #12 with two lengths.
#include "../_shared/attn_core.cuh"

void solve(const float* Q, const float* K, const float* V, float* output, int Lq, int Lk, int d_model, int h) {
  attn::Params p;
  p.q = Q; p.k = K; p.v = V; p.o = output;
  p.H = p.Hkv = h; p.Lq = Lq; p.Lk = Lk; p.d = d_model / h;
  p.qi = p.ki = d_model; p.qh = p.kh = p.d; p.qb = p.kb = 0;
  attn::launch(p);
}
