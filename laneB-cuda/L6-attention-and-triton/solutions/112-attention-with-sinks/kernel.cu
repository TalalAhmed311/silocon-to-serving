// LeetGPU #112 Attention with Sinks — Lane B L6 solution. Q, K, V: [H, N, d], causal; each head has a learned sink
// logit s_h that joins the softmax normaliser but has no value vector (gpt-oss style): out_i = Σ_j e^{a_ij}·v_j /
// (e^{s_h} + Σ_j e^{a_ij}). Implemented by starting the online-softmax state at (m, l) = (s_h, 1). Check the statement:
// "attention sinks" also names StreamingLLM's keep-the-first-k-tokens window, which is a mask, not a logit.
#include "../_shared/attn_core.cuh"

void solve(const float* Q, const float* K, const float* V, const float* sinks, float* output, int N, int d, int H) {
  attn::Params p;
  p.q = Q; p.k = K; p.v = V; p.o = output;
  p.H = p.Hkv = H; p.Lq = p.Lk = N; p.d = d; p.causal = true; p.sinks = sinks;
  attn::set_bhld(p);
  attn::launch(p);
}
