// LeetGPU #80 Grouped Query Attention — Lane B L6 solution. Q: [H, N, d], K, V: [Hkv, N, d] with H % Hkv == 0;
// query head h reads KV head h / (H / Hkv). Non-causal by default; pass causal for decoder use. Our layout; check the
// statement (it may pack heads like #12 — then only the strides change).
#include "../_shared/attn_core.cuh"

void solve(const float* Q, const float* K, const float* V, float* output, int N, int d, int H, int Hkv, bool causal) {
  attn::Params p;
  p.q = Q; p.k = K; p.v = V; p.o = output;
  p.H = H; p.Hkv = Hkv; p.Lq = p.Lk = N; p.d = d; p.causal = causal;
  attn::set_bhld(p);
  attn::launch(p);
}
