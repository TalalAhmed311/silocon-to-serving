// LeetGPU #53 Causal Self-Attention — Lane B L6 solution. Q, K, V: N×d single head; query i attends to keys j ≤ i.
// Fully masked K/V tiles are skipped (≈ half the work).
#include "../_shared/attn_core.cuh"

void solve(const float* Q, const float* K, const float* V, float* output, int N, int d) {
  attn::Params p;
  p.q = Q; p.k = K; p.v = V; p.o = output;
  p.Lq = p.Lk = N; p.d = d; p.causal = true;
  attn::set_bhld(p);
  attn::launch(p);
}
