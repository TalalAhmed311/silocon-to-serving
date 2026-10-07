// LeetGPU #6 Softmax Attention — Lane B L6 solution (exit check is P5.8: FA-2 ≥ 5× naive at seq 4k).
// out = softmax(Q·Kᵀ/√d)·V, Q: M×d, K: N×d, V: N×d (single head), fp32. Our signature; check the statement.
#include "../_shared/attn_core.cuh"

void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
  attn::Params p;
  p.q = Q; p.k = K; p.v = V; p.o = output;
  p.Lq = M; p.Lk = N; p.d = d;
  attn::set_bhld(p);
  attn::launch(p);
}
