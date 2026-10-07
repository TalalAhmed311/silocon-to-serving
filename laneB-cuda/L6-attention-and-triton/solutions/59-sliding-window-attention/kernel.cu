// LeetGPU #59 Sliding Window Self-Attention — Lane B L6 solution. Q, K, V: N×d single head; causal + window w:
// query i attends to keys j with i − w < j ≤ i (Mistral-style). K/V tiles entirely outside the band are skipped, so
// cost is O(N·w), not O(N²). Check the statement: some variants use a symmetric window (|i − j| < w, non-causal).
#include "../_shared/attn_core.cuh"

void solve(const float* Q, const float* K, const float* V, float* output, int N, int d, int window) {
  attn::Params p;
  p.q = Q; p.k = K; p.v = V; p.o = output;
  p.Lq = p.Lk = N; p.d = d; p.causal = true; p.window = window;
  attn::set_bhld(p);
  attn::launch(p);
}
