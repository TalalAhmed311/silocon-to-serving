#pragma once
#include <d4/attention.cuh>
inline void attention(const __half* Q, const __half* K, const __half* V, __half* O, float* S, int BH, int N, bool causal) {
  d4::attn_naive(Q, K, V, O, S, BH, N, causal);
}
