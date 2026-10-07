#pragma once
#include <d4/attention.cuh>
inline void attention(const __half* Q, const __half* K, const __half* V, __half* O, int BH, int N, bool causal) {
  d4::attn_flash2(Q, K, V, O, BH, N, causal);
}
