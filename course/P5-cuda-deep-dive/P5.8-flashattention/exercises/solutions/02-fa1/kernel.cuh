#pragma once
#include <d4/attention.cuh>
inline void attention(const __half* Q, const __half* K, const __half* V, __half* O, float* Oacc, float* M, float* L,
                      int BH, int N, bool causal) {
  d4::attn_flash1(Q, K, V, O, Oacc, M, L, BH, N, causal);
}
