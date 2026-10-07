// Exercise 1 starter: naive attention. Q, K, V, O: [BH, N, 64] fp16. S: a B·H·N·N fp32 workspace you may use.
// Steps: S = Q·Kᵀ / sqrt(64) (mask j > i with -inf if causal) → row softmax → O = P·V.
#pragma once
#include <cuda_fp16.h>

inline void attention(const __half* Q, const __half* K, const __half* V, __half* O, float* S, int BH, int N, bool causal) {
  (void)Q; (void)K; (void)V; (void)O; (void)S; (void)BH; (void)N; (void)causal;   // TODO
}
