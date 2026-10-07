// Exercise 3 starter: FlashAttention-2 forward + causal. One block per tile of query rows; loop over K/V tiles held in
// shared memory; (m, l, o) in registers; rescale once per K/V tile; skip K/V tiles that are fully masked (causal).
#pragma once
#include <cuda_fp16.h>

inline void attention(const __half* Q, const __half* K, const __half* V, __half* O, int BH, int N, bool causal) {
  (void)Q; (void)K; (void)V; (void)O; (void)BH; (void)N; (void)causal;   // TODO
}
