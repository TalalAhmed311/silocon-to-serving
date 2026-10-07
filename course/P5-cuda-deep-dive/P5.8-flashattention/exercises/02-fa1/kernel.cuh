// Exercise 2 starter: FlashAttention-1 structure. Outer loop over K/V tiles; for each, update every query row's running
// (m, l, O_unnormalised) — kept in the global workspaces Oacc [BH·N·64], M and L [BH·N] — then normalise at the end.
// Never materialise the N×N score matrix. Same signature as exercise 1 (S is unused).
#pragma once
#include <cuda_fp16.h>

inline void attention(const __half* Q, const __half* K, const __half* V, __half* O, float* Oacc, float* M, float* L,
                      int BH, int N, bool causal) {
  (void)Q; (void)K; (void)V; (void)O; (void)Oacc; (void)M; (void)L; (void)BH; (void)N; (void)causal;   // TODO
}
