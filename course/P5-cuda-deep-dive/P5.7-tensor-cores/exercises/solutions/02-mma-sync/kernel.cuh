// S2S_MIN_SM 80
#pragma once
#include <d4/hgemm.cuh>
inline void hgemm_mma(int M, int N, int K, const __half* A, const __half* B, float* C) { d4::hgemm(1, M, N, K, A, B, C); }
