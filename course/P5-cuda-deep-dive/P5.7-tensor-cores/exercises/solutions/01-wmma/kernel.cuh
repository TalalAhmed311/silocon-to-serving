#pragma once
#include <d4/hgemm.cuh>
inline void hgemm(int M, int N, int K, const __half* A, const __half* B, float* C) { d4::hgemm(0, M, N, K, A, B, C); }
