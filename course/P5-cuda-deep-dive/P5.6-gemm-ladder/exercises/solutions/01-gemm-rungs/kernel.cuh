#pragma once
#include <d4/gemm.cuh>
inline bool sgemm_supported(int rung, int M, int N, int K) { return d4::sgemm_supported(rung, M, N, K); }
inline void sgemm(int rung, int M, int N, int K, const float* A, const float* B, float* C) { d4::sgemm(rung, M, N, K, A, B, C); }
