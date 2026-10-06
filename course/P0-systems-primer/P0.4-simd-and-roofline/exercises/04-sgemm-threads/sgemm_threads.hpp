// Exercise 4 starter.
#pragma once
#include "sgemm.hpp"
#include "thread_pool.hpp"

namespace d1 {
inline void sgemm_threads(int M, int N, int K, const float* A, const float* B, float* C, s2s::ThreadPool& pool) {
  (void)M; (void)N; (void)K; (void)A; (void)B; (void)C; (void)pool;  // TODO: split rows of C across the pool
}
}  // namespace d1
