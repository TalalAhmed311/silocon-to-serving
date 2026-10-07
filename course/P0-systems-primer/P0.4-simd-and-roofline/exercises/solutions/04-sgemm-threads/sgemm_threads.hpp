// Exercise 4 solution.
#pragma once
#include "sgemm.hpp"
#include "thread_pool.hpp"

namespace d1 {
inline void sgemm_threads(int M, int N, int K, const float* A, const float* B, float* C, s2s::ThreadPool& pool) {
  pool.parallel_for(M, [&](int64_t i0, int64_t i1) {
    // Each worker owns rows [i0, i1) of C: disjoint writes, shared read-only B.
    sgemm_simd(int(i1 - i0), N, K, A + i0 * K, B, C + i0 * N);
  });
}
}  // namespace d1
