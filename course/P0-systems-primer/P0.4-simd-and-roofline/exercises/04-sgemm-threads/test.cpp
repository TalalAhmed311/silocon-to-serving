#include <cstdio>
#include <random>
#include <vector>

#include <s2s/bench.hpp>
#include <s2s/check.hpp>
#include "sgemm_threads.hpp"

S2S_TEST(matches_single_thread) {
  const int M = 257, N = 300, K = 129;
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> d(-1, 1);
  std::vector<float> A(size_t(M) * K), B(size_t(K) * N), C1(size_t(M) * N, 0.f), C2(size_t(M) * N, 0.f);
  for (auto& x : A) x = d(rng);
  for (auto& x : B) x = d(rng);
  d1::sgemm_simd(M, N, K, A.data(), B.data(), C1.data());
  s2s::ThreadPool pool(4);
  d1::sgemm_threads(M, N, K, A.data(), B.data(), C2.data(), pool);
  CHECK_ALLCLOSE(C2.data(), C1.data(), C1.size(), 1e-6, 1e-6);
}

S2S_TEST(scaling_table) {
  const int n = 768;
  std::vector<float> A(size_t(n) * n, 1.f), B(size_t(n) * n, 1.f), C(size_t(n) * n);
  std::printf("| threads | ms (N=%d) | GFLOP/s |\n|---|---|---|\n", n);
  for (unsigned t = 1; t <= std::thread::hardware_concurrency(); t *= 2) {
    s2s::ThreadPool pool(t);
    auto tm = s2s::time_fn([&] { d1::sgemm_threads(n, n, n, A.data(), B.data(), C.data(), pool); }, 1, 3);
    std::printf("| %u | %.1f | %.1f |\n", t, tm.median_ms, 2.0 * n * n * n / (tm.median_ms * 1e6));
  }
  CHECK(true);
}

S2S_TEST_MAIN()
