#include <functional>
#include <random>
#include <vector>

#include <s2s/check.hpp>
#include "sgemm.hpp"

using Fn = void (*)(int, int, int, const float*, const float*, float*);

static void check(Fn f, int M, int N, int K) {
  std::mt19937 rng(M * 131 + N * 7 + K);
  std::uniform_real_distribution<float> d(-1, 1);
  std::vector<float> A(size_t(M) * K), B(size_t(K) * N), C(size_t(M) * N);
  for (auto& x : A) x = d(rng);
  for (auto& x : B) x = d(rng);
  for (auto& x : C) x = d(rng);             // C += A·B: start from non-zero C
  std::vector<double> want(C.begin(), C.end());
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double s = 0;
      for (int k = 0; k < K; ++k) s += double(A[size_t(i) * K + k]) * B[size_t(k) * N + j];
      want[size_t(i) * N + j] += s;
    }
  f(M, N, K, A.data(), B.data(), C.data());
  // atol grows with K: a loose, safe bound on fp32 rounding of a K-term sum of O(1) products.
  CHECK_ALLCLOSE(C.data(), want.data(), C.size(), 1e-4, 1e-5 * K);
}

static const int shapes[][3] = {{1, 1, 1}, {3, 5, 7}, {64, 64, 64}, {65, 33, 129}, {128, 256, 96}, {17, 300, 520}};

S2S_TEST(naive) { for (auto& s : shapes) check(d1::sgemm_naive, s[0], s[1], s[2]); }
S2S_TEST(reorder) { for (auto& s : shapes) check(d1::sgemm_reorder, s[0], s[1], s[2]); }
S2S_TEST(tiled) { for (auto& s : shapes) check(d1::sgemm_tiled, s[0], s[1], s[2]); }
S2S_TEST(simd) { for (auto& s : shapes) check(d1::sgemm_simd, s[0], s[1], s[2]); }

S2S_TEST_MAIN()
