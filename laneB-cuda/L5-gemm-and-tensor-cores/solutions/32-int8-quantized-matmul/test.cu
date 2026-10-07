#include <cmath>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int M, int N, int K) {
  auto a = s2s::random_vec<int>(size_t(M) * K, -128, 127, 1), b = s2s::random_vec<int>(size_t(K) * N, -128, 127, 2);
  std::vector<int8_t> a8(a.begin(), a.end()), b8(b.begin(), b.end());
  const float sA = 0.02f, sB = 0.03f, sC = 0.5f * std::sqrt(float(K));     // keeps most outputs inside int8 range
  std::vector<int> want(size_t(M) * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      long long s = 0;
      for (int k = 0; k < K; ++k) s += int(a8[size_t(m) * K + k]) * int(b8[size_t(k) * N + n]);
      const float v = std::nearbyint(float(s) * (sA * sB / sC));
      want[size_t(m) * N + n] = int(std::fmin(std::fmax(v, -128.f), 127.f));
    }
  s2s::DeviceBuffer<int8_t> A(a8), B(b8), C(want.size());
  solve(A.get(), B.get(), C.get(), M, N, K, sA, sB, sC);
  CUDA_CHECK_LAUNCH();
  auto c8 = C.download();
  std::vector<int> got(c8.begin(), c8.end());
  // int32 accumulation is exact; the float requantisation is the same expression on both sides → exact match expected
  // (allow ±1 for a value landing exactly on .5 after different float rounding of the scale product).
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 1.0);
}

S2S_TEST(shapes) { check(1, 1, 4); check(32, 32, 32); check(70, 45, 100); check(128, 64, 512); }

S2S_TEST_MAIN()
