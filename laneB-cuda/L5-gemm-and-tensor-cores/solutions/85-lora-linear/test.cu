#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int M, int N, int K, int r, float alpha) {
  auto x = s2s::random_vec<float>(size_t(M) * K, -1, 1, 1), W = s2s::random_vec<float>(size_t(K) * N, -1, 1, 2);
  auto A = s2s::random_vec<float>(size_t(K) * r, -1, 1, 3), B = s2s::random_vec<float>(size_t(r) * N, -1, 1, 4);
  std::vector<float> want(size_t(M) * N);
  for (int m = 0; m < M; ++m) {
    std::vector<double> t(size_t(r), 0.0);
    for (int q = 0; q < r; ++q) for (int k = 0; k < K; ++k) t[size_t(q)] += double(x[size_t(m) * K + k]) * A[size_t(k) * r + q];
    for (int n = 0; n < N; ++n) {
      double s = 0, lo = 0;
      for (int k = 0; k < K; ++k) s += double(x[size_t(m) * K + k]) * W[size_t(k) * N + n];
      for (int q = 0; q < r; ++q) lo += t[size_t(q)] * B[size_t(q) * N + n];
      want[size_t(m) * N + n] = float(s + alpha / r * lo);
    }
  }
  s2s::DeviceBuffer<float> dx(x), dW(W), dA(A), dB(B), dy(want.size());
  solve(dx.get(), dW.get(), dA.get(), dB.get(), dy.get(), M, N, K, r, alpha);
  CUDA_CHECK_LAUNCH();
  auto got = dy.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-3);
}

S2S_TEST(shapes) { check(1, 8, 16, 1, 1); check(33, 70, 100, 8, 16); check(128, 256, 512, 16, 32); }

S2S_TEST_MAIN()
