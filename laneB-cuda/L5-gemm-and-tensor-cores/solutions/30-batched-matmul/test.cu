#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static std::vector<float> ref_gemm(const std::vector<float>& a, const std::vector<float>& b, int M, int N, int K, size_t ao = 0, size_t bo = 0) {
  std::vector<float> c(size_t(M) * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double s = 0;
      for (int k = 0; k < K; ++k) s += double(a[ao + size_t(m) * K + k]) * b[bo + size_t(k) * N + n];
      c[size_t(m) * N + n] = float(s);
    }
  return c;
}

static void check(int batch, int M, int N, int K) {
  auto a = s2s::random_vec<float>(size_t(batch) * M * K, -1, 1, 1), b = s2s::random_vec<float>(size_t(batch) * K * N, -1, 1, 2);
  std::vector<float> want;
  for (int i = 0; i < batch; ++i) { auto c = ref_gemm(a, b, M, N, K, size_t(i) * M * K, size_t(i) * K * N); want.insert(want.end(), c.begin(), c.end()); }
  s2s::DeviceBuffer<float> A(a), B(b), C(want.size());
  solve(A.get(), B.get(), C.get(), batch, M, N, K);
  CUDA_CHECK_LAUNCH();
  auto got = C.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-4);
}

S2S_TEST(shapes) { check(1, 5, 7, 3); check(8, 64, 64, 64); check(3, 100, 33, 129); check(32, 16, 16, 128); }

S2S_TEST_MAIN()
