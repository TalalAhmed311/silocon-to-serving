#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int batch, int M, int N, int K) {
  auto fa = s2s::random_vec<float>(size_t(batch) * M * K, -1, 1, 1), fb = s2s::random_vec<float>(size_t(batch) * K * N, -1, 1, 2);
  std::vector<__half> a(fa.size()), b(fb.size());
  for (size_t i = 0; i < a.size(); ++i) a[i] = __float2half(fa[i]);
  for (size_t i = 0; i < b.size(); ++i) b[i] = __float2half(fb[i]);
  std::vector<float> want(size_t(batch) * M * N);
  for (int z = 0; z < batch; ++z)
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        double s = 0;
        for (int k = 0; k < K; ++k) s += double(__half2float(a[(size_t(z) * M + m) * K + k])) * __half2float(b[(size_t(z) * K + k) * N + n]);
        want[(size_t(z) * M + m) * N + n] = float(s);
      }
  s2s::DeviceBuffer<__half> A(a), B(b), C(want.size());
  solve(A.get(), B.get(), C.get(), batch, M, N, K);
  CUDA_CHECK_LAUNCH();
  auto ch = C.download();
  std::vector<float> got(ch.size());
  for (size_t i = 0; i < ch.size(); ++i) got[i] = __half2float(ch[i]);
  // exact fp16 products, fp32 sums, ONE fp16 rounding of the output (u = 2^-11): rtol 2e-3 on |C| ≲ sqrt(K)
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 2e-3, 2e-3);
}

S2S_TEST(shapes) { check(1, 16, 16, 16); check(4, 64, 64, 64); check(3, 70, 33, 100); check(8, 128, 256, 64); }

S2S_TEST_MAIN()
