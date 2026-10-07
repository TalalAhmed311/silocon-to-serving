#include <cstdio>

#include <d4/hgemm.cuh>
#include <s2s/check.hpp>
#include "s2s_cuda.cuh"

// fp16 inputs (exactly representable), fp32 accumulation on both sides: only summation order differs.
// Tolerance: K = 512 terms of magnitude ≤ 1 → |C| ≲ 25; fp32 reordering error ≪ 1e-4 relative; 2e-3 leaves room for
// tensor-core accumulation details (documented in P5.7 §4).
static void check(int variant, int M, int N, int K) {
  if (!d4::hgemm_supported(variant, M, N, K)) { std::printf("  (skip variant %d: needs a newer GPU)\n", variant); return; }
  auto fa = s2s::random_vec<float>(size_t(M) * K, -1, 1, 1), fb = s2s::random_vec<float>(size_t(K) * N, -1, 1, 2);
  std::vector<__half> a(fa.size()), b(fb.size());
  for (size_t i = 0; i < fa.size(); ++i) a[i] = __float2half(fa[i]);
  for (size_t i = 0; i < fb.size(); ++i) b[i] = __float2half(fb[i]);
  s2s::DeviceBuffer<__half> A(a), B(b);
  s2s::DeviceBuffer<float> C(size_t(M) * N), R(size_t(M) * N);
  cublasHandle_t h;
  cublasCreate(&h);
  d4::hgemm_cublas(h, M, N, K, A.get(), B.get(), R.get());
  d4::hgemm(variant, M, N, K, A.get(), B.get(), C.get());
  CUDA_CHECK_LAUNCH();
  cublasDestroy(h);
  auto got = C.download(), want = R.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 2e-3, 2e-3);
}

S2S_TEST(variants) {
  for (int v = 0; v <= 3; ++v) { check(v, 128, 128, 32); check(v, 256, 384, 512); }
}

S2S_TEST(igemm_dp4a) {
  const int M = 70, N = 45, K = 96;
  auto a = s2s::random_vec<int>(size_t(M) * K, -128, 127, 3), b = s2s::random_vec<int>(size_t(N) * K, -128, 127, 4);
  std::vector<int8_t> a8(a.begin(), a.end()), b8(b.begin(), b.end());
  std::vector<int32_t> want(size_t(M) * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int s = 0;
      for (int k = 0; k < K; ++k) s += int(a8[size_t(m) * K + k]) * int(b8[size_t(n) * K + k]);
      want[size_t(m) * N + n] = s;
    }
  s2s::DeviceBuffer<int8_t> A(a8), Bt(b8);
  s2s::DeviceBuffer<int32_t> C(size_t(M) * N);
  d4::igemm(M, N, K, A.get(), Bt.get(), C.get());
  CUDA_CHECK_LAUNCH();
  auto got = C.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
}

S2S_TEST_MAIN()
