#include <d4/gemm.cuh>
#include <s2s/check.hpp>
#include "s2s_cuda.cuh"

// Reference: cuBLAS SGEMM. fp32 with different summation orders: rtol 1e-3 relative to the magnitude of the result.
static void check(int rung, int M, int N, int K) {
  auto a = s2s::random_vec<float>(size_t(M) * K, -1, 1, 1), b = s2s::random_vec<float>(size_t(K) * N, -1, 1, 2);
  s2s::DeviceBuffer<float> A(a), B(b), C(size_t(M) * N), R(size_t(M) * N);
  cublasHandle_t h;
  cublasCreate(&h);
  d4::gemm_cublas(h, M, N, K, A.get(), B.get(), R.get());
  d4::sgemm(rung, M, N, K, A.get(), B.get(), C.get());
  CUDA_CHECK_LAUNCH();
  cublasDestroy(h);
  auto got = C.download(), want = R.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-3, 1e-3);
}

S2S_TEST(rungs_square) { for (int r = 1; r <= 7; ++r) check(r, 256, 256, 256); }
S2S_TEST(rungs_rectangular) { for (int r = 1; r <= 7; ++r) check(r, 384, 512, 136); }
S2S_TEST(small_rungs_odd_sizes) { for (int r = 1; r <= 3; ++r) check(r, 77, 33, 51); }

S2S_TEST_MAIN()
