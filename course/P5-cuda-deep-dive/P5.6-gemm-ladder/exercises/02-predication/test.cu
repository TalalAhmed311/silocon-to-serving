#include <cublas_v2.h>
#include <s2s/check.hpp>
#include "kernel.cuh"
#include "s2s_cuda.cuh"

static void check(int M, int N, int K) {
  s2s::DeviceBuffer<float> A(s2s::random_vec<float>(size_t(M) * K, -1, 1, 1)), B(s2s::random_vec<float>(size_t(K) * N, -1, 1, 2));
  s2s::DeviceBuffer<float> C(size_t(M) * N), R(size_t(M) * N);
  C.zero();
  cublasHandle_t h;
  cublasCreate(&h);
  const float one = 1.f, zero = 0.f;
  cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B.get(), N, A.get(), K, &zero, R.get(), N);
  sgemm_any(M, N, K, A.get(), B.get(), C.get());
  CUDA_CHECK_LAUNCH();
  cublasDestroy(h);
  auto got = C.download(), want = R.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-3, 1e-3);
}

S2S_TEST(awkward_shapes) {
  check(1, 1, 1); check(127, 129, 7); check(300, 17, 1000); check(1, 4096, 4096); check(1000, 1000, 3);
}

S2S_TEST_MAIN()
