#include <cmath>
#include <cstring>

#include <cublas_v2.h>
#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void check(int M, int N, int K, float alpha, float beta) {
  auto a = s2s::random_vec<float>(size_t(M) * K, -1, 1, 1), b = s2s::random_vec<float>(size_t(K) * N, -1, 1, 2);
  auto c0 = s2s::random_vec<float>(size_t(M) * N, -1, 1, 3);
  std::vector<float> want(c0.size());
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double s = 0;
      for (int k = 0; k < K; ++k) s += double(a[size_t(m) * K + k]) * b[size_t(k) * N + n];
      want[size_t(m) * N + n] = float(alpha * s + beta * c0[size_t(m) * N + n]);
    }
  s2s::DeviceBuffer<float> A(a), B(b), C(c0);
  solve(A.get(), B.get(), C.get(), M, N, K, alpha, beta);
  CUDA_CHECK_LAUNCH();
  auto got = C.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-4);
}

S2S_TEST(shapes) { check(1, 1, 1, 1, 0); check(64, 64, 64, 1, 0); check(100, 37, 129, 2, 0.5f); check(1, 300, 77, 1, 1); }

// L5 exit check is on the D4 ladder (P5.6, >= 70% cuBLAS); here: where this any-shape kernel stands vs cuBLAS.
S2S_TEST(bench) {
  if (!g_bench) return;
  const int n = 4096;
  s2s::DeviceBuffer<float> A(size_t(n) * n), B(size_t(n) * n), C(size_t(n) * n);
  A.zero(); B.zero();
  cublasHandle_t h;
  cublasCreate(&h);
  const float one = 1.f, zero = 0.f;
  auto tc = s2s::time_gpu([&] { cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, B.get(), n, A.get(), n, &zero, C.get(), n); });
  auto t = s2s::time_gpu([&] { solve(A.get(), B.get(), C.get(), n, n, n, 1.f, 0.f); });
  std::printf("N=4096: ours %.2f TFLOP/s, cuBLAS %.2f TFLOP/s (%.0f%%)\n", 2.0 * n * n * n / (t.median_ms * 1e9),
              2.0 * n * n * n / (tc.median_ms * 1e9), 100 * tc.median_ms / t.median_ms);
  cublasDestroy(h);
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
