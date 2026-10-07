#include <cstring>
#include <string>

#include <cublas_v2.h>
#include <s2s/check.hpp>
#include "kernel.cuh"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void gemm_ref(cublasHandle_t h, int M, int N, int K, const float* A, const float* B, float* C) {
  const float one = 1.f, zero = 0.f;
  cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N);   // row-major via Cᵀ = BᵀAᵀ
}

static void check(int rung, int M, int N, int K) {
  if (!sgemm_supported(rung, M, N, K)) return;
  s2s::DeviceBuffer<float> A(s2s::random_vec<float>(size_t(M) * K, -1, 1, 1)), B(s2s::random_vec<float>(size_t(K) * N, -1, 1, 2));
  s2s::DeviceBuffer<float> C(size_t(M) * N), R(size_t(M) * N);
  C.zero();
  cublasHandle_t h;
  cublasCreate(&h);
  gemm_ref(h, M, N, K, A.get(), B.get(), R.get());
  sgemm(rung, M, N, K, A.get(), B.get(), C.get());
  CUDA_CHECK_LAUNCH();
  cublasDestroy(h);
  auto got = C.download(), want = R.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-3, 1e-3);   // fp32, different summation order
}

S2S_TEST(rung1) { check(1, 256, 256, 256); check(1, 77, 33, 51); }
S2S_TEST(rung2) { check(2, 256, 256, 256); check(2, 77, 33, 51); }
S2S_TEST(rung3) { check(3, 256, 256, 256); check(3, 77, 33, 51); }
S2S_TEST(rung4) { check(4, 256, 256, 256); check(4, 384, 512, 136); }
S2S_TEST(rung5) { check(5, 256, 256, 256); check(5, 384, 512, 136); }
S2S_TEST(rung6) { check(6, 256, 256, 256); check(6, 384, 512, 136); }
S2S_TEST(rung7) { check(7, 256, 256, 256); check(7, 384, 512, 136); }

// L5 exit check: best rung ≥ 70% of cuBLAS SGEMM.
S2S_TEST(bench) {
  if (!g_bench) return;
  cublasHandle_t h;
  cublasCreate(&h);
  for (int n : {1024, 2048, 4096}) {
    s2s::DeviceBuffer<float> A(size_t(n) * n), B(size_t(n) * n), C(size_t(n) * n);
    A.zero(); B.zero();
    auto tc = s2s::time_gpu([&] { gemm_ref(h, n, n, n, A.get(), B.get(), C.get()); });
    std::printf("\nN = %d, cuBLAS %.2f TFLOP/s\n| rung | TFLOP/s | %% cuBLAS |\n|---|---|---|\n", n, 2.0 * n * n * n / (tc.median_ms * 1e9));
    double best = 0;
    for (int r = 3; r <= 7; ++r) {
      auto t = s2s::time_gpu([&] { sgemm(r, n, n, n, A.get(), B.get(), C.get()); });
      const double pct = 100.0 * tc.median_ms / t.median_ms;
      best = pct > best ? pct : best;
      std::printf("| %d | %.2f | %.0f%% |\n", r, 2.0 * n * n * n / (t.median_ms * 1e9), pct);
    }
    std::printf("exit check (best >= 70%% of cuBLAS): %s\n", best >= 70 ? "PASS" : "not yet");
  }
  cublasDestroy(h);
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
