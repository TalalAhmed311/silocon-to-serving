// S2S_MIN_SM 80
#include <cstring>

#include <cublas_v2.h>
#include <s2s/check.hpp>
#include "kernel.cuh"
#include <d4/common.cuh>
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void ref(cublasHandle_t h, int M, int N, int K, const __half* A, const __half* B, float* C) {
  const float one = 1.f, zero = 0.f;
  cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, CUDA_R_16F, N, A, CUDA_R_16F, K, &zero, C, CUDA_R_32F, N,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

static void check(int M, int N, int K) {
  auto fa = s2s::random_vec<float>(size_t(M) * K, -1, 1, 1), fb = s2s::random_vec<float>(size_t(K) * N, -1, 1, 2);
  std::vector<__half> a(fa.size()), b(fb.size());
  for (size_t i = 0; i < a.size(); ++i) a[i] = __float2half(fa[i]);
  for (size_t i = 0; i < b.size(); ++i) b[i] = __float2half(fb[i]);
  s2s::DeviceBuffer<__half> A(a), B(b);
  s2s::DeviceBuffer<float> C(size_t(M) * N), R(size_t(M) * N);
  C.zero();
  cublasHandle_t h;
  cublasCreate(&h);
  ref(h, M, N, K, A.get(), B.get(), R.get());
  if (d4::compute_capability() < 80) { std::printf("skip: needs sm_80+\n"); return; }
  hgemm_mma(M, N, K, A.get(), B.get(), C.get());
  CUDA_CHECK_LAUNCH();
  cublasDestroy(h);
  auto got = C.download(), want = R.download();
  // fp16 inputs are exact; products are exact in fp32; only the fp32 summation order differs → 2e-3 is generous.
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 2e-3, 2e-3);
}

S2S_TEST(shapes) { check(128, 128, 32); check(256, 384, 512); check(512, 128, 1024); }

S2S_TEST(bench) {
  if (!g_bench) return;
  cublasHandle_t h;
  cublasCreate(&h);
  const int n = 4096;
  s2s::DeviceBuffer<__half> A(size_t(n) * n), B(size_t(n) * n);
  s2s::DeviceBuffer<float> C(size_t(n) * n);
  A.zero(); B.zero();
  auto tc = s2s::time_gpu([&] { ref(h, n, n, n, A.get(), B.get(), C.get()); });
  if (d4::compute_capability() < 80) return;
  auto t = s2s::time_gpu([&] { hgemm_mma(n, n, n, A.get(), B.get(), C.get()); });
  std::printf("N=4096: yours %.1f TFLOP/s, cuBLAS %.1f → %.0f%% (L5 exit check needs >= 50%%)\n",
              2.0 * n * n * n / (t.median_ms * 1e9), 2.0 * n * n * n / (tc.median_ms * 1e9), 100 * tc.median_ms / t.median_ms);
  cublasDestroy(h);
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
