// hgemm_bench.cu — tensor-core HGEMM variants vs cuBLAS (fp16 in, fp32 accumulate/out), N = 1024 … 8192.
// Variants: WMMA (sm_70+), mma.sync + ldmatrix (sm_80+), + cp.async 2-stage, + cp.async 3-stage. On a T4 (sm_75) only
// WMMA runs; use a g6.xlarge (L4, sm_89) for the full table. L5 exit check: best ≥ 50% of cuBLAS.
#include <cstdio>
#include <string>

#include <d4/hgemm.cuh>
#include "s2s_cuda.cuh"

int main() {
  const char* names[] = {"WMMA 16x16x16", "mma.sync m16n8k16 + ldmatrix", "+ cp.async, 2 stages", "+ cp.async, 3 stages"};
  cublasHandle_t h;
  cublasCreate(&h);
  std::printf("GPU: %s\n| N | variant | TFLOP/s | %% cuBLAS |\n|---|---|---|---|\n", s2s::gpu_name().c_str());
  for (int n : {1024, 2048, 4096, 8192}) {
    s2s::DeviceBuffer<__half> A(size_t(n) * n), B(size_t(n) * n);
    s2s::DeviceBuffer<float> C(size_t(n) * n);
    A.zero(); B.zero();
    const double flops = 2.0 * n * n * n;
    auto tc = s2s::time_gpu([&] { d4::hgemm_cublas(h, n, n, n, A.get(), B.get(), C.get()); });
    std::printf("| %d | cuBLAS GemmEx | %.1f | 100%% |\n", n, flops / (tc.median_ms * 1e9));
    for (int v = 0; v <= 3; ++v) {
      if (!d4::hgemm_supported(v, n, n, n)) continue;
      auto t = s2s::time_gpu([&] { d4::hgemm(v, n, n, n, A.get(), B.get(), C.get()); });
      std::printf("| %d | %s | %.1f | %.0f%% |\n", n, names[v], flops / (t.median_ms * 1e9), 100.0 * tc.median_ms / t.median_ms);
      s2s::report("hgemm", std::string(names[v]) + " N=" + std::to_string(n), n, t, flops / (t.median_ms * 1e9), "TFLOP/s", 0);
    }
  }
  cublasDestroy(h);
}
