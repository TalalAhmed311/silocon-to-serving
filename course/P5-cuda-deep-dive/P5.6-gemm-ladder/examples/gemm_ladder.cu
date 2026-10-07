// gemm_ladder.cu — the one runner for all seven SGEMM rungs (d4/gemm.cuh) + cuBLAS, at N = 1024 … 8192 (sm_75+).
// Usage: ./p5.6_gemm_ladder [N ...]        default: 1024 2048 4096 8192 (rungs 1–2 skipped above 4096: too slow)
// Prints | N | rung | ms | TFLOP/s | % cuBLAS | and appends JSON lines to results/gemm.jsonl for gemm_plot.py.
// "% peak" uses the fp32 peak from P1.4 gpu_specs.yaml if you pass it as S2S_FP32_TFLOPS=<value>, else omitted.
#include <cstdio>
#include <cstdlib>
#include <string>

#include <d4/gemm.cuh>
#include "s2s_cuda.cuh"

int main(int argc, char** argv) {
  std::vector<int> sizes;
  for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
  if (sizes.empty()) sizes = {1024, 2048, 4096, 8192};
  const char* names[] = {"cuBLAS", "1 naive", "2 coalesced", "3 smem tiles", "4 1D blocking", "5 2D blocking",
                         "6 vectorized", "7 warptile + double buffer"};
  const double peak = std::getenv("S2S_FP32_TFLOPS") ? std::atof(std::getenv("S2S_FP32_TFLOPS")) : 0;
  cublasHandle_t h;
  cublasCreate(&h);
  std::printf("GPU: %s\n| N | rung | ms | TFLOP/s | %% cuBLAS | %% peak |\n|---|---|---|---|---|---|\n", s2s::gpu_name().c_str());
  for (int n : sizes) {
    s2s::DeviceBuffer<float> A(s2s::random_vec<float>(size_t(n) * n)), B(s2s::random_vec<float>(size_t(n) * n)), C(size_t(n) * n);
    const double flops = 2.0 * n * n * n;
    double cublas_ms = 0;
    for (int r = 0; r <= 7; ++r) {
      if (r >= 1 && r <= 2 && n > 4096) continue;
      if (r >= 1 && !d4::sgemm_supported(r, n, n, n)) continue;
      auto t = s2s::time_gpu([&] { if (r == 0) d4::gemm_cublas(h, n, n, n, A.get(), B.get(), C.get());
                                   else d4::sgemm(r, n, n, n, A.get(), B.get(), C.get()); }, 2, r <= 2 ? 3 : 20);
      if (r == 0) cublas_ms = t.median_ms;
      const double tf = flops / (t.median_ms * 1e9);
      std::printf("| %d | %s | %.3f | %.2f | %.0f%% | %s |\n", n, names[r], t.median_ms, tf, 100.0 * cublas_ms / t.median_ms,
                  peak > 0 ? (std::to_string(int(100 * tf / peak)) + "%").c_str() : "—");
      s2s::report("gemm", std::string(names[r]) + " N=" + std::to_string(n), n, t, tf, "TFLOP/s", peak);
    }
  }
  cublasDestroy(h);
}
