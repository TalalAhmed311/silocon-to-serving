// attention_bench.cu — naive (materialised S) vs FA-1 structure vs FA-2 forward, fp16 in/out, D = 64, B·H = 16,
// N = 1024 … 4096, full and causal (sm_75+). Prints ms, effective TFLOP/s (4·BH·N²·D FLOPs, halved when causal) and
// the speedup over naive. L6 exit check: FA-2 ≥ 5× naive at N = 4096.
#include <cstdio>

#include <d4/attention.cuh>
#include "s2s_cuda.cuh"

int main() {
  const int BH = 16;
  std::printf("GPU: %s\n| N | causal | naive ms | FA-1 ms | FA-2 ms | FA-2 TFLOP/s | FA-2 vs naive |\n|---|---|---|---|---|---|---|\n",
              s2s::gpu_name().c_str());
  for (int N : {1024, 2048, 4096}) {
    const size_t n = size_t(BH) * N * d4::AD;
    s2s::DeviceBuffer<__half> Q(n), K(n), V(n), O(n);
    Q.zero(); K.zero(); V.zero();
    s2s::DeviceBuffer<float> S(size_t(BH) * N * N), Oacc(n), M(size_t(BH) * N), L(size_t(BH) * N);
    for (bool causal : {false, true}) {
      auto tn = s2s::time_gpu([&] { d4::attn_naive(Q.get(), K.get(), V.get(), O.get(), S.get(), BH, N, causal); }, 1, 5);
      auto t1 = s2s::time_gpu([&] { d4::attn_flash1(Q.get(), K.get(), V.get(), O.get(), Oacc.get(), M.get(), L.get(), BH, N, causal); }, 1, 5);
      auto t2 = s2s::time_gpu([&] { d4::attn_flash2(Q.get(), K.get(), V.get(), O.get(), BH, N, causal); }, 2, 10);
      const double flops = 4.0 * BH * N * double(N) * d4::AD * (causal ? 0.5 : 1.0);
      std::printf("| %d | %s | %.2f | %.2f | %.2f | %.2f | %.1fx |\n", N, causal ? "yes" : "no", tn.median_ms, t1.median_ms,
                  t2.median_ms, flops / (t2.median_ms * 1e9), tn.median_ms / t2.median_ms);
    }
  }
  std::printf("Naive moves the B·H·N² score matrix through HBM 3+ times (write S, softmax read+write, read P);\n"
              "FA-2 never writes it. FA-2 here runs on CUDA cores — a tensor-core version (exercise 3 stretch) is the next 5–10×.\n");
}
