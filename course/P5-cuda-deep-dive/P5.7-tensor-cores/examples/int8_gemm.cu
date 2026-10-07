// int8_gemm.cu — int8 × int8 → int32 GEMM with __dp4a (sm_61+) vs an fp32 SGEMM of the same shape (d4 rung 6).
// dp4a does 4 multiply-adds per instruction on the CUDA cores (not tensor cores — those are IMMA, sm_72+, via WMMA
// int8 fragments or mma.sync .s8; extension exercise). Reports TOPS = 2·M·N·K / t.
#include <cstdio>

#include <d4/gemm.cuh>
#include <d4/hgemm.cuh>
#include "s2s_cuda.cuh"

int main() {
  const int n = 4096;
  s2s::DeviceBuffer<int8_t> A(size_t(n) * n), Bt(size_t(n) * n);
  s2s::DeviceBuffer<int32_t> C(size_t(n) * n);
  s2s::DeviceBuffer<float> Af(size_t(n) * n), Bf(size_t(n) * n), Cf(size_t(n) * n);
  A.zero(); Bt.zero(); Af.zero(); Bf.zero();
  const double ops = 2.0 * n * n * n;
  auto ti = s2s::time_gpu([&] { d4::igemm(n, n, n, A.get(), Bt.get(), C.get()); });
  auto tf = s2s::time_gpu([&] { d4::sgemm(6, n, n, n, Af.get(), Bf.get(), Cf.get()); });
  std::printf("| kernel | TOPS / TFLOPS |\n|---|---|\n| int8 dp4a (32×32 tiles) | %.2f |\n| fp32 SGEMM rung 6 | %.2f |\n",
              ops / (ti.median_ms * 1e9), ops / (tf.median_ms * 1e9));
  std::printf("The dp4a kernel here is a simple tiled one: register-block it like rung 5 for a fair comparison.\n");
}
