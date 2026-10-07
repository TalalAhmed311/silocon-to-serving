// 03_fused_residual_rmsnorm.cu — residual add then RMSNorm as two kernels vs one fused kernel (bf16, sm_75+).
// Unfused traffic: add reads x, r and writes r (3 passes), norm reads r and writes y (2) = 5 row-passes.
// Fused: reads x, r; writes r, y = 4 row-passes, and one launch fewer. Expected speedup ≈ 5/4 or a bit more.
#include <cstdio>

#include <d4/norms.cuh>
#include "s2s_cuda.cuh"

__global__ void add_inplace(const __nv_bfloat16* x, __nv_bfloat16* r, long long n) {
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
    r[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(r[i]));
}

int main() {
  const int rows = 16384, cols = 4096;
  const long long n = (long long)rows * cols;
  s2s::DeviceBuffer<__nv_bfloat16> x(size_t(n)), r(size_t(n)), w(size_t(cols)), y(size_t(n));
  x.zero(); r.zero(); w.zero();
  int sms = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  std::printf("| variant | median ms | speedup |\n|---|---|---|\n");
  auto unfused = s2s::time_gpu([&] {
    add_inplace<<<sms * 8, 256>>>(x.get(), r.get(), n);
    d4::rmsnorm(r.get(), w.get(), y.get(), rows, cols);
  });
  auto fused = s2s::time_gpu([&] { d4::fused_add_rmsnorm(x.get(), r.get(), w.get(), y.get(), rows, cols); });
  std::printf("| add + rmsnorm (2 kernels) | %.3f | 1.00 |\n| fused_add_rmsnorm | %.3f | %.2f |\n",
              unfused.median_ms, fused.median_ms, unfused.median_ms / fused.median_ms);
}
