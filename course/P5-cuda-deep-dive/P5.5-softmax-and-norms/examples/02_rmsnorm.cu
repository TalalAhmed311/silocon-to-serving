// 02_rmsnorm.cu — RMSNorm fp32 (scalar vs float4) and bf16, hidden sizes 4096 and 8192, vs copy bandwidth (sm_75+;
// bf16 math via cuda_bf16.h works on sm_75 through conversions — native bf16 arithmetic needs sm_80).
#include <cstdio>

#include <d4/norms.cuh>
#include "s2s_cuda.cuh"

int main() {
  const double copy = s2s::measure_copy_gbs();
  for (int cols : {4096, 8192}) {
    const int rows = 16384;
    s2s::DeviceBuffer<float> x(s2s::random_vec<float>(size_t(rows) * cols)), w(s2s::random_vec<float>(size_t(cols))), y(size_t(rows) * cols);
    s2s::DeviceBuffer<__nv_bfloat16> xb(size_t(rows) * cols), wb(size_t(cols)), yb(size_t(rows) * cols);
    std::printf("\nrows %d × hidden %d\n", rows, cols);
    s2s::table_header("GB/s");
    auto t1 = s2s::time_gpu([&] { d4::rmsnorm_k<float><<<rows, 256>>>(x.get(), w.get(), y.get(), cols, 1e-6f); });
    s2s::report("rmsnorm", "fp32 scalar", double(rows) * cols, t1, 8.0 * rows * cols / (t1.median_ms * 1e6), "GB/s", copy);
    auto t2 = s2s::time_gpu([&] { d4::rmsnorm(x.get(), w.get(), y.get(), rows, cols); });
    s2s::report("rmsnorm", "fp32 float4", double(rows) * cols, t2, 8.0 * rows * cols / (t2.median_ms * 1e6), "GB/s", copy);
    auto t3 = s2s::time_gpu([&] { d4::rmsnorm(xb.get(), wb.get(), yb.get(), rows, cols); });
    s2s::report("rmsnorm", "bf16 (fp32 math)", double(rows) * cols, t3, 4.0 * rows * cols / (t3.median_ms * 1e6), "GB/s", copy);
  }
}
