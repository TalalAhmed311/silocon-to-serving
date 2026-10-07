// 02_transpose_ladder.cu — naive → shared-memory tile → padded tile, vs a plain copy (sm_75+).
// Expected: naive well below copy (strided writes); smem tile much better but held back by 32-way bank conflicts on
// the column read; padded ≥ 80% of copy (the L2 exit check). Kernels: platform/kernels/include/d4/transpose.cuh.
#include <cstdio>

#include <d4/transpose.cuh>
#include "s2s_cuda.cuh"

int main() {
  const int R = 8192, C = 8192;
  s2s::DeviceBuffer<float> in(s2s::random_vec<float>(size_t(R) * C)), out(size_t(R) * C);
  const double copy = s2s::measure_copy_gbs();
  s2s::table_header("GB/s");
  const char* names[] = {"", "1 naive", "2 smem tile", "3 smem tile + pad"};
  for (int rung = 1; rung <= 3; ++rung) {
    auto t = s2s::time_gpu([&] { d4::transpose(in.get(), out.get(), R, C, rung); });
    s2s::report("transpose", names[rung], double(R) * C, t, 8.0 * R * C / (t.median_ms * 1e6), "GB/s", copy);
  }
}
