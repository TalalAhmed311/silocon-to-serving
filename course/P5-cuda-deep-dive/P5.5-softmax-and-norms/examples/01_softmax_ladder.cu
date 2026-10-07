// 01_softmax_ladder.cu — safe 3-pass vs online 2-pass vs single-read (row in registers), in GB/s (sm_75+).
// Bytes counted: the minimum, 1 read + 1 write of the matrix; extra passes show up as lower GB/s.
// Expected: v1 < v2 < v3 for rows that fit in registers (≤ 1024 cols); for long rows v3 falls back to v2.
#include <cstdio>

#include <d4/softmax.cuh>
#include "s2s_cuda.cuh"

int main() {
  const double copy = s2s::measure_copy_gbs();
  for (auto [rows, cols] : {std::pair{65536, 1024}, std::pair{8192, 8192}, std::pair{128, 131072}}) {
    s2s::DeviceBuffer<float> x(s2s::random_vec<float>(size_t(rows) * cols, -10, 10)), y(size_t(rows) * cols);
    std::printf("\nrows %d × cols %d\n", rows, cols);
    s2s::table_header("GB/s");
    const char* names[] = {"", "v1 safe (3 reads)", "v2 online (2 reads)", "v3 row in registers (1 read)"};
    for (int v = 1; v <= 3; ++v) {
      auto t = s2s::time_gpu([&] { d4::softmax(v, x.get(), y.get(), rows, cols); });
      s2s::report("softmax", names[v], double(rows) * cols, t, 8.0 * rows * cols / (t.median_ms * 1e6), "GB/s", copy);
    }
  }
}
