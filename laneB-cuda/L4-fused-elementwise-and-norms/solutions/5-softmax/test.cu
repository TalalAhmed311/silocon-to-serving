#include <algorithm>
#include <cmath>
#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void check(int rows, int cols, float lo, float hi) {
  auto x = s2s::random_vec<float>(size_t(rows) * cols, lo, hi, unsigned(rows + cols));
  std::vector<float> want(x.size());
  for (int r = 0; r < rows; ++r) {
    double m = -1e300, s = 0;
    for (int c = 0; c < cols; ++c) m = std::max(m, double(x[size_t(r) * cols + c]));
    for (int c = 0; c < cols; ++c) s += std::exp(x[size_t(r) * cols + c] - m);
    for (int c = 0; c < cols; ++c) want[size_t(r) * cols + c] = float(std::exp(x[size_t(r) * cols + c] - m) / s);
  }
  s2s::DeviceBuffer<float> dx(x), dy(x.size());
  solve(dx.get(), dy.get(), rows, cols);
  CUDA_CHECK_LAUNCH();
  auto got = dy.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-7);
}

S2S_TEST(shapes) { check(1, 1, -1, 1); check(3, 31, -5, 5); check(16, 4096, -10, 10); check(2, 32768, -10, 10); }
S2S_TEST(huge_logits) { check(4, 1000, 1000, 2000); }

// L4 exit check: rows of 4k–32k elements, GB/s and % of copy.
S2S_TEST(bench) {
  if (!g_bench) return;
  const double copy = s2s::measure_copy_gbs();
  s2s::table_header("GB/s");
  for (int cols : {4096, 8192, 32768}) {
    const int rows = (1 << 26) / cols;
    s2s::DeviceBuffer<float> x(size_t(rows) * cols), y(size_t(rows) * cols);
    x.zero();
    auto t = s2s::time_gpu([&] { solve(x.get(), y.get(), rows, cols); });
    s2s::report("softmax_l4", "online, cols=" + std::to_string(cols), double(rows) * cols, t,
                12.0 * rows * cols / (t.median_ms * 1e6), "GB/s", copy * 1.5);   // 2 reads + 1 write
  }
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
