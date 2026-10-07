#include <algorithm>
#include <cmath>
#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cuh"
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
  softmax_rows(dx.get(), dy.get(), rows, cols);
  CUDA_CHECK_LAUNCH();
  auto got = dy.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-7);
}

S2S_TEST(shapes) { check(1, 1, -1, 1); check(5, 31, -1, 1); check(64, 1024, -5, 5); check(4, 32000, -5, 5); }
S2S_TEST(large_logits_stay_finite) { check(16, 512, 500, 1000); }   // naive exp(x) overflows here: max-subtraction matters

S2S_TEST(bench) {     // L4 exit check: report GB/s (1 read + 1 write) vs copy
  if (!g_bench) return;
  const double copy = s2s::measure_copy_gbs();
  s2s::table_header("GB/s");
  for (auto [rows, cols] : {std::pair{65536, 1024}, std::pair{4096, 32000}}) {
    s2s::DeviceBuffer<float> x(size_t(rows) * cols), y(size_t(rows) * cols);
    x.zero();
    auto t = s2s::time_gpu([&] { softmax_rows(x.get(), y.get(), rows, cols); });
    s2s::report("softmax", "yours " + std::to_string(cols) + " cols", double(rows) * cols, t,
                8.0 * rows * cols / (t.median_ms * 1e6), "GB/s", copy);
  }
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
