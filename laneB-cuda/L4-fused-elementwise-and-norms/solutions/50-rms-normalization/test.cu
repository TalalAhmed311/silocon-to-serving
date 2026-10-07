#include <cmath>
#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;

static void check(int rows, int cols) {
  auto x = s2s::random_vec<float>(size_t(rows) * cols, -3, 3, 1), w = s2s::random_vec<float>(size_t(cols), 0.5, 1.5, 2);
  std::vector<float> want(x.size());
  for (int r = 0; r < rows; ++r) {
    double ss = 0;
    for (int c = 0; c < cols; ++c) ss += double(x[size_t(r) * cols + c]) * x[size_t(r) * cols + c];
    const double inv = 1 / std::sqrt(ss / cols + 1e-5);
    for (int c = 0; c < cols; ++c) want[size_t(r) * cols + c] = float(x[size_t(r) * cols + c] * inv * w[size_t(c)]);
  }
  s2s::DeviceBuffer<float> dx(x), dw(w), dy(x.size());
  solve(dx.get(), dw.get(), dy.get(), rows, cols, 1e-5f);
  CUDA_CHECK_LAUNCH();
  auto got = dy.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-5, 1e-6);
}

S2S_TEST(shapes) { check(1, 3); check(5, 1000); check(64, 4096); check(8, 8191); }

S2S_TEST(bench) {
  if (!g_bench) return;
  const int rows = 16384, cols = 4096;
  s2s::DeviceBuffer<float> x(size_t(rows) * cols), w(size_t(cols)), y(size_t(rows) * cols);
  x.zero(); w.zero();
  s2s::table_header("GB/s");
  auto t = s2s::time_gpu([&] { solve(x.get(), w.get(), y.get(), rows, cols, 1e-5f); });
  s2s::report("rmsnorm_l4", "block per row, float4", double(rows) * cols, t, 8.0 * rows * cols / (t.median_ms * 1e6), "GB/s", s2s::measure_copy_gbs());
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
