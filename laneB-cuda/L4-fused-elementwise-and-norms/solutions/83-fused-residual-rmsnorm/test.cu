#include <cmath>
#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static bool g_bench = false;

S2S_TEST(correct) {
  for (auto [rows, cols] : {std::pair{1, 4}, std::pair{9, 1000}, std::pair{32, 4096}}) {
    auto x = s2s::random_vec<float>(size_t(rows) * cols, -2, 2, 1), r = s2s::random_vec<float>(x.size(), -2, 2, 2);
    auto w = s2s::random_vec<float>(size_t(cols), 0.5, 1.5, 3);
    std::vector<float> want_r(x.size()), want_y(x.size());
    for (int i = 0; i < rows; ++i) {
      double ss = 0;
      for (int c = 0; c < cols; ++c) { const float v = x[size_t(i) * cols + c] + r[size_t(i) * cols + c]; want_r[size_t(i) * cols + c] = v; ss += double(v) * v; }
      const double inv = 1 / std::sqrt(ss / cols + 1e-6);
      for (int c = 0; c < cols; ++c) want_y[size_t(i) * cols + c] = float(want_r[size_t(i) * cols + c] * inv * w[size_t(c)]);
    }
    s2s::DeviceBuffer<float> dx(x), dr(r), dw(w), dro(x.size()), dy(x.size());
    solve(dx.get(), dr.get(), dw.get(), dro.get(), dy.get(), rows, cols, 1e-6f);
    CUDA_CHECK_LAUNCH();
    auto gr = dro.download(), gy = dy.download();
    CHECK_ALLCLOSE(gr.data(), want_r.data(), want_r.size(), 0.0, 0.0);
    CHECK_ALLCLOSE(gy.data(), want_y.data(), want_y.size(), 1e-5, 1e-6);
  }
}

S2S_TEST(bench) {      // fused: 2 reads + 2 writes; unfused (add, then norm) would be 3 + 2 row passes
  if (!g_bench) return;
  const int rows = 16384, cols = 4096;
  s2s::DeviceBuffer<float> x(size_t(rows) * cols), r(size_t(rows) * cols), w(size_t(cols)), ro(size_t(rows) * cols), y(size_t(rows) * cols);
  x.zero(); r.zero(); w.zero();
  s2s::table_header("GB/s");
  auto t = s2s::time_gpu([&] { solve(x.get(), r.get(), w.get(), ro.get(), y.get(), rows, cols, 1e-6f); });
  s2s::report("fused_add_rmsnorm_l4", "fused", double(rows) * cols, t, 16.0 * rows * cols / (t.median_ms * 1e6), "GB/s", s2s::measure_copy_gbs());
}

int main(int argc, char** argv) { g_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0; return s2s::run_all(); }
