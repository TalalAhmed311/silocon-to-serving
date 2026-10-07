#include <cmath>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int rows, int cols, float offset) {
  auto x = s2s::random_vec<float>(size_t(rows) * cols, offset - 1, offset + 1, 1);
  auto g = s2s::random_vec<float>(size_t(cols), 0.5, 1.5, 2), b = s2s::random_vec<float>(size_t(cols), -1, 1, 3);
  std::vector<float> want(x.size());
  for (int r = 0; r < rows; ++r) {
    double mu = 0, var = 0;
    for (int c = 0; c < cols; ++c) mu += x[size_t(r) * cols + c];
    mu /= cols;
    for (int c = 0; c < cols; ++c) { const double d = x[size_t(r) * cols + c] - mu; var += d * d; }
    var /= cols;
    for (int c = 0; c < cols; ++c)
      want[size_t(r) * cols + c] = float((x[size_t(r) * cols + c] - mu) / std::sqrt(var + 1e-5) * g[size_t(c)] + b[size_t(c)]);
  }
  s2s::DeviceBuffer<float> dx(x), dg(g), db(b), dy(x.size());
  solve(dx.get(), dg.get(), db.get(), dy.get(), rows, cols, 1e-5f);
  CUDA_CHECK_LAUNCH();
  auto got = dy.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-4);
}

S2S_TEST(shapes) { check(1, 2, 0); check(7, 1000, 0); check(32, 4096, 0); }
// Large offset, small variance: the naive E[x²] − E[x]² formula cancels catastrophically in fp32; Welford doesn't.
S2S_TEST(large_mean) { check(8, 4096, 1000.f); }

S2S_TEST_MAIN()
