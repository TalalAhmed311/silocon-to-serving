#include <cmath>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int N, int C, float offset) {
  auto x = s2s::random_vec<float>(size_t(N) * C, offset - 1, offset + 1, 1);
  auto g = s2s::random_vec<float>(size_t(C), 0.5, 1.5, 2), b = s2s::random_vec<float>(size_t(C), -1, 1, 3);
  std::vector<float> want(x.size());
  for (int c = 0; c < C; ++c) {
    double mu = 0, var = 0;
    for (int r = 0; r < N; ++r) mu += x[size_t(r) * C + c];
    mu /= N;
    for (int r = 0; r < N; ++r) { const double d = x[size_t(r) * C + c] - mu; var += d * d; }
    var /= N;
    for (int r = 0; r < N; ++r) want[size_t(r) * C + c] = float((x[size_t(r) * C + c] - mu) / std::sqrt(var + 1e-5) * g[size_t(c)] + b[size_t(c)]);
  }
  s2s::DeviceBuffer<float> dx(x), dg(g), db(b), dy(x.size());
  solve(dx.get(), dg.get(), db.get(), dy.get(), N, C, 1e-5f);
  CUDA_CHECK_LAUNCH();
  auto got = dy.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-3, 1e-3);
}

S2S_TEST(shapes) { check(1, 1, 0); check(64, 33, 0); check(4096, 256, 0); }
S2S_TEST(offset_mean) { check(1024, 64, 100.f); }   // the shift keeps E[d²] − E[d]² well-conditioned

S2S_TEST_MAIN()
