#include <cmath>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int M, int D, int F) {
  auto x = s2s::random_vec<float>(size_t(M) * D, -1, 1, 1);
  auto Wg = s2s::random_vec<float>(size_t(D) * F, -0.2, 0.2, 2), Wu = s2s::random_vec<float>(size_t(D) * F, -0.2, 0.2, 3);
  auto Wd = s2s::random_vec<float>(size_t(F) * D, -0.2, 0.2, 4);
  std::vector<float> want(size_t(M) * D);
  for (int m = 0; m < M; ++m) {
    std::vector<double> h(size_t(F));
    for (int f = 0; f < F; ++f) {
      double g = 0, u = 0;
      for (int d = 0; d < D; ++d) { g += double(x[size_t(m) * D + d]) * Wg[size_t(d) * F + f]; u += double(x[size_t(m) * D + d]) * Wu[size_t(d) * F + f]; }
      h[size_t(f)] = g / (1 + std::exp(-g)) * u;
    }
    for (int d = 0; d < D; ++d) {
      double s = 0;
      for (int f = 0; f < F; ++f) s += h[size_t(f)] * Wd[size_t(f) * D + d];
      want[size_t(m) * D + d] = float(s);
    }
  }
  s2s::DeviceBuffer<float> dx(x), dg(Wg), du(Wu), dd(Wd), dy(want.size());
  solve(dx.get(), dg.get(), du.get(), dd.get(), dy.get(), M, D, F);
  CUDA_CHECK_LAUNCH();
  auto got = dy.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-3, 1e-4);
}

S2S_TEST(shapes) { check(1, 16, 40); check(17, 64, 172); check(64, 256, 688); }

S2S_TEST_MAIN()
