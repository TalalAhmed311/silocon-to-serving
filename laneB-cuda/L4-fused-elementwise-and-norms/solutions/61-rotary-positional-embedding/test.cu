#include <cmath>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int T, int H, int D, int max_pos) {
  auto x = s2s::random_vec<float>(size_t(T) * H * D, -1, 1, 3);
  auto pos = s2s::random_vec<int>(size_t(T), 0, max_pos, 4);
  std::vector<float> want(x);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < H; ++h)
      for (int i = 0; i < D / 2; ++i) {
        const double th = pos[size_t(t)] * std::pow(10000.0, -2.0 * i / D);
        const size_t o = (size_t(t) * H + h) * D;
        const double a = x[o + i], b = x[o + i + D / 2];
        want[o + i] = float(a * std::cos(th) - b * std::sin(th));
        want[o + i + D / 2] = float(b * std::cos(th) + a * std::sin(th));
      }
  s2s::DeviceBuffer<float> dx(x);
  s2s::DeviceBuffer<int> dp(pos);
  solve(dx.get(), dp.get(), T, H, D, 10000.f);
  CUDA_CHECK_LAUNCH();
  auto got = dx.download();
  // sincosf at angles up to ~max_pos radians: absolute error grows with |angle| (≈ ulp(angle)), so atol scales too.
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-6 * max_pos + 1e-5);
}

S2S_TEST(shapes) { check(1, 1, 2, 0); check(5, 4, 64, 100); check(128, 8, 128, 8192); }

S2S_TEST(position_zero_is_identity) {
  std::vector<float> x = s2s::random_vec<float>(64);
  s2s::DeviceBuffer<float> dx(x);
  s2s::DeviceBuffer<int> dp(std::vector<int>{0});
  solve(dx.get(), dp.get(), 1, 1, 64, 10000.f);
  CUDA_CHECK_LAUNCH();
  auto got = dx.download();
  CHECK_ALLCLOSE(got.data(), x.data(), x.size(), 0.0, 0.0);
}

S2S_TEST_MAIN()
