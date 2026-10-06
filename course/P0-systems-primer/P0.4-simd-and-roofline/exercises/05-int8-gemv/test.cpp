#include <cmath>
#include <random>
#include <vector>

#include <s2s/check.hpp>
#include "q8.hpp"

S2S_TEST(round_trip_within_half_scale) {
  std::mt19937 rng(3);
  std::normal_distribution<float> d(0, 1);
  std::vector<float> x(256);
  for (auto& v : x) v = d(rng);
  auto q = quantize(x.data(), 256);
  bool ok = true;
  for (int b = 0; b < 8; ++b)
    for (int i = 0; i < 32; ++i) {
      float back = q[size_t(b)].scale * q[size_t(b)].q[i];
      ok &= std::fabs(back - x[size_t(32 * b + i)]) <= q[size_t(b)].scale * 0.5f + 1e-7f;
    }
  CHECK(ok);
}

S2S_TEST(zero_block) {
  std::vector<float> x(32, 0.0f);
  auto q = quantize(x.data(), 32);
  CHECK_EQ(q[0].scale, 0.0f);
  bool all0 = true;
  for (int i = 0; i < 32; ++i) all0 &= q[0].q[i] == 0;
  CHECK(all0);
}

S2S_TEST(gemv_within_derived_bound) {
  const int R = 64, Cn = 256;
  std::mt19937 rng(5);
  std::normal_distribution<float> d(0, 1);
  std::vector<float> W(size_t(R) * Cn), x(Cn), y(R);
  for (auto& v : W) v = d(rng);
  for (auto& v : x) v = d(rng);
  auto Wq = quantize(W.data(), R * Cn);
  auto xq = quantize(x.data(), Cn);
  gemv_q8(Wq, R, Cn, xq, y.data());
  bool ok = true;
  for (int r = 0; r < R; ++r) {
    double ref = 0, bound = 1e-4;
    for (int c = 0; c < Cn; ++c) {
      const auto& wb = Wq[size_t(r) * (Cn / 32) + c / 32];
      const auto& xb = xq[size_t(c / 32)];
      const double xhat = double(xb.scale) * xb.q[c % 32];
      ref += double(W[size_t(r) * Cn + c]) * x[size_t(c)];
      bound += std::fabs(W[size_t(r) * Cn + c]) * xb.scale / 2 + std::fabs(xhat) * wb.scale / 2;
    }
    ok &= std::fabs(y[size_t(r)] - ref) <= bound;
  }
  CHECK(ok);
}

S2S_TEST_MAIN()
