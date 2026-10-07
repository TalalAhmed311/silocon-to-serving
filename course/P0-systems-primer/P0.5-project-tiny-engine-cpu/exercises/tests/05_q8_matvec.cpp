#include <cmath>
#include <random>
#include <vector>

#include <s2s/check.hpp>
#include "q8_matvec.hpp"

static std::string dir;

S2S_TEST(q8_matvec_within_bound) {
  const int R = 96, C = 256;
  std::mt19937 rng(11);
  std::normal_distribution<float> d(0, 1);
  std::vector<float> W(size_t(R) * C), x(C), y(R);
  for (auto& v : W) v = d(rng);
  for (auto& v : x) v = d(rng);
  auto Q = quantize_matrix(W.data(), R, C);
  matvec_q8(Q, x.data(), y.data());
  // Bound: |w x - ŵ x̂| <= |w| δx + |x̂| δw with δ = scale / 2 (round-to-nearest), summed over the row.
  // We use |x| + δx for |x̂| so the bound does not depend on the implementation's own quantization of x.
  bool ok = true;
  for (int r = 0; r < R; ++r) {
    double ref = 0, bound = 1e-3;
    for (int b = 0; b < C / 32; ++b) {
      float amax = 0;
      for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[size_t(32 * b + i)]));
      const double dx = amax / 127.0 / 2, dw = Q.scales[size_t(r) * (C / 32) + b] / 2;
      for (int i = 0; i < 32; ++i) {
        const int c = 32 * b + i;
        ref += double(W[size_t(r) * C + c]) * x[size_t(c)];
        bound += std::fabs(W[size_t(r) * C + c]) * dx + (std::fabs(x[size_t(c)]) + dx) * dw;
      }
    }
    ok &= std::fabs(y[size_t(r)] - ref) <= bound;
  }
  CHECK(ok);
  CHECK(std::fabs(y[0]) > 0);  // a starter that writes zeros does not pass by accident
}

int main(int argc, char** argv) { dir = argc > 1 ? argv[1] : ""; return s2s::run_all(); }
