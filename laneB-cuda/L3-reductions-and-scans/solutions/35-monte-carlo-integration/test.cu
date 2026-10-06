#include <cmath>
#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

S2S_TEST(given_samples) {
  for (int n : {1, 100, 1 << 20}) {
    auto y = s2s::random_vec<float>(size_t(n), 0, 3, unsigned(n));
    double s = 0;
    for (float v : y) s += v;
    const float a = -1.f, b = 2.5f;
    s2s::DeviceBuffer<float> dy(y), dr(1);
    solve(dy.get(), dr.get(), a, b, n);
    CUDA_CHECK_LAUNCH();
    CHECK_NEAR(dr.download()[0], (b - a) * s / n, 1e-5, 1e-5);
  }
}

S2S_TEST(device_rng_converges) {
  // ∫_0^3 x^2 dx = 9. Std error of the estimate ~ 3 * sd(x^2) / sqrt(n) ≈ 8e-3 at n = 2^24; allow 5 sigma.
  s2s::DeviceBuffer<float> dr(1);
  mc_integrate_device(dr.get(), 1LL << 24, 0.f, 3.f, 1234);
  CUDA_CHECK_LAUNCH();
  CHECK_NEAR(dr.download()[0], 9.0, 0.0, 0.05);
}

int main() { return s2s::run_all(); }
