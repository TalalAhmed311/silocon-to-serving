#include <cmath>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int n) {
  auto in = s2s::random_vec<float>(size_t(2) * n, -5, 5, unsigned(n));
  std::vector<float> want(size_t(n));
  for (int i = 0; i < n; ++i) { const double a = in[size_t(i)]; want[size_t(i)] = float(0.5 * a * (1 + std::erf(a / std::sqrt(2.0))) * in[size_t(n + i)]); }
  s2s::DeviceBuffer<float> din(in), dout(size_t(n));
  solve(din.get(), dout.get(), n);
  CUDA_CHECK_LAUNCH();
  auto got = dout.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-5, 1e-6);
}

S2S_TEST(sizes) { check(1); check(77); check(1 << 20); }

S2S_TEST_MAIN()
