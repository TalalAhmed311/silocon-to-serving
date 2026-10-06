#include <cstring>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int N) {
  auto p = s2s::random_vec<float>(size_t(N), -2, 2, 7), t = s2s::random_vec<float>(size_t(N), -2, 2, 8);
  double want = 0;
  for (int i = 0; i < N; ++i) { const double d = double(p[size_t(i)]) - t[size_t(i)]; want += d * d; }
  want /= N;
  s2s::DeviceBuffer<float> dp(p), dt(t), dm(1);
  solve(dp.get(), dt.get(), dm.get(), N);
  CUDA_CHECK_LAUNCH();
  const float a = dm.download()[0];
  CHECK_NEAR(a, want, 1e-5, 1e-6);
  solve(dp.get(), dt.get(), dm.get(), N);                    // deterministic: no atomics
  CUDA_CHECK_LAUNCH();
  CHECK_EQ(dm.download()[0], a);
}

S2S_TEST(sizes) { for (int n : {1, 2, 777, 1 << 16, (1 << 21) + 3}) check(n); }

int main() { return s2s::run_all(); }
