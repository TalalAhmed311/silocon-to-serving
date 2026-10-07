#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int N, float gamma, float lambda) {
  auto r = s2s::random_vec<float>(size_t(N), -1, 1, unsigned(N)), v = s2s::random_vec<float>(size_t(N) + 1, -2, 2, unsigned(N + 1));
  auto di = s2s::random_vec<int>(size_t(N), 0, 49, unsigned(N + 2));
  std::vector<float> d(size_t(N)), want(size_t(N));
  for (int t = 0; t < N; ++t) d[size_t(t)] = di[size_t(t)] == 0 ? 1.f : 0.f;   // ~2% episode ends
  double A = 0;
  for (int t = N - 1; t >= 0; --t) {
    const double nd = 1.0 - d[size_t(t)];
    const double delta = r[size_t(t)] + gamma * v[size_t(t) + 1] * nd - v[size_t(t)];
    A = delta + gamma * lambda * nd * A;
    want[size_t(t)] = float(A);
  }
  s2s::DeviceBuffer<float> dr(r), dv(v), dd(d), da(size_t(N));
  solve(dr.get(), dv.get(), dd.get(), da.get(), N, gamma, lambda);
  CUDA_CHECK_LAUNCH();
  auto got = da.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-4);
}

S2S_TEST(gae) {
  check(1, 0.99f, 0.95f);
  check(1000, 0.99f, 0.95f);
  check(1 << 20, 0.99f, 0.95f);
  check(5000, 1.0f, 1.0f);   // undiscounted: plain reverse segmented sums of delta, cut at dones
}

int main() { return s2s::run_all(); }
