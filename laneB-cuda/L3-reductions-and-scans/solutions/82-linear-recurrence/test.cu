#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int N, double alo, double ahi) {
  auto a = s2s::random_vec<float>(size_t(N), alo, ahi, unsigned(N)), b = s2s::random_vec<float>(size_t(N), -1, 1, unsigned(N + 1));
  std::vector<float> want(size_t(N));
  double x = 0;
  for (int t = 0; t < N; ++t) { x = double(a[size_t(t)]) * x + b[size_t(t)]; want[size_t(t)] = float(x); }
  s2s::DeviceBuffer<float> da(a), db(b), dx(size_t(N));
  solve(da.get(), db.get(), dx.get(), N);
  CUDA_CHECK_LAUNCH();
  auto got = dx.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 1e-4, 1e-4);   // |a| < 1 keeps the recurrence stable
}

S2S_TEST(stable) {
  check(1, 0.0, 0.9);
  check(5000, -0.95, 0.95);
  check(1 << 20, 0.5, 0.99);   // long memory: errors from far back must stay damped
}

S2S_TEST(cumsum_special_case) {   // a = 1 → x is the inclusive prefix sum of b
  const int N = 10000;
  std::vector<float> a(size_t(N), 1.f), b(size_t(N), 1.f);
  s2s::DeviceBuffer<float> da(a), db(b), dx(size_t(N));
  solve(da.get(), db.get(), dx.get(), N);
  CUDA_CHECK_LAUNCH();
  CHECK_EQ(dx.download()[size_t(N) - 1], float(N));
}

int main() { return s2s::run_all(); }
