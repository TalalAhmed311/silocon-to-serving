#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int N, int seg_every) {
  auto vi = s2s::random_vec<int>(size_t(N), -4, 4, unsigned(N));
  std::vector<float> v(vi.begin(), vi.end());
  auto r = s2s::random_vec<int>(size_t(N), 0, seg_every - 1, unsigned(N + 1));
  std::vector<int> f(size_t(N));
  for (int i = 0; i < N; ++i) f[size_t(i)] = (r[size_t(i)] == 0) ? 1 : 0;
  std::vector<float> want(size_t(N));
  double run = 0;
  for (int i = 0; i < N; ++i) {
    if (i == 0 || f[size_t(i)]) run = 0;
    want[size_t(i)] = float(run);
    run += v[size_t(i)];
  }
  s2s::DeviceBuffer<float> dv(v), dout(size_t(N));
  s2s::DeviceBuffer<int> df(f);
  solve(dv.get(), df.get(), dout.get(), N);
  CUDA_CHECK_LAUNCH();
  auto got = dout.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);   // small ints as floats: exact
}

S2S_TEST(segments) {
  check(1, 2);
  check(100, 1);              // every element starts a segment → all zeros
  check(5000, 7);
  check(1 << 20, 3000);       // segments spanning tile boundaries
  check(1 << 20, 1 << 30);    // (almost surely) one segment = plain exclusive scan
}

int main() { return s2s::run_all(); }
