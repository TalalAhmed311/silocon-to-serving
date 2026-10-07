#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int N, float lo) {
  auto in = s2s::random_vec<float>(size_t(N), lo, 1, unsigned(N));
  std::vector<float> want;
  for (float x : in) if (x > 0.f) want.push_back(x);
  s2s::DeviceBuffer<float> din(in), dout(size_t(N));
  s2s::DeviceBuffer<int> dc(1);
  solve(din.get(), dout.get(), dc.get(), N);
  CUDA_CHECK_LAUNCH();
  const int c = dc.download()[0];
  CHECK_EQ(c, int(want.size()));
  auto got = dout.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);   // same values, same order
}

S2S_TEST(densities) {
  check(1, -1);
  check(1000, -1);       // ~50% kept
  check(100000, -10);    // ~10% kept
  check(1 << 20, 0.5f);  // all kept
  check(5000, -1e9f);    // ~none kept
}

int main() { return s2s::run_all(); }
