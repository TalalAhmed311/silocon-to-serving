#include <algorithm>
#include <functional>

#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int N, int k) {
  auto in = s2s::random_vec<float>(size_t(N), -1e4, 1e4, unsigned(N + k));
  auto want = in;
  std::sort(want.begin(), want.end(), std::greater<float>());
  want.resize(size_t(k));
  s2s::DeviceBuffer<float> din(in), dout(size_t(k));
  solve(din.get(), dout.get(), N, k);
  CUDA_CHECK_LAUNCH();
  auto got = dout.download();
  CHECK_ALLCLOSE(got.data(), want.data(), want.size(), 0.0, 0.0);
}

S2S_TEST(small_k) {
  check(1, 1);
  check(10, 3);
  check(5000, 1);
  check(1 << 20, 10);
  check(1 << 20, 1000);       // keep = 1024: shrinks by 2× per round
  check(3000, 1024);
}

int main() { return s2s::run_all(); }
