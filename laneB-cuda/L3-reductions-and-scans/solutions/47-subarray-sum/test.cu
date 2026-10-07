#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int N, int S, int E) {
  auto in = s2s::random_vec<int>(size_t(N), -100, 100, unsigned(N + S));
  long long want = 0;
  for (int i = S; i <= E; ++i) want += in[size_t(i)];
  s2s::DeviceBuffer<int> din(in), dout(1);
  solve(din.get(), dout.get(), N, S, E);
  CUDA_CHECK_LAUNCH();
  CHECK_EQ(dout.download()[0], int(want));
}

S2S_TEST(ranges) {
  check(1, 0, 0);
  check(10, 3, 3);
  check(1000, 0, 999);
  check(1 << 20, 12345, 1000000);
}

int main() { return s2s::run_all(); }
