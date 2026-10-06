#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int N, int M, int r0, int r1, int c0, int c1) {
  auto in = s2s::random_vec<int>(size_t(N) * M, -10, 10, unsigned(N * 7 + M));
  long long want = 0;
  for (int r = r0; r <= r1; ++r)
    for (int c = c0; c <= c1; ++c) want += in[size_t(r) * M + c];
  s2s::DeviceBuffer<int> din(in), dout(1);
  solve(din.get(), dout.get(), N, M, r0, r1, c0, c1);
  CUDA_CHECK_LAUNCH();
  CHECK_EQ(dout.download()[0], int(want));
}

S2S_TEST(rects) {
  check(1, 1, 0, 0, 0, 0);
  check(7, 9, 2, 5, 1, 1);
  check(100, 37, 0, 99, 0, 36);
  check(2048, 3000, 17, 2000, 5, 2999);
}

int main() { return s2s::run_all(); }
