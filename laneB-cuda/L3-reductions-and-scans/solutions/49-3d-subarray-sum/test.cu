#include <s2s/check.hpp>
#include "kernel.cu"
#include "s2s_cuda.cuh"

static void check(int N, int M, int K, int d0, int d1, int r0, int r1, int c0, int c1) {
  auto in = s2s::random_vec<int>(size_t(N) * M * K, -5, 5, unsigned(N + M + K));
  long long want = 0;
  for (int d = d0; d <= d1; ++d)
    for (int r = r0; r <= r1; ++r)
      for (int c = c0; c <= c1; ++c) want += in[(size_t(d) * M + r) * K + c];
  s2s::DeviceBuffer<int> din(in), dout(1);
  solve(din.get(), dout.get(), N, M, K, d0, d1, r0, r1, c0, c1);
  CUDA_CHECK_LAUNCH();
  CHECK_EQ(dout.download()[0], int(want));
}

S2S_TEST(boxes) {
  check(1, 1, 1, 0, 0, 0, 0, 0, 0);
  check(4, 5, 6, 1, 2, 0, 4, 2, 3);
  check(64, 64, 300, 3, 60, 10, 50, 0, 299);
}

int main() { return s2s::run_all(); }
